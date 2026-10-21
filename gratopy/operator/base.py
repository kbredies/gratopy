"""Generic operators and expression nodes for operator algebra."""

from __future__ import annotations

from math import prod
from numbers import Integral, Number
from typing import Any, Literal, Sequence
from weakref import ReferenceType, ref

import numpy as np
import numpy.typing as npt

from gratopy.operator.space import Space
from gratopy.utilities import Numeric


def _compute_sum_shapes(
    operands: Sequence[Operator],
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    input_shape = None
    output_shape = None

    for op in operands:
        if op.input_shape is not None:
            if input_shape is not None and input_shape != op.input_shape:
                raise ValueError(
                    f"Input shape mismatch in sum: expected {input_shape}, "
                    f"but {op} has input shape {op.input_shape}"
                )
            input_shape = op.input_shape

        if op.output_shape is not None:
            if output_shape is not None and output_shape != op.output_shape:
                raise ValueError(
                    f"Output shape mismatch in sum: expected {output_shape}, "
                    f"but {op} has output shape {op.output_shape}"
                )
            output_shape = op.output_shape

    return input_shape, output_shape


def _compute_product_shapes(
    operands: Sequence[Operator],
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    for left, right in zip(operands, operands[1:]):
        if left.input_shape is not None and right.output_shape is not None:
            if left.input_shape != right.output_shape:
                raise ValueError(
                    f"Shape mismatch in composition: {left} expects input "
                    f"{left.input_shape}, but {right} produces {right.output_shape}"
                )

    input_shape = operands[-1].input_shape
    output_shape = operands[0].output_shape
    return input_shape, output_shape


def _common_space(spaces: Sequence[Space | None], description: str) -> Space | None:
    """Return the common specified space, raising if specified spaces differ."""
    common = None
    for space in spaces:
        if space is None:
            continue
        if common is not None and space != common:
            raise ValueError(f"{description} mismatch: {common} and {space}")
        common = space
    return common


def _validate_product_spaces(operands: Sequence[Operator]) -> None:
    for left, right in zip(operands, operands[1:]):
        _common_space(
            [left.domain, right.codomain],
            f"Space mismatch in composition of {left} and {right}",
        )


def _scalar_repr(scalar: Numeric) -> str:
    scalar_repr = repr(scalar)
    if scalar < 0:
        return f"({scalar_repr})"
    return scalar_repr


def _with_return_event(result: Any, return_event: bool) -> Any:
    if not return_event:
        return result
    return result, list(getattr(result, "events", []))


class Operator:
    """Base class for operators and operator expression nodes."""

    def __init__(
        self,
        name: str | None = None,
        state: dict[str, Any] | None = None,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        domain: Space | None = None,
        codomain: Space | None = None,
    ):
        self.name = self.__class__.__name__ if name is None else name
        self.state = {} if state is None else state
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._domain: Space | None = None
        self._codomain: Space | None = None
        self.domain = domain
        self.codomain = codomain
        self._adjoint_ref: ReferenceType[AdjointOperator] | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Return serializable state without the weak adjoint memoization."""
        state = self.__dict__.copy()
        state["_adjoint_ref"] = None
        return state

    @property
    def domain(self) -> Space | None:
        """Space of the operator's inputs, or ``None`` if unspecified.

        Solvers treat an unspecified space as Euclidean.
        """
        if self._domain is not None:
            return self._domain
        return self._default_domain()

    @domain.setter
    def domain(self, space: Space | None) -> None:
        self._domain = self._checked_space(space, self.input_shape, "domain")

    @property
    def codomain(self) -> Space | None:
        """Space of the operator's outputs, or ``None`` if unspecified."""
        if self._codomain is not None:
            return self._codomain
        return self._default_codomain()

    @codomain.setter
    def codomain(self, space: Space | None) -> None:
        self._codomain = self._checked_space(space, self.output_shape, "codomain")

    @staticmethod
    def _checked_space(
        space: Space | None,
        shape: tuple[int, ...] | None,
        description: str,
    ) -> Space | None:
        if space is None:
            return None
        if not isinstance(space, Space):
            raise TypeError(f"{description} must be a Space, got {type(space)}")
        if shape is not None and space.shape != tuple(shape):
            raise ValueError(
                f"The {description} shape {space.shape} does not match the "
                f"operator shape {tuple(shape)}"
            )
        return space

    def _default_domain(self) -> Space | None:
        """Return the domain implied by the operator's structure."""
        return None

    def _default_codomain(self) -> Space | None:
        """Return the codomain implied by the operator's structure."""
        return None

    def _repr_name_(self) -> str:
        """Return the operator name used in string representations."""
        return self.name

    def __repr__(self) -> str:
        return self._repr_name_()

    def __eq__(self, other: Any) -> bool:
        """Check structural equality of two operators."""
        if not isinstance(other, Operator):
            return False
        return all(
            [
                type(self) is type(other),
                self.name == other.name,
                self.state == other.state,
                self.input_shape == other.input_shape,
                self.output_shape == other.output_shape,
            ]
        )

    @property
    def T(self) -> Operator:
        """Return a memoized adjoint wrapper referencing this operator."""
        if self._adjoint_ref is not None:
            adjoint = self._adjoint_ref()
            if adjoint is not None:
                return adjoint

        adjoint = AdjointOperator(self)
        self._adjoint_ref = ref(adjoint)
        return adjoint

    def norm_estimate(
        self,
        number_iterations: int = 50,
        dtype: npt.DTypeLike = np.float32,
        input_shape: tuple[int, ...] | None = None,
        algorithm: Literal["poweriteration", "naive"] = "poweriteration",
        queue: Any | None = None,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> float:
        """Estimate the operator norm.

        Parameters
        ----------
        number_iterations:
            Positive number of power-iteration updates.
        dtype:
            Data type of the random starting vector.
        input_shape:
            Shape of the starting vector. Defaults to :attr:`input_shape`.
        algorithm:
            ``"poweriteration"`` applies power iteration to ``self.T * self``,
            normalizing in the norm of :attr:`domain` (Euclidean if
            unspecified). The norm is taken with respect to the operator's
            spaces; if :attr:`T` is the adjoint in these spaces, each iterate
            is a lower bound of the norm (up to rounding).
            ``"naive"`` combines power-iteration estimates for expression leaves
            using the triangle inequality for sums and submultiplicativity for
            compositions. Since leaf norms are themselves estimates, the result
            is a heuristic and not a guaranteed upper bound.
        queue:
            Optional backend-specific execution queue forwarded to applications.
        rng:
            Random-number generator used to construct the starting vector.
        """
        if algorithm not in {"poweriteration", "naive"}:
            raise ValueError(
                "Unknown norm-estimation algorithm. Expected 'poweriteration' or 'naive'."
            )
        if (
            not isinstance(number_iterations, Integral)
            or isinstance(number_iterations, bool)
            or number_iterations < 1
        ):
            raise ValueError("number_iterations must be a positive integer")
        number_iterations = int(number_iterations)

        exact_norm = self._exact_norm()
        if exact_norm is not None:
            return exact_norm

        if algorithm == "naive":
            naive_norm = self._naive_norm_estimate(
                number_iterations=number_iterations,
                dtype=dtype,
                queue=queue,
                rng=rng,
                **kwargs,
            )
            if naive_norm is not None:
                return naive_norm

        return self._power_iteration_norm_estimate(
            number_iterations=number_iterations,
            dtype=dtype,
            input_shape=input_shape,
            queue=queue,
            rng=rng,
            **kwargs,
        )

    def _exact_norm(self) -> float | None:
        """Return an exact norm when one is available for this operator."""
        return None

    def _naive_norm_estimate(self, **kwargs: Any) -> float | None:
        """Return a structural norm estimate, or defer to power iteration."""
        return None

    def _power_iteration_norm_estimate(
        self,
        number_iterations: int,
        dtype: npt.DTypeLike,
        input_shape: tuple[int, ...] | None,
        queue: Any | None,
        rng: np.random.Generator | None,
        **kwargs: Any,
    ) -> float:
        if input_shape is None:
            input_shape = self.input_shape
        if input_shape is None:
            raise ValueError(
                "Cannot estimate the norm without an input shape. Set "
                "operator.input_shape or pass input_shape explicitly."
            )

        if rng is None:
            rng = np.random.default_rng()

        # .T is self-adjoint with respect to the domain's inner product, so
        # normalizing in that norm makes every iterate a lower bound of the norm.
        space = self.domain if self.domain is not None else Space(tuple(input_shape))
        x = np.asarray(rng.standard_normal(input_shape), dtype=dtype)
        x_norm = space.norm(x)
        if x_norm == 0:
            return 0.0
        x = x / x_norm

        estimate = 0.0
        adjoint = self.T
        for _ in range(number_iterations):
            y = (
                self.apply_to(x, queue=queue, **kwargs)
                if queue is not None
                else self.apply_to(x, **kwargs)
            )
            z = (
                adjoint.apply_to(y, queue=queue, **kwargs)
                if queue is not None
                else adjoint.apply_to(y, **kwargs)
            )
            z_norm = space.norm(z)
            if z_norm == 0:
                return 0.0

            # Since x is normalized, ||(A.T * A)x|| converges to the dominant
            # eigenvalue of A.T * A. Its square root is the operator norm.
            estimate = float(np.sqrt(z_norm))
            x = z / z_norm

        return estimate

    def apply_to(
        self,
        argument: Any,
        output: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply the operator in its forward direction."""
        raise NotImplementedError(
            "apply_to needs to be implemented in specialized subclasses"
        )

    def apply_adjoint_to(
        self,
        argument: Any,
        output: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply the adjoint operator."""
        raise NotImplementedError(
            "apply_adjoint_to needs to be implemented in specialized subclasses"
        )

    def is_composite(self) -> bool:
        """Return whether this is a sum or composition expression node."""
        return isinstance(self, (SumOperator, CompositionOperator))

    def __add__(self, other: Operator) -> Operator:
        """Add another operator to this one."""
        if not isinstance(other, Operator):
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")
        if isinstance(other, _ZeroOperator):
            return self
        if isinstance(self, _ZeroOperator):
            return other
        return SumOperator.create(self, other)

    def __neg__(self) -> Operator:
        """Negate this operator."""
        return ScaledOperator.create(-1, self)

    def __sub__(self, other: Operator) -> Operator:
        """Subtract another operator from this one."""
        return self + (-other)

    def __rmul__(self, other: Numeric) -> Operator:  # type: ignore[misc]
        """Multiply this operator by a scalar."""
        if not isinstance(other, Number):
            return NotImplemented
        return ScaledOperator.create(other, self)  # type: ignore[arg-type]

    def __mul__(self, other: Operator | Any) -> Operator | Any:
        """Compose, scale, or apply this operator."""
        if isinstance(other, Operator):
            if isinstance(self, _ZeroOperator) or isinstance(other, _ZeroOperator):
                return ZERO
            if isinstance(self, _IdentityOperator):
                return other
            if isinstance(other, _IdentityOperator):
                return self
            return CompositionOperator.create(self, other)

        if isinstance(other, Number):
            return ScaledOperator.create(other, self)  # type: ignore[arg-type]

        return self.apply_to(other)


class AdjointOperator(Operator):
    """Adjoint view of an underlying operator."""

    def __init__(self, operand: Operator):
        super().__init__(
            name=f"{operand.name}.T",
            input_shape=operand.output_shape,
            output_shape=operand.input_shape,
        )
        self._operand = operand

    @property
    def operand(self) -> Operator:
        return self._operand

    def _default_domain(self) -> Space | None:
        return self.operand.codomain

    def _default_codomain(self) -> Space | None:
        return self.operand.domain

    @property
    def T(self) -> Operator:
        """The adjoint of an adjoint is the underlying operator."""
        return self.operand

    def __repr__(self) -> str:
        if isinstance(self.operand, (SumOperator, CompositionOperator)):
            return f"({self.operand!r}).T"
        return f"{self.operand!r}.T"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AdjointOperator) and self.operand == other.operand

    def _exact_norm(self) -> float | None:
        return self.operand._exact_norm()

    def _naive_norm_estimate(self, **kwargs: Any) -> float | None:
        return self.operand._naive_norm_estimate(**kwargs)

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        return self.operand.apply_adjoint_to(argument, output=output, **kwargs)

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self.operand.apply_to(argument, output=output, **kwargs)


class ScaledOperator(Operator):
    """An operator multiplied by a scalar."""

    def __init__(self, scalar: Numeric, operand: Operator):
        super().__init__(
            name=operand.name,
            input_shape=operand.input_shape,
            output_shape=operand.output_shape,
        )
        self._scalar = scalar
        self._operand = operand

    @property
    def scalar(self) -> Numeric:
        return self._scalar

    @property
    def operand(self) -> Operator:
        return self._operand

    def _default_domain(self) -> Space | None:
        return self.operand.domain

    def _default_codomain(self) -> Space | None:
        return self.operand.codomain

    @classmethod
    def create(cls, scalar: Numeric, operand: Operator) -> Operator:
        if scalar == 0 or isinstance(operand, _ZeroOperator):
            return ZERO
        if scalar == 1:
            return operand
        if isinstance(operand, ScaledOperator):
            return cls.create(scalar * operand.scalar, operand.operand)
        return cls(scalar, operand)

    def __repr__(self) -> str:
        operand_repr = repr(self.operand)
        if isinstance(self.operand, SumOperator):
            operand_repr = f"({operand_repr})"
        return f"{_scalar_repr(self.scalar)}*{operand_repr}"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, ScaledOperator)
            and self.scalar == other.scalar
            and self.operand == other.operand
        )

    def _exact_norm(self) -> float | None:
        operand_norm = self.operand._exact_norm()
        if operand_norm is None:
            return None
        return abs(float(self.scalar)) * operand_norm

    def _naive_norm_estimate(self, **kwargs: Any) -> float:
        return abs(float(self.scalar)) * self.operand.norm_estimate(
            algorithm="naive", **kwargs
        )

    def _apply_scaled(
        self,
        argument: Any,
        output: Any | None,
        adjoint: bool,
        **kwargs: Any,
    ) -> Any:
        return_event = bool(kwargs.pop("return_event", False))
        apply = self.operand.apply_adjoint_to if adjoint else self.operand.apply_to
        result = apply(argument, output=output, **kwargs)

        scalar = np.conjugate(self.scalar) if adjoint else self.scalar
        try:
            scalar = result.dtype.type(scalar)
        except AttributeError:
            pass
        result *= scalar
        return _with_return_event(result, return_event)

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        return self._apply_scaled(argument, output, False, **kwargs)

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self._apply_scaled(argument, output, True, **kwargs)


class SumOperator(Operator):
    """A sum of operators with compatible input and output shapes."""

    def __init__(self, operands: tuple[Operator, ...]):
        input_shape, output_shape = _compute_sum_shapes(operands)
        _common_space([op.domain for op in operands], "Domain in sum")
        _common_space([op.codomain for op in operands], "Codomain in sum")
        super().__init__(input_shape=input_shape, output_shape=output_shape)
        self._operands = operands

    @property
    def operands(self) -> tuple[Operator, ...]:
        return self._operands

    def _default_domain(self) -> Space | None:
        return _common_space([op.domain for op in self.operands], "Domain in sum")

    def _default_codomain(self) -> Space | None:
        return _common_space([op.codomain for op in self.operands], "Codomain in sum")

    @classmethod
    def create(cls, *operators: Operator) -> Operator:
        operands: list[Operator] = []
        for operator in operators:
            if isinstance(operator, _ZeroOperator):
                continue
            if isinstance(operator, SumOperator):
                operands.extend(operator.operands)
            else:
                operands.append(operator)
        if not operands:
            return ZERO
        if len(operands) == 1:
            return operands[0]
        return cls(tuple(operands))

    def __repr__(self) -> str:
        return " + ".join(repr(op) for op in self.operands)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SumOperator) and self.operands == other.operands

    def _naive_norm_estimate(self, **kwargs: Any) -> float:
        return sum(
            operand.norm_estimate(algorithm="naive", **kwargs)
            for operand in self.operands
        )

    def _apply_sum(
        self,
        argument: Any,
        output: Any | None,
        adjoint: bool,
        **kwargs: Any,
    ) -> Any:
        return_event = bool(kwargs.pop("return_event", False))

        def apply(operand: Operator, child_output: Any | None = None) -> Any:
            if adjoint:
                return operand.apply_adjoint_to(
                    argument,
                    output=child_output,
                    **kwargs,
                )
            return operand.apply_to(argument, output=child_output, **kwargs)

        result = apply(self.operands[0], output)
        for operand in self.operands[1:]:
            result += apply(operand)
        return _with_return_event(result, return_event)

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        return self._apply_sum(argument, output, False, **kwargs)

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self._apply_sum(argument, output, True, **kwargs)


class CompositionOperator(Operator):
    """A composition ``A*B`` representing ``A(B(x))``."""

    def __init__(self, operands: tuple[Operator, ...]):
        input_shape, output_shape = _compute_product_shapes(operands)
        _validate_product_spaces(operands)
        super().__init__(input_shape=input_shape, output_shape=output_shape)
        self._operands = operands

    @property
    def operands(self) -> tuple[Operator, ...]:
        return self._operands

    def _default_domain(self) -> Space | None:
        return self.operands[-1].domain

    def _default_codomain(self) -> Space | None:
        return self.operands[0].codomain

    @classmethod
    def create(cls, *operators: Operator) -> Operator:
        operands: list[Operator] = []
        for operator in operators:
            if isinstance(operator, _ZeroOperator):
                return ZERO
            if isinstance(operator, _IdentityOperator):
                continue
            if isinstance(operator, CompositionOperator):
                operands.extend(operator.operands)
            else:
                operands.append(operator)
        if not operands:
            return IDENTITY
        if len(operands) == 1:
            return operands[0]
        return cls(tuple(operands))

    def __repr__(self) -> str:
        representations = []
        for operand in self.operands:
            operand_repr = repr(operand)
            if isinstance(operand, SumOperator):
                operand_repr = f"({operand_repr})"
            representations.append(operand_repr)
        return "*".join(representations)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CompositionOperator) and self.operands == other.operands

    def _naive_norm_estimate(self, **kwargs: Any) -> float:
        return prod(
            operand.norm_estimate(algorithm="naive", **kwargs)
            for operand in self.operands
        )

    def _apply_composition(
        self,
        argument: Any,
        output: Any | None,
        adjoint: bool,
        **kwargs: Any,
    ) -> Any:
        return_event = bool(kwargs.pop("return_event", False))
        result = argument
        application_order = self.operands if adjoint else tuple(reversed(self.operands))
        for index, operand in enumerate(application_order):
            child_output = output if index == len(application_order) - 1 else None
            apply = operand.apply_adjoint_to if adjoint else operand.apply_to
            result = apply(result, output=child_output, **kwargs)
        return _with_return_event(result, return_event)

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        return self._apply_composition(argument, output, False, **kwargs)

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self._apply_composition(argument, output, True, **kwargs)


class _IdentityOperator(Operator):
    """Shape-agnostic identity operator."""

    @property
    def T(self) -> Operator:
        return self

    def _exact_norm(self) -> float:
        return 1.0

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        result = 1 * argument
        if output is not None:
            output[...] = result
            return output
        return result

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self.apply_to(argument, output=output, **kwargs)


class _ZeroOperator(Operator):
    """Shape-agnostic zero operator."""

    @property
    def T(self) -> Operator:
        return self

    def _exact_norm(self) -> float:
        return 0.0

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        if output is not None:
            output[...] = 0
            return output

        try:
            return 0 * argument  # type: ignore
        except TypeError:
            pass

        try:
            return np.zeros_like(argument)
        except (ValueError, TypeError):
            pass

        raise TypeError(f"Cannot apply zero operator to {type(argument)}")

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self.apply_to(argument, output=output, **kwargs)


IDENTITY = _IdentityOperator(name="[Id]")
ZERO = _ZeroOperator(name="[0]")
