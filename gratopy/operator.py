"""Generic implementation of operators including basic arithmetic."""

from __future__ import annotations

from enum import Enum
from typing import Sequence, Any
from numbers import Number
from copy import copy, deepcopy 

import numpy as np

class OperatorArithmeticOperation(Enum):
    """Enum for operations that can be performed on operators."""
    ADDITION = "sum"
    MULTIPLICATION = "prod"

class Operator:
    """Base class for all operators."""

    def __init__(
        self,
        name: str | None = None,
        scalar: float = 1,
        state: dict[str, Any] | None = None,
        arithmetic_operation: OperatorArithmeticOperation | None = None,
        operands: list[Operator] | None = None
    ):
        if name is None:
            name = self.__class__.__name__
        self.name = name

        if state is None:
            state = {}
        self.state = state

        self._arithmetic_operation = arithmetic_operation
        if operands is None:
            operands = []
        self._operands = operands

        self._scalar = 1
        self.scalar = scalar
    
    def __repr__(self) -> str:
        scalar_repr = ""
        if self.scalar != 1:
            scalar_repr = repr(self.scalar)
            if self.scalar < 0:
                scalar_repr = f"({scalar_repr})"

        if not self.is_composite():
            if scalar_repr:
                return f"{scalar_repr}*{self.name}"
            return self.name
        
        if self._arithmetic_operation == OperatorArithmeticOperation.ADDITION:
            op_repr = " + ".join(repr(op) for op in self._operands)
            if scalar_repr:
                return f"{scalar_repr}*({op_repr})"
            return op_repr
        elif self._arithmetic_operation == OperatorArithmeticOperation.MULTIPLICATION:
            op_reprs = []
            for op in self._operands:
                if op.is_composite():
                    op_reprs.append(f"({repr(op)})")
                else:
                    op_reprs.append(repr(op))
            op_repr = "*".join(op_reprs)
            if scalar_repr:
                return f"{scalar_repr}*{op_repr}"
            return op_repr
        raise ValueError(f"Unknown arithmetic operation: {self._arithmetic_operation}")
    
    def __eq__(self, other: Any) -> bool:
        """Check equality of two operators."""
        if not isinstance(other, Operator):
            return False
        
        return all([
            type(self) == type(other),
            self.name == other.name,
            self.scalar == other.scalar,
            self.state == other.state,
            self._arithmetic_operation == other._arithmetic_operation,
            self._operands == other._operands
        ])
    
    @property
    def scalar(self) -> float:
        return self._scalar
    
    @scalar.setter
    def scalar(self, value: float):
        """Set the scalar value of the operator."""
        if self.is_composite():
            if self._arithmetic_operation == OperatorArithmeticOperation.ADDITION:
                for child_operator in self._operands:
                    child_operator.scalar *= value
            elif self._arithmetic_operation == OperatorArithmeticOperation.MULTIPLICATION:
                self._operands[0].scalar *= value
        else:
            self._scalar = value
    
    def apply_to(self, argument: Sequence):
        """Application of this operator to some given argument."""
        raise NotImplementedError("apply_to needs to be implemented in specialized subclasses")

    def is_composite(self) -> bool:
        """Check if the operator is composite."""
        return self._arithmetic_operation is not None
    
    def __add__(self, other: Operator) -> Operator:
        """Add another operator to this one."""
        if not isinstance(other, Operator):
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")
        
        if isinstance(other, _ZeroOperator):
            return self

        operands = []
        for operator in [copy(self), copy(other)]:
            if operator.is_composite() and operator._arithmetic_operation == OperatorArithmeticOperation.ADDITION:
                for child_operator in operator._operands:
                    child_operator.scalar *= operator.scalar
                    operands.append(child_operator)
            else:
                operands.append(operator)

        return Operator(
            name=None,
            scalar=1,
            arithmetic_operation=OperatorArithmeticOperation.ADDITION,
            operands=operands,
        )
    
    def __neg__(self) -> Operator:
        """Negate this operator."""
        return (-1)*self
    
    def __sub__(self, other: Operator) -> Operator:
        """Subtract another operator from this one."""
        return self + (-1)*other
    
    def __rmul__(self, other: Operator | Number | float) -> Operator:
        """Right-multiply this operator by a scalar or another operator."""
        if isinstance(other, Number):
            if other == 0:
                return ZERO
            if other == 1:
                return self
            
            operator_copy = deepcopy(self)
            operator_copy.scalar = operator_copy.scalar * other
            return operator_copy
        
        elif isinstance(other, Operator):
            return other.__mul__(self)
        
        return NotImplemented
    
    def __mul__(self, other: Operator | Sequence) -> Operator | Any:
        """Multiply this operator by another operator, or apply it to appropriate input."""
        if not isinstance(other, Operator):
            # attempt to apply the operator to the input
            if not self.is_composite():
                return self.scalar * self.apply_to(other)
            
            if self._arithmetic_operation == OperatorArithmeticOperation.ADDITION:
                return self.scalar * sum(child_op * other for child_op in self._operands)
            
            if self._arithmetic_operation == OperatorArithmeticOperation.MULTIPLICATION:
                result = other
                for child_op in reversed(self._operands):
                    result = child_op * result
                
                return self.apply_to(other)
            
            raise TypeError(f"Cannot multiply {type(other)} with {type(self)}")
        
        if isinstance(other, _ZeroOperator):
            return other
        
        if isinstance(other, _IdentityOperator):
            return self
        
        operands = []
        scalar = 1
        for operator in [copy(self), copy(other)]:
            if operator.is_composite() and operator._arithmetic_operation == OperatorArithmeticOperation.MULTIPLICATION:
                operands.extend(operator._operands)
            else:
                scalar *= operator.scalar
                operator.scalar = 1
                operands.append(operator)
        
        return Operator(
            name=None,
            scalar=scalar,
            arithmetic_operation=OperatorArithmeticOperation.MULTIPLICATION,
            operands=operands
        )


class _IdentityOperator(Operator):
    """Base class for identity operator."""
    def __mul__(self, other: Operator | Sequence) -> Operator | Any:
        """Multiplying the identity operator with another operator returns
        the other operator."""
        if isinstance(other, Operator):
            return other
        return super().__mul__(other)
        
        
    def apply_to(self, argument: Sequence) -> Sequence:
        """The identity operator does not change the input."""
        return argument

class _ZeroOperator(Operator):
    """Base class for zero operator."""
    def __add__(self, other: Operator) -> Operator:
        """Adding zero operator to any operator returns the other operator."""
        return other

    @Operator.scalar.setter
    def scalar(self, value: float):
        pass

    def apply_to(self, argument: Sequence) -> Sequence:
        """Applying the zero operator returns a zero-multiplied version of the input."""
        try:
            return 0 * argument
        except TypeError:
            pass
        
        try:
            return np.zeros_like(argument)
        except (ValueError, TypeError):
            pass

        raise TypeError(f"Cannot apply zero operator to {type(argument)}")


IDENTITY = _IdentityOperator(name="[Id]")
ZERO = _ZeroOperator(name="[0]")