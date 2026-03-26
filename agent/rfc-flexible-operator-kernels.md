# RFC: Slim kernel-spec architecture for operator-era gratopy

Status: draft  
Author: AI-assisted design note  
Date: 2026-03-25

## 1. Summary

This RFC proposes a **minimal, release-oriented** design for preliminary operator support in gratopy.

The key decisions are:

- keep `Radon` as the **public concrete operator**,
- **do not** introduce a public `AbstractRadon`,
- keep `Operator` backend-agnostic and mostly unchanged,
- introduce a small internal **`_OpenCLOperator`** base for shared OpenCL execution plumbing,
- introduce a small **`OpenCLKernelSpec`** for configurable kernel sources,
- move `Radon` away from depending on `ProjectionSettings` at execution time,
- let `Radon` own its geometry state and launch kernels directly.

This is intended to be simple enough to implement in the next few days while still solving the main prototyping problem: **swapping out kernels without rewriting central internals**.

---

## 2. Motivation

The current operator syntax is promising, but for a release that advertises preliminary operator support, the implementation should be a little cleaner and a lot more flexible.

### Current good parts

- `Operator` already provides useful arithmetic and composition.
- `Radon` already exposes the desired user-facing syntax.
- Shape propagation works.
- The operator test suite is already in decent shape.

### Current pain points

- `Radon` still routes execution through `ProjectionSettings`.
- Kernel selection and compilation are tied to the legacy code path.
- Prototyping custom kernel implementations is awkward.
- Placeholder support is incomplete.
- `Fanbeam` is not implemented yet, so a large operator hierarchy would be premature.

### Design constraint

This should not become a large abstraction exercise right before a release.

The release goal is explicitly:

> Preliminary but solid operator syntax for Radon, with a flexible kernel interface for experimentation.

---

## 3. Main design decisions

## 3.1 Keep `Radon` concrete

`Radon` should remain the public class users instantiate.

We do **not** introduce:

- `AbstractRadon` as a public class,
- a public `AbstractGratopyOperator`,
- any deeper geometry hierarchy unless it becomes necessary later.

Reason: there is currently only one real operator implementation (`Radon`). `Fanbeam` is still a stub, so a large hierarchy would mostly anticipate future structure instead of simplifying present code.

## 3.2 Keep `Operator` generic

`Operator` in `gratopy/operator/base.py` is currently a clean algebraic abstraction.

It should stay responsible for:

- arithmetic,
- shape propagation,
- composite application semantics.

It should **not** become responsible for:

- OpenCL queue inference,
- kernel specs,
- program caches,
- device buffers,
- contiguity-dependent kernel selection.

Those concerns belong in a gratopy-specific OpenCL layer.

## 3.3 Add a small internal `_OpenCLOperator`

Shared OpenCL execution behavior between `Radon` now and `Fanbeam` later should live in a **private** internal base class, tentatively named `_OpenCLOperator`.

This base should only cover genuinely shared concerns:

- queue inference,
- NumPy/array-like coercion to `clarray.Array`,
- output allocation,
- kernel program compilation/cache,
- kernel lookup from a kernel spec,
- optional event-return behavior.

It should **not** know Radon geometry formulas.

## 3.4 Add a small `OpenCLKernelSpec`

A concrete operator should be able to specify which kernels it uses via a small declarative object.

This solves the main prototyping problem without needing a broad plugin framework.

---

## 4. Goals

## Primary goals

1. Make `Radon` operator support release-ready.
2. Support configurable kernel source paths for experimentation.
3. Decouple operator execution from `ProjectionSettings`.
4. Keep the implementation small and understandable.
5. Leave a clean path for a future `Fanbeam` implementation.

## Secondary goals

6. Implement proper placeholder handling (`FULL`, `VALID`).
7. Improve composite operator usability where feasible.
8. Preserve compatibility with the shipped kernel naming scheme.

---

## 5. Non-goals

This RFC does **not** require before release:

- a full `Fanbeam` operator implementation,
- a public abstract gratopy operator hierarchy,
- generic plugin contracts for arbitrary kernel signatures,
- migration of reconstruction algorithms to generic operators,
- a rewrite of the legacy `ProjectionSettings` API.

---

## 6. Proposed architecture

## 6.1 Public classes

### `Operator`
Remains the generic mathematical operator base.

### `Radon`
Remains the public Radon transform operator.

### `OpenCLKernelSpec`
New small public helper describing the kernel source bundle for an OpenCL-backed operator.

`OpenCLKernelSpec` should likely be public because it is part of the prototyping interface.

## 6.2 Internal classes/helpers

### `_OpenCLOperator`
New internal base class for gratopy OpenCL-backed operators.

Responsibilities:

- queue inference,
- argument coercion,
- output allocation,
- program caching,
- kernel lookup.

### Module-level geometry helpers

Geometry-specific helpers should stay close to `Radon`, either:

- as private methods on `Radon`, or
- as module-level private helpers in `gratopy/operator/projection.py`.

No separate public geometry base class is needed.

---

## 7. `OpenCLKernelSpec`

## 7.1 Minimal shape

For the release, `OpenCLKernelSpec` should stay intentionally small.

Suggested form:

```python
@dataclass(frozen=True)
class OpenCLKernelSpec:
    paths: tuple[str, ...]
    base_name: str
    build_options: tuple[str, ...] = ()
```

This is enough for:

- shipped kernels,
- custom local kernels,
- future extension if needed.

## 7.2 Why keep it this small?

Because anything larger would likely be speculative right now.

In particular, I would **not** add yet:

- naming scheme strategy objects,
- argument contract objects,
- templating strategy objects,
- kernel source string support unless it is immediately useful.

If later needed, these can be added without breaking the basic design.

## 7.3 Construction helpers

A few convenience constructors are useful:

```python
OpenCLKernelSpec.from_path(path, base_name, build_options=())
OpenCLKernelSpec.from_paths(paths, base_name, build_options=())
```

For now, file-path-based prototyping is sufficient.

---

## 8. `_OpenCLOperator`

## 8.1 Scope

`_OpenCLOperator` should be a **small private base**, not a new abstraction layer for its own sake.

Suggested responsibilities:

- infer queue,
- coerce array-like input to `clarray.Array`,
- allocate output arrays in the right shape/order/dtype,
- compile and cache programs from `kernel_spec`,
- resolve concrete kernel names based on dtype, contiguity, and adjointness,
- launch kernels.

Suggested non-responsibilities:

- geometry formulas,
- placeholder semantics,
- Radon/Fanbeam-specific payload creation,
- reconstruction algorithms.

## 8.2 Why private?

Because the exact shared surface between Radon and Fanbeam is not stable yet.

Making this base private keeps the implementation flexible while avoiding premature API commitments.

---

## 9. `Radon` after refactor

After the refactor, `Radon` should:

1. normalize constructor input (`ImageDomain`, `Angles`, `Detectors`),
2. store `kernel_spec` with a shipped default,
3. resolve placeholders,
4. compute and cache Radon geometry payloads,
5. lazily upload device buffers per queue/context and dtype,
6. launch kernels directly through `_OpenCLOperator`.

Crucially, it should no longer depend on creating a `ProjectionSettings` for normal execution.

---

## 10. Proposed `Radon` constructor shape

A likely constructor is:

```python
class Radon(_OpenCLOperator):
    def __init__(
        self,
        image_domain: int | tuple[int, int] | ImageDomain,
        angles: Angles | int,
        detectors: Detectors | int | None = None,
        adjoint: bool = False,
        kernel_spec: OpenCLKernelSpec | None = None,
    ):
        ...
```

Behavior:

- `kernel_spec=None` uses shipped gratopy Radon kernels,
- custom `kernel_spec` allows path-based experimental kernels.

Example:

```python
R = gratopy.operator.Radon(
    image_domain=256,
    angles=180,
    kernel_spec=gratopy.operator.OpenCLKernelSpec.from_path(
        "scratch/my_radon.cl",
        base_name="radon",
    ),
)
```

---

## 11. Kernel naming convention

For the release, keep the existing naming convention.

Forward kernel:

```text
{base_name}_{dtype}_{out_order}{in_order}
```

Adjoint kernel:

```text
{base_name}_ad_{dtype}_{out_order}{in_order}
```

Where:

- `dtype` is `float` or `double`,
- `out_order` is `f` or `c`,
- `in_order` is `f` or `c`.

For Radon this means shipped kernels continue to match names like:

- `radon_float_ff`
- `radon_float_fc`
- `radon_ad_float_ff`
- etc.

This is the simplest path and already aligns with the existing `.cl` files.

---

## 12. Geometry handling in `Radon`

## 12.1 No `AbstractRadon`

All Radon-specific geometry logic should stay in `Radon` or in private helpers next to it.

That includes:

- shape inference,
- placeholder resolution,
- geometry buffer preparation,
- maybe `show_geometry()` if desired,
- maybe a legacy conversion helper.

This keeps the code easier to follow than splitting one still-small operator into multiple public layers.

## 12.2 Reuse existing `radon_struct()` logic where practical

The current `gratopy.gratopy.radon_struct()` already computes the payloads needed by the shipped kernels.

For the release, the pragmatic path is:

- either call `radon_struct()` directly from `Radon`,
- or move/copy the relevant logic into private operator-side helpers.

The main architectural point is not where the math lives physically; it is that **`Radon` should own the geometry path instead of creating a `ProjectionSettings` and going through legacy execution**.

## 12.3 Recommended geometry payloads

For shipped Radon kernels, keep the same payload semantics as today:

- `ofs`
- `geometry`
- optionally separate `angle_weights` cache if useful

This minimizes migration cost and makes custom-kernel prototyping easier because the shipped kernels remain the reference implementation.

---

## 13. Placeholder handling

## 13.1 Required improvement

`ExtentPlaceholder.FULL` and `ExtentPlaceholder.VALID` should be implemented properly for the operator API.

Current behavior is incomplete and `VALID` is effectively unsupported.

## 13.2 Recommendation

Resolve placeholders in `Radon` itself, using the configured discrete angle set.

That means:

- placeholders are interpreted with respect to the actual operator geometry,
- limited-angle operators get sensible results,
- the logic stays local to the operator.

## 13.3 Scope for release

For the Radon release, this needs to work for:

- detector extent from image extent,
- image extent from detector extent,
- `FULL`,
- `VALID`,
- shifted image center / detector center.

This is a release-quality issue, not an optional enhancement.

---

## 14. Program cache design

## 14.1 Why the current global cache is not sufficient for prototyping

The current kernel cache is organized around the global generated code path used by the legacy API.

That makes it awkward to support multiple different kernel source bundles for the same operator class.

## 14.2 Recommended cache key

`_OpenCLOperator` should use an operator-local or module-local cache keyed by:

```text
(
    cl_context,
    kernel_spec_signature,
)
```

Where `kernel_spec_signature` includes:

- source file contents or a content hash,
- file ordering,
- base name,
- build options.

This allows multiple custom kernel specs to coexist safely.

## 14.3 Invalidation

Optional but useful:

```python
gratopy.operator.invalidate_kernel_cache()
```

Not strictly required for the first release, but nice if easy.

---

## 15. Queue and array handling

## 15.1 Shared OpenCL behavior

This is a good fit for `_OpenCLOperator`.

Queue inference priority should be:

1. explicit `queue=...`,
2. input array queue,
3. output array queue,
4. last-used queue stored on the operator,
5. otherwise error.

## 15.2 NumPy coercion

If input is not a `clarray.Array`, coerce via:

- `np.asarray(...)`,
- contiguity normalization if needed,
- `clarray.to_device(queue, ...)`.

## 15.3 Output allocation

The current operator implementation should be improved to auto-handle 3D data.

Required behavior:

- forward on `(Nx, Ny, Nz)` allocates `(Ns, Na, Nz)`,
- adjoint on `(Ns, Na, Nz)` allocates `(Nx, Ny, Nz)`.

This should be handled centrally in `_OpenCLOperator` where possible.

---

## 16. Small recommended `Operator` improvement

Although `Operator` should remain backend-agnostic, there is one small improvement worth making.

### Problem

Composite operators currently cannot conveniently pass through execution kwargs like `queue=...`.

This hurts usability for expressions like:

```python
A = R.T * R
A.apply_to(x_np, queue=queue)
```

### Recommendation

Allow `Operator.apply_to()` to accept and forward `**kwargs` in the composite case.

That keeps `Operator` generic while making execution of OpenCL-backed composite operators much more ergonomic.

This is the only base-class change I would currently recommend.

---

## 17. Fanbeam implications

This RFC does not implement `Fanbeam`, but it should make the future implementation simpler.

A later `Fanbeam` operator can likely reuse `_OpenCLOperator` for:

- queue handling,
- program cache,
- kernel lookup,
- output allocation.

But `Fanbeam` should own its own geometry logic, just as `Radon` does.

That is another reason not to introduce `AbstractRadon` now.

---

## 18. Transitional compatibility

A compatibility helper may still be useful during refactoring, e.g.:

```python
def to_projection_settings(self, queue: cl.CommandQueue) -> ProjectionSettings:
    ...
```

This can help with:

- testing against legacy output,
- temporary reuse of geometry visualization,
- migration confidence.

But this should be considered a **bridge**, not the target execution path.

---

## 19. Suggested module layout

A simple layout would be:

```text
gratopy/operator/
    __init__.py
    base.py
    opencl.py        # _OpenCLOperator, OpenCLKernelSpec, cache helpers
    projection.py    # Radon, later Fanbeam
```

This keeps the split focused:

- generic math in `base.py`,
- OpenCL execution plumbing in `opencl.py`,
- concrete operator implementations in `projection.py`.

---

## 20. Implementation plan

## Phase 1: release-critical work

1. Introduce `OpenCLKernelSpec`.
2. Introduce internal `_OpenCLOperator`.
3. Refactor `Radon` to inherit from `_OpenCLOperator`.
4. Add `kernel_spec` to `Radon` with sane shipped defaults.
5. Replace `ProjectionSettings`-based execution with direct geometry payload creation and direct kernel launch.
6. Implement proper placeholder handling.
7. Fix 3D auto-allocation.

## Phase 2: polish

8. Add tests for custom kernel path loading.
9. Add tests for `FULL` / `VALID`.
10. Update docs/examples to show `kernel_spec` usage.
11. Fix current operator docs (`R.T` vs `R.adjoint`, etc.).
12. Optionally add a legacy compatibility helper.

## Phase 3: later

13. Implement `Fanbeam` on the same `_OpenCLOperator` base.
14. Reassess whether any public abstract operator layer is actually needed.

---

## 21. Testing strategy

## Release-blocking tests

- Radon constructor shape inference,
- `R * img` against legacy output,
- `R.T * sino` against legacy output,
- NumPy coercion,
- explicit output buffers,
- automatic 3D allocation,
- `FULL` placeholder behavior,
- `VALID` placeholder behavior,
- custom kernel spec loading,
- program cache isolation for different kernel specs.

## Nice-to-have tests

- composite execution with forwarded `queue`,
- limited-angle placeholder cases,
- last-used queue reuse.

---

## 22. Public API examples

## Standard usage

```python
import gratopy

R = gratopy.operator.Radon(image_domain=256, angles=180)
sino = R * img
backproj = R.T * sino
```

## Custom kernel path

```python
from gratopy.operator import OpenCLKernelSpec, Radon

spec = OpenCLKernelSpec.from_path("scratch/my_radon.cl", base_name="radon")
R = Radon(image_domain=256, angles=180, kernel_spec=spec)
```

This is the core prototyping UX this RFC is meant to enable.

---

## 23. Alternatives considered

## Alternative A: `AbstractRadon`

Rejected for now.

Reason:

- adds another public layer without enough current benefit,
- overcomplicates a still-small concrete implementation,
- can be introduced later if real duplication emerges.

## Alternative B: put OpenCL behavior into `Operator`

Rejected.

Reason:

- mixes backend-specific concerns into a clean generic abstraction,
- reduces future flexibility.

## Alternative C: no shared OpenCL base at all

Possible, but less attractive.

Reason:

- queue handling, compilation, cacheing, and output allocation are real shared concerns,
- a small internal `_OpenCLOperator` is justified.

---

## 24. Recommendation

The simplest and most elegant solution is:

- keep `Operator` generic,
- keep `Radon` concrete,
- add a small public `OpenCLKernelSpec`,
- add a small private `_OpenCLOperator`,
- let `Radon` own its geometry and placeholder logic,
- remove `ProjectionSettings` from the normal Radon execution path.

This is enough to make operator syntax feel real, flexible, and releaseable without overengineering the architecture.

---

## 25. Immediate next steps

If implementation starts now, I would do this first:

1. create `gratopy/operator/opencl.py`,
2. implement `OpenCLKernelSpec`,
3. implement `_OpenCLOperator`,
4. refactor `Radon` onto that base,
5. switch `Radon` to direct kernel launches,
6. add placeholder tests,
7. add one custom-kernel test,
8. update docs/release notes.

That should be enough for a strong preliminary Radon operator release.
