# Operator kernel-spec refactor roadmap

Goal: prepare a **Radon-first release** with preliminary operator syntax support that is polished enough for users, while introducing a flexible kernel-loading path for prototyping alternative OpenCL implementations.

## Design decisions

- Keep `gratopy.operator.base.Operator` **generic** and backend-agnostic.
- Keep `gratopy.operator.Radon` **concrete and public**.
- Do **not** introduce a public `AbstractRadon`.
- Introduce a small public `OpenCLKernelSpec`.
- Introduce a small internal `_OpenCLOperator` base for shared OpenCL execution plumbing.
- Refactor `Radon` later to use `_OpenCLOperator` directly instead of routing through `ProjectionSettings`.
- `Fanbeam` can later reuse `_OpenCLOperator`, but should still own its own geometry logic.

---

## Phase 1 — groundwork (now)

### 1. Add `gratopy/operator/opencl.py`
Implement prototypes for:

- `OpenCLKernelSpec`
  - minimal fields:
    - `paths: tuple[str, ...]`
    - `base_name: str`
    - `build_options: tuple[str, ...] = ()`
  - convenience constructors:
    - `from_path(...)`
    - `from_paths(...)`
  - normalized absolute paths
  - content-based signature/hash for cacheing

- `_OpenCLOperator`
  - inherit from `Operator`
  - shared helper methods only:
    - queue inference
    - NumPy/array-like coercion to `clarray.Array`
    - output allocation helper
    - template expansion for gratopy-style kernels
    - program compilation/cache keyed by `(context, kernel_spec signature)`
    - kernel name construction based on dtype / contiguity / adjoint flag
    - kernel lookup helper
  - no geometry-specific logic yet
  - no Radon integration yet

### 2. Export `OpenCLKernelSpec`
Update `gratopy/operator/__init__.py` to export the public spec.

### 3. Keep `Radon` unchanged for now
The prototype step should not yet refactor execution. This phase is just to establish the new building blocks for review.

---

## Phase 2 — Radon migration

### 4. Move Radon execution off `ProjectionSettings`
Refactor `Radon` to inherit from `_OpenCLOperator` and use:

- direct geometry payload creation,
- direct device buffer upload,
- direct kernel lookup/launch via `kernel_spec`.

### 5. Add `kernel_spec` argument to `Radon`
- default to shipped gratopy Radon kernels,
- allow custom path-based specs for experiments.

### 6. Keep geometry local to `Radon`
Implement or reuse helpers for:

- shape inference,
- `FULL` / `VALID` placeholder resolution,
- host-side Radon geometry payloads,
- lazy queue/dtype-specific device buffers.

No public geometry base class should be added.

---

## Phase 3 — release polish

### 7. Improve array handling
- automatic 3D output allocation,
- better queue reuse behavior,
- consistent dtype/order handling.

### 8. Improve composite usability
Consider a small `Operator.apply_to(..., **kwargs)` change so composite operators can forward `queue=...`.

### 9. Fix docs/examples
- `R.T` instead of stale `R.adjoint` doc examples,
- add operator-era Radon example,
- add custom kernel example.

---

## Phase 4 — tests

### Groundwork tests
- `OpenCLKernelSpec.from_path/from_paths`
- path normalization
- content-hash/signature stability
- `_OpenCLOperator` kernel name generation

### Radon migration tests
- `R * img` matches legacy output
- `R.T * sino` matches legacy output
- NumPy coercion
- explicit output buffers
- automatic 3D allocation
- `FULL` and `VALID` placeholders
- custom kernel path loading
- cache separation for different kernel specs

---

## Immediate deliverables

For the current step, implement only:

1. `agent/rfc-flexible-operator-kernels.md` update ✅
2. `agent/task.md` roadmap ✅
3. prototype `gratopy/operator/opencl.py`
4. export `OpenCLKernelSpec`

No `Radon` refactor yet.
