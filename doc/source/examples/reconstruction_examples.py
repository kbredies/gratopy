"""Reconstruction examples for the documentation page ``reconstruction_examples.rst``.

The page includes the code sections between the ``# [name]`` and
``# [end-name]`` markers. Running this script executes all examples and
renders the figures shown on the page into ``doc/source/graphics``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# [setup]
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

import gratopy
from gratopy.operator import Radon
from gratopy.reconstruction import conjugate_gradients, landweber, total_variation

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

R = Radon(image_domain=256, angles=60)
phantom = gratopy.phantom(queue, 256, dtype="float32")

# Simulated data with Gaussian noise of 2% of the maximal sinogram value
clean = R * phantom
rng = np.random.default_rng(1)
noise_scale = 0.02 * np.abs(clean.get()).max()
noise = clarray.to_device(
    queue, (noise_scale * rng.standard_normal(R.output_shape)).astype(np.float32)
)
sino = clean + noise
# [end-setup]

# [comparison]
norm = R.norm_estimate(queue=queue)  # estimated once, reused below

x_landweber = landweber(R, sino, 100, operator_norm=norm)
x_cg = conjugate_gradients(R, sino, 10)
x_tv = total_variation(R, sino, mu=1e4, number_iterations=1000, operator_norm=norm)
# [end-comparison]


# [monitoring]
def relative_error(x):
    return R.domain.norm(x - phantom) / R.domain.norm(phantom)


errors = {"Landweber": [], "CG": []}
landweber(
    R,
    sino,
    300,
    operator_norm=norm,
    callback=lambda iteration, x: errors["Landweber"].append(relative_error(x)),
)
conjugate_gradients(
    R,
    sino,
    60,
    callback=lambda iteration, x: errors["CG"].append(relative_error(x)),
)
# [end-monitoring]

# [discrepancy]
delta = R.codomain.norm(noise)  # noise level, measured in the data norm
stop = {}


def discrepancy_principle(iteration, x, tau=1.1):
    if R.codomain.norm(R * x - sino) <= tau * delta:
        stop["iteration"] = iteration
        return True  # stop the iteration


x_stopped = conjugate_gradients(R, sino, 60, callback=discrepancy_principle)
# [end-discrepancy]

# [parameters]
tv_reconstructions = {
    mu: total_variation(R, sino, mu, 1000, operator_norm=norm) for mu in (1e3, 1e4, 1e6)
}
# [end-parameters]

# [geometries]
from gratopy.operator import Fanbeam
from gratopy.utilities import Detectors, ExtentPlaceholder

F = Fanbeam(
    image_domain=256,
    angles=180,
    detectors=Detectors(number=400, extent=ExtentPlaceholder.FULL),
    source_detector_distance=5.0,
    source_origin_distance=3.0,
)
x_fanbeam = conjugate_gradients(F, F * phantom, 20)
# [end-geometries]


def show_images(images, filename):
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3.3))
    for ax, (title, image) in zip(axes, images.items()):
        ax.imshow(image.get(), cmap="gray", vmin=0, vmax=phantom.get().max())
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(GRAPHICS / filename, dpi=80)
    plt.close(fig)


def with_error(name, x):
    return f"{name} (error {relative_error(x):.2f})"


if __name__ == "__main__":
    GRAPHICS = Path(__file__).resolve().parent.parent / "graphics"

    show_images(
        {
            "phantom": phantom,
            with_error("Landweber", x_landweber): x_landweber,
            with_error("CG", x_cg): x_cg,
            with_error("TV", x_tv): x_tv,
        },
        "reconstruction-comparison.png",
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, history in errors.items():
        ax.plot(np.arange(1, len(history) + 1), history, label=name)
    stopped = stop["iteration"]
    ax.plot(stopped, errors["CG"][stopped - 1], "ko", label="CG, discrepancy principle")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHICS / "reconstruction-semiconvergence.png", dpi=80)
    plt.close(fig)

    show_images(
        {
            with_error(f"TV, $\\mu = 10^{{{np.log10(mu):.0f}}}$", x): x
            for mu, x in tv_reconstructions.items()
        },
        "reconstruction-tv-parameters.png",
    )

    print("discrepancy principle stopped CG at iteration", stopped)
    print("fanbeam CG relative error", relative_error(x_fanbeam))
