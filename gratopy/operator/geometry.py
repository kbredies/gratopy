"""Physical geometry of projection operators.

:class:`ProjectionGeometry` describes where the image domain, the detector and
(for fan-beam geometries) the source lie in physical coordinates for a set of
projection angles. It is pure NumPy and follows the conventions of the OpenCL
kernels:

- the beam points along ``theta = (cos(phi), sin(phi))`` (from the source
  towards the detector),
- the detector axis ``e = (sin(phi), -cos(phi))`` points towards increasing
  detector pixel indices (flipped for reversed fan-beam detectors),
- detector pixel ``i`` is centered at detector coordinate
  ``t_i = (i - (N - 1) / 2) * delta_s + detector_center`` along ``e``.

It is an internal helper of :meth:`show_geometry`: operators build it via
their private ``_geometry_at`` method.

:func:`plot_projection_geometry` draws a geometry with matplotlib, which is
imported only when drawing. Every artist carries a ``gid`` naming its role
(``"image"``, ``"detector"``, ``"ray-hit"``, ``"ray-miss"``, ...), which
makes the drawing inspectable.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes


@dataclass(frozen=True, eq=False)
class ProjectionGeometry:
    """Physical geometry of a projection operator at given angles.

    All per-angle arrays have the angles along their first axis.

    **Attributes**

    ``angles``:
        Projection angles in radians, shape ``(Na,)``.
    ``image_center``, ``image_dims``:
        Center and side lengths ``(Dx, Dy)`` of the image domain.
    ``detector_center``, ``detector_width``, ``detector_number``:
        Detector coordinate of the detector center, physical detector width
        and number of detector pixels.
    ``detector_origin``:
        Position of detector coordinate ``t = 0``, shape ``(Na, 2)``.
    ``detector_axis``:
        Unit vector towards increasing detector pixel indices, shape
        ``(Na, 2)``.
    ``sources``:
        Fan-beam source positions, shape ``(Na, 2)``; ``None`` for parallel
        beams.
    ``beam_direction``:
        Unit ray direction of parallel beams, shape ``(Na, 2)``; ``None`` for
        fan beams.
    """

    angles: np.ndarray
    image_center: np.ndarray
    image_dims: np.ndarray
    detector_center: float
    detector_width: float
    detector_number: int
    detector_origin: np.ndarray
    detector_axis: np.ndarray
    sources: np.ndarray | None = None
    beam_direction: np.ndarray | None = None

    @property
    def is_fanbeam(self) -> bool:
        return self.sources is not None

    @property
    def detector_interval(self) -> tuple[float, float]:
        """Detector coordinates of the outer edges of the first/last pixel."""
        half = self.detector_width / 2
        return (self.detector_center - half, self.detector_center + half)

    def pixel_positions(self) -> np.ndarray:
        """Detector coordinates of the pixel centers, shape ``(N,)``."""
        delta_s = self.detector_width / self.detector_number
        index = np.arange(self.detector_number)
        return (index - (self.detector_number - 1) / 2) * delta_s + self.detector_center

    def image_corners(self) -> np.ndarray:
        """Corners of the image rectangle (counter-clockwise), shape ``(4, 2)``."""
        half = self.image_dims / 2
        signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        return self.image_center + signs * half

    def detector_points(self, t: npt.ArrayLike) -> np.ndarray:
        """Physical positions of detector coordinates ``t``, shape ``(Na, Nt, 2)``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return (
            self.detector_origin[:, None, :]
            + t[None, :, None] * self.detector_axis[:, None, :]
        )

    def ray_directions(self, t: npt.ArrayLike) -> np.ndarray:
        """Unit directions of the rays hitting detector coordinates ``t``."""
        points = self.detector_points(t)
        if self.sources is None:
            assert self.beam_direction is not None
            return np.broadcast_to(self.beam_direction[:, None, :], points.shape)
        direction = points - self.sources[:, None, :]
        return direction / np.linalg.norm(direction, axis=-1, keepdims=True)

    def rays_hit_image(self, t: npt.ArrayLike) -> np.ndarray:
        """Return which rays hitting detector coordinates ``t`` meet the image.

        A ray (as a line) meets the rectangle iff its signed distance to the
        image center is at most the rectangle's half width in the ray normal.
        """
        points = self.detector_points(t)
        direction = self.ray_directions(t)
        normal = np.stack([-direction[..., 1], direction[..., 0]], axis=-1)
        offset = np.einsum("atk,atk->at", normal, points - self.image_center)
        half_width = (np.abs(normal) * (self.image_dims / 2)).sum(axis=-1)
        return np.abs(offset) <= half_width

    def image_detector_range(self) -> np.ndarray:
        """Detector coordinates hit by rays through the image, shape ``(Na, 2)``.

        Returns the smallest and largest detector coordinate over all rays
        passing through the image domain. The detector coordinate is a
        linear-fractional function of the image point, so the extremes are
        attained at the corners.
        """
        corners = self.image_corners()
        if self.sources is None:
            relative = corners[None, :, :] - self.detector_origin[:, None, :]
            t = np.einsum("ack,ak->ac", relative, self.detector_axis)
        else:
            towards_detector = self.detector_origin - self.sources
            distance = np.linalg.norm(towards_detector, axis=-1)
            towards_detector = towards_detector / distance[:, None]
            relative = corners[None, :, :] - self.sources[:, None, :]
            along_axis = np.einsum("ack,ak->ac", relative, self.detector_axis)
            along_beam = np.einsum("ack,ak->ac", relative, towards_detector)
            t = distance[:, None] * along_axis / along_beam
        return np.stack([t.min(axis=1), t.max(axis=1)], axis=1)


# -- Drawing ------------------------------------------------------------------

MISS_COLOR = "tab:red"
IMAGE_COLOR = "0.35"


def plot_projection_geometry(
    geometry: ProjectionGeometry,
    ax: matplotlib.axes.Axes | None = None,
    *,
    n_rays: int = 15,
    coverage: bool = True,
    trajectory: np.ndarray | None = None,
    legend: bool = True,
    title: str | None = None,
) -> matplotlib.axes.Axes:
    """Draw a projection geometry into ``ax`` (a new figure if ``None``).

    :param geometry: The geometry to draw, typically from
        :meth:`_geometry_at`.
    :param ax: Axes to draw into. Existing content is kept.
    :param n_rays: Number of sample rays per angle, equispaced over the
        detector and including both detector edges.
    :param coverage: Shade the illuminated region and draw rays missing the
        image domain dashed in red.
    :param trajectory: Fan-beam source positions of the whole scan, drawn as
        the source circle with the sampled positions.
    :param legend: Whether to add a legend to the right of the axes.
    :param title: Axes title; only set when given.
    :return: The axes drawn into.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6) if legend else None, layout="constrained")

    n_angles = len(geometry.angles)
    colors: list[Any] = ["tab:blue"]
    if n_angles > 1:
        colors = [plt.get_cmap("viridis")(k / (n_angles - 1)) for k in range(n_angles)]

    rays = _RaySamples(geometry, n_rays)
    scene_points = [
        geometry.image_corners(),
        rays.edges.reshape(-1, 2),
        rays.starts.reshape(-1, 2),
        np.zeros((1, 2)),
    ]
    scale = np.ptp(np.concatenate(scene_points), axis=0).max()

    # Source trajectory (fan beam) below everything else.
    if trajectory is not None:
        radius = np.linalg.norm(trajectory, axis=1).max()
        ax.add_patch(
            Circle((0, 0), radius, fill=False, ls=":", color="0.6", gid="trajectory")
        )
        ax.plot(*trajectory.T, ".", ms=3, color="0.6", gid="trajectory-samples")
        scene_points.append(np.array([[-radius, -radius], [radius, radius]]))

    ax.add_patch(
        Rectangle(
            tuple(geometry.image_center - geometry.image_dims / 2),
            *geometry.image_dims,
            facecolor=IMAGE_COLOR,
            alpha=0.25,
            edgecolor=IMAGE_COLOR,
            lw=1.5,
            gid="image",
        )
    )
    ax.plot(0, 0, "+", color="k", ms=10, gid="rotation-center")

    for a, color in enumerate(colors):
        _draw_angle(ax, geometry, rays, a, color, coverage=coverage, scale=scale)

    points = np.concatenate(scene_points)
    margin = 0.08 * scale
    ax.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
    ax.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if legend:
        ax.legend(
            handles=_legend_handles(geometry, colors, coverage, trajectory is not None),
            fontsize="small",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
        )
    if title is not None:
        ax.set_title(title)
    return ax


class _RaySamples:
    """Sample rays over the detector, for all angles of a geometry."""

    def __init__(self, geometry: ProjectionGeometry, n_rays: int) -> None:
        lower, upper = geometry.detector_interval
        t = np.linspace(lower, upper, max(n_rays, 2))
        self.ends = geometry.detector_points(t)
        self.hits = geometry.rays_hit_image(t)
        self.edges = geometry.detector_points([lower, upper])
        self.centers = geometry.detector_points(geometry.detector_center)[:, 0]
        self.first_pixel = geometry.detector_points(geometry.pixel_positions()[0])[:, 0]
        if geometry.sources is not None:
            self.starts = np.broadcast_to(geometry.sources[:, None, :], self.ends.shape)
            self.center_starts = geometry.sources
        else:
            assert geometry.beam_direction is not None
            # Start parallel rays on the far side of the scene.
            back = 2 * np.linalg.norm(geometry.detector_origin, axis=1)
            shift = back[:, None] * geometry.beam_direction
            self.starts = self.ends - shift[:, None, :]
            self.center_starts = self.centers - shift


def _draw_angle(ax, geometry, rays, a, color, *, coverage, scale):
    """Draw illuminated region, rays, detector and source for angle ``a``."""
    from matplotlib.patches import Polygon

    edges = rays.edges[a]
    if coverage:
        if geometry.sources is not None:
            region = [geometry.sources[a], edges[0], edges[1]]
        else:
            region = [edges[0], edges[1], rays.starts[a, -1], rays.starts[a, 0]]
        ax.add_patch(
            Polygon(region, facecolor=color, alpha=0.08, lw=0, gid="illuminated")
        )

    for start, end, hit in zip(rays.starts[a], rays.ends[a], rays.hits[a]):
        missed = coverage and not hit
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=MISS_COLOR if missed else color,
            ls="--" if missed else "-",
            lw=0.8,
            alpha=0.9 if missed else 0.6,
            gid="ray-miss" if missed else "ray-hit",
        )

    center, center_start = rays.centers[a], rays.center_starts[a]
    ax.plot(
        [center_start[0], center[0]],
        [center_start[1], center[1]],
        color=color,
        ls="-.",
        lw=1.2,
        gid="central-ray",
    )

    # Detector with orientation: arrow towards increasing pixel index,
    # marker at pixel 0 and a tick at the detector center.
    ax.plot(*edges.T, color=color, lw=3.5, solid_capstyle="butt", gid="detector")
    axis = geometry.detector_axis[a]
    ax.annotate(
        "",
        xy=edges[1] + 0.06 * scale * axis,
        xytext=edges[1] - 0.02 * scale * axis,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5},
        gid="detector-direction",
    )
    ax.plot(*rays.first_pixel[a], "o", mfc="white", mec=color, ms=6, gid="first-pixel")
    normal = np.array([-axis[1], axis[0]])
    tick = center + 0.02 * scale * np.array([-normal, normal])
    ax.plot(*tick.T, color="k", lw=1.5, gid="detector-center")

    if geometry.sources is not None:
        ax.plot(*geometry.sources[a], "*", color=color, ms=12, mec="k", gid="source")


def _legend_handles(geometry, colors, coverage, trajectory):
    """Return proxy artists describing the drawn elements."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Elements drawn in the angle's color: with one angle, show that color;
    # with several, show them neutrally next to one color patch per angle.
    angle_color = colors[0] if len(colors) == 1 else "k"
    handles = [
        Patch(
            facecolor=IMAGE_COLOR, alpha=0.4, edgecolor=IMAGE_COLOR, label="image domain"
        ),
        Line2D(
            [], [], ls="", marker="+", color="k", ms=10, label="rotation center (0, 0)"
        ),
        Line2D([], [], color=angle_color, lw=3.5, label="detector, arrow: index ↑"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            mfc="white",
            mec=angle_color,
            label="detector pixel 0",
        ),
        Line2D(
            [], [], color="k", lw=1.5, marker="|", ls="", ms=10, label="detector center"
        ),
        Line2D([], [], color=angle_color, ls="-.", label="central ray"),
        Line2D([], [], color=angle_color, lw=0.8, alpha=0.6, label="ray meeting image"),
    ]
    if coverage:
        handles.append(
            Line2D([], [], color=MISS_COLOR, ls="--", lw=0.8, label="ray missing image")
        )
    if geometry.sources is not None:
        handles.append(
            Line2D(
                [],
                [],
                ls="",
                marker="*",
                color=angle_color,
                mec="k",
                ms=10,
                label="source",
            )
        )
    if trajectory:
        handles.append(Line2D([], [], color="0.6", ls=":", label="source circle"))
    if 1 < len(colors) <= 8:
        handles += [
            Patch(color=color, label=f"φ = {np.degrees(angle):.1f}°")
            for angle, color in zip(geometry.angles, colors)
        ]
    return handles
