from typing import List, Tuple
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Polygon
from fastdtw import dtw

from matplotlib.patches import Arc
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox


import scienceplots

plt.style.use("science")
plt.rc("text.latex", preamble=r"\usepackage{physics}")

from pedestrians_social_binding.trajectory_utils import (
    compute_turning_angles,
    compute_velocity,
    rediscretize_position_v2,
    compute_steps,
    compute_curvature_gradient,
    compute_normals,
    compute_tangents,
    compute_dynamic_time_warping_distance,
    compute_euclidean_distance,
)


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    Adapted from
    """

    def __init__(
        self,
        xy,
        p1,
        p2,
        size=75,
        unit="points",
        ax=None,
        text="",
        textposition="inside",
        text_kw=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(
            self._xydata,
            size,
            size,
            angle=0.0,
            theta1=self.theta1,
            theta2=self.theta2,
            **kwargs,
        )

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(
            ha="center",
            va="center",
            xycoords=IdentityTransform(),
            xytext=(0, 0),
            textcoords="offset points",
            annotation_clip=True,
        )
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.0
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.0
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {
                "max": max(b.width, b.height),
                "min": min(b.width, b.height),
                "width": b.width,
                "height": b.height,
            }
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180], [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":

            def R90(a, r, w, h):
                if a < np.arctan(h / 2 / (r + w / 2)):
                    return np.sqrt((r + w / 2) ** 2 + (np.tan(a) * (r + w / 2)) ** 2)
                else:
                    c = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                    T = np.arcsin(c * np.cos(np.pi / 2 - a + np.arcsin(h / 2 / c)) / r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w / 2, h / 2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi / 4)) * ((a % (np.pi / 2)) <= np.pi / 4) + (
                    np.pi / 4 - (a % (np.pi / 4))
                ) * ((a % (np.pi / 2)) >= np.pi / 4)
                return R90(aa, r, *[w, h][:: int(np.sign(np.cos(2 * a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X - s / 2), 0))[0] * 72
            self.text.set_position([offs * np.cos(angle), offs * np.sin(angle)])


def make_trajectory(
    control_points,
    velocity,
    sampling_time,
    gait_frequency=1.9,
    swaying=0.045,
    noise_sigma=0.01,
):
    distances = np.linalg.norm(control_points[1:] - control_points[:-1], axis=1)
    times = np.cumsum(distances / velocity)
    times = np.insert(times, 0, 0)
    spline = CubicSpline(times, control_points)

    t = np.arange(0, times[-1], sampling_time)
    positions = spline(t)

    # add gait fluctuations
    displacements = swaying * np.sin(2 * np.pi * gait_frequency * t)
    normals = np.diff(positions, axis=0)
    normals = np.insert(normals, 0, normals[0], axis=0)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    tangents = np.array([-normals[:, 1], normals[:, 0]]).T
    positions += displacements[:, None] * tangents

    # add some noise
    positions += np.random.normal(0, noise_sigma, positions.shape)

    trajectory = np.zeros((len(t), 7))
    trajectory[:, 0] = t
    trajectory[:, 1:3] = positions * 1000

    return trajectory


def plot_trajectory(ax, opacity=1.0, label=True, scatter=False):
    if scatter:
        ax.scatter(
            trajectory[:, 1] / 1000,
            trajectory[:, 2] / 1000,
            color="cornflowerblue",
            s=20,
        )
    else:
        ax.plot(
            trajectory[:, 1] / 1000,
            trajectory[:, 2] / 1000,
            marker="o",
            linestyle=(0, (2, 1)),
            linewidth=2,
            color="cornflowerblue",
            alpha=opacity,
        )
    if label:
        ax.text(
            trajectory[7, 1] / 1000 + 0.1,
            trajectory[7, 2] / 1000 + 0.5,
            "$T$",
            fontsize=16,
            color="cornflowerblue",
        )
    return ax


def plot_intended_direction(ax):
    # plot line
    ax.plot(
        [
            start_point[0] / 1000,
            (start_point[0] + t_final * start_vel[0]) / 1000,
        ],
        [
            start_point[1] / 1000,
            (start_point[1] + t_final * start_vel[1]) / 1000,
        ],
        color="black",
        linestyle="-",
    )
    # plot arrow
    ax.arrow(
        start_point[0] / 1000,
        start_point[1] / 1000,
        start_vel[0] / 1000,
        start_vel[1] / 1000,
        head_width=0.1,
        head_length=0.2,
        fc="hotpink",
        ec="hotpink",
        zorder=10,
    )
    # add label
    ax.text(
        start_point[0] / 1000 + 0.3,
        start_point[1] / 1000 - 0.2,
        "$\\vb{v_0}$",
        fontsize=16,
        color="hotpink",
    )

    ax.text(
        straight_line_trajectory[7, 1] / 1000 + 0.1,
        straight_line_trajectory[7, 2] / 1000 - 0.4,
        "$L_0$",
        fontsize=16,
        color="black",
    )

    return ax


def plot_straight_line_trajectory(ax):
    ax.plot(
        straight_line_trajectory[:, 1] / 1000,
        straight_line_trajectory[:, 2] / 1000,
        marker="o",
        linestyle=(0, (2, 1)),
        linewidth=2,
        color="lightcoral",
    )
    ax.text(
        straight_line_trajectory[7, 1] / 1000 + 0.1,
        straight_line_trajectory[7, 2] / 1000 - 0.4,
        "$T_0$",
        fontsize=16,
        color="lightcoral",
    )
    return ax


def format_plot(ax):
    # # plot control points
    # ax.scatter(
    #     control_points[:, 0],
    #     control_points[:, 1],
    #     color="black",
    #     s=20,
    # )

    # draw axes
    plt.arrow(
        0,
        2,
        0.5,
        0,
        head_width=0.1,
        head_length=0.2,
        fc="k",
        ec="k",
        overhang=0.8,
    )
    plt.arrow(
        0,
        2,
        0,
        0.5,
        head_width=0.1,
        head_length=0.2,
        fc="k",
        ec="k",
        overhang=0.8,
    )
    # add labels
    plt.text(0.5, 2.1, "$x$", fontsize=16)
    plt.text(0.1, 2.5, "$y$", fontsize=16)

    ax.axis("off")
    ax.set_aspect("equal")
    return ax


if __name__ == "__main__":
    start_point = np.array([0, 0])

    sampling_time = 0.3
    velocity = 1.2  # m/s

    smoothing_window_duration = 3  # seconds
    smoothing_window = int(smoothing_window_duration / sampling_time)

    time_entrance = 0.5  # seconds
    n_points_entrance = int(time_entrance / sampling_time)

    control_points = np.array(
        [
            [0, 0],
            [0.1, 0.0],
            [0.2, 0.0],
            [1, 0.7],
            [1.5, 1],
            [3, 0.5],
            [5, 1.3],
            [7.5, 1.4],
        ]
    )

    trajectory = make_trajectory(
        control_points, velocity, sampling_time, swaying=0, noise_sigma=0
    )
    positions = trajectory[:, 1:3]
    n_points = len(trajectory)
    t = trajectory[:, 0]
    t_final = t[-1]

    velocity = compute_velocity(trajectory)
    entrance_velocities = velocity[:n_points_entrance]
    start_vel = np.mean(entrance_velocities, axis=0)
    start_direction = start_vel / np.linalg.norm(start_vel)

    straight_line_trajectory = np.zeros((n_points, 7))
    straight_line_trajectory[:, 0] = trajectory[:, 0]
    straight_line_trajectory[:, 1:3] = (
        start_vel * (trajectory[:, 0][:, None] - trajectory[0, 0]) + trajectory[0, 1:3]
    )

    # ------------------------------
    # maximum lateral deviation
    # ------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))
    distances_to_straight_line = np.abs(
        np.cross(start_vel, positions - start_point)
    ) / np.linalg.norm(start_vel)

    idx_max = np.argmax(distances_to_straight_line)
    position_max = positions[np.argmax(distances_to_straight_line), :]

    max_distance = np.max(distances_to_straight_line) / 1000

    ax = plot_trajectory(ax)
    ax = plot_intended_direction(ax)

    # show furthest point in red
    ax.scatter(
        position_max[0] / 1000,
        position_max[1] / 1000,
        color="red",
        s=20,
    )
    # show distance to line
    point_on_line_furthest = start_point + start_vel * (
        np.dot(start_vel, (position_max - start_point)) / np.dot(start_vel, start_vel)
    )

    # double head arrow
    arrow_distance = FancyArrowPatch(
        (position_max[0] / 1000, position_max[1] / 1000),
        (point_on_line_furthest[0] / 1000, point_on_line_furthest[1] / 1000),
        color="purple",
        arrowstyle="<->",
        mutation_scale=20,
        lw=2,
        zorder=10,
        shrinkA=0,
        shrinkB=0,
    )

    ax.text(
        position_max[0] / 1000 - 0.1,
        position_max[1] / 1000 + 0.2,
        "$\\vb{p}(t_k)$",
        fontsize=14,
        color="cornflowerblue",
    )

    ax.scatter(
        point_on_line_furthest[0] / 1000,
        point_on_line_furthest[1] / 1000,
        color="purple",
        s=20,
    )
    ax.text(
        point_on_line_furthest[0] / 1000 + 0.1,
        point_on_line_furthest[1] / 1000 - 0.3,
        "$\\vb{h}(t_k)$",
        fontsize=14,
        color="purple",
    )

    ax.add_patch(arrow_distance)

    ax.text(
        position_max[0] / 1000 + 0.3,
        position_max[1] / 1000 - 0.7,
        "$d_{max}$",
        fontsize=16,
        color="purple",
    )

    ax = format_plot(ax)
    plt.savefig("../data/figures/measures/maximum_lateral_deviation.pdf")
    plt.close()

    # ------------------------------
    # lockstep maximum deviation
    # ------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_trajectory(ax)
    ax = plot_straight_line_trajectory(ax)

    for i in range(n_points):
        ax.plot(
            [
                trajectory[i, 1] / 1000,
                straight_line_trajectory[i, 1] / 1000,
            ],
            [
                trajectory[i, 2] / 1000,
                straight_line_trajectory[i, 2] / 1000,
            ],
            color="black",
            linestyle="--",
        )

    lockstep_distances = np.linalg.norm(
        trajectory[:, 1:3] - straight_line_trajectory[:, 1:3], axis=1
    )

    idx_max = np.argmax(lockstep_distances)

    arrow_distance = FancyArrowPatch(
        (trajectory[idx_max, 1] / 1000, trajectory[idx_max, 2] / 1000),
        (
            straight_line_trajectory[idx_max, 1] / 1000,
            straight_line_trajectory[idx_max, 2] / 1000,
        ),
        color="purple",
        arrowstyle="<->",
        mutation_scale=20,
        lw=2,
        zorder=10,
        shrinkA=0,
        shrinkB=0,
    )

    ax.add_patch(arrow_distance)

    ax.text(
        trajectory[idx_max, 1] / 1000 + 0.1,
        trajectory[idx_max, 2] / 1000 - 0.3,
        "$\\delta_{max}$",
        fontsize=16,
        color="purple",
    )

    ax = format_plot(ax)

    plt.savefig("../data/figures/measures/lockstep_maximum_deviation.pdf")
    plt.close()

    # draw each trapezoid

    # ------------------------------
    # integral of the lateral deviation
    # ------------------------------

    distances_to_origin = np.linalg.norm(positions - start_point, axis=1)
    distances_to_projection = np.sqrt(
        distances_to_origin**2 - distances_to_straight_line**2
    )

    points_on_line = start_point + start_direction * distances_to_projection[:, None]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_trajectory(ax)
    ax = plot_intended_direction(ax)

    # draw each trapezoid
    for i in range(n_points - 1):
        p = Polygon(
            np.array(
                [
                    [positions[i, 0], positions[i, 1]],
                    [positions[i + 1, 0], positions[i + 1, 1]],
                    [points_on_line[i + 1, 0], points_on_line[i + 1, 1]],
                    [points_on_line[i, 0], points_on_line[i, 1]],
                ]
            )
            / 1000,  # type: ignore
            facecolor="lightgrey",
            edgecolor="black",
            zorder=40,
            alpha=0.3,
        )
        ax.add_patch(p)

    ax.text(
        position_max[0] / 1000 + 0.2,
        position_max[1] / 1000 - 0.5,
        "$\\Delta$",
        fontsize=16,
        color="gray",
    )

    ax = format_plot(ax)

    # plt.show()
    plt.savefig("../data/figures/measures/integral_lateral_deviation.pdf")
    plt.close()

    # ------------------------------
    # euclidean distance
    # ------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_trajectory(ax)
    ax = plot_straight_line_trajectory(ax)

    for i in range(n_points):
        ax.plot(
            [
                trajectory[i, 1] / 1000,
                straight_line_trajectory[i, 1] / 1000,
            ],
            [
                trajectory[i, 2] / 1000,
                straight_line_trajectory[i, 2] / 1000,
            ],
            color="black",
            linestyle="--",
        )

    ax = format_plot(ax)

    # plt.show()
    plt.savefig("../data/figures/measures/euclidean_distance.pdf")

    # ------------------------------
    # dynamic time warping deviation
    # ------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_trajectory(ax)
    ax = plot_straight_line_trajectory(ax)

    _, path = dtw(
        trajectory[:, 1:3],
        straight_line_trajectory[:, 1:3],
        dist=lambda x, y: np.linalg.norm(x - y),
    )

    for i, j in path:
        ax.plot(
            [
                trajectory[i, 1] / 1000,
                straight_line_trajectory[j, 1] / 1000,
            ],
            [
                trajectory[i, 2] / 1000,
                straight_line_trajectory[j, 2] / 1000,
            ],
            color="black",
            linestyle="--",
        )

    # print(
    #     compute_euclidean_distance(trajectory, straight_line_trajectory),
    #     compute_dynamic_time_warping_distance(trajectory, straight_line_trajectory)
    #     / 1000,
    # )

    ax = format_plot(ax)

    # plt.show()
    plt.savefig("../data/figures/measures/dynamic_time_warping_deviation.pdf")
    plt.close()

    # ------------------------------
    # sinuosity
    # ------------------------------

    # compute turning angles
    rediscretized_positions = rediscretize_position_v2(positions, step_length=800)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_trajectory(ax, opacity=0.5, label=False)

    ax.plot(
        rediscretized_positions[:, 0] / 1000,
        rediscretized_positions[:, 1] / 1000,
        "s-",
        linewidth=2,
        color="seagreen",
    )
    ax.text(
        rediscretized_positions[7, 0] / 1000 + 0.1,
        rediscretized_positions[7, 1] / 1000 + 0.3,
        "$\\tilde{T}$",
        fontsize=16,
        color="seagreen",
    )

    for i in range(1, len(rediscretized_positions) - 1):
        dp = rediscretized_positions[i] - rediscretized_positions[i - 1]
        end_point = rediscretized_positions[i] + dp
        angle = np.arctan2(dp[1], dp[0]) - np.arctan2(
            rediscretized_positions[i + 1, 1] - rediscretized_positions[i, 1],
            rediscretized_positions[i + 1, 0] - rediscretized_positions[i, 0],
        )
        if angle > 0:
            p1, p2 = rediscretized_positions[i + 1], end_point
        else:
            p1, p2 = end_point, rediscretized_positions[i + 1]
        ax.plot(
            [
                rediscretized_positions[i, 0] / 1000,
                end_point[0] / 1000,
            ],
            [
                rediscretized_positions[i, 1] / 1000,
                end_point[1] / 1000,
            ],
            color="seagreen",
            linestyle="--",
        )
        angle_annotation = AngleAnnotation(
            rediscretized_positions[i] / 1000,
            p1 / 1000,
            p2 / 1000,
            ax=ax,
            size=50,
            color="seagreen",
        )

    arrow = FancyArrowPatch(
        (rediscretized_positions[4, 0] / 1000, rediscretized_positions[4, 1] / 1000),
        (
            rediscretized_positions[5, 0] / 1000,
            rediscretized_positions[5, 1] / 1000,
        ),
        color="coral",
        arrowstyle="<->",
        mutation_scale=20,
        lw=2,
        zorder=10,
        shrinkA=0,
        shrinkB=0,
    )

    ax.add_patch(arrow)

    ax.text(
        (rediscretized_positions[4, 0] + rediscretized_positions[5, 0]) / 2000,
        (rediscretized_positions[4, 1] + rediscretized_positions[5, 1]) / 2000 - 0.3,
        "$q$",
        fontsize=16,
        color="coral",
    )

    ax = format_plot(ax)

    # plt.show()
    plt.savefig("../data/figures/measures/sinuosity.pdf")

    plt.close()

    # ------------------------------
    # maximum cumulative turning angle
    # ------------------------------

    turning_angles = compute_turning_angles(positions)
    cumulative_turning_angles = np.cumsum(turning_angles)

    argmax_cumulative_turning_angle = np.argmax(np.abs(cumulative_turning_angles))

    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_trajectory(ax)

    for i in range(1, len(positions) - 1):
        dp = positions[i] - positions[i - 1]
        end_point = positions[i] + 1.5 * dp
        angle = np.arctan2(dp[1], dp[0]) - np.arctan2(
            positions[i + 1, 1] - positions[i, 1],
            positions[i + 1, 0] - positions[i, 0],
        )
        if angle > 0:
            p1, p2 = positions[i + 1], end_point
        else:
            p1, p2 = end_point, positions[i + 1]
        ax.plot(
            [
                positions[i, 0] / 1000,
                end_point[0] / 1000,
            ],
            [
                positions[i, 1] / 1000,
                end_point[1] / 1000,
            ],
            color="gray",
            linestyle="--",
        )
        angle_annotation = AngleAnnotation(
            positions[i] / 1000,
            p1 / 1000,
            p2 / 1000,
            ax=ax,
            size=30,
            linewidth=2,
            color="orangered",
        )
        if i == 3:
            ax.text(
                positions[i, 0] / 1000 + 0.35,
                positions[i, 1] / 1000 + 0.15,
                "$d\\theta_j$",
                fontsize=14,
                color="orangered",
            )

    # add position of maximum cumulative turning angle
    ax.plot(
        positions[
            argmax_cumulative_turning_angle + 1 : argmax_cumulative_turning_angle + 3,
            0,
        ]
        / 1000,
        positions[
            argmax_cumulative_turning_angle + 1 : argmax_cumulative_turning_angle + 3,
            1,
        ]
        / 1000,
        "o-",
        linewidth=2,
        color="purple",
        zorder=30,
    )

    ax = format_plot(ax)

    plt.savefig("../data/figures/measures/maximum_cumulative_turning_angle.pdf")
    plt.close()

    # temporary increase font size
    plt.rcParams.update({"font.size": 36})

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(
        t[1:-1],
        np.rad2deg(turning_angles),
        marker="o",
        linestyle=(0, (2, 1)),
        color="orangered",
        label="$d\\theta$",
        markersize=10,
        linewidth=2,
    )
    ax.plot(
        t[1:-1],
        np.rad2deg(cumulative_turning_angles),
        "x--",
        color="olivedrab",
        label="$\\theta$",
        markersize=10,
        linewidth=2,
    )
    ax.plot(
        t[1:-1],
        np.abs(np.rad2deg(cumulative_turning_angles)),
        "s--",
        color="seagreen",
        label="$|\\theta|$",
        markersize=10,
        linewidth=2,
    )

    ax.axhline(0, color="black", linestyle="--")

    arrow_distance = FancyArrowPatch(
        (t[argmax_cumulative_turning_angle + 1], 0),
        (
            t[argmax_cumulative_turning_angle + 1],
            np.rad2deg(
                np.abs(cumulative_turning_angles[argmax_cumulative_turning_angle])
            ),
        ),
        color="purple",
        arrowstyle="<->",
        mutation_scale=20,
        lw=3,
        zorder=10,
    )

    ax.add_patch(arrow_distance)

    ax.text(
        t[argmax_cumulative_turning_angle + 1] + 0.1,
        np.rad2deg(abs(cumulative_turning_angles[argmax_cumulative_turning_angle])) / 2,
        "$\\theta_{max}$",
        fontsize=30,
        color="purple",
    )

    # draw trapezoids
    # for i in range(n_points - 3):
    #     ax.fill(
    #         [
    #             t[i + 1],
    #             t[i + 2],
    #             t[i + 2],
    #             t[i + 1],
    #         ],
    #         [
    #             0,
    #             0,
    #             np.rad2deg(cumulative_turning_angles[i + 1]),
    #             np.rad2deg(cumulative_turning_angles[i]),
    #         ],
    #         color="lightgrey",
    #         alpha=0.5,
    #     )

    # ax.text(
    #     t[7] + 0.1,
    #     np.rad2deg(cumulative_turning_angles[5]) / 2,
    #     "$\\Theta$",
    #     fontsize=30,
    #     color="dimgrey",
    # )

    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("angle [°]")

    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    ax.legend()

    plt.tight_layout()
    plt.savefig(
        "../data/figures/measures/maximum_cumulative_turning_angle_breakthrough.pdf"
    )
    plt.close()

    # reset font size
    plt.rcParams.update({"font.size": 10})

    # ------------------------------
    # turn intensity
    # ------------------------------
    start_angle = np.arctan2(start_direction[1], start_direction[0])
    steps_positions = compute_steps(trajectory, n_average=n_points_entrance)
    steps = np.diff(steps_positions, axis=0)
    step_directions = steps / np.linalg.norm(steps, axis=1)[:, None]
    step_angles = np.arctan2(step_directions[:, 1], step_directions[:, 0])
    step_angles_to_desired = step_angles - start_angle
    step_deviation = np.linalg.norm(steps, axis=1) * np.sin(step_angles_to_desired)
    orthogonal_direction = np.array([-start_direction[1], start_direction[0]])

    projection_step = (
        steps_positions[1:] - step_deviation[:, None] * orthogonal_direction
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_trajectory(ax)
    ax = plot_intended_direction(ax)

    ax.plot(
        steps_positions[:, 0] / 1000,
        steps_positions[:, 1] / 1000,
        "s-",
        linewidth=2,
        color="coral",
    )

    # plot angles
    # for i in range(1, n_points - 1):
    #     end_point = positions[i] + start_direction * 500
    #     ax.plot(
    #         [
    #             positions[i, 0] / 1000,
    #             end_point[0] / 1000,
    #         ],
    #         [
    #             positions[i, 1] / 1000,
    #             end_point[1] / 1000,
    #         ],
    #         color="gray",
    #         linestyle="--",
    #     )
    #     angle = np.arctan2(start_direction[1], start_direction[0]) - np.arctan2(
    #         positions[i + 1, 1] - positions[i, 1],
    #         positions[i + 1, 0] - positions[i, 0],
    #     )

    #     if angle > 0:
    #         p1, p2 = positions[i + 1], end_point
    #     else:
    #         p1, p2 = end_point, positions[i + 1]
    #     angle_annotation = AngleAnnotation(
    #         positions[i] / 1000,
    #         p1 / 1000,
    #         p2 / 1000,
    #         ax=ax,
    #         size=20,
    #         color="gray",
    #     )

    # plot steps angles
    for i in range(len(steps_positions) - 1):
        ax.plot(
            [steps_positions[i, 0] / 1000, projection_step[i, 0] / 1000],
            [
                steps_positions[i, 1] / 1000,
                projection_step[i, 1] / 1000,
            ],
            color="black",
            linestyle="--",
        )

        fancy_arrow = FancyArrowPatch(
            (steps_positions[i + 1, 0] / 1000, steps_positions[i + 1, 1] / 1000),
            (projection_step[i, 0] / 1000, projection_step[i, 1] / 1000),
            color="purple",
            arrowstyle="<->",
            mutation_scale=20,
            lw=2,
            zorder=10,
            shrinkA=0,
            shrinkB=0,
        )

        mid_point_arrow = (steps_positions[i + 1] + projection_step[i]) / 2 / 1000

        ax.text(
            mid_point_arrow[0] + 0.2,
            mid_point_arrow[1],
            f"$\\lambda_{i}$",
            fontsize=16,
            color="purple",
        )

        ax.add_patch(fancy_arrow)

        if step_angles_to_desired[i] < 0:
            p1, p2 = steps_positions[i + 1], projection_step[i]
            mid_point_line = (
                steps_positions[i] + start_direction * 700 - orthogonal_direction * 150
            ) / 1000
        else:
            p1, p2 = projection_step[i], steps_positions[i + 1]
            mid_point_line = (
                steps_positions[i] + start_direction * 700 + orthogonal_direction * 150
            ) / 1000

        ax.text(
            mid_point_line[0],
            mid_point_line[1],
            f"$\\omega_{i}$",
            fontsize=16,
            color="darkgreen",
        )
        angle_annotation = AngleAnnotation(
            steps_positions[i] / 1000,
            p1 / 1000,
            p2 / 1000,
            ax=ax,
            size=70,
            color="darkgreen",
            linewidth=3,
            zorder=40,
        )

    ax = format_plot(ax)

    plt.savefig("../data/figures/measures/turn_intensity.pdf")
    plt.close()

    # ------------------------------
    # Frechet distance
    # ------------------------------

    D: List[List[Tuple[float, int, int]]] = [
        [(float(np.inf), 0, 0) for _ in range(n_points + 1)]
        for _ in range(n_points + 1)
    ]
    D[0][0] = (0.0, 0, 0)
    curr_max = 0
    for i in range(1, n_points + 1):
        for j in range(1, n_points + 1):
            dt = float(
                np.linalg.norm(positions[i - 1] - straight_line_trajectory[j - 1, 1:3])
            )
            m = min(
                [
                    (D[i - 1][j][0], i - 1, j),
                    (D[i][j - 1][0], i, j - 1),
                    (D[i - 1][j - 1][0], i - 1, j - 1),
                ],
                key=lambda x: x[0],
            )
            D[i][j] = (max(m[0], dt), m[1], m[2])

    cur_max, i_frechet, j_frechet = 0, 0, 0
    curr_i, curr_j = n_points, n_points
    mapping = []
    while curr_i > 0 and curr_j > 0:
        if D[curr_i][curr_j][0] >= cur_max:
            cur_max = D[curr_i][curr_j][0]
            i_frechet, j_frechet = curr_i, curr_j

        mapping.append((curr_i - 1, curr_j - 1))
        curr_i, curr_j = D[curr_i][curr_j][1], D[curr_i][curr_j][2]

    mapping = mapping[::-1]
    # free_space = np.zeros((n_points, n_points))
    # for i in range(n_points):
    #     for j in range(n_points):
    #         free_space[i, j] = D[i + 1][j + 1][0]

    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(free_space, cmap="viridis")
    # ax.set_xlabel("straight line trajectory")
    # ax.set_ylabel("trajectory")
    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_trajectory(ax)
    ax = plot_straight_line_trajectory(ax)

    # plot mapping
    for i, j in mapping:
        ax.plot(
            [
                positions[i, 0] / 1000,
                straight_line_trajectory[j, 1] / 1000,
            ],
            [
                positions[i, 1] / 1000,
                straight_line_trajectory[j, 2] / 1000,
            ],
            color="black",
            linestyle="--",
        )

    # draw frechet distance
    arrow = FancyArrowPatch(
        (positions[i_frechet - 1, 0] / 1000, positions[i_frechet - 1, 1] / 1000),
        (
            straight_line_trajectory[j_frechet - 1, 1] / 1000,
            straight_line_trajectory[j_frechet - 1, 2] / 1000,
        ),
        color="purple",
        arrowstyle="<->",
        mutation_scale=20,
        lw=2,
        zorder=30,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)

    ax.text(
        positions[i_frechet - 1, 0] / 1000 + 0.1,
        positions[i_frechet - 1, 1] / 1000 - 0.5,
        "$\\delta_{F}$",
        fontsize=16,
        color="purple",
    )

    ax = format_plot(ax)

    plt.savefig("../data/figures/measures/frechet_distance.pdf")
    plt.close()

    # ------------------------------
    # deviation index
    # ------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_trajectory(ax, scatter=True)

    # plot distance between first and last point
    arrow = FancyArrowPatch(
        (positions[0, 0] / 1000, positions[0, 1] / 1000),
        (positions[-1, 0] / 1000, positions[-1, 1] / 1000),
        color="coral",
        arrowstyle="<->",
        mutation_scale=20,
        lw=2,
        zorder=30,
        shrinkA=0,
        shrinkB=0,
    )

    ax.add_patch(arrow)

    ax.text(
        (positions[0, 0] + positions[-1, 0]) / 2000,
        (positions[0, 1] + positions[-1, 1]) / 2000 + 0.1,
        "$D$",
        fontsize=16,
        color="coral",
    )

    # plot all distances
    for i in range(n_points - 1):
        arrow = FancyArrowPatch(
            (positions[i, 0] / 1000, positions[i, 1] / 1000),
            (positions[i + 1, 0] / 1000, positions[i + 1, 1] / 1000),
            color="darkgreen",
            arrowstyle="<->",
            mutation_scale=10,
            lw=1.5,
            zorder=30,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arrow)

    ax = format_plot(ax)

    plt.savefig("../data/figures/measures/deviation_index.pdf")
    plt.close()

    # ------------------------------
    # LCSS distance
    # ------------------------------

    eps = 800
    L = [[0] * (n_points + 1) for i in range(n_points + 1)]
    for i in range(n_points + 1):
        for j in range(n_points + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif (
                np.linalg.norm(trajectory[i - 1] - straight_line_trajectory[j - 1])
                < eps
            ):
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # find the mapping
    mapping = []
    i, j = n_points, n_points
    while i > 0 and j > 0:
        if np.linalg.norm(trajectory[i - 1] - straight_line_trajectory[j - 1]) < eps:
            mapping.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    fig, ax = plt.subplots(figsize=(10, 10))

    ax = plot_trajectory(ax)
    ax = plot_straight_line_trajectory(ax)

    # add epsilon band around straight trajectory
    for i in range(n_points):
        circle = Circle(
            (
                straight_line_trajectory[i, 1] / 1000,
                straight_line_trajectory[i, 2] / 1000,
            ),
            eps / 1000,
            # edgecolor="purple",
            facecolor="lavender",
            # linestyle="--",
            # alpha=0.7,
        )
        ax.add_patch(circle)

    # add radius of epsilon
    circle_idx = 16

    circle = Circle(
        (
            straight_line_trajectory[circle_idx, 1] / 1000,
            straight_line_trajectory[circle_idx, 2] / 1000,
        ),
        eps / 1000,
        edgecolor="purple",
        fill=False,
        # linestyle="--",
        zorder=30,
    )

    ax.add_patch(circle)
    direction_arrow = np.array([-start_direction[1], start_direction[0]])
    arrow = FancyArrowPatch(
        (
            straight_line_trajectory[circle_idx, 1] / 1000,
            straight_line_trajectory[circle_idx, 2] / 1000,
        ),
        (
            straight_line_trajectory[circle_idx, 1] / 1000
            + direction_arrow[0] * eps / 1000,
            straight_line_trajectory[circle_idx, 2] / 1000
            + direction_arrow[1] * eps / 1000,
        ),
        color="purple",
        arrowstyle="<->",
        mutation_scale=20,
        lw=1,
        zorder=30,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)
    ax.text(
        straight_line_trajectory[circle_idx, 1] / 1000 + 0.15,
        straight_line_trajectory[circle_idx, 2] / 1000 + 0.25,
        "$\\epsilon$",
        fontsize=16,
        color="purple",
    )

    for i, j in mapping:
        ax.plot(
            [
                trajectory[i, 1] / 1000,
                straight_line_trajectory[j, 1] / 1000,
            ],
            [
                trajectory[i, 2] / 1000,
                straight_line_trajectory[j, 2] / 1000,
            ],
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

    ax = format_plot(ax)

    plt.savefig("../data/figures/measures/lcss_distance.pdf")
    # plt.show()
    plt.close()

    # ------------------------------
    # Curvature
    # ------------------------------

    fig, ax = plt.subplots(figsize=(10, 10))

    curvature = np.array(compute_curvature_gradient(trajectory)) * 1000

    normals = compute_normals(trajectory)
    tangents = compute_tangents(trajectory)

    ax.plot(
        trajectory[:, 1] / 1000,
        trajectory[:, 2] / 1000,
        linestyle=(0, (2, 1)),
        linewidth=2,
        color="cornflowerblue",
    )

    scatter = ax.scatter(
        trajectory[:, 1] / 1000,
        trajectory[:, 2] / 1000,
        s=100,
        c=curvature,
        cmap="gnuplot2",
        vmin=-1,
        vmax=1,
        zorder=10,
        edgecolors="cornflowerblue",
        linewidths=2,
    )
    cb = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cb.set_label("curvature [1/m]", fontsize=16)
    cb.ax.tick_params(labelsize=16)

    centers = trajectory[:, 1:3] + 1000 / curvature[:, None] * normals

    # plot radius of curvature
    circle = Circle(
        (
            centers[10, 0] / 1000,
            centers[10, 1] / 1000,
        ),
        1 / np.abs(curvature[10]),
        # edgecolor="purple",
        facecolor="lavender",
        linestyle="--",
        linewidth=2,
        zorder=0,
    )
    ax.add_patch(circle)

    # add arrow
    arrow = FancyArrowPatch(
        (
            trajectory[10, 1] / 1000,
            trajectory[10, 2] / 1000,
        ),
        (
            centers[10, 0] / 1000,
            centers[10, 1] / 1000,
        ),
        color="purple",
        arrowstyle="<->",
        mutation_scale=20,
        lw=2,
        zorder=30,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)

    # add tangent
    end1 = trajectory[10, 1:3] - 1500 * tangents[10]
    end2 = trajectory[10, 1:3] + 1500 * tangents[10]

    ax.plot(
        [end1[0] / 1000, end2[0] / 1000],
        [end1[1] / 1000, end2[1] / 1000],
        color="gray",
        linestyle="--",
        linewidth=1.5,
    )

    ax = format_plot(ax)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/measures/curvature.pdf")
