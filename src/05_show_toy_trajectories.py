import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")
plt.rcParams.update({"font.size": 12})


import pandas as pd


from parameters import MEASURES
from utils import compute_measures, get_latex_scientific_notation

from pedestrians_social_binding.trajectory_utils import (
    compute_turning_angles,
    compute_velocity,
)


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
    trajectory[:, 0] = t * 1000
    trajectory[:, 1:3] = positions * 1000

    return trajectory


def make_breakdown_plots(trajectory, name):
    turning_angles = compute_turning_angles(trajectory[:, 1:3])
    cumulative_turning_angles = np.cumsum(turning_angles)
    idx_max_cumulative_turning_angle = np.argmax(np.abs(cumulative_turning_angles))
    t = trajectory[1:-1, 0] / 1000
    integrals = []
    for i in range(len(cumulative_turning_angles)):
        integrals += [np.trapz(cumulative_turning_angles[: i + 1], t[: i + 1])]
    integrals = np.abs(integrals)
    idx_max_integral_cumulative_turning_angle = np.argmax(integrals)

    start_point = trajectory[0, 1:3]
    velocities = compute_velocity(trajectory)
    start_vel = np.nanmean(velocities[:n_points_entrance], axis=0)
    start_vel /= np.linalg.norm(start_vel)

    distances_to_straight_line = np.cross(start_vel, trajectory[:, 1:3] - start_point)
    idx_max_distance_to_straight_line = np.argmax(np.abs(distances_to_straight_line))

    _, axes = plt.subplots(4, 1, figsize=(10, 12))
    axes[0].scatter(trajectory[:, 1], trajectory[:, 2], s=1)
    axes[0].scatter(
        trajectory[idx_max_cumulative_turning_angle, 1],
        trajectory[idx_max_cumulative_turning_angle, 2],
        color="red",
        s=100,
    )
    axes[0].scatter(
        trajectory[idx_max_integral_cumulative_turning_angle, 1],
        trajectory[idx_max_integral_cumulative_turning_angle, 2],
        color="green",
        s=100,
    )
    axes[0].scatter(
        trajectory[idx_max_distance_to_straight_line, 1],
        trajectory[idx_max_distance_to_straight_line, 2],
        color="blue",
        s=100,
    )
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")

    axes[1].plot(t, cumulative_turning_angles)
    axes[1].plot(t, abs(cumulative_turning_angles))
    axes[1].scatter(
        t[idx_max_cumulative_turning_angle],
        cumulative_turning_angles[idx_max_cumulative_turning_angle],
        color="red",
        s=100,
    )
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("cumulative turning angle [rad]")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].grid(color="black", linestyle="--", linewidth=0.5)

    axes[2].plot(t, integrals)
    axes[2].scatter(
        t[idx_max_integral_cumulative_turning_angle],
        integrals[idx_max_integral_cumulative_turning_angle],
        color="green",
        s=100,
    )
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("integral cumulative turning angle [rad]")
    axes[2].axhline(0, color="black", linestyle="--")
    axes[2].grid(color="black", linestyle="--", linewidth=0.5)

    axes[3].plot(trajectory[:, 0], distances_to_straight_line)
    axes[3].scatter(
        trajectory[idx_max_distance_to_straight_line, 0],
        distances_to_straight_line[idx_max_distance_to_straight_line],
        color="blue",
        s=100,
    )
    axes[3].set_xlabel("t [s]")
    axes[3].set_ylabel("distance to straight line [m]")
    axes[3].grid(color="black", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"../data/figures/toy_trajectories/breakdown_{name}.pdf")
    plt.close()


def make_measure_table(measures):

    df = pd.DataFrame(measures).T
    # find the order of the values for each measure
    arg_sort = df.values.argsort(axis=0)
    order = np.zeros_like(arg_sort)
    for i in range(arg_sort.shape[1]):
        order[arg_sort[:, i], i] = np.arange(arg_sort.shape[0])

    print("\\begin{table}[!htb]")
    print(
        "\\caption{Measures of deviation for the toy trajectories. The maximum value for each measure is highlighted in bold and the second smallest value is underlined (since the straight line trajectory is the smallest value for all the measures). The ranking of the trajectories (from the straightest to the most deviated) is also indicated in parenthesis.}"
    )
    print("\\label{tab:toy_measures}")
    print("\\centering")
    print("\\begin{adjustbox}{angle=90}")
    print("\\scalebox{0.5}{")
    col = "l" + "r" * len(MEASURES)
    print("\\begin{tabular}{" + col + "}")
    print("\\toprule")
    header = (
        "Measures &" + " & ".join([measure["symbol"] for measure in MEASURES]) + "\\\\"
    )
    print(header)
    print("\\midrule")
    for i, (index, row) in enumerate(df.iterrows()):
        line = trajectories_data[i]["abbreviation"]
        for measure in MEASURES:
            j = df.columns.get_loc(measure["name"])
            s = get_latex_scientific_notation(row[measure["name"]])
            if order[i, j] == len(df) - 1:
                line += f" & $\\mathbf{{{s}}}$"
            elif order[i, j] == 1:
                line += f" & $\\underline{{{s}}}$"
            else:
                line += f" & ${s}$"
            line += f" ({order[i, j] + 1})"
        line += "\\\\"
        print(line)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\end{adjustbox}")
    print("\\end{table}")


if __name__ == "__main__":
    dx = 4
    start_point = np.array([0, 0])

    lateral_deviation = 0.3  # m
    sampling_time = 0.03
    velocity = 1.2  # m/s

    smoothing_window_duration = 3  # seconds
    smoothing_window = int(smoothing_window_duration / sampling_time)

    time_entrance = 0.5  # seconds
    n_points_entrance = int(time_entrance / sampling_time)

    trajectories_data = [
        {
            "name": "straight line",
            "control_points": np.array([start_point, start_point + np.array([dx, 0])]),
            "colors": "#1f77b4",
            "noise_sigma": 0,
            "abbreviation": "a",
        },
        # {
        #     "name": "noisy straight line",
        #     "control_points": np.array([start_point, start_point + np.array([dx, 0])]),
        #     "colors": "#1A92E5",
        #     "noise_sigma": 0.01,
        # },
        {
            "name": "small deviation without recovery",
            "control_points": np.array(
                [
                    start_point,
                    np.array([dx / 6, 0]),
                    np.array([5 * dx / 6, lateral_deviation]),
                    np.array([dx, lateral_deviation]),
                ]
            ),
            "colors": "#ff7f0e",
            "abbreviation": "b",
        },
        {
            "name": "small deviation with recovery",
            "control_points": np.array(
                [
                    start_point,
                    np.array([dx / 12, 0]),
                    np.array([dx / 6, 0]),
                    np.array([dx / 2, lateral_deviation]),
                    np.array([5 * dx / 6, 0]),
                    np.array([11 * dx / 12, 0]),
                    np.array([dx, 0]),
                ]
            ),
            "colors": "#2ca02c",
            "abbreviation": "c",
        },
        {
            "name": "big deviation without recovery",
            "control_points": np.array(
                [
                    start_point,
                    np.array([dx / 12, 0]),
                    np.array([dx / 6, 0]),
                    np.array([5 * dx / 6, 2 * lateral_deviation]),
                    np.array([dx, 2 * lateral_deviation]),
                ]
            ),
            "colors": "#d62728",
            "abbreviation": "d",
        },
        {
            "name": "big deviation with recovery",
            "control_points": np.array(
                [
                    start_point,
                    np.array([dx / 12, 0]),
                    np.array([dx / 6, 0]),
                    np.array([dx / 2, 2 * lateral_deviation]),
                    np.array([5 * dx / 6, 0]),
                    np.array([11 * dx / 12, 0]),
                    np.array([dx, 0]),
                ]
            ),
            "colors": "#e377c2",
            "abbreviation": "e",
        },
        {
            "name": "fast deviation with recovery",
            "control_points": np.array(
                [
                    start_point,
                    np.array([dx / 12, 0]),
                    np.array([2 * dx / 12, 0]),
                    np.array([3 * dx / 12, 0]),
                    np.array([4 * dx / 12, 0]),
                    np.array([4.5 * dx / 12, 0]),
                    np.array([5 * dx / 12, 0]),
                    np.array([6 * dx / 12, lateral_deviation]),
                    np.array([7 * dx / 12, 0]),
                    np.array([7.5 * dx / 12, 0]),
                    np.array([8 * dx / 12, 0]),
                    np.array([9 * dx / 12, 0]),
                    np.array([10 * dx / 12, 0]),
                    np.array([11 * dx / 12, 0]),
                    np.array([12 * dx / 12, 0]),
                ]
            ),
            "colors": "#9467bd",
            "abbreviation": "f",
        },
        {
            "name": "deviation both sides",
            "control_points": np.array(
                [
                    start_point,
                    np.array([dx / 12, 0]),
                    np.array([2 * dx / 12, 0]),
                    np.array([4 * dx / 12, lateral_deviation]),
                    np.array([8 * dx / 12, -2 * lateral_deviation]),
                    np.array([11 * dx / 12, 0]),
                    np.array([11.5 * dx / 12, 0]),
                    np.array([12 * dx / 12, 0]),
                ]
            ),
            "colors": "#8c564b",
            "abbreviation": "g",
        },
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    measures = {}
    for trajectory_data in trajectories_data:
        control_points = trajectory_data["control_points"]
        if "noise_sigma" in trajectory_data:
            noise_sigma = trajectory_data["noise_sigma"]
        else:
            noise_sigma = 0
        trajectory = make_trajectory(
            control_points, velocity, sampling_time, swaying=0, noise_sigma=noise_sigma
        )

        # trajectory = smooth_trajectory_savitzy_golay(
        #     trajectory, window_size=smoothing_window
        # )

        # make_breakdown_plots(trajectory, trajectory_data["name"])

        measures[trajectory_data["name"]] = compute_measures(
            trajectory, n_points_entrance
        )

        ax.plot(
            trajectory[:, 1] / 1000,
            trajectory[:, 2] / 1000,
            color=trajectory_data["colors"],
            label=trajectory_data["abbreviation"],
            linewidth=2,
        )
        # ax.plot(
        #     smooth_trajectory[:, 1],
        #     smooth_trajectory[:, 2],
        #     color=trajectory_data["colors"],
        # )
        # ax.scatter(
        #     control_points[:, 0],
        #     control_points[:, 1],
        #     color=trajectory_data["colors"],
        #     s=20,
        # )

    # add arrow to indicate direction
    ax.annotate(
        "",
        xy=(0.4, 0.3),
        xytext=(0.1, 0.3),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=2),
    )
    ax.annotate(
        "direction of motion",
        xy=(0, 0.2),
        color="black",
        fontsize=12,
    )

    ax.set_ylim(-1, 1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(which="major", color="grey", linestyle="--", linewidth=0.5)
    ax.grid(which="minor", color="grey", linestyle="--", linewidth=0.1, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), prop={"size": 16})
    # plt.show()
    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/toy_trajectories/trajectories.pdf")
    plt.close()

    make_measure_table(measures)
