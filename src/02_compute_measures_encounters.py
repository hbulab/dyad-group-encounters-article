from matplotlib import pyplot as plt
from pedestrians_social_binding.environment import Environment
from pedestrians_social_binding.trajectory_utils import (
    compute_simultaneous_observations,
    compute_relative_direction,
    smooth_trajectory_savitzy_golay,
    resample_trajectory,
    compute_net_displacement,
    compute_lateral_distance_obstacle,
    align_trajectories_at_origin,
    compute_average_velocity,
)
from pedestrians_social_binding.plot_utils import plot_static_2D_trajectories

from utils import (
    pickle_save,
    get_pedestrian_thresholds,
    get_groups_thresholds,
    get_all_days,
    get_social_values,
    compute_measures,
)

from tqdm import tqdm
import numpy as np

from parameters import MEASURES, SAMPLING, SMOOTHING_WINDOW, N_POINTS_ENTRANCE


def plot_trajectories(
    traj_group, traj_non_group, traj_group_aligned, traj_non_group_aligned
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(traj_group[:, 1], traj_group[:, 2], label="Group", s=1)
    axes[0].quiver(
        traj_group[::10, 1],
        traj_group[::10, 2],
        traj_group[::10, 5],
        traj_group[::10, 6],
    )
    axes[0].scatter(traj_non_group[:, 1], traj_non_group[:, 2], label="Individual", s=1)
    axes[0].quiver(
        traj_non_group[::10, 1],
        traj_non_group[::10, 2],
        traj_non_group[::10, 5],
        traj_non_group[::10, 6],
    )
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_title("Trajectories")
    axes[0].legend()
    axes[0].set_aspect("equal")

    axes[1].scatter(
        traj_group_aligned[:, 1], traj_group_aligned[:, 2], label="Group", s=1
    )
    axes[1].scatter(
        traj_non_group_aligned[:, 1],
        traj_non_group_aligned[:, 2],
        label="Individual",
        s=1,
    )
    axes[1].quiver(
        traj_non_group_aligned[::10, 1],
        traj_non_group_aligned[::10, 2],
        traj_non_group_aligned[::10, 5],
        traj_non_group_aligned[::10, 6],
    )
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    axes[1].set_title("Trajectories")
    axes[1].legend()
    axes[1].set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    env_name = "diamor:corridor"
    env = Environment(
        env_name, data_dir="../../atc-diamor-pedestrians/data/formatted", raw=True
    )
    env_name_short = env_name.split(":")[0]

    (
        soc_binding_type,
        soc_binding_names,
        soc_binding_values,
        colors,
    ) = get_social_values(env_name)
    days = get_all_days(env_name)

    thresholds_indiv = get_pedestrian_thresholds(env_name)
    thresholds_groups = get_groups_thresholds()

    measures = {
        measure["name"]: {
            "groups": {soc_binding: [] for soc_binding in soc_binding_values},
            "individuals": {soc_binding: [] for soc_binding in soc_binding_values},
        }
        for measure in MEASURES
    }
    measures["entrance"] = {
        "groups": {soc_binding: [] for soc_binding in soc_binding_values},
        "individuals": {soc_binding: [] for soc_binding in soc_binding_values},
    }
    measures["sizes"] = {
        "groups": {soc_binding: [] for soc_binding in soc_binding_values},
        "individuals": {soc_binding: [] for soc_binding in soc_binding_values},
    }
    measures["n_points"] = {
        "groups": {soc_binding: [] for soc_binding in soc_binding_values},
        "individuals": {soc_binding: [] for soc_binding in soc_binding_values},
    }
    measures["velocities"] = {
        "groups": {soc_binding: [] for soc_binding in soc_binding_values},
        "individuals": {soc_binding: [] for soc_binding in soc_binding_values},
    }

    for day in days:
        # print(f"Day {day}:")
        groups = env.get_groups(
            days=[day],
            size=2,
            ped_thresholds=thresholds_indiv,
            group_thresholds=thresholds_groups,
        )

        all_pedestrians = env.get_pedestrians(
            days=[day],
            no_groups=True,
            thresholds=thresholds_indiv,
        )

        for group in tqdm(groups):
            group_id = group.get_id()
            group_members_id = [m.get_id() for m in group.get_members()]
            group_as_indiv = group.get_as_individual()
            group_members = group.get_members()
            soc_binding = group.get_annotation(soc_binding_type)
            if soc_binding not in soc_binding_values:
                continue

            group_encounters = group_as_indiv.get_encountered_pedestrians(
                all_pedestrians,
                proximity_threshold=None,
                skip=group_members_id,
                alone=None,
            )

            for non_group in group_encounters:
                non_group_id = non_group.get_id()

                trajectories = [
                    group_members[0].get_trajectory(),
                    group_members[1].get_trajectory(),
                    group_as_indiv.get_trajectory(),
                    non_group.get_trajectory(),
                ]
                [
                    traj_A,
                    traj_B,
                    traj_group,
                    traj_non_group,
                ] = compute_simultaneous_observations(trajectories)

                if len(traj_group) < 2:
                    continue

                # resample trajectories
                traj_A = resample_trajectory(traj_A, SAMPLING, interpolation="spline")
                traj_B = resample_trajectory(traj_B, SAMPLING, interpolation="spline")
                traj_group = resample_trajectory(
                    traj_group, SAMPLING, interpolation="spline"
                )
                traj_non_group = resample_trajectory(
                    traj_non_group, SAMPLING, interpolation="spline"
                )

                # smooth trajectories
                if (
                    len(traj_A) < SMOOTHING_WINDOW
                    or len(traj_B) < SMOOTHING_WINDOW
                    or len(traj_non_group) < SMOOTHING_WINDOW
                    or len(traj_group) < SMOOTHING_WINDOW
                ):
                    # print("Not enough points for smoothing")
                    continue
                traj_A = smooth_trajectory_savitzy_golay(traj_A, SMOOTHING_WINDOW)
                traj_B = smooth_trajectory_savitzy_golay(traj_B, SMOOTHING_WINDOW)
                traj_non_group = smooth_trajectory_savitzy_golay(
                    traj_non_group, SMOOTHING_WINDOW
                )
                traj_group = smooth_trajectory_savitzy_golay(
                    traj_group, SMOOTHING_WINDOW
                )

                # find the points where the trajectories are close enough
                in_vicinity = np.logical_and(
                    np.abs(traj_group[:, 1] - traj_non_group[:, 1]) <= 4000,
                    np.abs(traj_group[:, 2] - traj_non_group[:, 2]) <= 4000,
                )

                traj_A_vicinity = traj_A[in_vicinity]
                traj_B_vicinity = traj_B[in_vicinity]
                traj_group_vicinity = traj_group[in_vicinity]
                traj_non_group_vicinity = traj_non_group[in_vicinity]

                # verify that trajectories have enough points
                if len(traj_group_vicinity) < 6:
                    continue

                # verify that the trajectories are in opposite directions
                relative_direction = compute_relative_direction(
                    traj_group_vicinity[:N_POINTS_ENTRANCE],
                    traj_non_group_vicinity[:N_POINTS_ENTRANCE],
                    rel_dir_angle_cos=np.cos(np.pi / 4),
                    rel_dir_min_perc=0.75,
                )
                if relative_direction != "opposite":
                    continue

                # verify that the group does not separate
                group_interpersonal_distance = np.linalg.norm(
                    traj_A_vicinity[:, 1:3] - traj_B_vicinity[:, 1:3],
                    axis=1,
                )
                if np.any(group_interpersonal_distance > 3000):
                    continue

                # verify if entrance distance is large enough
                start_distance = np.linalg.norm(
                    traj_group_vicinity[0, 1:3] - traj_non_group_vicinity[0, 1:3]
                ).astype(float)
                if start_distance < 3000:
                    continue

                # verify if the exit distance is large enough
                end_distance = np.linalg.norm(
                    traj_group_vicinity[-1, 1:3] - traj_non_group_vicinity[-1, 1:3]
                ).astype(float)
                if end_distance < 3000:
                    continue

                # verify if the trajectories are long enough
                traj_size_A = compute_net_displacement(traj_A_vicinity[:, 1:3])
                traj_size_B = compute_net_displacement(traj_B_vicinity[:, 1:3])
                traj_size_non_group = compute_net_displacement(
                    traj_non_group_vicinity[:, 1:3]
                )
                if (
                    traj_size_A < 3000
                    or traj_size_B < 3000
                    or traj_size_non_group < 3000
                    or traj_size_A > 5000
                    or traj_size_B > 5000
                    or traj_size_non_group > 5000
                ):
                    continue
                measures["sizes"]["groups"][soc_binding].extend(
                    [traj_size_A, traj_size_B]
                )
                measures["sizes"]["individuals"][soc_binding].append(
                    traj_size_non_group
                )
                # compute the impact parameter
                # fix velocities
                velocities_group = np.diff(traj_group_vicinity[:, 1:3], axis=0) / (
                    SAMPLING / 1000
                )
                velocities_group = np.concatenate(
                    [velocities_group, velocities_group[-1:]], axis=0
                )
                velocities_non_group = np.diff(
                    traj_non_group_vicinity[:, 1:3], axis=0
                ) / (SAMPLING / 1000)
                velocities_non_group = np.concatenate(
                    [velocities_non_group, velocities_non_group[-1:]], axis=0
                )
                traj_group_vicinity[:, 5:7] = velocities_group
                traj_non_group_vicinity[:, 5:7] = velocities_non_group

                traj_group_aligned, [traj_non_group_aligned] = (
                    align_trajectories_at_origin(
                        traj_group_vicinity,
                        [traj_non_group_vicinity],
                    )
                )

                # compute impact parameter
                entrance_distance = (
                    compute_lateral_distance_obstacle(
                        traj_non_group_aligned,
                        np.array([0, 0]),
                        N_POINTS_ENTRANCE,
                    )
                    / 1000
                )
                measures["entrance"]["groups"][soc_binding].extend(
                    [entrance_distance, entrance_distance]
                )
                measures["entrance"]["individuals"][soc_binding].append(
                    entrance_distance
                )

                # plot_trajectories(
                #     traj_group_vicinity,
                #     traj_non_group_vicinity,
                #     traj_group_aligned,
                #     traj_non_group_aligned,
                # )

                # plot_static_2D_trajectories(
                #     [traj_A_vicinity, traj_B_vicinity, traj_non_group_vicinity],
                #     title=f"{entrance_distance_A:.2f} - {entrance_distance_B:.2f} - {entrance_distance_non_group:.2f}",
                #     colors=["cornflowerblue", "cornflowerblue", "orange"],
                #     gradient=True,
                # )

                measures_A = compute_measures(traj_A_vicinity, N_POINTS_ENTRANCE)
                measures_B = compute_measures(traj_B_vicinity, N_POINTS_ENTRANCE)
                measures_non_group = compute_measures(
                    traj_non_group_vicinity, N_POINTS_ENTRANCE
                )

                for measure in MEASURES:
                    measures[measure["name"]]["groups"][soc_binding].extend(
                        [measures_A[measure["name"]], measures_B[measure["name"]]]
                    )
                    measures[measure["name"]]["individuals"][soc_binding].append(
                        measures_non_group[measure["name"]]
                    )

                n_points_A = len(traj_A_vicinity)
                n_points_B = len(traj_B_vicinity)
                n_points_non_group = len(traj_non_group_vicinity)

                measures["n_points"]["groups"][soc_binding].extend(
                    [n_points_A, n_points_B]
                )
                measures["n_points"]["individuals"][soc_binding].append(
                    n_points_non_group
                )

                velocity_A = compute_average_velocity(traj_A_vicinity)
                velocity_B = compute_average_velocity(traj_B_vicinity)
                velocity_non_group = compute_average_velocity(traj_non_group_vicinity)

                measures["velocities"]["groups"][soc_binding].extend(
                    [velocity_A, velocity_B]
                )
                measures["velocities"]["individuals"][soc_binding].append(
                    velocity_non_group
                )

    pickle_save(f"../data/pickle/measures_encounters.pkl", measures)
