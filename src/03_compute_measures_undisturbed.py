from pedestrians_social_binding.environment import Environment
from pedestrians_social_binding.constants import DAYS_DIAMOR
from pedestrians_social_binding.utils import pickle_load, pickle_save
from pedestrians_social_binding.trajectory_utils import (
    get_trajectory_at_times,
    get_pieces,
    compute_continuous_sub_trajectories_using_time,
    smooth_trajectory_savitzy_golay,
    compute_trajectory_direction,
    compute_net_displacement,
    resample_trajectory,
)
from pedestrians_social_binding.plot_utils import plot_static_2D_trajectories

from utils import (
    get_pedestrian_thresholds,
    get_groups_thresholds,
    get_social_values,
    compute_measures,
)

from parameters import MEASURES, SAMPLING, SMOOTHING_WINDOW, N_POINTS_ENTRANCE


from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    env_name = "diamor:corridor"
    # Setup relevant variables
    env_name_short = env_name.split(":")[0]
    env = Environment(
        env_name, data_dir="../../atc-diamor-pedestrians/data/formatted", raw=True
    )
    days = DAYS_DIAMOR
    (
        soc_binding_type,
        soc_binding_names,
        soc_binding_values,
        colors,
    ) = get_social_values(env_name)
    thresholds_ped = get_pedestrian_thresholds(env_name)
    thresholds_group = get_groups_thresholds()

    undisturbed_masks = pickle_load("../data/pickle/undisturbed_masks.pkl")

    measures = {
        measure["name"]: {
            "groups": {soc_binding: [] for soc_binding in soc_binding_values},
            "individuals": [],
        }
        for measure in MEASURES
    }
    measures["sizes"] = {
        "groups": {soc_binding: [] for soc_binding in soc_binding_values},
        "individuals": [],
    }

    for day in days:
        # print(f"Day {day}:")

        groups = env.get_groups(
            days=[day],
            size=2,
            ped_thresholds=thresholds_ped,
            group_thresholds=thresholds_group,
        )

        individuals = env.get_pedestrians(
            days=[day],
            no_groups=True,
            thresholds=thresholds_ped,
        )

        # compute straightness for all the groups
        for group in tqdm(groups):
            # get the times where the group is undisturbed
            group_id = group.get_id()
            soc_binding = group.get_annotation(soc_binding_type)
            if soc_binding not in soc_binding_values:
                continue

            group_times = group.get_as_individual().get_trajectory()[:, 0]
            if group_id not in undisturbed_masks[day]["group"]:
                continue
            undisturbed_mask = undisturbed_masks[day]["group"][group_id]

            # compute the straightness for each pedestrian in the group
            for group_member in group.get_members():
                # get the trajectory of the pedestrian, filter it to keep only the times where the group is in the corridor
                group_member_id = group_member.get_id()

                group_member_trajectory = group_member.get_trajectory()

                pedestrian_trajectory = get_trajectory_at_times(
                    group_member_trajectory, group_times
                )

                pedestrian_undisturbed_trajectory = pedestrian_trajectory[
                    undisturbed_mask
                ]

                # cut the trajectory
                sub_trajectories = compute_continuous_sub_trajectories_using_time(
                    pedestrian_undisturbed_trajectory, max_gap=0.5
                )

                measures_traj = {measure["name"]: [] for measure in MEASURES}

                for sub_trajectory in sub_trajectories:
                    if sub_trajectory.shape[0] < 2:
                        continue
                    sub_trajectory = resample_trajectory(
                        sub_trajectory,
                        SAMPLING,
                        interpolation="spline",
                    )

                    if sub_trajectory.shape[0] < 2:
                        continue
                    sub_trajectory = resample_trajectory(
                        sub_trajectory,
                        SAMPLING,
                        interpolation="spline",
                    )
                    if len(sub_trajectory) < SMOOTHING_WINDOW:
                        # print("Not enough points for smoothing")
                        continue
                    sub_trajectory = smooth_trajectory_savitzy_golay(
                        sub_trajectory, SMOOTHING_WINDOW
                    )
                    pieces = get_pieces(sub_trajectory, piece_size=4000)

                    for i, piece_trajectory in enumerate(pieces):
                        direction_start = compute_trajectory_direction(
                            piece_trajectory[:N_POINTS_ENTRANCE]
                        )
                        direction_end = compute_trajectory_direction(
                            piece_trajectory[-N_POINTS_ENTRANCE:]
                        )
                        if (
                            direction_start is None
                            or direction_end is None
                            or direction_start != direction_end
                        ):
                            continue
                        # if direction_start is None:
                        #     continue

                        # compute trajectory size
                        traj_size = compute_net_displacement(piece_trajectory[:, 1:3])
                        measures["sizes"]["groups"][soc_binding].append(traj_size)

                        measures_group = compute_measures(
                            piece_trajectory, N_POINTS_ENTRANCE
                        )

                        for measure in MEASURES:
                            measures_traj[measure["name"]].append(
                                measures_group[measure["name"]]
                            )

                for measure in MEASURES:
                    if len(measures_traj[measure["name"]]) == 0:
                        continue
                    measures[measure["name"]]["groups"][soc_binding].append(
                        np.nanmean(measures_traj[measure["name"]])
                    )

        # compute straightness for the non groups
        for non_group in tqdm(individuals):
            non_group_id = non_group.get_id()

            non_group_trajectory = non_group.get_trajectory()

            if non_group_id not in undisturbed_masks[day]["non_group"]:
                continue
            undisturbed_mask = undisturbed_masks[day]["non_group"][non_group_id]

            # pedestrian_undisturbed_trajectory = non_group_trajectory
            pedestrian_undisturbed_trajectory = non_group_trajectory[undisturbed_mask]

            # cut the trajectory
            sub_trajectories = compute_continuous_sub_trajectories_using_time(
                pedestrian_undisturbed_trajectory, max_gap=0.5
            )

            measures_traj = {measure["name"]: [] for measure in MEASURES}

            for sub_trajectory in sub_trajectories:
                if sub_trajectory.shape[0] < 2:
                    continue
                sub_trajectory = resample_trajectory(
                    sub_trajectory,
                    SAMPLING,
                    interpolation="linear",
                )
                if len(sub_trajectory) < SMOOTHING_WINDOW:
                    # print("Not enough points for smoothing")
                    continue
                sub_trajectory = smooth_trajectory_savitzy_golay(
                    sub_trajectory, SMOOTHING_WINDOW
                )
                pieces = get_pieces(sub_trajectory, piece_size=4000)

                for i, piece_trajectory in enumerate(pieces):
                    direction_start = compute_trajectory_direction(
                        piece_trajectory[:N_POINTS_ENTRANCE]
                    )
                    direction_end = compute_trajectory_direction(
                        piece_trajectory[-N_POINTS_ENTRANCE:]
                    )
                    if (
                        direction_start is None
                        or direction_end is None
                        or direction_start != direction_end
                    ):
                        continue
                    # if direction_start is None:
                    #     continue

                    # compute trajectory size
                    traj_size = compute_net_displacement(piece_trajectory[:, 1:3])
                    measures["sizes"]["individuals"].append(traj_size)

                    measures_individual = compute_measures(
                        piece_trajectory, N_POINTS_ENTRANCE
                    )
                    for measure in MEASURES:
                        measures_traj[measure["name"]].append(
                            measures_individual[measure["name"]]
                        )

            for measure in MEASURES:
                if len(measures_traj[measure["name"]]) == 0:
                    continue
                measures[measure["name"]]["individuals"].append(
                    np.nanmean(measures_traj[measure["name"]])
                )

    pickle_save(f"../data/pickle/measures_undisturbed.pkl", measures)
