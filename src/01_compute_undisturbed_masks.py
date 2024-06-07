from pedestrians_social_binding.environment import Environment
from pedestrians_social_binding.trajectory_utils import (
    compute_simultaneous_observations,
)

from tqdm import tqdm
import numpy as np

from parameters import VICINITY

from utils import (
    get_pedestrian_thresholds,
    get_groups_thresholds,
    get_social_values,
    get_all_days,
    pickle_save,
)

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
    thresholds_groups = get_groups_thresholds(corridor=True)

    masks_undisturbed = {}

    for day in days:

        groups = env.get_groups(
            days=[day],
            size=2,
            ped_thresholds=thresholds_indiv,
            group_thresholds=thresholds_groups,
        )

        all_pedestrians = env.get_pedestrians(
            days=[day],
            no_groups=False,
            # thresholds=thresholds_indiv,
        )

        individuals = env.get_pedestrians(
            days=[day],
            no_groups=True,
            thresholds=thresholds_indiv,
        )

        masks_undisturbed[day] = {"group": {}, "non_group": {}}

        encountered = []

        for group in tqdm(groups):
            group_id = group.get_id()
            group_members_id = [m.get_id() for m in group.get_members()]
            group_as_indiv = group.get_as_individual()
            group_members = group.get_members()
            soc_binding = group.get_annotation(soc_binding_type)
            if soc_binding not in soc_binding_values:
                soc_binding = "other"

            group_trajectory = group_as_indiv.get_trajectory()
            group_times = group_trajectory[:, 0]

            group_encounters = group_as_indiv.get_encountered_pedestrians(
                all_pedestrians, proximity_threshold=VICINITY, skip=group_members_id
            )

            mask_undisturbed = np.ones(len(group_times), dtype=bool)

            vicinity_trajectories = []
            for encountered_pedestrian in group_encounters:
                encountered.append(encountered_pedestrian.get_id())
                [
                    traj_group_simultaneous,
                    traj_encountered_pedestrian_simultaneous,
                ] = compute_simultaneous_observations(
                    [
                        group_trajectory,
                        encountered_pedestrian.get_trajectory(),
                    ]
                )

                in_vicinity_simultaneous = (
                    np.linalg.norm(
                        traj_group_simultaneous[:, 1:3]
                        - traj_encountered_pedestrian_simultaneous[:, 1:3],
                        axis=1,
                    )
                    < VICINITY
                )
                times_in_vicinity_simultaneous = traj_group_simultaneous[
                    in_vicinity_simultaneous, 0
                ]

                # find the times in the global trajectory that correspond to the times in the simultaneous trajectory
                _, indices, _ = np.intersect1d(
                    group_times, times_in_vicinity_simultaneous, return_indices=True
                )
                mask_in_vicinity = np.zeros(len(group_times), dtype=bool)
                mask_in_vicinity[indices] = True

                # plot_static_2D_trajectories(
                #     [group_trajectory[mask_in_vicinity]], boundaries=env.boundaries
                # )

                mask_undisturbed = np.logical_and(
                    mask_undisturbed, np.logical_not(mask_in_vicinity)
                )

            masks_undisturbed[day]["group"][group_id] = mask_undisturbed

        for individual in tqdm(individuals):
            individual_id = individual.get_id()
            if individual_id not in encountered:
                continue
            individual_trajectory = individual.get_trajectory()
            individual_times = individual_trajectory[:, 0]

            individual_encounters = individual.get_encountered_pedestrians(
                all_pedestrians, proximity_threshold=VICINITY
            )

            mask_undisturbed = np.ones(len(individual_times), dtype=bool)

            vicinity_trajectories = []

            for encountered_pedestrian in individual_encounters:
                [
                    traj_individual_simultaneous,
                    traj_encountered_pedestrian_simultaneous,
                ] = compute_simultaneous_observations(
                    [individual_trajectory, encountered_pedestrian.get_trajectory()]
                )

                in_vicinity_simultaneous = (
                    np.linalg.norm(
                        traj_individual_simultaneous[:, 1:3]
                        - traj_encountered_pedestrian_simultaneous[:, 1:3],
                        axis=1,
                    )
                    < VICINITY
                )
                times_in_vicinity_simultaneous = traj_individual_simultaneous[
                    in_vicinity_simultaneous, 0
                ]

                # find the times in the global trajectory that correspond to the times in the simultaneous trajectory
                _, indices, _ = np.intersect1d(
                    individual_times,
                    times_in_vicinity_simultaneous,
                    return_indices=True,
                )
                mask_in_vicinity = np.zeros(len(individual_times), dtype=bool)
                mask_in_vicinity[indices] = True

                mask_undisturbed = np.logical_and(
                    mask_undisturbed, np.logical_not(mask_in_vicinity)
                )

            # plot_static_2D_trajectories(
            #     [individual_trajectory, individual_trajectory[mask_undisturbed]],
            #     labels=[
            #         "all",
            #         "undisturbed",
            #     ],
            #     boundaries=env.boundaries,
            # )

            masks_undisturbed[day]["non_group"][individual_id] = mask_undisturbed
    # save the masks
    pickle_save(f"../data/pickle/undisturbed_masks.pkl", masks_undisturbed)
