from pedestrians_social_binding.environment import Environment

from parameters import *
from utils import *


if __name__ == "__main__":

    env_name = "diamor"
    env = Environment(env_name, data_dir="../../atc-diamor-pedestrians/data/formatted")

    env_name_short = env_name.split(":")[0]
    XMIN, XMAX, YMIN, YMAX = env.get_boundaries()
    days = get_all_days(env_name)
    soc_binding_type, soc_binding_names, soc_binding_values, _ = get_social_values(
        env_name
    )

    thresholds_ped = get_pedestrian_thresholds(env_name)
    thresholds_group = get_groups_thresholds()

    sizes = {}
    breadths = {}
    depths = {}

    groups = env.get_groups(
        size=2,
        ped_thresholds=thresholds_ped,
        group_thresholds=thresholds_group,
        with_social_binding=True,
    )

    for group in groups:
        group_id = group.get_id()

        soc_binding = group.get_annotation(soc_binding_type)

        if soc_binding not in sizes:
            sizes[soc_binding] = []
            breadths[soc_binding] = []
            depths[soc_binding] = []

        size = group.get_interpersonal_distance()
        depth, breadth = group.get_depth_and_breadth()

        sizes[soc_binding] += list(size)
        breadths[soc_binding] += list(breadth)
        depths[soc_binding] += list(depth)

        # group.plot_2D_trajectory()

    pickle_save(f"../data/pickle/group_sizes.pkl", sizes)
