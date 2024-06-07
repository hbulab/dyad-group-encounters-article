import pickle as pkl
import numpy as np
from scipy.stats import f_oneway, ttest_ind

from parameters import (
    BOUNDARIES_ATC_CORRIDOR,
    BOUNDARIES_DIAMOR_CORRIDOR,
    DAYS_ATC,
    DAYS_DIAMOR,
    GROUP_BREADTH_MAX_CORRIDOR,
    SOCIAL_RELATIONS_EN,
    INTENSITIES_OF_INTERACTION_NUM,
    COLORS_SOC_REL,
    COLORS_INTERACTION,
    VEL_MIN,
    VEL_MAX,
    GROUP_BREADTH_MIN,
    GROUP_BREADTH_MAX,
)


from pedestrians_social_binding.threshold import Threshold


from pedestrians_social_binding.trajectory_utils import (
    compute_sinuosity,
    compute_sinuosity_without_rediscretization,
    compute_max_cumulative_turning_angle,
    compute_straightness_index,
    compute_maximum_lateral_deviation_using_vel,
    compute_integral_cumulative_turning_angle,
    compute_integral_deviation,
    compute_dynamic_time_warping_deviation,
    compute_lcss_deviation,
    compute_edr_deviation,
    compute_erp_deviation,
    compute_discrete_frechet_deviation,
    compute_curvature_gradient,
    compute_turn_intensity,
    compute_suddenness_turn,
    compute_tortuosity_from_curvature,
    compute_simultaneous_frechet_deviation,
    compute_euclidean_deviation,
)


def pickle_load(file_path: str):
    """Load the content of a pickle file

    Parameters
    ----------
    file_path : str
        The path to the file which will be unpickled

    Returns
    -------
    obj
        The content of the pickle file
    """
    with open(file_path, "rb") as f:
        data = pkl.load(f)
    return data


def pickle_save(file_path: str, data):
    """Save data to pickle file

    Parameters
    ----------
    file_path : str
        The path to the file where the data will be saved
    data : obj
        The data to save
    """
    with open(file_path, "wb") as f:
        pkl.dump(data, f)


def get_pedestrian_thresholds(env_name):
    """Return the thresholds for the pedestrians

    Parameters
    ----------
    env_name : str
        The name of the environment

    Returns
    -------
    list
        The list of thresholds for the pedestrians
    """

    thresholds_indiv = []
    thresholds_indiv += [
        Threshold("v", min=VEL_MIN, max=VEL_MAX)
    ]  # velocity in [0.5; 3]m/s
    thresholds_indiv += [Threshold("n", min=16)]

    # corridor threshold for ATC
    if env_name == "atc:corridor":
        thresholds_indiv += [
            Threshold(
                "x", BOUNDARIES_ATC_CORRIDOR["xmin"], BOUNDARIES_ATC_CORRIDOR["xmax"]
            )
        ]
        thresholds_indiv += [
            Threshold(
                "y", BOUNDARIES_ATC_CORRIDOR["ymin"], BOUNDARIES_ATC_CORRIDOR["ymax"]
            )
        ]
    elif env_name == "diamor:corridor":
        thresholds_indiv += [
            Threshold(
                "x",
                BOUNDARIES_DIAMOR_CORRIDOR["xmin"],
                BOUNDARIES_DIAMOR_CORRIDOR["xmax"],
            )
        ]
        thresholds_indiv += [
            Threshold(
                "y",
                BOUNDARIES_DIAMOR_CORRIDOR["ymin"],
                BOUNDARIES_DIAMOR_CORRIDOR["ymax"],
            )
        ]

    return thresholds_indiv


def get_groups_thresholds(corridor=False):
    """Return the thresholds for the groups

    Parameters
    ----------
    corridor : bool
        Whether the groups are in a corridor

    Returns
    -------
    list
        The list of thresholds for the groups
    """
    # threshold on the distance between the group members
    group_thresholds = [
        Threshold(
            "delta",
            min=GROUP_BREADTH_MIN,
            max=GROUP_BREADTH_MAX if not corridor else GROUP_BREADTH_MAX_CORRIDOR,
        )
    ]
    return group_thresholds


def get_all_days(env_name):
    """Return the days for the environment

    Parameters
    ----------
    env_name : str
        The name of the environment

    Returns
    -------
    int
        The list of days for the environment
    """
    if "atc" in env_name:
        return DAYS_ATC
    elif "diamor" in env_name:
        return DAYS_DIAMOR
    else:
        raise ValueError(f"Unknown env {env_name}")


def get_social_values(env_name):
    """Return the social values for the environment

    Parameters
    ----------
    env_name : str
        The name of the environment

    Returns
    -------
    tuple
        The social values for the environment
    """
    if "atc" in env_name:
        return "soc_rel", SOCIAL_RELATIONS_EN, [2, 1, 3, 4], COLORS_SOC_REL
    elif "diamor" in env_name:
        return (
            "interaction",
            INTENSITIES_OF_INTERACTION_NUM,
            [0, 1, 2, 3],
            COLORS_INTERACTION,
        )
    else:
        raise ValueError(f"Unknown env {env_name}")


def compute_measures(trajectory, n_points_entrance):
    """
    Compute measures for a trajectory

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory of the pedestrian
    n_points_entrance : int
        The number of points at the entrance

    Returns
    -------
    dict
        The measures for the trajectory
    """
    measures = {}
    # compute vertical deviation
    max_deviation = compute_maximum_lateral_deviation_using_vel(
        trajectory, n_points_entrance  # , plot=True
    )
    measures["max_lateral_deviation"] = max_deviation
    # compute integral deviation
    integral_deviation = compute_integral_deviation(
        trajectory, n_points_entrance, normalize=False
    )
    measures["integral_deviation"] = integral_deviation
    # compute cumulative turning angle
    max_cumulative_turning_angle = compute_max_cumulative_turning_angle(
        trajectory[:, 1:3], rediscretize=False
    )
    measures["max_cumulative_turning_angle"] = max_cumulative_turning_angle
    # compute integral cumulative turning angle
    integral_cumulative_turning_angle = compute_integral_cumulative_turning_angle(
        trajectory, rediscretize=False, normalize=True
    )
    measures["integral_cumulative_turning_angle"] = integral_cumulative_turning_angle
    # compute sinuosity
    sinuosity = compute_sinuosity(trajectory[:, 1:3], step_length=50)
    measures["sinuosity"] = sinuosity
    sinuosity_without_rediscretization = compute_sinuosity_without_rediscretization(
        trajectory[:, 1:3]
    )
    measures["sinuosity_without_rediscretization"] = sinuosity_without_rediscretization
    # compute straightness
    straightness = 1 - compute_straightness_index(trajectory[:, 1:3])
    measures["straightness_index"] = straightness
    # compute time warping
    dynamic_time_warping_deviation = compute_dynamic_time_warping_deviation(
        trajectory, n_average=n_points_entrance
    )
    measures["dynamic_time_warping_deviation"] = dynamic_time_warping_deviation
    # compute lcss
    lcss_deviation = compute_lcss_deviation(trajectory, n_average=n_points_entrance)
    measures["lcss_deviation"] = lcss_deviation
    # compute edr
    edr_deviation = compute_edr_deviation(trajectory, n_average=n_points_entrance)
    measures["edit_distance_deviation"] = edr_deviation
    # compute erp
    erp_deviation = compute_erp_deviation(trajectory, n_average=n_points_entrance)
    measures["edit_distance_real_penalty_deviation"] = erp_deviation
    # compute discrete frechet
    frechet_deviation = compute_discrete_frechet_deviation(
        trajectory, n_average=n_points_entrance
    )
    measures["frechet_deviation"] = frechet_deviation
    # compute simultaneous frechet
    simultaneous_frechet_deviation = compute_simultaneous_frechet_deviation(
        trajectory, n_average=n_points_entrance
    )
    measures["simultaneous_frechet_deviation"] = simultaneous_frechet_deviation
    # compute euclidean distance
    euclidean_distance = compute_euclidean_deviation(
        trajectory, n_average=n_points_entrance
    )
    measures["euclidean_distance"] = euclidean_distance
    # compute curvature
    curvature = compute_curvature_gradient(trajectory)
    measures["mean_curvature"] = np.nanmean(np.abs(curvature))
    measures["total_curvature"] = np.trapz(np.abs(curvature), trajectory[:, 0])
    measures["energy_curvature"] = np.trapz(np.array(curvature) ** 2, trajectory[:, 0])

    # compute turn intensity
    turn_intensity = compute_turn_intensity(trajectory, n_average=n_points_entrance)
    measures["turn_intensity"] = np.nanmean(turn_intensity)
    # compute suddenness turn
    suddenness_turn = compute_suddenness_turn(trajectory, n_average=n_points_entrance)
    measures["suddenness_turn"] = np.nanmean(suddenness_turn)

    # compute tortuosity curvature
    tortuosity_curvature = compute_tortuosity_from_curvature(trajectory)
    measures["tortuosity_curvature"] = np.nanmean(tortuosity_curvature)

    return measures


def without_nan(values):
    """Remove nan values from a list

    Parameters
    ----------
    values : list
        The list of values

    Returns
    -------
    np.ndarray
        The list of values without nan
    """
    values = np.array(values)
    return values[~np.isnan(values)]


def get_scientific_notation(f):
    """Return the scientific notation of a number

    Parameters
    ----------
    f : float
        The number

    Returns
    -------
    tuple
        The scientific notation of the number
    """

    scientific_notation = f"{f:.2e}"
    v, exp = scientific_notation.split("e")
    exp_sign = exp[0]
    exp_value = exp[1:]
    # remove leading 0
    exp_value = exp_value.lstrip("0")
    if exp_sign == "+":
        exp = exp_value
    else:
        exp = f"-{exp_value}"
    return v, exp


def get_latex_scientific_notation(f, threshold=None):
    """Return the latex scientific notation of a number

    Parameters
    ----------
    f : float
        The number
    threshold : float
        A threshold (number above which the number is not formatted)

    Returns
    -------
    str
        The latex scientific notation of the number
    """
    if threshold is not None and f > threshold:
        return f"{f:.2f}"
    if f == 0:
        return "0"
    v, exp = get_scientific_notation(f)
    if exp == "":
        return v
    if exp == "1":
        return f"{v} \\times 10"
    return f"{v} \\times 10^{{{exp}}}"


def get_formatted_p_value(p_value):
    """Format the p-value

    Parameters
    ----------
    p_value : float
        The p-value

    Returns
    -------
    str
        The formatted p-value
    """
    if p_value < 1e-4:
        formatted_p_value = f"\\mathbf{{< 10^{{-4}}}}"
    else:
        s = get_latex_scientific_notation(p_value)
        if p_value < 0.05:
            formatted_p_value = f"\\mathbf{{{s}}}"
        else:
            formatted_p_value = s
    return formatted_p_value


def compute_binned_values(x_values, y_values, min_v, max_v, n_bins):
    """Compute binned values of y_values with respect to x_values

    Parameters
    ----------
    x_values : np.ndarray
        The x values
    y_values : np.ndarray
        The y values
    min_v : float
        The minimum value
    max_v : float
        The maximum value
    n_bins : int
        The number of bins

    Returns
    -------
    tuple
        The bin centers, the means, the stds, the errors, the number of values
    """

    pdf_edges = np.linspace(min_v, max_v, n_bins + 1)
    bin_centers = 0.5 * (pdf_edges[0:-1] + pdf_edges[1:])

    indices = np.digitize(x_values, pdf_edges) - 1

    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    errors = np.full(n_bins, np.nan)
    n_values = np.zeros(n_bins)

    for i in range(n_bins):
        if np.sum(indices == i) == 0:
            continue
        means[i] = np.nanmean(y_values[indices == i])
        stds[i] = np.nanstd(y_values[indices == i])
        errors[i] = stds[i] / np.sqrt(np.sum(indices == i))
        n_values[i] = np.sum(indices == i)

    return bin_centers, means, stds, errors, n_values


def compute_p_values_per_bin(x_values, y_values, min_v, max_v, n_bins):
    """Compute p-values for each bin

    Parameters
    ----------
    x_values : dict
        The x values
    y_values : dict
        The y values
    min_v : float
        The minimum value
    max_v : float
        The maximum value
    n_bins : int
        The number of bins

    Returns
    -------
    tuple
        The bin centers, the p-values
    """

    values_per_bins = {soc_binding: [] for soc_binding in [0, 1, 2, 3]}
    pdf_edges = np.linspace(min_v, max_v, n_bins + 1)
    bins_centers = (pdf_edges[1:] + pdf_edges[:-1]) / 2

    for soc_binding in [0, 1, 2, 3]:
        x_values_soc = np.array(x_values[soc_binding])
        y_values_soc = np.array(y_values[soc_binding])
        indices = np.digitize(x_values_soc, pdf_edges) - 1
        for idx_bin in range(n_bins):
            values_per_bins[soc_binding].append(
                y_values_soc[indices == idx_bin].tolist()
            )

    p_values = []
    for idx_bin in range(n_bins):
        values_for_p_values = []
        for interaction in [0, 1, 2, 3]:
            if len(values_per_bins[interaction][idx_bin]) > 0:
                values_for_p_values.append(
                    without_nan(values_per_bins[interaction][idx_bin]).tolist()
                )
            # else:
            #     print(
            #         f"WARNING: not enough values for p-value for bin {idx_bin}, {interaction}"
            #     )
        if len(values_for_p_values) > 1:
            p_values.append(f_oneway(*values_for_p_values)[1][0])  # type: ignore
        else:
            p_values.append(np.nan)
    return bins_centers, p_values


def compute_p_values_per_bin_2_groups(
    x_values_a, y_values_a, x_values_b, y_values_b, min_v, max_v, n_bins
):
    """Compute p-values for each bin

    Parameters
    ----------
    x_values_a : np.ndarray
        The x values for the first group
    y_values_a : np.ndarray
        The y values for the first group
    x_values_b : np.ndarray
        The x values for the second group
    y_values_b : np.ndarray
        The y values for the second group
    min_v : float
        The minimum value
    max_v : float
        The maximum value
    n_bins : int
        The number of bins

    Returns
    -------
    tuple
        The bin centers, the p-values
    """

    pdf_edges = np.linspace(min_v, max_v, n_bins + 1)
    bins_centers = (pdf_edges[1:] + pdf_edges[:-1]) / 2

    p_values = []
    for idx_bin in range(n_bins):
        indices_0 = np.digitize(x_values_a, pdf_edges) - 1
        indices_123 = np.digitize(x_values_b, pdf_edges) - 1

        values_0 = y_values_a[indices_0 == idx_bin]
        values_123 = y_values_b[indices_123 == idx_bin]

        if len(values_0) > 0 and len(values_123) > 0:
            p_values.append(ttest_ind(values_0, values_123, equal_var=False)[1])
        else:
            p_values.append(np.nan)

    return bins_centers, p_values
