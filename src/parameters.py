import numpy as np

# ---------- ATC constants ----------
DAYS_ATC = ["0109", "0217", "0424", "0505", "0508"]
SOCIAL_RELATIONS_JP = ["idontknow", "koibito", "doryo", "kazoku", "yuujin"]
SOCIAL_RELATIONS_EN = ["idontknow", "Couples", "Colleagues", "Families", "Friends"]
BOUNDARIES_ATC = {"xmin": -41000, "xmax": 49000, "ymin": -27500, "ymax": 24000}
BOUNDARIES_ATC_CORRIDOR = {"xmin": 5000, "xmax": 48000, "ymin": -27000, "ymax": 8000}


# ---------- DIAMOR constants ----------
DAYS_DIAMOR = ["06", "08"]
INTENSITIES_OF_INTERACTION_NUM = ["0", "1", "2", "3"]
BOUNDARIES_DIAMOR = {"xmin": -200, "xmax": 60300, "ymin": -5300, "ymax": 12000}
BOUNDARIES_DIAMOR_CORRIDOR = {
    "xmin": 20000,
    "xmax": 60300,
    "ymin": -5300,
    "ymax": 12000,
}
BOUNDARIES_DIAMOR_CENTER_CORRIDOR = {
    "xmin": 30000,
    "xmax": 50000,
    "ymin": 0,
    "ymax": 7500,
}


COLORS_SOC_REL = ["black", "red", "blue", "green", "orange"]
COLORS_INTERACTION = ["blue", "red", "green", "orange"]

# velocity threshold
VEL_MIN = 500
VEL_MAX = 3000

# group breath threshold
GROUP_BREADTH_MIN = 100
GROUP_BREADTH_MAX = 5000
GROUP_BREADTH_MAX_CORRIDOR = 3000


SAMPLING = 0.03

VICINITY = 4000


SMOOTHING_WINDOW_DURATION = 3  # seconds
SMOOTHING_WINDOW = int(SMOOTHING_WINDOW_DURATION / SAMPLING)

TIME_ENTRANCE = 0.5  # seconds
N_POINTS_ENTRANCE = int(TIME_ENTRANCE / SAMPLING)


MEASURES = [
    {
        "name": "euclidean_distance",
        "symbol": "$\\delta_{E}$",
        "latex_name": "Euclidean distance",
        "limits": [0, 3000],
        "units": "(in~m)",
    },
    {
        "name": "simultaneous_frechet_deviation",
        "symbol": "$\\delta_{max}$",
        "latex_name": "lockstep maximum deviation",
        "limits": [0, 1],
        "units": "(in~m)",
    },
    {
        "name": "frechet_deviation",
        "symbol": "$\\delta_{F}$",
        "latex_name": "Fréchet deviation",
        "limits": [0, 1],
        "units": "(in~m)",
    },
    {
        "name": "max_lateral_deviation",
        "symbol": "$d_{max}$",
        "latex_name": "maximum lateral deviation",
        "limits": [0, 1],
        "units": "(in~m)",
    },
    {
        "name": "integral_deviation",
        "symbol": "$\\Delta$",
        "latex_name": "integral of lateral deviation",
        "limits": [0, 0.5],
        "units": "(in~m$^2$)",
    },
    # {
    #     "name": "tortuosity_curvature",
    #     "symbol": "$\\tau_{\\kappa}$",
    #     "latex_name": "tortuosity curvature",
    #     "limits": [0, 0.002],
    # },
    {
        "name": "dynamic_time_warping_deviation",
        "symbol": "$\\delta_{DTW}$",
        "latex_name": "dynamic time warping deviation",
        "limits": [0, 1],
        "units": "(in~m)",
    },
    {
        "name": "lcss_deviation",
        "symbol": "$\\delta_{LCSS}$",
        "latex_name": "longest common subsequence deviation",
        "limits": [0, 1],
        "units": "(dimensionless)",
    },
    {
        "name": "edit_distance_deviation",
        "symbol": "$\\delta_{Lev}$",
        "latex_name": "Levenshtein deviation",
        "limits": [0, 1],
        "units": "(dimensionless)",
    },
    # {
    #     "name": "edit_distance_real_penalty_deviation",
    #     "symbol": "$\\delta_{ERP}$",
    #     "latex_name": "edit distance with real penalty deviation",
    #     "limits": [0, 1],
    # },
    {
        "name": "straightness_index",
        "symbol": "$\\tilde{\\tau}$",
        "latex_name": "deviation index",
        "limits": [0, 0.015],
        "units": "(dimensionless)",
    },
    {
        "name": "max_cumulative_turning_angle",
        "symbol": "$\\theta_{max}$",
        "latex_name": "maximum cumulative turning angle",
        "limits": [0, np.pi / 5],
        "units": "(in~rad)",
    },
    {
        "name": "integral_cumulative_turning_angle",
        "symbol": "$\\Theta$",
        "latex_name": "integral cumulative turning angle",
        "limits": [0, np.pi / 5],
        "units": "(in~rad)",
    },
    {
        "name": "sinuosity",
        "symbol": "$S$",
        "latex_name": "sinuosity",
        "limits": [0, 0.0025],
        "units": "(in rad/m$^{\\frac{1}{2}}$)",
    },
    {
        "name": "energy_curvature",
        "symbol": "$E_{\\kappa}$",
        "latex_name": "energy curvature",
        "limits": [0, 0.002],
        "units": "(in s/m$^2$)",
    },
    # {
    #     "name": "total_curvature",
    #     "symbol": "$K$",
    #     "latex_name": "energy curvature",
    #     "limits": [0, 0.002],
    # },
    # {
    #     "name": "mean_curvature",
    #     "symbol": "$K$",
    #     "latex_name": "energy curvature",
    #     "limits": [0, 0.002],
    # },
    {
        "name": "turn_intensity",
        "symbol": "$I$",
        "latex_name": "turn intensity",
        "limits": [0, 0.05],
        "units": "(in~deg$\\times$cm)",
    },
    {
        "name": "suddenness_turn",
        "symbol": "$\\sigma$",
        "latex_name": "suddenness of turn",
        "limits": [0, 0.05],
        "units": "(in~deg$\\times$cm/s)",
    },
]
