from copy import deepcopy
from pedestrians_social_binding.utils import pickle_load

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, kruskal, alexandergovern

from utils import (
    without_nan,
    get_formatted_p_value,
    compute_binned_values,
    compute_p_values_per_bin,
    compute_p_values_per_bin_2_groups,
    get_latex_scientific_notation,
)

from parameters import MEASURES

import scienceplots

plt.style.use("science")


def make_undisturbed_latex_table(
    values,
    measure_info,
    color=None,
    info="",
    type_multi="ANOVA",
    type_pairwise="student",
):
    print("\\begin{table}[!htb]")
    if color is not None:
        print(f"\\color{{{color}}}")
    print(
        f"\\caption{{Average and standard deviation of the {measure_info['latex_name']} {measure_info['symbol']} {measure_info['units']}, for dyads and individuals in undisturbed situations{f' ({info})' if info else ''}. The {'ANOVA' if type_multi == 'ANOVA' else 'Kruskal-Wallis'} $p$-value for the difference of means between dyads with various level of interaction and the {'Student' if type_pairwise == 'student' else 'Welch'} T-test $p$-value for the difference of means between all individuals against all dyads are also shown.}}"
    )
    print(f"\\label{{tab:undisturbed_{measure_info['name']}}}")
    print("\\begin{center}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print(
        f"Intensity of interaction & {measure_info['symbol']} & Normalized & $p$-value \\\\"
    )
    print("\\midrule")
    values_for_p_values = []
    for interaction in [0, 1, 2, 3]:
        if len(values["groups"][interaction]) > 0:
            values_for_p_values.append(without_nan(values["groups"][interaction]))
        else:
            print(f"WARNING: not enough values for p-value for bin {interaction}")

    if type_multi == "ANOVA":
        p_value_03 = f_oneway(*values_for_p_values)[1]
    elif type_multi == "kruskal":
        p_value_03 = kruskal(*values_for_p_values)[1]
    elif type_multi == "alexandergovern":
        p_value_03 = alexandergovern(*values_for_p_values)[1]
    else:
        raise ValueError(f"Unknown type {type}")
    formated_p_value_03 = get_formatted_p_value(p_value_03)
    all_values = []
    for soc_binding in [0, 1, 2, 3]:
        ratio = np.array(values["groups"][soc_binding]) / np.mean(
            np.array(values["individuals"])
        )
        line = f"{soc_binding} & ${get_latex_scientific_notation(np.nanmean(values['groups'][soc_binding]))}\\pm{get_latex_scientific_notation(np.nanstd(values['groups'][soc_binding]))}$ & ${get_latex_scientific_notation(np.nanmean(ratio), threshold=0.1)}$ &"
        if soc_binding == 0:
            line += f"\\multirow{{4}}{{*}}{{${formated_p_value_03}$}}"
        line += " \\\\"
        print(line)
        all_values.extend(values["groups"][soc_binding])

    print("\\midrule")
    p_value = ttest_ind(
        without_nan(all_values),
        without_nan(values["individuals"]),
        equal_var=type_pairwise == "student",
    )[1]
    # print(np.isnan(values["individuals"][measure]).any())
    formated_p_value = get_formatted_p_value(p_value)
    ratio_all = np.array(all_values) / np.mean(np.array(values["individuals"]))
    print(
        f"All dyads & ${get_latex_scientific_notation(np.nanmean(all_values))}\\pm{get_latex_scientific_notation(np.nanstd(all_values))}$ &  $\\num{{{np.nanmean(ratio_all):.2f}}}$ & \\multirow{{2}}{{*}}{{${formated_p_value}$}}  \\\\"
    )
    ratio_indiv = np.array(values["individuals"]) / np.mean(
        np.array(values["individuals"])
    )
    print(
        f"Individuals & ${get_latex_scientific_notation(np.nanmean(values['individuals']))}\\pm{get_latex_scientific_notation(np.nanstd(values['individuals']))}$  & $\\num{{{np.nanmean(ratio_indiv):.2f}}}$ &  \\\\"
    )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")


def make_encounter_latex_table(
    values,
    measure_info,
    color=None,
    info="",
    type_multi="ANOVA",
    type_pairwise="student",
):
    print("\\begin{table}[!htb]")
    if color is not None:
        print(f"\\color{{{color}}}")
    print(
        f"\\caption{{Average and standard deviation of the {measure_info['latex_name']} {measure_info['symbol']} {measure_info['units']}, for dyads and individuals during encounters{f' ({info})' if info else ''}. {'ANOVA' if type_multi == 'ANOVA' else 'Kruskal-Wallis'} $p$-values for the difference of means between dyads (and individuals encountering dyads) with various level of interaction and the {'Student' if type_pairwise == 'student' else 'Welch'} T-test $p$-value for the difference of means between all individuals against all dyads are also shown.}}"
    )
    print(f"\\label{{tab:encounter_{measure_info['name']}}}")
    print("\\begin{center}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("& \\multicolumn{2}{c}{Individual} & \\multicolumn{2}{c}{Dyad} \\\\")
    print("\\midrule")
    print(
        f"Intensity of & {measure_info['symbol']} & {'ANOVA' if type_multi == 'ANOVA' else 'Kruskal-Wallis'} & {measure_info['symbol']} & {'ANOVA' if type_multi == 'ANOVA' else 'Kruskal-Wallis'} \\\\"
    )
    print("interaction & & $p$-value & & $p$-value \\\\")
    print("\\midrule")
    values_for_p_values_group = []
    values_for_p_values_individual = []
    for soc_binding in [0, 1, 2, 3]:
        if len(values["groups"][soc_binding]) > 0:
            values_for_p_values_group.append(without_nan(values["groups"][soc_binding]))
        # else:
        #     print(
        #         f"WARNING: not enough values for p-value for bin {soc_binding}, groups"
        #     )
        if len(values["individuals"][soc_binding]) > 0:
            values_for_p_values_individual.append(
                without_nan(values["individuals"][soc_binding])
            )
        # else:
        #     print(
        #         f"WARNING: not enough values for p-value for bin {soc_binding}, individuals"
        #     )
    if type_multi == "ANOVA":
        p_value_03_group = f_oneway(*values_for_p_values_group)[1]
        p_value_03_individual = f_oneway(*values_for_p_values_individual)[1]
    elif type_multi == "kruskal":
        p_value_03_group = kruskal(*values_for_p_values_group)[1]
        p_value_03_individual = kruskal(*values_for_p_values_individual)[1]
    elif type_multi == "alexandergovern":
        p_value_03_group = alexandergovern(*values_for_p_values_group)[1]
        p_value_03_individual = alexandergovern(*values_for_p_values_individual)[1]
    else:
        raise ValueError(f"Unknown type {type_multi}")
    formated_p_value_03_group = get_formatted_p_value(p_value_03_group)
    formated_p_value_03_individual = get_formatted_p_value(p_value_03_individual)

    all_values_individuals = []
    all_values_groups = []
    values_interaction_individuals = []
    values_interaction_groups = []
    for soc_binding in [0, 1, 2, 3]:
        if soc_binding == 0:
            print(
                f"{soc_binding} & ${get_latex_scientific_notation(np.nanmean(values['individuals'][soc_binding]))}\\pm{get_latex_scientific_notation(np.nanstd(values['individuals'][soc_binding]))}$ & \\multirow{{4}}{{*}}{{${formated_p_value_03_individual}$}} & ${get_latex_scientific_notation(np.nanmean(values['groups'][soc_binding]))}\\pm{get_latex_scientific_notation(np.nanstd(values['groups'][soc_binding]))}$ & \\multirow{{4}}{{*}}{{${formated_p_value_03_group}$}} \\\\"
            )
        else:
            print(
                f"{soc_binding} & ${get_latex_scientific_notation(np.nanmean(values['individuals'][soc_binding]))}\\pm{get_latex_scientific_notation(np.nanstd(values['individuals'][soc_binding]))}$ & & ${get_latex_scientific_notation(np.nanmean(values['groups'][soc_binding]))}\\pm{get_latex_scientific_notation(np.nanstd(values['groups'][soc_binding]))}$ & \\\\"
            )
            values_interaction_individuals.extend(values["individuals"][soc_binding])
            values_interaction_groups.extend(values["groups"][soc_binding])
        all_values_individuals.extend(values["individuals"][soc_binding])
        all_values_groups.extend(values["groups"][soc_binding])
    print("\\midrule")
    # print(
    #     f"interaction (1-3) & ${get_latex_scientific_notation(np.nanmean(values_interaction_individuals))}\\pm{get_latex_scientific_notation(np.nanstd(values_interaction_individuals))}$ & & ${get_latex_scientific_notation(np.nanmean(values_interaction_groups))}\\pm{get_latex_scientific_notation(np.nanstd(values_interaction_groups))}$ & \\\\"
    # )
    print(
        f"All & ${get_latex_scientific_notation(np.nanmean(all_values_individuals))}\\pm{get_latex_scientific_notation(np.nanstd(all_values_individuals))}$ & & ${get_latex_scientific_notation(np.nanmean(all_values_groups))}\\pm{get_latex_scientific_notation(np.nanstd(all_values_groups))}$ & \\\\"
    )
    formatted_p_value_interaction_groups = get_formatted_p_value(
        ttest_ind(
            without_nan(values_interaction_groups),
            without_nan(values["groups"][0]),
            equal_var=type_pairwise == "student",
        )[1]
    )
    formatted_p_value_interaction_individuals = get_formatted_p_value(
        ttest_ind(
            without_nan(values_interaction_individuals),
            without_nan(values["individuals"][0]),
            equal_var=type_pairwise == "student",
        )[1]
    )
    # print(
    #     f"interaction (1-3) vs 0 & & ${formatted_p_value_interaction_individuals}$ & & ${formatted_p_value_interaction_groups}$ \\\\"
    # )
    formatted_p_value_all = get_formatted_p_value(
        ttest_ind(
            without_nan(all_values_individuals),
            without_nan(all_values_groups),
            equal_var=type_pairwise == "student",
        )[1]
    )
    print(
        f"T-test $p$-value  & \multicolumn{{4}}{{c}}{{${formatted_p_value_all}$}} \\\\"
    )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")


def make_count_table_encounters(values, color=None, info=""):
    print("\\begin{table}[!htb]")
    if color is not None:
        print(f"\\color{{{color}}}")
    print(
        f"\\caption{{Number of trajectories of individuals and dyads during encounters{f' ({info})' if info else ''}.}}"
    )
    print("\\begin{center}")
    print("\\label{tab:encounters_count}")
    print("\\begin{tabular}{lc}")
    print(f"Intensity of interaction & Count \\\\")
    print("\\midrule")
    total = 0
    for soc_binding in [0, 1, 2, 3]:
        print(
            f"{soc_binding} & {len(values['sinuosity']['individuals'][soc_binding])} \\\\"
        )
        total += len(values["sinuosity"]["individuals"][soc_binding])
    print("\\midrule")
    print(f"All & {total} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")


def make_count_table_undisturbed(values, color=None, info=""):
    print("\\begin{table}[!htb]")
    if color is not None:
        print(f"\\color{{{color}}}")
    print(
        f"\\caption{{Number of trajectories of individuals and dyads in undisturbed situations{f' ({info})' if info else ''}.}}"
    )
    print("\\label{tab:undisturbed_count}")
    print("\\begin{center}")
    print("\\begin{tabular}{lc}")
    print(f"Intensity of interaction & Count \\\\")
    print("\\midrule")
    total = 0
    for soc_binding in [0, 1, 2, 3]:
        print(f"{soc_binding} & {len(values['sinuosity']['groups'][soc_binding])} \\\\")
        total += len(values["sinuosity"]["groups"][soc_binding])
    print("\\midrule")
    print(f"All dyads & {total} \\\\")
    print("\\midrule")
    print(f"Individuals & {len(values['sinuosity']['individuals'])} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")


def make_p_value_tables_encounters_undisturbed(
    values_encounters,
    values_undisturbed,
    measure_info,
    color=None,
    type_pairwise="student",
):
    print("\\begin{table}[!htb]")
    if color is not None:
        print(f"\\color{{{color}}}")
    print(
        f"\\caption{{{'Student' if type_pairwise == 'student' else 'Welch'} T-test $p$-values for the difference of means of the {measure_info['latex_name']} {measure_info['symbol']} between undisturbed individuals (resp. dyads) and individuals (resp. dyads) during encounters.}}"
    )
    print("\\begin{center}")
    print(f"\\label{{tab:p_values_encounters_undisturbed_{measure_info['name']}}}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Intensity of interaction & Individual & Dyad \\\\")
    print("\\midrule")
    all_values_undisturbed_indiv = []
    all_values_encounters_indiv = []
    all_values_undisturbed_group = []
    all_values_encounters_group = []
    for soc_binding in [0, 1, 2, 3]:
        p_value_indiv = ttest_ind(
            without_nan(values_encounters["individuals"][soc_binding]),
            without_nan(values_undisturbed["individuals"]),
            equal_var=type_pairwise == "student",
        )[1]
        formated_p_value_indiv = get_formatted_p_value(p_value_indiv)
        p_value_group = ttest_ind(
            without_nan(values_encounters["groups"][soc_binding]),
            without_nan(values_undisturbed["groups"][soc_binding]),
            equal_var=type_pairwise == "student",
        )[1]
        formated_p_value_group = get_formatted_p_value(p_value_group)
        print(
            f"{soc_binding} & ${formated_p_value_indiv}$ & ${formated_p_value_group}$ \\\\"
        )
        all_values_undisturbed_indiv.extend(values_undisturbed["individuals"])
        all_values_encounters_indiv.extend(
            values_encounters["individuals"][soc_binding]
        )
        all_values_undisturbed_group.extend(values_undisturbed["groups"])
        all_values_encounters_group.extend(values_encounters["groups"][soc_binding])
    print("\\midrule")
    p_value_indiv = ttest_ind(
        without_nan(all_values_encounters_indiv),
        without_nan(all_values_undisturbed_indiv),
        equal_var=type_pairwise == "student",
    )[1]
    formated_p_value_indiv = get_formatted_p_value(p_value_indiv)
    p_value_group = ttest_ind(
        without_nan(all_values_encounters_group),
        without_nan(all_values_undisturbed_group),
        equal_var=type_pairwise == "student",
    )[1]
    formated_p_value_group = get_formatted_p_value(p_value_group)
    print(f"All & ${formated_p_value_indiv}$ & ${formated_p_value_group}$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")


def make_p_value_tables_encounters(
    values_encounters,
    measure_info,
    color=None,
    type_pairwise="student",
):
    print("\\begin{table}[!htb]")
    if color is not None:
        print(f"\\color{{{color}}}")
    print(
        f"\\caption{{{'Student' if type_pairwise == 'student' else 'Welch'} T-test $p$-values for the difference of means of the {measure_info['latex_name']} {measure_info['symbol']} between dyads and individuals during encounters.}}"
    )
    print("\\begin{center}")
    print(f"\\label{{tab:p_values_encounters_{measure_info['name']}}}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("Intensity of interaction & $p$-value \\\\")
    print("\\midrule")
    all_values_indiv = []
    all_values_group = []
    for soc_binding in [0, 1, 2, 3]:
        p_value = ttest_ind(
            without_nan(values_encounters["groups"][soc_binding]),
            without_nan(values_encounters["individuals"][soc_binding]),
            equal_var=type_pairwise == "student",
        )[1]
        formated_p_value = get_formatted_p_value(p_value)
        print(f"{soc_binding} & ${formated_p_value}$ \\\\")
        all_values_indiv.extend(values_encounters["individuals"][soc_binding])
        all_values_group.extend(values_encounters["groups"][soc_binding])
    print("\\midrule")
    p_value = ttest_ind(
        without_nan(all_values_group),
        without_nan(all_values_indiv),
        equal_var=type_pairwise == "student",
    )[1]
    formated_p_value = get_formatted_p_value(p_value)
    print(f"All & ${formated_p_value}$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")


def make_ratio_table(
    values_encounters,
    values_undisturbed,
    measure_info,
    color=None,
    info="",
    type_multi="ANOVA",
    type_pairwise="student",
):
    baseline_individuals = np.nanmean(values_undisturbed["individuals"])

    print("\\begin{table}[!htb]")
    if color is not None:
        print(f"\\color{{{color}}}")
    print(
        f"\\caption{{Ratio of the average value for {measure_info['latex_name']} {measure_info['symbol']}, for dyads and individuals during encounters{f' ({info})' if info else ''} to the undisturbed value. {'ANOVA' if type_multi == 'ANOVA' else 'Kruskal-Wallis'} $p$-values for the difference of means between dyads (and individuals encountering dyads) with various level of interaction and the {'Student' if type_pairwise == 'student' else 'Welch'} T-test $p$-value for the difference of means between all individuals against all dyads are also shown.}}"
    )
    print(f"\\label{{tab:ratio_{measure_info['name']}}}")
    print("\\begin{center}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("& \\multicolumn{2}{c}{Individual} & \\multicolumn{2}{c}{Dyad} \\\\")
    print("\\midrule")
    print(
        f"Intensity of & Ratio {measure_info['symbol']} & {'ANOVA' if type_multi == 'ANOVA' else 'Kruskal-Wallis'} & Ratio {measure_info['symbol']} & {'ANOVA' if type_multi == 'ANOVA' else 'Kruskal-Wallis'} \\\\"
    )
    print("interaction & & $p$-value & & $p$-value \\\\")
    print("\\midrule")
    values_for_p_values_group = []
    values_for_p_values_individual = []
    for soc_binding in [0, 1, 2, 3]:
        ratio_individuals = (
            values_encounters["individuals"][soc_binding] / baseline_individuals
        )
        ratio_groups = values_encounters["groups"][soc_binding] / np.nanmean(
            values_undisturbed["groups"][soc_binding]
        )
        values_for_p_values_group.append(without_nan(ratio_groups))
        values_for_p_values_individual.append(without_nan(ratio_individuals))
    if type_multi == "ANOVA":
        p_value_03_group = f_oneway(*values_for_p_values_group)[1]
        p_value_03_individual = f_oneway(*values_for_p_values_individual)[1]
    elif type_multi == "kruskal":
        p_value_03_group = kruskal(*values_for_p_values_group)[1]
        p_value_03_individual = kruskal(*values_for_p_values_individual)[1]
    elif type_multi == "alexandergovern":
        p_value_03_group = alexandergovern(*values_for_p_values_group)[1]
        p_value_03_individual = alexandergovern(*values_for_p_values_individual)[1]
    else:
        raise ValueError(f"Unknown type {type_multi}")
    formated_p_value_03_group = get_formatted_p_value(p_value_03_group)
    formated_p_value_03_individual = get_formatted_p_value(p_value_03_individual)

    all_values_individuals = []
    all_values_groups = []
    values_interaction_individuals = []
    values_interaction_groups = []
    values_no_interaction_individuals = []
    values_no_interaction_groups = []
    for soc_binding in [0, 1, 2, 3]:
        ratio_individuals = (
            values_encounters["individuals"][soc_binding] / baseline_individuals
        )
        ratio_groups = values_encounters["groups"][soc_binding] / np.nanmean(
            values_undisturbed["groups"][soc_binding]
        )
        mean_ratio_individuals = np.nanmean(ratio_individuals)
        mean_ratio_groups = np.nanmean(ratio_groups)
        if soc_binding == 0:
            print(
                f"{soc_binding} & ${get_latex_scientific_notation(np.nanmean(mean_ratio_individuals), threshold=0.1)}$ & \\multirow{{4}}{{*}}{{${formated_p_value_03_individual}$}} & ${get_latex_scientific_notation(np.nanmean(mean_ratio_groups), threshold=0.1)}$ & \\multirow{{4}}{{*}}{{${formated_p_value_03_group}$}} \\\\"
            )
            values_no_interaction_groups.extend(ratio_groups)
            values_no_interaction_individuals.extend(ratio_individuals)
        else:
            print(
                f"{soc_binding} & ${get_latex_scientific_notation(np.nanmean(mean_ratio_individuals), threshold=0.1)}$ & & ${get_latex_scientific_notation(np.nanmean(mean_ratio_groups), threshold=0.1)}$ & \\\\"
            )
            values_interaction_individuals.extend(ratio_individuals)
            values_interaction_groups.extend(ratio_groups)
        all_values_individuals.extend(ratio_individuals)
        all_values_groups.extend(ratio_groups)
    print("\\midrule")
    formatted_p_value_all = get_formatted_p_value(
        ttest_ind(
            without_nan(all_values_individuals),
            without_nan(all_values_groups),
            equal_var=type_pairwise == "student",
        )[1]
    )
    print(
        f"T-test $p$-value  & \multicolumn{{4}}{{c}}{{${formatted_p_value_all}$}} \\\\"
    )
    formatted_p_value_interaction_individuals = get_formatted_p_value(
        ttest_ind(
            without_nan(values_interaction_individuals),
            without_nan(values_no_interaction_individuals),
            equal_var=type_pairwise == "student",
        )[1]
    )
    formatted_p_value_interaction_groups = get_formatted_p_value(
        ttest_ind(
            without_nan(values_interaction_groups),
            without_nan(values_no_interaction_groups),
            equal_var=type_pairwise == "student",
        )[1]
    )
    # print(
    #     f"Interaction 1-3 vs 0 & \multicolumn{{2}}{{c}}{{${formatted_p_value_interaction_individuals}$}} & \multicolumn{{2}}{{c}}{{${formatted_p_value_interaction_groups}$}} \\\\"
    # )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\end{table}")


def make_summary_table_undisturbed(
    measures_undisturbed, type_multi="ANOVA", type_pairwise="student"
):
    print("\\begin{table}[!htb]")
    print(
        "\\caption{Summary of the measures for the undisturbed situations. The value of the average deviation of the individuals is shown. This value is used to scale the deviation of the dyads. The {'ANOVA' if type_multi == 'ANOVA' else 'Kruskal-Wallis'} $p$-value for the difference of means between dyads with various level of interaction and the {'Student' if type_pairwise == 'student' else 'Welch'} T-test $p$-value for the difference of means between all individuals against all dyads are also shown.}"
    )
    print("\\label{tab:summary_undisturbed}")
    print("\\begin{center}")
    col = "l" + "c" * len(MEASURES)
    print("\\begin{adjustbox}{angle=90}")
    print("\\scalebox{0.6}{")
    print(f"\\begin{{tabular}}{{{col}}}")
    print("\\toprule")
    print(
        "Intensity of interaction & "
        + " & ".join([m["symbol"] for m in MEASURES])
        + " \\\\"
    )
    print("\\midrule")
    all_measures_for_anova = {}
    all_measures_for_means = {}
    # print ratios for each interaction level
    for soc_binding in [0, 1, 2, 3]:
        line = f"{soc_binding}"
        for measure_info in MEASURES:
            measure_name = measure_info["name"]
            mean_individuals = np.nanmean(
                measures_undisturbed[measure_info["name"]]["individuals"]
            )
            measure_groups = measures_undisturbed[measure_info["name"]]["groups"][
                soc_binding
            ]
            ratios = np.array(measure_groups) / mean_individuals
            mean_ratio = np.nanmean(ratios)
            line += f" & $\\num{{{mean_ratio:.2f}}}$"
            if measure_name not in all_measures_for_anova:
                all_measures_for_anova[measure_name] = []
            all_measures_for_anova[measure_name].append(ratios)
            if measure_name not in all_measures_for_means:
                all_measures_for_means[measure_name] = []
            all_measures_for_means[measure_name].extend(measure_groups)
        line += " \\\\"
        print(line)
    print("\\midrule")
    # print anova p-values
    line = "ANOVA $p$-value"
    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        if type_multi == "ANOVA":
            p_value = f_oneway(*all_measures_for_anova[measure_name])[1]
        elif type_multi == "kruskal":
            p_value = kruskal(*all_measures_for_anova[measure_name])[1]
        elif type_multi == "alexandergovern":
            p_value = alexandergovern(*all_measures_for_anova[measure_name])[1]
        else:
            raise ValueError(f"Unknown type {type_multi}")
        formated_p_value = get_formatted_p_value(p_value)
        line += f" & ${formated_p_value}$"
    line += " \\\\"
    print(line)
    print("\\midrule")
    # print ratio all groups
    line = "Group means"
    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        mean_individuals = np.nanmean(
            measures_undisturbed[measure_info["name"]]["individuals"]
        )
        measure_groups = all_measures_for_means[measure_name]
        mean_ratio = np.nanmean(measure_groups) / mean_individuals
        line += f" & $\\num{{{mean_ratio:.2f}}}$"
    line += " \\\\"
    print(line)
    # print individual means
    line = "Individual means"
    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        mean_individuals = np.nanmean(
            measures_undisturbed[measure_info["name"]]["individuals"]
        )
        line += f" & ${get_latex_scientific_notation(mean_individuals)}$"
    line += " \\\\"
    print(line)
    print("\\midrule")
    # print t-test p-values
    line = "T-test $p$-value"
    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        p_value = ttest_ind(
            all_measures_for_means[measure_name],
            measures_undisturbed[measure_info["name"]]["individuals"],
            equal_var=type_pairwise == "student",
        )[1]
        formated_p_value = get_formatted_p_value(p_value)
        line += f" & ${formated_p_value}$"
    line += " \\\\"
    print(line)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\end{adjustbox}")
    print("\\end{center}")
    print("\\end{table}")


def make_summary_table_encounters(
    measures_encounters,
    measures_undisturbed,
    type_multi="ANOVA",
    type_pairwise="student",
):
    print("\\begin{table}[!htb]")
    print(
        "\\caption{Summary of the measures for the encounter situations. The ratio of the average value for dyads and individuals during encounters to the undisturbed value is shown. The {'ANOVA' if type_multi == 'ANOVA' else 'Kruskal-Wallis'} $p$-value for the difference of means between dyads (and individuals encountering dyads) with various level of interaction and the {'Student' if type_pairwise == 'student' else 'Welch'} T-test $p$-value for the difference of means between all individuals against all dyads are also shown.}"
    )
    print("\\label{tab:summary_encounters}")
    print("\\begin{center}")
    col = "ll" + "c" * len(MEASURES)
    print("\\begin{adjustbox}{angle=90}")
    print("\\scalebox{0.6}{")
    print(f"\\begin{{tabular}}{{{col}}}")
    print("\\toprule")
    print(
        "& Intensity of interaction & "
        + " & ".join([m["symbol"] for m in MEASURES])
        + " \\\\"
    )
    print("\\midrule")
    all_measures_for_anova = {"groups": {}, "individuals": {}}
    all_measures_for_means = {"groups": {}, "individuals": {}}
    # print ratios for each interaction level
    for party in ["individuals", "groups"]:
        party_name = "Dyads" if party == "groups" else "Individuals"
        for soc_binding in [0, 1, 2, 3]:
            if soc_binding == 0:
                line = f"\\multirow{{6}}{{*}}{{{party_name}}}"
            else:
                line = ""
            line += f" & {soc_binding}"
            for measure_info in MEASURES:
                measure_name = measure_info["name"]
                if party == "individuals":
                    mean_undisturbed = np.nanmean(
                        measures_undisturbed[measure_info["name"]]["individuals"]
                    )
                else:
                    mean_undisturbed = np.nanmean(
                        measures_undisturbed[measure_info["name"]][party][soc_binding]
                    )
                measure_encounters = measures_encounters[measure_info["name"]][party][
                    soc_binding
                ]
                ratios = np.array(measure_encounters) / mean_undisturbed
                mean_ratio = np.nanmean(ratios)
                line += f" & $\\num{{{mean_ratio:.2f}}}$"
                if measure_name not in all_measures_for_anova[party]:
                    all_measures_for_anova[party][measure_name] = []
                all_measures_for_anova[party][measure_name].append(ratios)
                if measure_name not in all_measures_for_means[party]:
                    all_measures_for_means[party][measure_name] = []
                all_measures_for_means[party][measure_name].extend(measure_encounters)
            line += " \\\\"
            print(line)
        print(f"\\cmidrule{{2-{len(MEASURES)+2}}}")
        # print anova p-values
        line = "& ANOVA $p$-value"
        for measure_info in MEASURES:
            measure_name = measure_info["name"]
            if type_multi == "ANOVA":
                p_value = f_oneway(*all_measures_for_anova[party][measure_name])[1]
            elif type_multi == "kruskal":
                p_value = kruskal(*all_measures_for_anova[party][measure_name])[1]
            elif type_multi == "alexandergovern":
                p_value = alexandergovern(*all_measures_for_anova[party][measure_name])[
                    1
                ]
            else:
                raise ValueError(f"Unknown type {type_multi}")
            formated_p_value = get_formatted_p_value(p_value)
            line += f" & ${formated_p_value}$"
        line += " \\\\"
        print(line)
        print(f"\\cmidrule{{2-{len(MEASURES)+2}}}")
        line = f" & Mean"
        for measure_info in MEASURES:
            measure_name = measure_info["name"]
            mean = np.nanmean(all_measures_for_means[party][measure_name])
            line += f" & ${get_latex_scientific_notation(mean)}$"
        line += " \\\\"
        print(line)
        print("\\midrule")
    # print t-test p-values
    line = "\\multicolumn{2}{c}{T-test $p$-value}"
    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        p_value = ttest_ind(
            all_measures_for_means["groups"][measure_name],
            all_measures_for_means["individuals"][measure_name],
            equal_var=type_pairwise == "student",
        )[1]
        formated_p_value = get_formatted_p_value(p_value)
        line += f" & ${formated_p_value}$"
    line += " \\\\"
    print(line)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\end{adjustbox}")
    print("\\end{center}")
    print("\\end{table}")


def make_summary_table_p_value(
    measures_encounters,
    measures_undisturbed,
    type_pairwise="student",
):
    print("\\begin{table}[!htb]")
    print(
        "\\caption{{'Student' if type_pairwise == 'student' else 'Welch'} T-test $p$-values for the difference of means of the measures between dyads (resp. individuals) during encounters and dyads (resp. individuals) in undisturbed situations.}"
    )
    print("\\label{tab:p_values_encounters_undisturbed}")
    print("\\begin{center}")
    print("\\begin{adjustbox}{angle=90}")
    print("\\scalebox{0.6}{")
    col = "ll" + "c" * len(MEASURES)
    print(f"\\begin{{tabular}}{{{col}}}")
    print("\\toprule")
    print(
        "& Intensity of interaction & "
        + " & ".join([m["symbol"] for m in MEASURES])
        + " \\\\"
    )
    print("\\midrule")
    for party in ["individuals", "groups"]:
        all_values_undisturbed = {}
        all_values_encounters = {}
        party_name = "Dyads" if party == "groups" else "Individuals"
        for soc_binding in [0, 1, 2, 3]:
            if soc_binding == 0:
                line = f"\\multirow{{5}}{{*}}{{{party_name}}}"
            line = f"& {soc_binding}"
            for measure_info in MEASURES:
                measure_name = measure_info["name"]
                if measure_name not in all_values_undisturbed:
                    all_values_undisturbed[measure_name] = []
                    all_values_encounters[measure_name] = []
                if party == "individuals":
                    all_values_undisturbed[measure_name].extend(
                        measures_undisturbed[measure_name]["individuals"]
                    )
                    p_value = ttest_ind(
                        without_nan(
                            measures_encounters[measure_name]["individuals"][
                                soc_binding
                            ]
                        ),
                        without_nan(measures_undisturbed[measure_name]["individuals"]),
                        equal_var=type_pairwise == "student",
                    )[1]
                else:
                    all_values_undisturbed[measure_name].extend(
                        measures_undisturbed[measure_name][party][soc_binding]
                    )
                    p_value = ttest_ind(
                        without_nan(
                            measures_encounters[measure_name][party][soc_binding]
                        ),
                        without_nan(
                            measures_undisturbed[measure_name][party][soc_binding]
                        ),
                        equal_var=type_pairwise == "student",
                    )[1]
                all_values_encounters[measure_name].extend(
                    measures_encounters[measure_name][party][soc_binding]
                )
                formated_p_value = get_formatted_p_value(p_value)
                line += f" & ${formated_p_value}$"
            line += " \\\\"
            print(line)
        print(f"\\cmidrule{{2-{len(MEASURES)+2}}}")
        line = f" & All"
        for measure_info in MEASURES:
            measure_name = measure_info["name"]
            p_value = ttest_ind(
                without_nan(all_values_encounters[measure_name]),
                without_nan(all_values_undisturbed[measure_name]),
                equal_var=type_pairwise == "student",
            )[1]
            formated_p_value = get_formatted_p_value(p_value)
            line += f" & ${formated_p_value}$"
        line += " \\\\"
        print(line)
        print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\end{adjustbox}")
    print("\\end{center}")
    print("\\end{table}")

    print("\\begin{table}[!htb]")
    print(
        "\\caption{{'Student' if type_pairwise == 'student' else 'Welch'} T-test $p$-values for the difference of means of the measures between dyads and individuals during encounters.}"
    )
    print("\\label{tab:p_values_encounters}")
    print("\\begin{center}")
    print("\\begin{adjustbox}{angle=90}")
    print("\\scalebox{0.6}{")
    col = "l" + "c" * len(MEASURES)
    print(f"\\begin{{tabular}}{{{col}}}")
    print("\\toprule")
    print(
        "Intensity of interaction & "
        + " & ".join([m["symbol"] for m in MEASURES])
        + " \\\\"
    )
    print("\\midrule")
    all_values_individuals = {}
    all_values_groups = {}
    for soc_binding in [0, 1, 2, 3]:
        line = f"{soc_binding}"
        for measure_info in MEASURES:
            measure_name = measure_info["name"]
            if measure_name not in all_values_individuals:
                all_values_individuals[measure_name] = []
                all_values_groups[measure_name] = []
            all_values_individuals[measure_name].extend(
                measures_encounters[measure_name]["individuals"][soc_binding]
            )
            all_values_groups[measure_name].extend(
                measures_encounters[measure_name]["groups"][soc_binding]
            )
            p_value = ttest_ind(
                without_nan(
                    measures_encounters[measure_name]["individuals"][soc_binding]
                ),
                without_nan(measures_encounters[measure_name]["groups"][soc_binding]),
                equal_var=type_pairwise == "student",
            )[1]
            formated_p_value = get_formatted_p_value(p_value)
            line += f" & ${formated_p_value}$"
        line += " \\\\"
        print(line)
    print("\\midrule")
    line = f"All"
    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        p_value = ttest_ind(
            without_nan(all_values_individuals[measure_name]),
            without_nan(all_values_groups[measure_name]),
            equal_var=type_pairwise == "student",
        )[1]
        formated_p_value = get_formatted_p_value(p_value)
        line += f" & ${formated_p_value}$"
    line += " \\\\"
    print(line)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\end{adjustbox}")
    print("\\end{center}")
    print("\\end{table}")


if __name__ == "__main__":
    env_name = "diamor:corridor"
    env_name_short = env_name.split(":")[0]

    measures_encounters = pickle_load(
        f"../data/pickle/measures_encounters.pkl",
    )
    measures_undisturbed = pickle_load(f"../data/pickle/measures_undisturbed.pkl")
    group_sizes = pickle_load(f"../data/pickle/group_sizes.pkl")

    # ===========================================================
    # ===========================================================

    # filter the encounters measures
    filtered_measures_encounters = deepcopy(measures_encounters)

    threshold_entrance = 2
    for party in ["individuals", "groups"]:
        for soc_binding in [0, 1, 2, 3]:
            entrance_distance = np.array(
                filtered_measures_encounters["entrance"][party][soc_binding]
            )
            mask_threshold = entrance_distance < threshold_entrance
            # print(np.sum(mask_threshold))
            for measure_info in MEASURES:
                filtered_measures_encounters[measure_info["name"]][party][
                    soc_binding
                ] = np.array(
                    filtered_measures_encounters[measure_info["name"]][party][
                        soc_binding
                    ]
                )[
                    mask_threshold
                ].tolist()

    # ===========================================================
    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        make_undisturbed_latex_table(
            measures_undisturbed[measure_name],
            measure_info,
            type_multi="kruskal",
            type_pairwise="welch",
        )

    # make_count_table_undisturbed(measures_undisturbed)

    print("\\clearpage")

    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        make_encounter_latex_table(
            filtered_measures_encounters[measure_name],
            measure_info,
            type_multi="kruskal",
            type_pairwise="welch",
        )

    print("\\clearpage")

    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        make_ratio_table(
            filtered_measures_encounters[measure_name],
            measures_undisturbed[measure_name],
            measure_info,
            type_multi="kruskal",
            type_pairwise="welch",
        )

    print("\\clearpage")

    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        make_p_value_tables_encounters_undisturbed(
            filtered_measures_encounters[measure_name],
            measures_undisturbed[measure_name],
            measure_info,
            type_pairwise="welch",
        )

    print("\\clearpage")

    for measure_info in MEASURES:
        measure_name = measure_info["name"]
        make_p_value_tables_encounters(
            filtered_measures_encounters[measure_name],
            measure_info,
            type_pairwise="welch",
        )

    print("\\clearpage")

    # make_count_table_encounters(filtered_measures_encounters)
    # make_count_table_undisturbed(measures_undisturbed)

    # make_summary_table_undisturbed(measures_undisturbed, type_multi="kruskal", type_pairwise="welch")
    # make_summary_table_encounters(filtered_measures_encounters, measures_undisturbed)
    # make_summary_table_p_value(
    #     filtered_measures_encounters,
    #     measures_undisturbed,
    #     type_pairwise="welch",
    # )

    # ===========================================================
    # ===========================================================
    # ====================== GRAPHS =============================
    # ===========================================================
    # ===========================================================

    # fig, ax = plt.subplots(len(MEASURES), 4, figsize=(20, 40))

    MIN_DIST = 0
    MAX_DIST = 4
    N_BINS = 4

    # for i, measure_info in enumerate(MEASURES):
    #     measure_name = measure_info["name"]
    #     for j, party in enumerate(["groups", "individuals"]):
    #         ratios_per_intensity = {}
    #         for soc_binding in [0, 1, 2, 3]:
    #             entrance_distance = np.array(
    #                 measures_encounters["entrance"][party][soc_binding]
    #             ) / (np.nanmean(group_sizes[soc_binding]) / 1000)
    #             measures = np.array(
    #                 measures_encounters[measure_name][party][soc_binding]
    #             )
    #             if party == "groups":
    #                 ratio = np.array(measures) / np.nanmean(
    #                     measures_undisturbed[measure_name][party][soc_binding]
    #                 )
    #             else:
    #                 ratio = np.array(measures) / np.nanmean(
    #                     measures_undisturbed[measure_name]["individuals"]
    #                 )
    #             ratios_per_intensity[soc_binding] = ratio
    #             (
    #                 entrance_bins_centers,
    #                 mean_measure_wrt_entrance,
    #                 std_measure_wrt_entrance,
    #                 error_measure_wrt_entrance,
    #                 n_values,
    #             ) = compute_binned_values(
    #                 entrance_distance,
    #                 ratio,
    #                 MIN_DIST,
    #                 MAX_DIST,
    #                 N_BINS,
    #             )

    #             ax[i][2 * j].errorbar(
    #                 entrance_bins_centers,
    #                 mean_measure_wrt_entrance,
    #                 yerr=error_measure_wrt_entrance,
    #                 label=soc_binding,
    #                 capsize=5,
    #             )

    #         entrance_bins_centers, p_values = compute_p_values_per_bin(
    #             measures_encounters["entrance"][party],
    #             ratios_per_intensity,
    #             MIN_DIST,
    #             MAX_DIST,
    #             N_BINS,
    #         )

    #         ax[i][2 * j + 1].scatter(
    #             entrance_bins_centers,
    #             p_values,
    #             marker="X",
    #         )

    #         # add threshold
    #         ax[i][2 * j + 1].plot(
    #             [MIN_DIST, MAX_DIST],
    #             [0.05, 0.05],
    #             color="red",
    #             linestyle="--",
    #             linewidth=1,
    #         )
    #         ax[i][2 * j + 1].text(
    #             1.5,
    #             0.08,
    #             "p=0.05",
    #             horizontalalignment="center",
    #             verticalalignment="center",
    #         )

    #         # styling
    #         ax[i][2 * j].set_xlabel("$\\bar{r}_{b}$")
    #         ax[i][2 * j].set_ylabel("ratio {measure_info['symbol']}")
    #         ax[i][2 * j].set_xlim([MIN_DIST, MAX_DIST])
    #         ax[i][2 * j].set_ylim([0, 3])
    #         ax[i][2 * j].legend()
    #         ax[i][2 * j].grid()

    #         ax[i][2 * j + 1].set_xlabel("$\\bar{r}_{b}$")
    #         ax[i][2 * j + 1].set_ylabel(f"p-value for {measure_info['symbol']}")
    #         ax[i][2 * j + 1].set_ylim([-0.05, 1.05])
    #         ax[i][2 * j + 1].grid()

    #         if i == 0:
    #             ax[i][2 * j].set_title(
    #                 f"Ratio for {'dyads' if party == 'groups' else 'individuals'}"
    #             )
    #             ax[i][2 * j + 1].set_title(
    #                 f"$p$-values for {'dyads' if party == 'groups' else 'individuals'}"
    #             )

    # fig.tight_layout()
    # # plt.show()
    # plt.savefig(
    #     f"../data/figures/202402_article/ratios_wrt_impact_parameter.pdf",
    #     bbox_inches="tight",
    # )
    # plt.close()

    # fig, ax = plt.subplots(len(MEASURES), 4, figsize=(20, 40))

    # for i, measure_info in enumerate(MEASURES):
    #     measure_name = measure_info["name"]
    #     for j, party in enumerate(["groups", "individuals"]):
    #         entrance_distance_0 = np.array(
    #             measures_encounters["entrance"][party][0]
    #         ) / (np.nanmean(group_sizes[0]) / 1000)
    #         measures_0 = np.array(measures_encounters[measure_name][party][0])
    #         if party == "groups":
    #             ratio_0 = np.array(measures_0) / np.nanmean(
    #                 measures_undisturbed[measure_name][party][0]
    #             )
    #         else:
    #             ratio_0 = np.array(measures_0) / np.nanmean(
    #                 measures_undisturbed[measure_name]["individuals"]
    #             )
    #         (
    #             entrance_bins_centers,
    #             mean_measure_wrt_entrance_0,
    #             std_measure_wrt_entrance_0,
    #             error_measure_wrt_entrance_0,
    #             n_values_0,
    #         ) = compute_binned_values(
    #             entrance_distance_0,
    #             ratio_0,
    #             MIN_DIST,
    #             MAX_DIST,
    #             N_BINS,
    #         )

    #         entrance_distance_123 = np.concatenate(
    #             [
    #                 np.array(measures_encounters["entrance"][party][1])
    #                 / (np.nanmean(group_sizes[1]) / 1000),
    #                 np.array(measures_encounters["entrance"][party][2])
    #                 / (np.nanmean(group_sizes[2]) / 1000),
    #                 np.array(measures_encounters["entrance"][party][3])
    #                 / (np.nanmean(group_sizes[3]) / 1000),
    #             ]
    #         )

    #         if party == "groups":
    #             ratio_123 = np.concatenate(
    #                 [
    #                     np.array(measures_encounters[measure_name][party][1])
    #                     / np.nanmean(measures_undisturbed[measure_name][party][1]),
    #                     np.array(measures_encounters[measure_name][party][2])
    #                     / np.nanmean(measures_undisturbed[measure_name][party][2]),
    #                     np.array(measures_encounters[measure_name][party][3])
    #                     / np.nanmean(measures_undisturbed[measure_name][party][3]),
    #                 ]
    #             )
    #         else:
    #             ratio_123 = np.concatenate(
    #                 [
    #                     np.array(measures_encounters[measure_name][party][1])
    #                     / np.nanmean(measures_undisturbed[measure_name]["individuals"]),
    #                     np.array(measures_encounters[measure_name][party][2])
    #                     / np.nanmean(measures_undisturbed[measure_name]["individuals"]),
    #                     np.array(measures_encounters[measure_name][party][3])
    #                     / np.nanmean(measures_undisturbed[measure_name]["individuals"]),
    #                 ]
    #             )

    #         (
    #             entrance_bins_centers,
    #             mean_measure_wrt_entrance_123,
    #             std_measure_wrt_entrance_123,
    #             error_measure_wrt_entrance_123,
    #             n_values_123,
    #         ) = compute_binned_values(
    #             entrance_distance_123,
    #             ratio_123,
    #             MIN_DIST,
    #             MAX_DIST,
    #             N_BINS,
    #         )

    #         ax[i][2 * j].errorbar(
    #             entrance_bins_centers,
    #             mean_measure_wrt_entrance_0,
    #             yerr=error_measure_wrt_entrance_0,
    #             label="0",
    #             capsize=5,
    #         )

    #         ax[i][2 * j].errorbar(
    #             entrance_bins_centers,
    #             mean_measure_wrt_entrance_123,
    #             yerr=error_measure_wrt_entrance_123,
    #             label="1-3",
    #             capsize=5,
    #         )

    #         entrance_bins_centers, p_values = compute_p_values_per_bin_2_groups(
    #             entrance_distance_0,
    #             ratio_0,
    #             entrance_distance_123,
    #             ratio_123,
    #             MIN_DIST,
    #             MAX_DIST,
    #             N_BINS,
    #         )

    #         ax[i][2 * j + 1].scatter(
    #             entrance_bins_centers,
    #             p_values,
    #             marker="X",
    #         )

    #         # add threshold
    #         ax[i][2 * j + 1].plot(
    #             [MIN_DIST, MAX_DIST],
    #             [0.05, 0.05],
    #             color="red",
    #             linestyle="--",
    #             linewidth=1,
    #         )
    #         ax[i][2 * j + 1].text(
    #             1.5,
    #             0.08,
    #             "p=0.05",
    #             horizontalalignment="center",
    #             verticalalignment="center",
    #         )

    #         # styling
    #         ax[i][2 * j].set_xlabel("$\\bar{r}_{b}$")
    #         ax[i][2 * j].set_ylabel("ratio {measure_info['symbol']}")
    #         ax[i][2 * j].set_xlim([MIN_DIST, MAX_DIST])
    #         ax[i][2 * j].set_ylim([0, 3])
    #         ax[i][2 * j].legend()
    #         ax[i][2 * j].grid()

    #         ax[i][2 * j + 1].set_xlabel("$\\bar{r}_{b}$")
    #         ax[i][2 * j + 1].set_ylabel(f"p-value for {measure_info['symbol']}")
    #         ax[i][2 * j + 1].set_ylim([-0.05, 1.05])
    #         ax[i][2 * j + 1].grid()

    #         if i == 0:
    #             ax[i][2 * j].set_title(
    #                 f"Ratio for {'dyads' if party == 'groups' else 'individuals'}"
    #             )
    #             ax[i][2 * j + 1].set_title(
    #                 f"$p$-values for {'dyads' if party == 'groups' else 'individuals'}"
    #             )

    # fig.tight_layout()
    # # plt.show()
    # plt.savefig(
    #     f"../data/figures/202402_article/ratios_wrt_impact_parameter_0_123.pdf",
    #     bbox_inches="tight",
    # )
    # plt.close()

    # fig, ax = plt.subplots(len(MEASURES), 4, figsize=(20, 40))

    plt.rcParams.update({"font.size": 14})
    for i, measure_info in enumerate(MEASURES):

        measure_name = measure_info["name"]
        measure_latex = measure_info["latex_name"]
        for j, party in enumerate(["individuals", "groups"]):

            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            entrance_distance_01 = np.concatenate(
                [
                    np.array(measures_encounters["entrance"][party][0])
                    / (np.nanmean(group_sizes[0]) / 1000),
                    np.array(measures_encounters["entrance"][party][1])
                    / (np.nanmean(group_sizes[1]) / 1000),
                ]
            )
            if party == "groups":
                ratio_01 = np.concatenate(
                    [
                        np.array(measures_encounters[measure_name][party][0])
                        / np.nanmean(measures_undisturbed[measure_name][party][0]),
                        np.array(measures_encounters[measure_name][party][1])
                        / np.nanmean(measures_undisturbed[measure_name][party][1]),
                    ]
                )
            else:
                ratio_01 = np.concatenate(
                    [
                        np.array(measures_encounters[measure_name][party][0])
                        / np.nanmean(measures_undisturbed[measure_name]["individuals"]),
                        np.array(measures_encounters[measure_name][party][1])
                        / np.nanmean(measures_undisturbed[measure_name]["individuals"]),
                    ]
                )

            (
                entrance_bins_centers,
                mean_measure_wrt_entrance_01,
                std_measure_wrt_entrance_01,
                error_measure_wrt_entrance_01,
                n_values_01,
            ) = compute_binned_values(
                entrance_distance_01,
                ratio_01,
                MIN_DIST,
                MAX_DIST,
                N_BINS,
            )

            entrance_distance_23 = np.concatenate(
                [
                    np.array(measures_encounters["entrance"][party][2])
                    / (np.nanmean(group_sizes[2]) / 1000),
                    np.array(measures_encounters["entrance"][party][3])
                    / (np.nanmean(group_sizes[3]) / 1000),
                ]
            )

            if party == "groups":
                ratio_23 = np.concatenate(
                    [
                        np.array(measures_encounters[measure_name][party][2])
                        / np.nanmean(measures_undisturbed[measure_name][party][2]),
                        np.array(measures_encounters[measure_name][party][3])
                        / np.nanmean(measures_undisturbed[measure_name][party][3]),
                    ]
                )
            else:
                ratio_23 = np.concatenate(
                    [
                        np.array(measures_encounters[measure_name][party][2])
                        / np.nanmean(measures_undisturbed[measure_name]["individuals"]),
                        np.array(measures_encounters[measure_name][party][3])
                        / np.nanmean(measures_undisturbed[measure_name]["individuals"]),
                    ]
                )

            (
                entrance_bins_centers,
                mean_measure_wrt_entrance_23,
                std_measure_wrt_entrance_23,
                error_measure_wrt_entrance_23,
                n_values_23,
            ) = compute_binned_values(
                entrance_distance_23,
                ratio_23,
                MIN_DIST,
                MAX_DIST,
                N_BINS,
            )

            ax[0].errorbar(
                entrance_bins_centers,
                mean_measure_wrt_entrance_01,
                yerr=error_measure_wrt_entrance_01,
                label="0-1",
                capsize=5,
            )

            ax[0].errorbar(
                entrance_bins_centers,
                mean_measure_wrt_entrance_23,
                yerr=error_measure_wrt_entrance_23,
                label="2-3",
                capsize=5,
            )

            entrance_bins_centers, p_values = compute_p_values_per_bin_2_groups(
                entrance_distance_01,
                ratio_01,
                entrance_distance_23,
                ratio_23,
                MIN_DIST,
                MAX_DIST,
                N_BINS,
            )

            ax[1].scatter(
                entrance_bins_centers,
                p_values,
                marker="X",
            )

            # add threshold
            ax[1].plot(
                [MIN_DIST, MAX_DIST],
                [0.05, 0.05],
                color="red",
                linestyle="--",
                linewidth=1.5,
            )
            ax[1].text(
                1.5,
                0.08,
                "p=0.05",
                horizontalalignment="center",
                verticalalignment="center",
            )

            # styling
            ax[0].set_xlabel("$\\bar{r}_{b}$")
            ax[0].set_ylabel(f"ratio {measure_info['symbol']}")
            ax[0].set_xlim([MIN_DIST, MAX_DIST])

            # if measure_name == ""
            # ax[0].set_ylim([0, 3])
            ax[0].legend()
            ax[0].grid()

            ax[1].set_xlabel("$\\bar{r}_{b}$")
            ax[1].set_ylabel(f"$p$-value for {measure_info['symbol']}")
            ax[1].set_ylim([-0.05, 1.05])
            ax[1].grid()

            fig.tight_layout()
            plt.savefig(
                f"../data/figures/impact_parameter/ratios_wrt_impact_parameter_01_23_{measure_name}_{party}.pdf",
                bbox_inches="tight",
            )
            plt.close()

        print("\\begin{figure}[htb]")
        print("\\centering")
        print("\\begin{subfigure}[t]{\\textwidth}")
        print("\\centering")
        print(
            f"\\includegraphics[width=\\textwidth]{{figures/impact_parameter/ratios_wrt_impact_parameter_01_23_{measure_name}_individuals.pdf}}"
        )
        print("\\caption{Individual}")
        print(
            f"\\label{{fig:ratios_{measure_name}_individuals_wrt_impact_parameter_01_23}}"
        )
        print("\\end{subfigure}")
        print("\\begin{subfigure}[t]{\\textwidth}")
        print("\\centering")
        print(
            f"\\includegraphics[width=\\textwidth]{{figures/impact_parameter/ratios_wrt_impact_parameter_01_23_{measure_name}_groups.pdf}}"
        )
        print("\\caption{Dyad}")
        print(f"\\label{{fig:ratios_{measure_name}_groups_wrt_impact_parameter_01_23}}")
        print("\\end{subfigure}")
        print(
            f"\\caption{{Ratio of the value of the {measure_latex} {measure_info['symbol']} in encounters to the undisturbed value for binned normalized impact parameter $\\bar{{r}}_b$. The ratio are shown separately for encounters involving dyads with a low (0-1, in blue) and high (2-3, in green) level of interaction. The error bars represent the standard error of the mean. The $p$-values for the difference of means between 0-1 and 2-3 are also shown. The red dashed line represents the threshold $p=0.05$.}}"
        )
        print(f"\\label{{fig:ratios_{measure_name}_wrt_impact_parameter_01_23}}")
        print("\\end{figure}")
        print("")
