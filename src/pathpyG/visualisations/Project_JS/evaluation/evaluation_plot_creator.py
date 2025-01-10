import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy


def boxplot(data, filename, title, ylabel):

    plt.figure(figsize=(6, 6))
    values = list(data.values())

    sns.boxplot(data=values, palette="Blues")
    
    plt.title(title)
    plt.ylabel(ylabel)

    plt.xticks(ticks=range(len(data)), labels=data.keys(), rotation=45)

    plt.tight_layout()
    plt.savefig("src/pathpyG/visualisations/Project_JS/evaluation/plots/" + filename, dpi=300, bbox_inches='tight')
    plt.close()


def bar_chart(data, filename, title, value_title, methods_title, figsize =(6,6)):

    plt.figure(figsize=figsize)
    
    plt.bar(methods_title, data, color='#003366', width=0.4)

    plt.title(title)
    plt.ylabel(value_title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("src/pathpyG/visualisations/Project_JS/evaluation/plots/" + filename, dpi=300, bbox_inches='tight')
    plt.close()


def aggregate_hotvis(data):
    for key, values in data.items():
        min_value = min(values[1:4])
        values[1:4] = [min_value] 
    return data

def create_heatmap(data, filename, title):
    methods = ["Fruchtermann-Reingold", "HOTVis", "Stress Min. SGD", "Stress Min. Adam", "Stress Min. SGD Torch"]

    data_aggregated = copy.deepcopy(data)
    data_aggregated = aggregate_hotvis(data_aggregated)

    df = pd.DataFrame(data_aggregated, index=methods)

    # sclae values
    def min_max_scale(df):
        return df.apply(lambda x: 1000 * (x - x.min()) / (x.max() - x.min()))

    scaled_df = min_max_scale(df)


    plt.figure(figsize=(6, 4))
    sns.heatmap(scaled_df, annot=False, cmap="Blues", cbar_kws={'label': title, 'ticks': [0, 1]}, cbar_ax=None, cbar=False)

    #cbar = plt.gca().collections[0].colorbar
    #cbar.set_ticks([0, 1000])
    #cbar.set_ticklabels(['Low', 'High'])


    plt.title(title)
    plt.tight_layout()
    plt.savefig("src/pathpyG/visualisations/Project_JS/evaluation/plots/heatmaps/" + filename, dpi=300, bbox_inches='tight')
    plt.close()


############### data ############

# order: FR, HOTVIS 2, HOTVIS 3, HOTVIS 5, Stress Paper, Stress Adam, Stress Torch

edge_crossing = {
    'Synthetic Graph': [22843, 22356, 22713, 22396, 23463, 23261, 23041],
    'HighSchool': [5691, 6344, 6621, 8538, 7820, 8877, 15653],
    'Hospital': [106955, 108437, 106345, 107218, 125042, 121986, 124209],
    'Office': [23452, 24538, 25688, 26027, 50925, 39702, 31320],
    'Tube': [219, 84, 111, 225, 181, 118, 2026],
    'Wikipedia': [85797, 106710, 124770, 135610, 91168, 89146, 136188],
    'Flights': [68189, 59035, 58626, 62315, 84047, 89160, 94107],
}

causal_path_dispersion = {
    'Synthetic Graph': [0.813381, 0.801798, 0.794910, 0.836765, 0.824449, 0.822558, 0.823231],
    'HighSchool': [0.120126, 0.050441, 0.033709, 0.019394, 0.179915, 0.241374, 0.307557],
    'Hospital': [0.516470, 0.486110, 0.453289, 0.535620, 0.632957, 0.639486, 0.532366],
    'Office': [0.377934, 0.430128, 0.439021, 0.426563, 0.610299, 0.580903, 0.477975],
    'Tube': [0.594804, 0.486773, 0.493302, 0.464800, 0.509085, 0.488161, 0.542099],
    'Wikipedia': [0.898011, 0.465788, 0.554954, 0.596721, 0.276180, 0.329340, 0.190717],
    'Flights': [0.353946, 0.286163, 0.282987, 0.280552, 0.359986, 0.356310, 0.357503],
}

closeness_eccentricity = {
    'Synthetic Graph': [1.066102, 0.805937, 1.070057, 1.118524, 0.981720, 1.055042, 1.151974],
    'HighSchool': [0.501410, 0.452422, 0.495577, 0.490840, 0.469445, 0.818984, 0.306253],
    'Hospital': [0.264690, 0.435542, 0.570272, 0.335727, 0.732924, 0.594882, 0.616378],
    'Office': [0.731372, 0.612239, 0.710338, 0.612839, 0.737031, 0.869107, 0.639850],
    'Tube': [0.513739, 0.305325, 0.343330, 0.410648, 0.526665, 0.310092, 0.244317],
    'Wikipedia': [0.618225, 0.418665, 0.394499, 0.423687, 0.278606, 0.293343, 0.146485],
    'Flights': [0.293634, 0.263079, 0.267177, 0.296842, 0.346378, 0.328997, 0.287378],
}

stress = {
    'Synthetic Graph': [395.722, 1251.105, 1241.878, 1227.460, 66.934, 66.736, 66.987],
    'HighSchool': [4636.800, 662191.0, 378685.969, 290050.781, 1758.104, 1287.413, 52910.031],
    'Hospital': [1819.250, 21084.021, 11667.377, 3328.326, 538.288, 541.820, 746.759],
    'Office': [747.402, 56938.457, 48996.328, 45603.281, 118.235, 122.668, 1661.183],
    'Tube': [25925.785, 38145.805, 38525.199, 23512.275, 4108.923, 2353.260, 39731.566],
    'Wikipedia': [40091.910, 4943849.0, 3969028.5, 2910347.25, 13949.055, 13476.835, 86547.531],
    'Flights': [10709.918, 106873.398, 98559.539, 95085.430, 2312.675, 2347.256, 2442.145],
}

cluster_distance_ratio_synthetic = {
        "Fruchtermann-Reingold": [0.9452, 1.0211, 0.9434],
    "HOTVis 2": [0.4915, 0.5772, 0.5174],
    "HOTVis 3": [0.5714, 0.6006, 0.5376],
    "HOTVis 5": [0.6327, 0.7042, 0.5619],
    "Stress Min. SGD": [0.9412, 1.0479, 0.9370],
    "Stress Min. Adam": [0.7708, 0.9500, 1.0467],
    "Stress Min. SGD Torch": [0.9456, 0.9723, 1.0138]

}

cluster_distance_ratio_highschool = {
    "Fruchtermann-Reingold": [0.6684, 0.1245, 0.1896, 0.1576, 0.2977, 0.2469, 0.6083, 0.1899, 1.0744],
    "HOTVis 2": [1.8490, 0.0551, 0.0716, 0.0627, 0.1498, 0.1245, 0.3516, 0.1227, 1.2090],
    "HOTVis 3": [1.6322, 0.0318, 0.0434, 0.0432, 0.1120, 0.1026, 0.3207, 0.0972, 1.3683],
    "HOTVis 5": [1.6456, 0.0180, 0.0221, 0.0370, 0.0996, 0.0813, 0.4408, 0.1020, 1.3998],
    "Stress Min. SGD": [0.9377, 0.2127, 0.2573, 0.2696, 0.5244, 0.2957, 0.7653, 0.2556, 1.5685],
    "Stress Min. Adam": [0.7813, 0.3451, 0.3172, 0.3506, 0.4894, 0.3764, 0.6899, 0.3826, 0.9058],
    "Stress Min. SGD Torch": [1.2637, 0.2077, 0.2945, 0.4156, 0.8256, 0.8156, 0.9941, 0.6986, 1.2953]
}

cluster_distance_ratio_hospital = {
    "Fruchtermann-Reingold": [0.7973, 0.2849, 0.6008, 1.1598],
    "HOTVis 2": [1.0323, 0.4577, 0.6970, 1.2067],
    "HOTVis 3": [0.7603, 0.2970, 0.6640, 1.2088],
    "HOTVis 5": [1.3991, 0.2349, 0.6807, 1.1351],
    "Stress Min. SGD": [0.9896, 0.7681, 0.6128, 1.3877],
    "Stress Min. Adam": [0.9032, 0.8265, 0.6222, 1.3895],
    "Stress Min. SGD Torch": [1.0010, 0.6647, 0.6532, 1.0732]
}

cluster_distance_ratio_office = {
    "Fruchtermann-Reingold": [0.3853, 0.4907, 0.4551, 0.4876, 0.1822],
    "HOTVis 2": [0.5400, 0.4073, 0.6413, 1.1908, 0.3100],
    "HOTVis 3": [0.5137, 0.4508, 0.6636, 1.2128, 0.2472],
    "HOTVis 5": [0.5161, 0.4355, 0.6754, 0.5952, 0.2096],
    "Stress Min. SGD": [0.6794, 0.7911, 0.7581, 0.6729, 0.7214],
    "Stress Min. Adam": [0.6591, 0.7057, 0.7636, 0.7226, 0.6942],
    "Stress Min. SGD Torch": [0.7793, 0.6486, 0.8545, 0.8779, 0.4272]
}



#boxplot(cluster_distance_ratio_synthetic, "boxplot_office", "Cluster Distance Ratio Office Data", "Cluster Distance Ratio")
############################################################################################
####################################                    ####################################
####################################      heatmaps      ####################################
####################################                    ####################################
############################################################################################


create_heatmap(edge_crossing, "edge_crossing_heatmap", "Edge Crossing")

create_heatmap(causal_path_dispersion, "causal_path_dispersion_heatmap", "Causal Path Dispersion")

create_heatmap(closeness_eccentricity, "closeness_eccentricity_heatmap", "Closeness Eccentircity")

create_heatmap(stress, "stress_heatmap", "Stress")


############################################################################################
####################################                    ####################################
####################################     bar charts     ####################################
####################################                    ####################################
############################################################################################

methods_names = ["Fruchtermann-Reingold", "HOTVis 2", "HOTVis 3", "HOTVis 5", "Stress Min. Baseline", "Stress Min. Torch Adam", "Stress Min. Torch SGD"]
file_names = ["closeness_synthetic", "closeness_highschool", "closeness_hospital", "closeness_office", "closeness_tube", "closeness_wiki", "closeness_flights"]
titles = ["Synthetic Dataset", "High School", "Hospital", "Office", "Tube", "Wikipedia", "Flights"]
values = list(closeness_eccentricity.values())

for i in range(len(values)):
    bar_chart(values[i], "closeness_eccentricity/" + file_names[i], "Closeness Eccentricity " + titles[i] , "Closeness Eccentricity", methods_title=methods_names)

values = list(edge_crossing.values())
file_names = ["edge_crossing_synthetic", "edge_crossing_highschool", "edge_crossing_hospital", "edge_crossing_office", "edge_crossing_tube", "edge_crossing_wiki", "edge_crossing_flights"]

for i in range(len(values)):
    bar_chart(values[i][:4], "edge_crossing/" + file_names[i], "Edge Crossing " + titles[i] , "Edge Crossing", methods_title=methods_names[:4], figsize = (4, 4))


############################################################################################
####################################                    ####################################
####################################      boxplots      ####################################
####################################                    ####################################
############################################################################################

boxplot(cluster_distance_ratio_synthetic, "cluster_distance_ratio/cluster_duistance_ratio_synthetic", "Cluster Distance Ratio Synthetic Data", "Cluster Distance Ratio")
boxplot(cluster_distance_ratio_highschool, "cluster_distance_ratio/cluster_duistance_ratio_highschool", "Cluster Distance Ratio Highschool", "Cluster Distance Ratio")
boxplot(cluster_distance_ratio_hospital, "cluster_distance_ratio/cluster_duistance_ratio_hospital", "Cluster Distance Ratio Hospital", "Cluster Distance Ratio")
boxplot(cluster_distance_ratio_office, "cluster_distance_ratio/cluster_duistance_ratio_office", "Cluster Distance Ratio Office", "Cluster Distance Ratio")

