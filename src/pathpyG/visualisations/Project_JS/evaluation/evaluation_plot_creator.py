import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt


def boxplot(data, filename, title, ylabel):

    plt.figure(figsize=(6, 6))
    values = list(data.values())

    sns.boxplot(data=values, palette="Blues")
    
    plt.title(title)
    plt.ylabel(ylabel)

    plt.xticks(ticks=range(len(data)), labels=data.keys(), rotation=45)

    plt.tight_layout()
    plt.savefig("src/pathpyG/visualisations/Project_JS/evaluation/plots/boxplots/" + filename, dpi=300, bbox_inches='tight')



def bar_chart(data, filename, title, value):
    methods = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(6, 6))
    
    # Balkenfarbe auf Dunkelblau setzen und die Breite anpassen
    plt.bar(methods, values, color='#003366', width=0.4)

    # Diagramm anpassen
    plt.title(title)
    plt.xlabel('Methods')
    plt.ylabel(value)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Diagramm speichern
    plt.savefig("src/pathpyG/visualisations/Project_JS/evaluation/plots/bar_charts/" + filename, dpi=300, bbox_inches='tight')
    


def aggregate_hotvis(data):
    for key, values in data.items():
        min_value = min(values[1:4])
        values[1:4] = [min_value] 
    return data

def create_heatmap(data, filename, title, remove_stress=[]):
    methods = ["Fruchtermann-Reingold", "HOTVis", "Stress Min. SGD", "Stress Min. Adam", "Stress Min. SGD Torch"]

    data_aggregated = data.copy()
    data_aggregated = aggregate_hotvis(data_aggregated)

    df = pd.DataFrame(data_aggregated, index=methods)

    # remove stress minimizing results
    for dataset in remove_stress:
        df.loc[df.index[-3:], dataset] = df[dataset].min()

    # sclae values
    def min_max_scale(df):
        return df.apply(lambda x: 1000 * (x - x.min()) / (x.max() - x.min()))

    scaled_df = min_max_scale(df)


    plt.figure(figsize=(10, 6))
    sns.heatmap(scaled_df, annot=False, cmap="Blues", cbar_kws={'label': title, 'ticks': [0, 1]}, cbar_ax=None)

    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticks([0, 1000])
    cbar.set_ticklabels(['Low', 'High'])


    plt.title(title)
    plt.xlabel("Datasets")
    plt.ylabel("Methods")
    plt.tight_layout()
    plt.savefig("src/pathpyG/visualisations/Project_JS/evaluation/plots/heatmaps/" + filename, dpi=300, bbox_inches='tight')


############################################################################################
####################################                    ####################################
####################################     bar charts     ####################################
####################################                    ####################################
############################################################################################
stress_values_synthetic = {
    "HOTVis 2": 1258.125,
    "HOTVis 3": 1244.7322998046875,
    "HOTVis 5": 1224.0843505859375,
    "Stress Min. SGD": 66.92721557617188,
    "Stress Min. Adam": 66.73497772216797,
    "Stress Min. SGD Torch": 66.91865539550781,
    "Fruchtermann-Reingold": 395.5680236816406,
}

bar_chart(stress_values_synthetic, "stress_synthetic", "Stress Values Synthtic Dataset", "Stress Values")

edge_crossing_office = {
    "HOTVis 2": 29757,
    "HOTVis 3": 22614,
    "HOTVis 5": 24853,
    "Fruchtermann-Reingold": 22437,
}

bar_chart(edge_crossing_office, "edge_crossing_office", "Edge Crossing Office Dataset", "Edge Crosing")

edge_crossing_tube = {
    "HOTVis 2": 72,
    "HOTVis 3": 127,
    "HOTVis 5": 237,
    "Fruchtermann-Reingold": 262,
}

bar_chart(edge_crossing_tube, "edge_crossing_tube", "Edge Crossing Tube Dataset", "Edge Crosing")

closeness_eccentricity_flights = {
    'HOTVis 2': 0.28546659810770825,
    'HOTVis 3': 0.25304183256297436,
    'HOTVis 5': 0.2611201009668535,
    'Stress Min. SGD': 0.3342815217001675,
    'Stress Min. Adam': 0.3395482963153705,
    'Stress Min. SGD Torch': 0.3073269287798831,
    'Fruchtermann-Reingold': 0.35214469779909907
}

bar_chart(closeness_eccentricity_flights, "closeness_eccentricity_flights", "Closeness Eccentricity Flights", "Closeness Eccentricity")

causal_path_dispersion_flights = {
    'HOTVis 2': 0.28537674407724717,
    'HOTVis 3': 0.2811776374595373,
    'HOTVis 5': 0.2816451431843459,
    'Stress Min. SGD': 0.3527449258487508,
    'Stress Min. Adam': 0.35512868840138523,
    'Stress Min. SGD Torch': 0.3511580202933725,
    'Fruchtermann-Reingold': 0.35385698767618246
}

bar_chart(causal_path_dispersion_flights, "causal_path_dispersion_flights", "Causal Path Dispersion Flights", "Causal Path Dispersion")

############################################################################################
####################################                    ####################################
####################################      boxplots      ####################################
####################################                    ####################################
############################################################################################

cluster_distance_ratio_synthetic = {
    "HOTVis 2": [0.5446, 0.4861, 0.5323],
    "HOTVis 3": [0.5545, 0.5492, 0.5276],
    "HOTVis 5": [0.5377, 0.8003, 0.8380],
    "Stress Min. SGD": [0.7910, 0.7097, 1.0156],
    "Stress Min. Adam" : [0.8446, 0.9854, 0.7929],
    "Stress Min. SGD Torch": [0.9745, 0.9505, 1.0119],
    "Fruchtermann-Reingold": [0.9567, 1.0054, 0.9228]
}

boxplot(cluster_distance_ratio_synthetic, "boxplot_synthetic", "Cluster Distance Ratio Synthetic Data", "Cluster Distance Ratio")


cluster_distance_ratio_office = {
    "HOTVis 2": [0.5831, 0.4564, 0.6285, 0.8642, 0.3670],
    "HOTVis 3": [0.4853, 0.4168, 0.5704, 0.4835, 0.2310],
    "HOTVis 5": [0.5614, 0.4441, 0.6285, 0.4415, 0.2102],
    "Fruchtermann-Reingold": [0.3998, 0.4373, 0.4503, 0.2718, 0.1809]
}

boxplot(cluster_distance_ratio_synthetic, "boxplot_office", "Cluster Distance Ratio Office Data", "Cluster Distance Ratio")
############################################################################################
####################################                    ####################################
####################################      heatmaps      ####################################
####################################                    ####################################
############################################################################################

#remove_stress = ["Office", "Tube", "Wikipedia", 'Highschool', 'Hospital']
remove_stress = []

# order: FR, HOTVIS 2, HOTVIS 3, HOTVIS 5, Stress Paper, Stress Adam, Stress Torch
edge_crossing = {
    'Synthetic Graph': [22772, 22584, 22567, 22496, 23427, 23174, 22909], 
    'Office': [22437, 29757, 22614, 24853, 66020, 68764, 66898], 
    'Tube': [262, 72, 127, 237, 11836, 12917, 12303], 
    'Wikipedia': [89958, 120757, 134573, 124993, 226268, 237178, 221729], 
    'Flights': [61184, 72235, 71611, 62526, 87778, 84562, 97415],
    'Highschool' : [5661, 5936, 6362, 7875, 80590, 75971, 80356],
    'Hospital': []
 }

create_heatmap(edge_crossing, "edge_crossing_heatmap", "Edge Crossing", remove_stress)

causal_path_dispersion = {
    'Synthetic Graph': [0.8495024879251144, 0.8044899496400337, 0.8063295082640137, 0.8108884310861977, 0.8337478679217781, 0.8253319482912406, 0.8192977887530488], 
    'Office': [0.35403453046994005, 0.4441712118439523, 0.3823763165405113, 0.3822591147406217, 0.6649192996148706, 0.7103102957374403, 0.7498126164916142], 
    'Tube': [0.5458288844308957, 0.4896376949218661, 0.5210443097980572, 0.5576612339261361, 0.9602115526150503, 0.9299884178277121, 0.8864323679622664], 
    'Wikipedia': [0.828137632034445, 0.5983817995840187, 0.5530487797434024, 0.4973251196381555, 1.0402130361071142, 0.833310015925265, 0.9276365521565368], 
    'Flights': [0.35385698767618246, 0.28537674407724717, 0.2811776374595373, 0.2816451431843459, 0.3527449258487508, 0.35512868840138523, 0.3511580202933725],
    'Highschool' : [0.12912287333251724, 0.044150490291686134, 0.03692216378598026, 0.0215969828517869, 0.79226092869011, 0.7719155053637443, 0.7735758227961815],
    'Hospital': []
}

create_heatmap(causal_path_dispersion, "causal_path_dispersion_heatmap", "Causal Path Dispersion", remove_stress)

closeness_eccentricity = {
    'Synthetic Graph': [1.1068207157410448, 0.9194850213469012, 1.1506570660099176, 1.2433069175995686, 0.9536136109165133, 1.0245949741314972, 0.8680959348202881], 
    'Office': [0.8200476454650562, 0.7921383884059948, 0.8278207641765829, 0.515536140595814, 0.8050763046553711, 0.9690875492183336, 1.133203401247905], 
    'Tube': [0.5013855871539737, 0.3238235777171696, 0.338341671820004, 0.5134950988973279, 0.9618471412673553, 0.8437931369244194, 0.8876891454696505], 
    'Wikipedia': [0.5958483458526941, 0.4475458688693087, 0.4512134376819096, 0.3811513354131433, 1.0610036526646478, 0.8619001076072907, 1.0125964440653348], 
    'Flights': [0.35214469779909907, 0.28546659810770825, 0.25304183256297436, 0.2611201009668535, 0.3342815217001675, 0.3395482963153705, 0.3073269287798831],
    'Highschool': [0.6168892440766709, 0.5077349144378733, 0.5041980849063269, 0.5377502409663389, 1.1071421498431089, 0.9682673302897055, 0.9281420469010454],

}
create_heatmap(closeness_eccentricity, "closeness_eccentricity_heatmap", "Closeness Eccentircity", remove_stress)

stress = {
    'Synthetic Graph': [395.5680236816406, 1258.125, 1244.7322998046875, 1224.0843505859375, 66.92721557617188, 66.73497772216797, 66.91865539550781], 
    'Flights': [10613.9775390625, 101851.0703125, 96797.296875, 96103.703125, 2372.35009765625, 2329.656982421875, 2453.7080078125]
}
create_heatmap(stress, "stress_heatmap", "Stress")
