import sys
import os

# Pfad zu deinem Modul hinzufügen
sys.path.append(os.path.abspath("/home/stud/schneller/projects/ML4Nets/pathpyG/src"))


#import pathpyG as pp
import numpy as np
import pandas as pd
from pathpyG.visualisations.Project_JS.HotVisFunctions import *
from pathpyG.visualisations.Project_JS.SGDStressFunctions import *
import csv

FILENAME_METRIC = "src/pathpyG/visualisations/Project_JS/evaluation/metrics/metrics.txt"

def from_ngram(file: str, sep: str = ",") -> pp.PathData:
    with open(file, "r", encoding="utf-8") as f:
        paths = [line.strip().split(sep) for line in f if len(line.strip().split(sep)) > 1]
        
    weights = [1.0] * len(paths)

    mapping = pp.IndexMap()
    mapping.add_ids(np.unique(np.concatenate([np.array(path) for path in paths])))

    pathdata = pp.PathData(mapping)
    pathdata.append_walks(node_seqs=paths, weights=weights)

    return pathdata

def tensor_from_layout(g: pp.TemporalGraph, layout: dict):
    tensor_size = (len(layout), 2)
    tensor_layout = torch.zeros(tensor_size)

    for key in layout.keys():
        tensor_layout[g.mapping.to_idx(key)] = torch.tensor(layout[key])

    return tensor_layout

###################################### Styling ######################################################

style = {}
style['node_color'] = 'blue'
style['edge_color'] = 'grey'
style['edge_opacity'] = 0.3



###################################### Synthetic Graph ######################################################
'''
FILENAME_PLOT = "src/pathpyG/visualisations/Project_JS/evaluation/plots/synthetic_graph/synthetic_graph_"
DELTA = 1

synthetic_graph = pp.io.read_csv_temporal_graph('src/pathpyG/visualisations/Project_JS/graphs/synthetic_graph/temporal_clusters_real_kregular.tedges', is_undirected = True, timestamp_format='%S')


colors = {}
with open('src/pathpyG/visualisations/Project_JS/graphs/synthetic_graph/colors.csv', mode="r") as file:
    reader = csv.reader(file)
    next(reader)  # Überspringt die Kopfzeile
    for row in reader:
        number, color = row
        colors[number] = color


colors_to_nodes = {}
for key, value in colors.items():
    if value not in colors_to_nodes:
        colors_to_nodes[value] = []
    
    colors_to_nodes[value].append(key)

# Extrahiere die Listen der Keys, die den gleichen Wert haben
clusters = list(colors_to_nodes.values())

# create style
style_synthetic = {}
style_synthetic['node_color'] = colors
style_synthetic['edge_color'] = 'grey'
style_synthetic['edge_opacity'] = 0.3

# create layouts
print("Creating layouts.")
layout_paper, worked_paper = SGD_stress_paper(synthetic_graph, iterations=30, delta=DELTA, learning_rate=0.01)
print("SGD Paper created")
layout_adam, worked_adam = Adam_stress_torch(synthetic_graph, iterations=500, delta=DELTA, learning_rate=0.5)
print("Adam created")
layout_torch, worked_torch = SGD_stress_torch(synthetic_graph, iterations=200, delta=DELTA, learning_rate=0.001)
print("SGD torch created")
layout_2 = HotVis(synthetic_graph, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
print("Layout 2 created")
layout_3 = HotVis(synthetic_graph, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
print("Layout 3 created")
layout_5 = HotVis(synthetic_graph, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)
print("Layout 5 created")


## plot
graph = synthetic_graph.to_static_graph()
print("static graph created")
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style_synthetic)
print("1th plot cerated")
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style_synthetic)
print("2th plot cerated")
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style_synthetic)
print("3th plot cerated")
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style_synthetic)
print("4th plot cerated")
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style_synthetic)
print("5th plot cerated")
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style_synthetic)
print("5th plot cerated")

# metrics
print("Determining edge crossings.")
edge_crossing_2 = edge_crossing(synthetic_graph, layout_2)
print("Edge crossings first layout finished.")
edge_crossing_3 = edge_crossing(synthetic_graph, layout_3)
print("Edge crossings second layout finished.")
edge_crossing_5 = edge_crossing(synthetic_graph, layout_5)
print("Edge crossings third layout finished.")
edge_crossing_paper = edge_crossing(synthetic_graph, layout_paper)
print("Edge crossings 4th layout finished.")
edge_crossing_adam = edge_crossing(synthetic_graph, layout_adam)
print("Edge crossings 5th layout finished.")
edge_crossing_torch = edge_crossing(synthetic_graph, layout_torch)
print("Edge crossings 6th layout finished.")


print("Determining causal path dispersion.")
causal_path_dispersion_2 = causal_path_dispersion(synthetic_graph, layout_2, DELTA, steps=[3], runs=[200])
print("Causal path dispersion first layout finished.")
causal_path_dispersion_3 = causal_path_dispersion(synthetic_graph, layout_3, DELTA, steps=[3], runs=[200])
print("Causal path dispersion second layout finished.")
causal_path_dispersion_5 = causal_path_dispersion(synthetic_graph, layout_5, DELTA, steps=[3], runs=[200])
print("Causal path dispersion third layout finished.")
causal_path_dispersion_paper = causal_path_dispersion(synthetic_graph, layout_paper, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 4th layout finished.")
causal_path_dispersion_adam = causal_path_dispersion(synthetic_graph, layout_adam, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 5th layout finished.")
causal_path_dispersion_torch = causal_path_dispersion(synthetic_graph, layout_torch, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 6th layout finished.")

print("Determining closeness eccentricity.")
closeness_eccentricity_2 = closeness_eccentricity(synthetic_graph, layout_2, DELTA, 0.1)
print("Closeness eccentricity first layout finished.")
closeness_eccentricity_3 = closeness_eccentricity(synthetic_graph, layout_3, DELTA, 0.1)
print("Closeness eccentricity second layout finished.")
closeness_eccentricity_5 = closeness_eccentricity(synthetic_graph, layout_5, DELTA, 0.1)
print("Closeness eccentricity third layout finished.")
closeness_eccentricity_paper = closeness_eccentricity(synthetic_graph, layout_paper, DELTA, 0.1)
print("Closeness eccentricity 4th layout finished.")
closeness_eccentricity_adam = closeness_eccentricity(synthetic_graph, layout_adam, DELTA, 0.1)
print("Closeness eccentricity 5th layout finished.")
closeness_eccentricity_torch = closeness_eccentricity(synthetic_graph, layout_torch, DELTA, 0.1)
print("Closeness eccentricity 6th layout finished.")

print("Determining cluster distance ratio.")
cluster_distance_ratio_2 = cluster_distance_ratio(synthetic_graph, clusters, layout_2)
print("Cluster distance ratio first layout finished.")
cluster_distance_ratio_3 = cluster_distance_ratio(synthetic_graph, clusters, layout_3)
print("Cluster distance ratio second layout finished.")
cluster_distance_ratio_5 = cluster_distance_ratio(synthetic_graph, clusters, layout_5)
print("Cluster distance ratio third layout finished.")
cluster_distance_ratio_paper = cluster_distance_ratio(synthetic_graph, clusters, layout_paper)
print("Cluster distance ratio 4th layout finished.")
cluster_distance_ratio_adam = cluster_distance_ratio(synthetic_graph, clusters, layout_adam)
print("Cluster distance ratio 5th layout finished.")
cluster_distance_ratio_torch = cluster_distance_ratio(synthetic_graph, clusters, layout_torch)
print("Cluster distance ratio 6th layout finished.")

print("Determining shortest paths")
dist, _ = pp.algorithms.temporal_shortest_paths(synthetic_graph, DELTA)
print("determining stress loss")
tensor_layout_2 = tensor_from_layout(synthetic_graph, layout_2)
tensor_layout_3 = tensor_from_layout(synthetic_graph, layout_3)
tensor_layout_5 = tensor_from_layout(synthetic_graph, layout_5)
tensor_layout_paper = tensor_from_layout(synthetic_graph, layout_paper)
tensor_layout_adam = tensor_from_layout(synthetic_graph, layout_adam)
tensor_layout_torch = tensor_from_layout(synthetic_graph, layout_torch)

stress_2 = stress_loss(tensor_layout_2, dist)
print("Stress loss first layout finished.")
stress_3 = stress_loss(tensor_layout_3, dist)
print("Stress loss second layout finished.")
stress_5 = stress_loss(tensor_layout_5, dist)
print("Stress loss third layout finished.")
stress_paper = stress_loss(tensor_layout_paper, dist)
print("Stress loss 4th layout finished.")
stress_adam = stress_loss(tensor_layout_adam, dist)
print("Stress loss 5th layout finished.")
stress_torch = stress_loss(tensor_layout_torch, dist)
print("Stress loss 6th layout finished.")

# write to file 

results = {
    # Layout 2
    "Edge Crossing 2": edge_crossing_2,
    "Causal Path Dispersion 2": causal_path_dispersion_2,
    "Closeness Eccentricity 2": closeness_eccentricity_2,
    "Cluster Distance Ratio 2": cluster_distance_ratio_2,
    "Stress 2": stress_2.item() if isinstance(stress_2, torch.Tensor) else stress_2,

    # Layout 3
    "Edge Crossing 3": edge_crossing_3,
    "Causal Path Dispersion 3": causal_path_dispersion_3,
    "Closeness Eccentricity 3": closeness_eccentricity_3,
    "Cluster Distance Ratio 3": cluster_distance_ratio_3,
    "Stress 3": stress_3.item() if isinstance(stress_3, torch.Tensor) else stress_3,

    # Layout 5
    "Edge Crossing 5": edge_crossing_5,
    "Causal Path Dispersion 5": causal_path_dispersion_5,
    "Closeness Eccentricity 5": closeness_eccentricity_5,
    "Cluster Distance Ratio 5": cluster_distance_ratio_5,
    "Stress 5": stress_5.item() if isinstance(stress_5, torch.Tensor) else stress_5,

    # Paper Layout
    "Edge Crossing Paper": edge_crossing_paper,
    "Causal Path Dispersion Paper": causal_path_dispersion_paper,
    "Closeness Eccentricity Paper": closeness_eccentricity_paper,
    "Cluster Distance Ratio Paper": cluster_distance_ratio_paper,
    "Stress Paper": stress_paper.item() if isinstance(stress_paper, torch.Tensor) else stress_paper,

    # Adam Layout
    "Edge Crossing Adam": edge_crossing_adam,
    "Causal Path Dispersion Adam": causal_path_dispersion_adam,
    "Closeness Eccentricity Adam": closeness_eccentricity_adam,
    "Cluster Distance Ratio Adam": cluster_distance_ratio_adam,
    "Stress Adam": stress_adam.item() if isinstance(stress_adam, torch.Tensor) else stress_adam,

    # Torch Layout
    "Edge Crossing Torch": edge_crossing_torch,
    "Causal Path Dispersion Torch": causal_path_dispersion_torch,
    "Closeness Eccentricity Torch": closeness_eccentricity_torch,
    "Cluster Distance Ratio Torch": cluster_distance_ratio_torch,
    "Stress Torch": stress_torch.item() if isinstance(stress_torch, torch.Tensor) else stress_torch,
}



with open(FILENAME_METRIC, 'a') as file:
    file.write("\n\n\n###################################### Synthetic Graph ######################################################\n")
    # Get the maximum description length for consistent alignment
    max_length = max(len(description) for description in results)
    for description, value in results.items():
        # Align descriptions using ljust
        file.write(f"{description.ljust(max_length)} : {value}\n")
    if not worked_paper:
        file.write("The graph wasn't connected, so the SGD paper versions returned random layout")
    if not worked_adam:
        file.write("The graph wasn't connected, so the SGD adam versions returned random layout")
    if not worked_torch:
        file.write("The graph wasn't connected, so the SGD torch versions returned random layout")

print("Dataset finished.")
'''
###################################### High School ######################################################

FILENAME_PLOT = "src/pathpyG/visualisations/Project_JS/evaluation/plots/high_school/high_school_"
DELTA = 60

highschool_graph = pp.io.read_csv_temporal_graph('src/pathpyG/visualisations/Project_JS/graphs/Highschool/proximity/edges.csv', is_undirected = False, timestamp_format='%S')
# load metadata
meta_data = pd.read_csv("src/pathpyG/visualisations/Project_JS/graphs/Highschool/proximity/nodes.csv")

# get colors of nodes
colors = {}
color_map = {'2BIO3': 'yellow', 'PC*': 'green', '2BIO2': 'blue', 'PSI*':'pink', 'PC':'gray', 'MP*1':'black', 'MP':'red', '2BIO1':'purple', 'MP*2':'orange'}
for index, row in meta_data.iterrows():
        colors[f"{row['index']}"] = color_map[row['class']]

# get clusters
clusters = meta_data.groupby('class')['index'].apply(list).values

clusters = [[str(i) for i in cluster] for cluster in clusters]
# filter nodes, wich doesn't occure in graph
valid_nodes = set(highschool_graph.mapping.node_ids)
clusters = [list(filter(lambda node: node in set(highschool_graph.mapping.node_ids), cluster)) for cluster in clusters]
# filter empty lists
clusters = [lst for lst in clusters if lst]

# create style
style_high_school = {}
style_high_school['node_color'] = colors
style_high_school['edge_color'] = 'grey'
style_high_school['edge_opacity'] = 0.3

# create layouts
print("Creating layouts.")
layout_paper, worked_paper = SGD_stress_paper(highschool_graph, iterations=30, delta=DELTA, learning_rate=0.01)
layout_adam, worked_adam = Adam_stress_torch(highschool_graph, iterations=500, delta=DELTA, learning_rate=0.5)
layout_torch, worked_torch = SGD_stress_torch(highschool_graph, iterations=200, delta=DELTA, learning_rate=0.001)

layout_2 = HotVis(highschool_graph, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
layout_3 = HotVis(highschool_graph, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
layout_5 = HotVis(highschool_graph, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)


## plot
graph = highschool_graph.to_static_graph()
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style_high_school)
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style_high_school)
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style_high_school)
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style_high_school)
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style_high_school)
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style_high_school)

# metrics
print("Determining edge crossings.")
edge_crossing_2 = edge_crossing(highschool_graph, layout_2)
print("Edge crossings first layout finished.")
edge_crossing_3 = edge_crossing(highschool_graph, layout_3)
print("Edge crossings second layout finished.")
edge_crossing_5 = edge_crossing(highschool_graph, layout_5)
print("Edge crossings third layout finished.")
edge_crossing_paper = edge_crossing(highschool_graph, layout_paper)
print("Edge crossings 4th layout finished.")
edge_crossing_adam = edge_crossing(highschool_graph, layout_adam)
print("Edge crossings 5th layout finished.")
edge_crossing_torch = edge_crossing(highschool_graph, layout_torch)
print("Edge crossings 6th layout finished.")


print("Determining causal path dispersion.")
causal_path_dispersion_2 = causal_path_dispersion(highschool_graph, layout_2, DELTA, steps=[3], runs=[200])
print("Causal path dispersion first layout finished.")
causal_path_dispersion_3 = causal_path_dispersion(highschool_graph, layout_3, DELTA, steps=[3], runs=[200])
print("Causal path dispersion second layout finished.")
causal_path_dispersion_5 = causal_path_dispersion(highschool_graph, layout_5, DELTA, steps=[3], runs=[200])
print("Causal path dispersion third layout finished.")
causal_path_dispersion_paper = causal_path_dispersion(highschool_graph, layout_paper, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 4th layout finished.")
causal_path_dispersion_adam = causal_path_dispersion(highschool_graph, layout_adam, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 5th layout finished.")
causal_path_dispersion_torch = causal_path_dispersion(highschool_graph, layout_torch, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 6th layout finished.")

print("Determining closeness eccentricity.")
closeness_eccentricity_2 = closeness_eccentricity(highschool_graph, layout_2, DELTA, 0.1)
print("Closeness eccentricity first layout finished.")
closeness_eccentricity_3 = closeness_eccentricity(highschool_graph, layout_3, DELTA, 0.1)
print("Closeness eccentricity second layout finished.")
closeness_eccentricity_5 = closeness_eccentricity(highschool_graph, layout_5, DELTA, 0.1)
print("Closeness eccentricity third layout finished.")
closeness_eccentricity_paper = closeness_eccentricity(highschool_graph, layout_paper, DELTA, 0.1)
print("Closeness eccentricity 4th layout finished.")
closeness_eccentricity_adam = closeness_eccentricity(highschool_graph, layout_adam, DELTA, 0.1)
print("Closeness eccentricity 5th layout finished.")
closeness_eccentricity_torch = closeness_eccentricity(highschool_graph, layout_torch, DELTA, 0.1)
print("Closeness eccentricity 6th layout finished.")

print("Determining cluster distance ratio.")
cluster_distance_ratio_2 = cluster_distance_ratio(highschool_graph, clusters, layout_2)
print("Cluster distance ratio first layout finished.")
cluster_distance_ratio_3 = cluster_distance_ratio(highschool_graph, clusters, layout_3)
print("Cluster distance ratio second layout finished.")
cluster_distance_ratio_5 = cluster_distance_ratio(highschool_graph, clusters, layout_5)
print("Cluster distance ratio third layout finished.")
cluster_distance_ratio_paper = cluster_distance_ratio(highschool_graph, clusters, layout_paper)
print("Cluster distance ratio 4th layout finished.")
cluster_distance_ratio_adam = cluster_distance_ratio(highschool_graph, clusters, layout_adam)
print("Cluster distance ratio 5th layout finished.")
cluster_distance_ratio_torch = cluster_distance_ratio(highschool_graph, clusters, layout_torch)
print("Cluster distance ratio 6th layout finished.")

print("Determining shortest paths")
dist, _ = pp.algorithms.temporal_shortest_paths(highschool_graph, DELTA)
print("determining stress loss")
tensor_layout_2 = tensor_from_layout(highschool_graph, layout_2)
tensor_layout_3 = tensor_from_layout(highschool_graph, layout_3)
tensor_layout_5 = tensor_from_layout(highschool_graph, layout_5)
tensor_layout_paper = tensor_from_layout(highschool_graph, layout_paper)
tensor_layout_adam = tensor_from_layout(highschool_graph, layout_adam)
tensor_layout_torch = tensor_from_layout(highschool_graph, layout_torch)

stress_2 = stress_loss(tensor_layout_2, dist)
print("Stress loss first layout finished.")
stress_3 = stress_loss(tensor_layout_3, dist)
print("Stress loss second layout finished.")
stress_5 = stress_loss(tensor_layout_5, dist)
print("Stress loss third layout finished.")
stress_paper = stress_loss(tensor_layout_paper, dist)
print("Stress loss 4th layout finished.")
stress_adam = stress_loss(tensor_layout_adam, dist)
print("Stress loss 5th layout finished.")
stress_torch = stress_loss(tensor_layout_torch, dist)
print("Stress loss 6th layout finished.")

# write to file 

results = {
    # Layout 2
    "Edge Crossing 2": edge_crossing_2,
    "Causal Path Dispersion 2": causal_path_dispersion_2,
    "Closeness Eccentricity 2": closeness_eccentricity_2,
    "Cluster Distance Ratio 2": cluster_distance_ratio_2,
    "Stress 2": stress_2.item() if isinstance(stress_2, torch.Tensor) else stress_2,

    # Layout 3
    "Edge Crossing 3": edge_crossing_3,
    "Causal Path Dispersion 3": causal_path_dispersion_3,
    "Closeness Eccentricity 3": closeness_eccentricity_3,
    "Cluster Distance Ratio 3": cluster_distance_ratio_3,
    "Stress 3": stress_3.item() if isinstance(stress_3, torch.Tensor) else stress_3,

    # Layout 5
    "Edge Crossing 5": edge_crossing_5,
    "Causal Path Dispersion 5": causal_path_dispersion_5,
    "Closeness Eccentricity 5": closeness_eccentricity_5,
    "Cluster Distance Ratio 5": cluster_distance_ratio_5,
    "Stress 5": stress_5.item() if isinstance(stress_5, torch.Tensor) else stress_5,

    # Paper Layout
    "Edge Crossing Paper": edge_crossing_paper,
    "Causal Path Dispersion Paper": causal_path_dispersion_paper,
    "Closeness Eccentricity Paper": closeness_eccentricity_paper,
    "Cluster Distance Ratio Paper": cluster_distance_ratio_paper,
    "Stress Paper": stress_paper.item() if isinstance(stress_paper, torch.Tensor) else stress_paper,

    # Adam Layout
    "Edge Crossing Adam": edge_crossing_adam,
    "Causal Path Dispersion Adam": causal_path_dispersion_adam,
    "Closeness Eccentricity Adam": closeness_eccentricity_adam,
    "Cluster Distance Ratio Adam": cluster_distance_ratio_adam,
    "Stress Adam": stress_adam.item() if isinstance(stress_adam, torch.Tensor) else stress_adam,

    # Torch Layout
    "Edge Crossing Torch": edge_crossing_torch,
    "Causal Path Dispersion Torch": causal_path_dispersion_torch,
    "Closeness Eccentricity Torch": closeness_eccentricity_torch,
    "Cluster Distance Ratio Torch": cluster_distance_ratio_torch,
    "Stress Torch": stress_torch.item() if isinstance(stress_torch, torch.Tensor) else stress_torch,
}



with open(FILENAME_METRIC, 'a') as file:
    file.write("\n\n\n###################################### HighSchool ######################################################\n")
    # Get the maximum description length for consistent alignment
    max_length = max(len(description) for description in results)
    for description, value in results.items():
        # Align descriptions using ljust
        file.write(f"{description.ljust(max_length)} : {value}\n")
    if not worked_paper:
        file.write("The graph wasn't connected, so the SGD paper versions returned random layout")
    if not worked_adam:
        file.write("The graph wasn't connected, so the SGD adam versions returned random layout")
    if not worked_torch:
        file.write("The graph wasn't connected, so the SGD torch versions returned random layout")

print("Dataset finished.")


###################################### Hospital ######################################################

FILENAME_PLOT = "src/pathpyG/visualisations/Project_JS/evaluation/plots/hospital/hospital_"
DELTA = 40

# load graph
hospital_graph = pp.io.read_csv_temporal_graph('src/pathpyG/visualisations/Project_JS/graphs/Hospital/network/edges.csv', is_undirected = True, timestamp_format='%S')
# load metadata
meta_data = pd.read_csv("src/pathpyG/visualisations/Project_JS/graphs/Hospital/network/nodes.csv")

# get colors of nodes
colors = {}
color_map = {'ADM': 'yellow', 'NUR': 'green', 'MED': 'blue', 'PAT':'pink'}
for index, row in meta_data.iterrows():
        colors[f"{row['index']}"] = color_map[row['status']]


clusters = meta_data.groupby('status')['index'].apply(list).values
clusters = [[str(i) for i in cluster] for cluster in clusters]
# filter nodes, wich doesn't occure in graph
valid_nodes = set(hospital_graph.mapping.node_ids)
clusters = [list(filter(lambda node: node in set(hospital_graph.mapping.node_ids), cluster)) for cluster in clusters]
# filter empty lists
clusters = [lst for lst in clusters if lst]

# create style
style_hospital = {}
style_hospital['node_color'] = colors
style_hospital['edge_color'] = 'grey'
style_hospital['edge_opacity'] = 0.3

# create layouts
print("Creating layouts.")
layout_paper, worked_paper = SGD_stress_paper(hospital_graph, iterations=30, delta=DELTA, learning_rate=0.01)
layout_adam, worked_adam = Adam_stress_torch(hospital_graph, iterations=500, delta=DELTA, learning_rate=0.5)
layout_torch, worked_torch = SGD_stress_torch(hospital_graph, iterations=200, delta=DELTA, learning_rate=0.001)

layout_2 = HotVis(hospital_graph, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
layout_3 = HotVis(hospital_graph, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
layout_5 = HotVis(hospital_graph, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)


## plot

graph = hospital_graph.to_static_graph()
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style_hospital)
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style_hospital)
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style_hospital)
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style_hospital)
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style_hospital)
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style_hospital)

# metrics
print("Determining edge crossings.")
edge_crossing_2 = edge_crossing(hospital_graph, layout_2)
print("Edge crossings first layout finished.")
edge_crossing_3 = edge_crossing(hospital_graph, layout_3)
print("Edge crossings second layout finished.")
edge_crossing_5 = edge_crossing(hospital_graph, layout_5)
print("Edge crossings third layout finished.")
edge_crossing_paper = edge_crossing(hospital_graph, layout_paper)
print("Edge crossings 4th layout finished.")
edge_crossing_adam = edge_crossing(hospital_graph, layout_adam)
print("Edge crossings 5th layout finished.")
edge_crossing_torch = edge_crossing(hospital_graph, layout_torch)
print("Edge crossings 6th layout finished.")


print("Determining causal path dispersion.")
causal_path_dispersion_2 = causal_path_dispersion(hospital_graph, layout_2, DELTA, steps=[3], runs=[200])
print("Causal path dispersion first layout finished.")
causal_path_dispersion_3 = causal_path_dispersion(hospital_graph, layout_3, DELTA, steps=[3], runs=[200])
print("Causal path dispersion second layout finished.")
causal_path_dispersion_5 = causal_path_dispersion(hospital_graph, layout_5, DELTA, steps=[3], runs=[200])
print("Causal path dispersion third layout finished.")
causal_path_dispersion_paper = causal_path_dispersion(hospital_graph, layout_paper, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 4th layout finished.")
causal_path_dispersion_adam = causal_path_dispersion(hospital_graph, layout_adam, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 5th layout finished.")
causal_path_dispersion_torch = causal_path_dispersion(hospital_graph, layout_torch, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 6th layout finished.")

print("Determining closeness eccentricity.")
closeness_eccentricity_2 = closeness_eccentricity(hospital_graph, layout_2, DELTA, 0.1)
print("Closeness eccentricity first layout finished.")
closeness_eccentricity_3 = closeness_eccentricity(hospital_graph, layout_3, DELTA, 0.1)
print("Closeness eccentricity second layout finished.")
closeness_eccentricity_5 = closeness_eccentricity(hospital_graph, layout_5, DELTA, 0.1)
print("Closeness eccentricity third layout finished.")
closeness_eccentricity_paper = closeness_eccentricity(hospital_graph, layout_paper, DELTA, 0.1)
print("Closeness eccentricity 4th layout finished.")
closeness_eccentricity_adam = closeness_eccentricity(hospital_graph, layout_adam, DELTA, 0.1)
print("Closeness eccentricity 5th layout finished.")
closeness_eccentricity_torch = closeness_eccentricity(hospital_graph, layout_torch, DELTA, 0.1)
print("Closeness eccentricity 6th layout finished.")

print("Determining shortest paths")
dist, _ = pp.algorithms.temporal_shortest_paths(hospital_graph, DELTA)
print("determining stress loss")
tensor_layout_2 = tensor_from_layout(hospital_graph, layout_2)
tensor_layout_3 = tensor_from_layout(hospital_graph, layout_3)
tensor_layout_5 = tensor_from_layout(hospital_graph, layout_5)
tensor_layout_paper = tensor_from_layout(hospital_graph, layout_paper)
tensor_layout_adam = tensor_from_layout(hospital_graph, layout_adam)
tensor_layout_torch = tensor_from_layout(hospital_graph, layout_torch)

stress_2 = stress_loss(tensor_layout_2, dist)
print("Stress loss first layout finished.")
stress_3 = stress_loss(tensor_layout_3, dist)
print("Stress loss second layout finished.")
stress_5 = stress_loss(tensor_layout_5, dist)
print("Stress loss third layout finished.")
stress_paper = stress_loss(tensor_layout_paper, dist)
print("Stress loss 4th layout finished.")
stress_adam = stress_loss(tensor_layout_adam, dist)
print("Stress loss 5th layout finished.")
stress_torch = stress_loss(tensor_layout_torch, dist)
print("Stress loss 6th layout finished.")

# write to file 

results = {
    # Layout 2
    "Edge Crossing 2": edge_crossing_2,
    "Causal Path Dispersion 2": causal_path_dispersion_2,
    "Closeness Eccentricity 2": closeness_eccentricity_2,
    "Stress 2": stress_2.item() if isinstance(stress_2, torch.Tensor) else stress_2,

    # Layout 3
    "Edge Crossing 3": edge_crossing_3,
    "Causal Path Dispersion 3": causal_path_dispersion_3,
    "Closeness Eccentricity 3": closeness_eccentricity_3,
    "Stress 3": stress_3.item() if isinstance(stress_3, torch.Tensor) else stress_3,

    # Layout 5
    "Edge Crossing 5": edge_crossing_5,
    "Causal Path Dispersion 5": causal_path_dispersion_5,
    "Closeness Eccentricity 5": closeness_eccentricity_5,
    "Stress 5": stress_5.item() if isinstance(stress_5, torch.Tensor) else stress_5,

    # Paper Layout
    "Edge Crossing Paper": edge_crossing_paper,
    "Causal Path Dispersion Paper": causal_path_dispersion_paper,
    "Closeness Eccentricity Paper": closeness_eccentricity_paper,
    "Stress Paper": stress_paper.item() if isinstance(stress_paper, torch.Tensor) else stress_paper,

    # Adam Layout
    "Edge Crossing Adam": edge_crossing_adam,
    "Causal Path Dispersion Adam": causal_path_dispersion_adam,
    "Closeness Eccentricity Adam": closeness_eccentricity_adam,
    "Stress Adam": stress_adam.item() if isinstance(stress_adam, torch.Tensor) else stress_adam,

    # Torch Layout
    "Edge Crossing Torch": edge_crossing_torch,
    "Causal Path Dispersion Torch": causal_path_dispersion_torch,
    "Closeness Eccentricity Torch": closeness_eccentricity_torch,
    "Stress Torch": stress_torch.item() if isinstance(stress_torch, torch.Tensor) else stress_torch,
}



with open(FILENAME_METRIC, 'a') as file:
    file.write("\n\n\n###################################### Hospital ######################################################\n")
    # Get the maximum description length for consistent alignment
    max_length = max(len(description) for description in results)
    for description, value in results.items():
        # Align descriptions using ljust
        file.write(f"{description.ljust(max_length)} : {value}\n")
    if not worked_paper:
        file.write("The graph wasn't connected, so the SGD paper versions returned random layout")
    if not worked_adam:
        file.write("The graph wasn't connected, so the SGD adam versions returned random layout")
    if not worked_torch:
        file.write("The graph wasn't connected, so the SGD torch versions returned random layout")

print("Dataset finished.")


###################################### Office ######################################################
'''
FILENAME_PLOT = "src/pathpyG/visualisations/Project_JS/evaluation/plots/office/office_"
DELTA = 50

# load graph
office_graph = pp.io.read_csv_temporal_graph('src/pathpyG/visualisations/Project_JS/graphs/office/network/edges.csv', is_undirected = True, timestamp_format='%S')
# load metadata
meta_data = pd.read_csv("src/pathpyG/visualisations/Project_JS/graphs/office/network/nodes.csv")

# get colors of nodes
colors = {}
color_map = {'DSE': 'yellow', 'DMCT': 'green', 'DISQ': 'blue', 'SRH':'pink', 'SFLE':'black'}
for index, row in meta_data.iterrows():
        colors[f"{row['index']}"] = color_map[row['department']]


clusters = meta_data.groupby('department')['index'].apply(list).values
clusters = [[str(i) for i in cluster] for cluster in clusters]
# filter nodes, wich doesn't occure in graph
valid_nodes = set(office_graph.mapping.node_ids)
clusters = [list(filter(lambda node: node in set(office_graph.mapping.node_ids), cluster)) for cluster in clusters]
# filter empty lists
clusters = [lst for lst in clusters if lst]

# create style
style_office = {}
style_office['node_color'] = colors
style_office['edge_color'] = 'grey'
style_office['edge_opacity'] = 0.3

# create layouts
print("Creating layouts.")
layout_paper, worked_paper = SGD_stress_paper(office_graph, iterations=30, delta=DELTA, learning_rate=0.01)
layout_adam, worked_adam = Adam_stress_torch(office_graph, iterations=500, delta=DELTA, learning_rate=0.5)
layout_torch, worked_torch = SGD_stress_torch(office_graph, iterations=200, delta=DELTA, learning_rate=0.001)

layout_2 = HotVis(office_graph, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
layout_3 = HotVis(office_graph, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
layout_5 = HotVis(office_graph, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)


## plot
graph = office_graph.to_static_graph()
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style_office)
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style_office)
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style_office)
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style_office)
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style_office)
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style_office)

# metrics
print("Determining edge crossings.")
edge_crossing_2 = edge_crossing(office_graph, layout_2)
print("Edge crossings first layout finished.")
edge_crossing_3 = edge_crossing(office_graph, layout_3)
print("Edge crossings second layout finished.")
edge_crossing_5 = edge_crossing(office_graph, layout_5)
print("Edge crossings third layout finished.")
edge_crossing_paper = edge_crossing(office_graph, layout_paper)
print("Edge crossings 4th layout finished.")
edge_crossing_adam = edge_crossing(office_graph, layout_adam)
print("Edge crossings 5th layout finished.")
edge_crossing_torch = edge_crossing(office_graph, layout_torch)
print("Edge crossings 6th layout finished.")


print("Determining causal path dispersion.")
causal_path_dispersion_2 = causal_path_dispersion(office_graph, layout_2, DELTA, steps=[3], runs=[200])
print("Causal path dispersion first layout finished.")
causal_path_dispersion_3 = causal_path_dispersion(office_graph, layout_3, DELTA, steps=[3], runs=[200])
print("Causal path dispersion second layout finished.")
causal_path_dispersion_5 = causal_path_dispersion(office_graph, layout_5, DELTA, steps=[3], runs=[200])
print("Causal path dispersion third layout finished.")
causal_path_dispersion_paper = causal_path_dispersion(office_graph, layout_paper, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 4th layout finished.")
causal_path_dispersion_adam = causal_path_dispersion(office_graph, layout_adam, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 5th layout finished.")
causal_path_dispersion_torch = causal_path_dispersion(office_graph, layout_torch, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 6th layout finished.")

print("Determining closeness eccentricity.")
closeness_eccentricity_2 = closeness_eccentricity(office_graph, layout_2, DELTA, 0.1)
print("Closeness eccentricity first layout finished.")
closeness_eccentricity_3 = closeness_eccentricity(office_graph, layout_3, DELTA, 0.1)
print("Closeness eccentricity second layout finished.")
closeness_eccentricity_5 = closeness_eccentricity(office_graph, layout_5, DELTA, 0.1)
print("Closeness eccentricity third layout finished.")
closeness_eccentricity_paper = closeness_eccentricity(office_graph, layout_paper, DELTA, 0.1)
print("Closeness eccentricity 4th layout finished.")
closeness_eccentricity_adam = closeness_eccentricity(office_graph, layout_adam, DELTA, 0.1)
print("Closeness eccentricity 5th layout finished.")
closeness_eccentricity_torch = closeness_eccentricity(office_graph, layout_torch, DELTA, 0.1)
print("Closeness eccentricity 6th layout finished.")

print("Determining shortest paths")
dist, _ = pp.algorithms.temporal_shortest_paths(office_graph, DELTA)
print("determining stress loss")
tensor_layout_2 = tensor_from_layout(office_graph, layout_2)
tensor_layout_3 = tensor_from_layout(office_graph, layout_3)
tensor_layout_5 = tensor_from_layout(office_graph, layout_5)
tensor_layout_paper = tensor_from_layout(office_graph, layout_paper)
tensor_layout_adam = tensor_from_layout(office_graph, layout_adam)
tensor_layout_torch = tensor_from_layout(office_graph, layout_torch)

stress_2 = stress_loss(tensor_layout_2, dist)
print("Stress loss first layout finished.")
stress_3 = stress_loss(tensor_layout_3, dist)
print("Stress loss second layout finished.")
stress_5 = stress_loss(tensor_layout_5, dist)
print("Stress loss third layout finished.")
stress_paper = stress_loss(tensor_layout_paper, dist)
print("Stress loss 4th layout finished.")
stress_adam = stress_loss(tensor_layout_adam, dist)
print("Stress loss 5th layout finished.")
stress_torch = stress_loss(tensor_layout_torch, dist)
print("Stress loss 6th layout finished.")

# write to file 

results = {
    # Layout 2
    "Edge Crossing 2": edge_crossing_2,
    "Causal Path Dispersion 2": causal_path_dispersion_2,
    "Closeness Eccentricity 2": closeness_eccentricity_2,
    "Stress 2": stress_2.item() if isinstance(stress_2, torch.Tensor) else stress_2,

    # Layout 3
    "Edge Crossing 3": edge_crossing_3,
    "Causal Path Dispersion 3": causal_path_dispersion_3,
    "Closeness Eccentricity 3": closeness_eccentricity_3,
    "Stress 3": stress_3.item() if isinstance(stress_3, torch.Tensor) else stress_3,

    # Layout 5
    "Edge Crossing 5": edge_crossing_5,
    "Causal Path Dispersion 5": causal_path_dispersion_5,
    "Closeness Eccentricity 5": closeness_eccentricity_5,
    "Stress 5": stress_5.item() if isinstance(stress_5, torch.Tensor) else stress_5,

    # Paper Layout
    "Edge Crossing Paper": edge_crossing_paper,
    "Causal Path Dispersion Paper": causal_path_dispersion_paper,
    "Closeness Eccentricity Paper": closeness_eccentricity_paper,
    "Stress Paper": stress_paper.item() if isinstance(stress_paper, torch.Tensor) else stress_paper,

    # Adam Layout
    "Edge Crossing Adam": edge_crossing_adam,
    "Causal Path Dispersion Adam": causal_path_dispersion_adam,
    "Closeness Eccentricity Adam": closeness_eccentricity_adam,
    "Stress Adam": stress_adam.item() if isinstance(stress_adam, torch.Tensor) else stress_adam,

    # Torch Layout
    "Edge Crossing Torch": edge_crossing_torch,
    "Causal Path Dispersion Torch": causal_path_dispersion_torch,
    "Closeness Eccentricity Torch": closeness_eccentricity_torch,
    "Stress Torch": stress_torch.item() if isinstance(stress_torch, torch.Tensor) else stress_torch,
}



with open(FILENAME_METRIC, 'a') as file:
    file.write("\n\n\n###################################### Office ######################################################\n")
    # Get the maximum description length for consistent alignment
    max_length = max(len(description) for description in results)
    for description, value in results.items():
        # Align descriptions using ljust
        file.write(f"{description.ljust(max_length)} : {value}\n")

    if not worked_paper:
        file.write("The graph wasn't connected, so the SGD paper versions returned random layout")
    if not worked_adam:
        file.write("The graph wasn't connected, so the SGD adam versions returned random layout")
    if not worked_torch:
        file.write("The graph wasn't connected, so the SGD torch versions returned random layout")

print("Dataset finished.")



###################################### Tube ######################################################

FILENAME_PLOT = "src/pathpyG/visualisations/Project_JS/evaluation/plots/tube/tube_"

DELTA = 1

tube = pp.PathData.from_ngram("src/pathpyG/visualisations/Project_JS/graphs/Tube/tube.ngram")

# create layouts
print("Creating layouts.")
layout_paper, worked_paper = SGD_stress_paper(tube, iterations=30, delta=DELTA, learning_rate=0.01)
layout_adam, worked_adam = Adam_stress_torch(tube, iterations=500, delta=DELTA, learning_rate=0.5)
layout_torch, worked_torch = SGD_stress_torch(tube, iterations=200, delta=DELTA, learning_rate=0.001)

layout_2 = HotVis(tube, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
layout_3 =  HotVis(tube, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
layout_5 =  HotVis(tube, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)

## plot
graph = pp.MultiOrderModel.from_PathData(tube, max_order=1).layers[1]
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style)
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style)
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style)
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style)
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style)
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style)

# metrics
print("Determining edge crossings.")
edge_crossing_2 = edge_crossing(tube, layout_2)
print("Edge crossings first layout finished.")
edge_crossing_3 = edge_crossing(tube, layout_3)
print("Edge crossings second layout finished.")
edge_crossing_5 = edge_crossing(tube, layout_5)
print("Edge crossings third layout finished.")
edge_crossing_paper = edge_crossing(tube, layout_paper)
print("Edge crossings 4th layout finished.")
edge_crossing_adam = edge_crossing(tube, layout_adam)
print("Edge crossings 5th layout finished.")
edge_crossing_torch = edge_crossing(tube, layout_torch)
print("Edge crossings 6th layout finished.")


print("Determining causal path dispersion.")
causal_path_dispersion_2 = causal_path_dispersion(tube, layout_2, DELTA, steps=[3], runs=[200])
print("Causal path dispersion first layout finished.")
causal_path_dispersion_3 = causal_path_dispersion(tube, layout_3, DELTA, steps=[3], runs=[200])
print("Causal path dispersion second layout finished.")
causal_path_dispersion_5 = causal_path_dispersion(tube, layout_5, DELTA, steps=[3], runs=[200])
print("Causal path dispersion third layout finished.")
causal_path_dispersion_paper = causal_path_dispersion(tube, layout_paper, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 4th layout finished.")
causal_path_dispersion_adam = causal_path_dispersion(tube, layout_adam, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 5th layout finished.")
causal_path_dispersion_torch = causal_path_dispersion(tube, layout_torch, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 6th layout finished.")

print("Determining closeness eccentricity.")
closeness_eccentricity_2 = closeness_eccentricity(tube, layout_2, DELTA, 0.1)
print("Closeness eccentricity first layout finished.")
closeness_eccentricity_3 = closeness_eccentricity(tube, layout_3, DELTA, 0.1)
print("Closeness eccentricity second layout finished.")
closeness_eccentricity_5 = closeness_eccentricity(tube, layout_5, DELTA, 0.1)
print("Closeness eccentricity third layout finished.")
closeness_eccentricity_paper = closeness_eccentricity(tube, layout_paper, DELTA, 0.1)
print("Closeness eccentricity 4th layout finished.")
closeness_eccentricity_adam = closeness_eccentricity(tube, layout_adam, DELTA, 0.1)
print("Closeness eccentricity 5th layout finished.")
closeness_eccentricity_torch = closeness_eccentricity(tube, layout_torch, DELTA, 0.1)
print("Closeness eccentricity 6th layout finished.")

print("Determining shortest paths")
dist = shortest_paths_path_data(tube)
print("determining stress loss")
tensor_layout_2 = tensor_from_layout(tube, layout_2)
tensor_layout_3 = tensor_from_layout(tube, layout_3)
tensor_layout_5 = tensor_from_layout(tube, layout_5)
tensor_layout_paper = tensor_from_layout(tube, layout_paper)
tensor_layout_adam = tensor_from_layout(tube, layout_adam)
tensor_layout_torch = tensor_from_layout(tube, layout_torch)

stress_2 = stress_loss(tensor_layout_2, dist)
print("Stress loss first layout finished.")
stress_3 = stress_loss(tensor_layout_3, dist)
print("Stress loss second layout finished.")
stress_5 = stress_loss(tensor_layout_5, dist)
print("Stress loss third layout finished.")
stress_paper = stress_loss(tensor_layout_paper, dist)
print("Stress loss 4th layout finished.")
stress_adam = stress_loss(tensor_layout_adam, dist)
print("Stress loss 5th layout finished.")
stress_torch = stress_loss(tensor_layout_torch, dist)
print("Stress loss 6th layout finished.")

# write to file 

results = {
    # Layout 2
    "Edge Crossing 2": edge_crossing_2,
    "Causal Path Dispersion 2": causal_path_dispersion_2,
    "Closeness Eccentricity 2": closeness_eccentricity_2,
    "Stress 2": stress_2.item() if isinstance(stress_2, torch.Tensor) else stress_2,

    # Layout 3
    "Edge Crossing 3": edge_crossing_3,
    "Causal Path Dispersion 3": causal_path_dispersion_3,
    "Closeness Eccentricity 3": closeness_eccentricity_3,
    "Stress 3": stress_3.item() if isinstance(stress_3, torch.Tensor) else stress_3,

    # Layout 5
    "Edge Crossing 5": edge_crossing_5,
    "Causal Path Dispersion 5": causal_path_dispersion_5,
    "Closeness Eccentricity 5": closeness_eccentricity_5,
    "Stress 5": stress_5.item() if isinstance(stress_5, torch.Tensor) else stress_5,

    # Paper Layout
    "Edge Crossing Paper": edge_crossing_paper,
    "Causal Path Dispersion Paper": causal_path_dispersion_paper,
    "Closeness Eccentricity Paper": closeness_eccentricity_paper,
    "Stress Paper": stress_paper.item() if isinstance(stress_paper, torch.Tensor) else stress_paper,

    # Adam Layout
    "Edge Crossing Adam": edge_crossing_adam,
    "Causal Path Dispersion Adam": causal_path_dispersion_adam,
    "Closeness Eccentricity Adam": closeness_eccentricity_adam,
    "Stress Adam": stress_adam.item() if isinstance(stress_adam, torch.Tensor) else stress_adam,

    # Torch Layout
    "Edge Crossing Torch": edge_crossing_torch,
    "Causal Path Dispersion Torch": causal_path_dispersion_torch,
    "Closeness Eccentricity Torch": closeness_eccentricity_torch,
    "Stress Torch": stress_torch.item() if isinstance(stress_torch, torch.Tensor) else stress_torch,
}


with open(FILENAME_METRIC, 'a') as file:
    file.write("\n\n\n###################################### Tube ######################################################\n")
    # Get the maximum description length for consistent alignment
    max_length = max(len(description) for description in results)
    for description, value in results.items():
        # Align descriptions using ljust
        file.write(f"{description.ljust(max_length)} : {value}\n")

    if not worked_paper:
        file.write("The graph wasn't connected, so the SGD paper versions returned random layout")
    if not worked_adam:
        file.write("The graph wasn't connected, so the SGD adam versions returned random layout")
    if not worked_torch:
        file.write("The graph wasn't connected, so the SGD torch versions returned random layout")

print("Dataset finished.")

###################################### Wikipedia ######################################################

FILENAME_PLOT = "src/pathpyG/visualisations/Project_JS/evaluation/plots/wikipedia/wikipedia_"

DELTA = 1

wiki = from_ngram("src/pathpyG/visualisations/Project_JS/graphs/Wikipedia/paths_asteroid_viking.ngram", sep=";")

# create layouts
print("Creating layouts.")
layout_paper, worked_paper = SGD_stress_paper(wiki, iterations=30, delta=DELTA, learning_rate=0.01)
layout_adam, worked_adam = Adam_stress_torch(wiki, iterations=500, delta=DELTA, learning_rate=0.5)
layout_torch, worked_torch = SGD_stress_torch(wiki, iterations=200, delta=DELTA, learning_rate=0.001)

layout_2 = HotVis(wiki, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
layout_3 =  HotVis(wiki, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
layout_5 =  HotVis(wiki, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)

## plot
graph = pp.MultiOrderModel.from_PathData(wiki, max_order=1).layers[1]
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style)
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style)
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style)
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style)
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style)
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style)

# metrics
print("Determining edge crossings.")
edge_crossing_2 = edge_crossing(wiki, layout_2)
print("Edge crossings first layout finished.")
edge_crossing_3 = edge_crossing(wiki, layout_3)
print("Edge crossings second layout finished.")
edge_crossing_5 = edge_crossing(wiki, layout_5)
print("Edge crossings third layout finished.")
edge_crossing_paper = edge_crossing(wiki, layout_paper)
print("Edge crossings 4th layout finished.")
edge_crossing_adam = edge_crossing(wiki, layout_adam)
print("Edge crossings 5th layout finished.")
edge_crossing_torch = edge_crossing(wiki, layout_torch)
print("Edge crossings 6th layout finished.")


print("Determining causal path dispersion.")
causal_path_dispersion_2 = causal_path_dispersion(wiki, layout_2, DELTA, steps=[3], runs=[200])
print("Causal path dispersion first layout finished.")
causal_path_dispersion_3 = causal_path_dispersion(wiki, layout_3, DELTA, steps=[3], runs=[200])
print("Causal path dispersion second layout finished.")
causal_path_dispersion_5 = causal_path_dispersion(wiki, layout_5, DELTA, steps=[3], runs=[200])
print("Causal path dispersion third layout finished.")
causal_path_dispersion_paper = causal_path_dispersion(wiki, layout_paper, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 4th layout finished.")
causal_path_dispersion_adam = causal_path_dispersion(wiki, layout_adam, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 5th layout finished.")
causal_path_dispersion_torch = causal_path_dispersion(wiki, layout_torch, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 6th layout finished.")

print("Determining closeness eccentricity.")
closeness_eccentricity_2 = closeness_eccentricity(wiki, layout_2, DELTA, 0.1)
print("Closeness eccentricity first layout finished.")
closeness_eccentricity_3 = closeness_eccentricity(wiki, layout_3, DELTA, 0.1)
print("Closeness eccentricity second layout finished.")
closeness_eccentricity_5 = closeness_eccentricity(wiki, layout_5, DELTA, 0.1)
print("Closeness eccentricity third layout finished.")
closeness_eccentricity_paper = closeness_eccentricity(wiki, layout_paper, DELTA, 0.1)
print("Closeness eccentricity 4th layout finished.")
closeness_eccentricity_adam = closeness_eccentricity(wiki, layout_adam, DELTA, 0.1)
print("Closeness eccentricity 5th layout finished.")
closeness_eccentricity_torch = closeness_eccentricity(wiki, layout_torch, DELTA, 0.1)
print("Closeness eccentricity 6th layout finished.")

print("Determining shortest paths")
dist = shortest_paths_path_data(wiki)
print("determining stress loss")
tensor_layout_2 = tensor_from_layout(wiki, layout_2)
tensor_layout_3 = tensor_from_layout(wiki, layout_3)
tensor_layout_5 = tensor_from_layout(wiki, layout_5)
tensor_layout_paper = tensor_from_layout(wiki, layout_paper)
tensor_layout_adam = tensor_from_layout(wiki, layout_adam)
tensor_layout_torch = tensor_from_layout(wiki, layout_torch)

stress_2 = stress_loss(tensor_layout_2, dist)
print("Stress loss first layout finished.")
stress_3 = stress_loss(tensor_layout_3, dist)
print("Stress loss second layout finished.")
stress_5 = stress_loss(tensor_layout_5, dist)
print("Stress loss third layout finished.")
stress_paper = stress_loss(tensor_layout_paper, dist)
print("Stress loss 4th layout finished.")
stress_adam = stress_loss(tensor_layout_adam, dist)
print("Stress loss 5th layout finished.")
stress_torch = stress_loss(tensor_layout_torch, dist)
print("Stress loss 6th layout finished.")

# write to file 

results = {
    # Layout 2
    "Edge Crossing 2": edge_crossing_2,
    "Causal Path Dispersion 2": causal_path_dispersion_2,
    "Closeness Eccentricity 2": closeness_eccentricity_2,
    "Stress 2": stress_2.item() if isinstance(stress_2, torch.Tensor) else stress_2,

    # Layout 3
    "Edge Crossing 3": edge_crossing_3,
    "Causal Path Dispersion 3": causal_path_dispersion_3,
    "Closeness Eccentricity 3": closeness_eccentricity_3,
    "Stress 3": stress_3.item() if isinstance(stress_3, torch.Tensor) else stress_3,

    # Layout 5
    "Edge Crossing 5": edge_crossing_5,
    "Causal Path Dispersion 5": causal_path_dispersion_5,
    "Closeness Eccentricity 5": closeness_eccentricity_5,
    "Stress 5": stress_5.item() if isinstance(stress_5, torch.Tensor) else stress_5,

    # Paper Layout
    "Edge Crossing Paper": edge_crossing_paper,
    "Causal Path Dispersion Paper": causal_path_dispersion_paper,
    "Closeness Eccentricity Paper": closeness_eccentricity_paper,
    "Stress Paper": stress_paper.item() if isinstance(stress_paper, torch.Tensor) else stress_paper,

    # Adam Layout
    "Edge Crossing Adam": edge_crossing_adam,
    "Causal Path Dispersion Adam": causal_path_dispersion_adam,
    "Closeness Eccentricity Adam": closeness_eccentricity_adam,
    "Stress Adam": stress_adam.item() if isinstance(stress_adam, torch.Tensor) else stress_adam,

    # Torch Layout
    "Edge Crossing Torch": edge_crossing_torch,
    "Causal Path Dispersion Torch": causal_path_dispersion_torch,
    "Closeness Eccentricity Torch": closeness_eccentricity_torch,
    "Stress Torch": stress_torch.item() if isinstance(stress_torch, torch.Tensor) else stress_torch,
}


with open(FILENAME_METRIC, 'a') as file:
    file.write("\n\n\n###################################### Wikipedia ######################################################\n")
    # Get the maximum description length for consistent alignment
    max_length = max(len(description) for description in results)
    for description, value in results.items():
        # Align descriptions using ljust
        file.write(f"{description.ljust(max_length)} : {value}\n")

    if not worked_paper:
        file.write("The graph wasn't connected, so the SGD paper versions returned random layout")
    if not worked_adam:
        file.write("The graph wasn't connected, so the SGD adam versions returned random layout")
    if not worked_torch:
        file.write("The graph wasn't connected, so the SGD torch versions returned random layout")

print("Dataset finished.")

###################################### Flights ######################################################

FILENAME_PLOT = "src/pathpyG/visualisations/Project_JS/evaluation/plots/flights/flights_"

DELTA = 1

flights = from_ngram("src/pathpyG/visualisations/Project_JS/graphs/Flights/flights.ngram")

# create layouts
print("Creating layouts.")
layout_paper, worked_paper = SGD_stress_paper(flights, iterations=30, delta=DELTA, learning_rate=0.01)
layout_adam, worked_adam = Adam_stress_torch(flights, iterations=500, delta=DELTA, learning_rate=0.5)
layout_torch, worked_torch = SGD_stress_torch(flights, iterations=200, delta=DELTA, learning_rate=0.001)

layout_2 = HotVis(flights, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
layout_3 =  HotVis(flights, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
layout_5 =  HotVis(flights, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)


## plot
graph = pp.MultiOrderModel.from_PathData(flights, max_order=1).layers[1]
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style)
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style)
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style)
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style)
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style)
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style)

# metrics
print("Determining edge crossings.")
edge_crossing_2 = edge_crossing(flights, layout_2)
print("Edge crossings first layout finished.")
edge_crossing_3 = edge_crossing(flights, layout_3)
print("Edge crossings second layout finished.")
edge_crossing_5 = edge_crossing(flights, layout_5)
print("Edge crossings third layout finished.")
edge_crossing_paper = edge_crossing(flights, layout_paper)
print("Edge crossings 4th layout finished.")
edge_crossing_adam = edge_crossing(flights, layout_adam)
print("Edge crossings 5th layout finished.")
edge_crossing_torch = edge_crossing(flights, layout_torch)
print("Edge crossings 6th layout finished.")


print("Determining causal path dispersion.")
causal_path_dispersion_2 = causal_path_dispersion(flights, layout_2, DELTA, steps=[3], runs=[200])
print("Causal path dispersion first layout finished.")
causal_path_dispersion_3 = causal_path_dispersion(flights, layout_3, DELTA, steps=[3], runs=[200])
print("Causal path dispersion second layout finished.")
causal_path_dispersion_5 = causal_path_dispersion(flights, layout_5, DELTA, steps=[3], runs=[200])
print("Causal path dispersion third layout finished.")
causal_path_dispersion_paper = causal_path_dispersion(flights, layout_paper, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 4th layout finished.")
causal_path_dispersion_adam = causal_path_dispersion(flights, layout_adam, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 5th layout finished.")
causal_path_dispersion_torch = causal_path_dispersion(flights, layout_torch, DELTA, steps=[3], runs=[200])
print("Causal path dispersion 6th layout finished.")

print("Determining closeness eccentricity.")
closeness_eccentricity_2 = closeness_eccentricity(flights, layout_2, DELTA, 0.1)
print("Closeness eccentricity first layout finished.")
closeness_eccentricity_3 = closeness_eccentricity(flights, layout_3, DELTA, 0.1)
print("Closeness eccentricity second layout finished.")
closeness_eccentricity_5 = closeness_eccentricity(flights, layout_5, DELTA, 0.1)
print("Closeness eccentricity third layout finished.")
closeness_eccentricity_paper = closeness_eccentricity(flights, layout_paper, DELTA, 0.1)
print("Closeness eccentricity 4th layout finished.")
closeness_eccentricity_adam = closeness_eccentricity(flights, layout_adam, DELTA, 0.1)
print("Closeness eccentricity 5th layout finished.")
closeness_eccentricity_torch = closeness_eccentricity(flights, layout_torch, DELTA, 0.1)
print("Closeness eccentricity 6th layout finished.")

print("Determining shortest paths")
dist = shortest_paths_path_data(flights)
print("determining stress loss")
tensor_layout_2 = tensor_from_layout(flights, layout_2)
tensor_layout_3 = tensor_from_layout(flights, layout_3)
tensor_layout_5 = tensor_from_layout(flights, layout_5)
tensor_layout_paper = tensor_from_layout(flights, layout_paper)
tensor_layout_adam = tensor_from_layout(flights, layout_adam)
tensor_layout_torch = tensor_from_layout(flights, layout_torch)

stress_2 = stress_loss(tensor_layout_2, dist)
print("Stress loss first layout finished.")
stress_3 = stress_loss(tensor_layout_3, dist)
print("Stress loss second layout finished.")
stress_5 = stress_loss(tensor_layout_5, dist)
print("Stress loss third layout finished.")
stress_paper = stress_loss(tensor_layout_paper, dist)
print("Stress loss 4th layout finished.")
stress_adam = stress_loss(tensor_layout_adam, dist)
print("Stress loss 5th layout finished.")
stress_torch = stress_loss(tensor_layout_torch, dist)
print("Stress loss 6th layout finished.")

# write to file 

results = {
    # Layout 2
    "Edge Crossing 2": edge_crossing_2,
    "Causal Path Dispersion 2": causal_path_dispersion_2,
    "Closeness Eccentricity 2": closeness_eccentricity_2,
    "Stress 2": stress_2.item() if isinstance(stress_2, torch.Tensor) else stress_2,

    # Layout 3
    "Edge Crossing 3": edge_crossing_3,
    "Causal Path Dispersion 3": causal_path_dispersion_3,
    "Closeness Eccentricity 3": closeness_eccentricity_3,
    "Stress 3": stress_3.item() if isinstance(stress_3, torch.Tensor) else stress_3,

    # Layout 5
    "Edge Crossing 5": edge_crossing_5,
    "Causal Path Dispersion 5": causal_path_dispersion_5,
    "Closeness Eccentricity 5": closeness_eccentricity_5,
    "Stress 5": stress_5.item() if isinstance(stress_5, torch.Tensor) else stress_5,

    # Paper Layout
    "Edge Crossing Paper": edge_crossing_paper,
    "Causal Path Dispersion Paper": causal_path_dispersion_paper,
    "Closeness Eccentricity Paper": closeness_eccentricity_paper,
    "Stress Paper": stress_paper.item() if isinstance(stress_paper, torch.Tensor) else stress_paper,

    # Adam Layout
    "Edge Crossing Adam": edge_crossing_adam,
    "Causal Path Dispersion Adam": causal_path_dispersion_adam,
    "Closeness Eccentricity Adam": closeness_eccentricity_adam,
    "Stress Adam": stress_adam.item() if isinstance(stress_adam, torch.Tensor) else stress_adam,

    # Torch Layout
    "Edge Crossing Torch": edge_crossing_torch,
    "Causal Path Dispersion Torch": causal_path_dispersion_torch,
    "Closeness Eccentricity Torch": closeness_eccentricity_torch,
    "Stress Torch": stress_torch.item() if isinstance(stress_torch, torch.Tensor) else stress_torch,
}


with open(FILENAME_METRIC, 'a') as file:
    file.write("\n\n\n###################################### Flights ######################################################\n")
    # Get the maximum description length for consistent alignment
    max_length = max(len(description) for description in results)
    for description, value in results.items():
        # Align descriptions using ljust
        file.write(f"{description.ljust(max_length)} : {value}\n")
    
    if not worked_paper:
        file.write("The graph wasn't connected, so the SGD paper versions returned random layout")
    if not worked_adam:
        file.write("The graph wasn't connected, so the SGD adam versions returned random layout")
    if not worked_torch:
        file.write("The graph wasn't connected, so the SGD torch versions returned random layout")

print("Dataset finished.")
'''