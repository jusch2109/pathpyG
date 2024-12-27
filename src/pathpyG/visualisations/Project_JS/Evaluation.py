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


def eval(g: pp.TemporalGraph|pp.PathData, layout_2, layout_3, layout_5, layout_paper, layout_adam, layout_torch, layout_fr, delta, clusters = []):
    # metrics
    print("Determining edge crossings.")
    edge_crossing_2 = edge_crossing(g, layout_2)
    print("Edge crossings first layout finished.")
    edge_crossing_3 = edge_crossing(g, layout_3)
    print("Edge crossings second layout finished.")
    edge_crossing_5 = edge_crossing(g, layout_5)
    print("Edge crossings third layout finished.")
    edge_crossing_paper = edge_crossing(g, layout_paper)
    print("Edge crossings 4th layout finished.")
    edge_crossing_adam = edge_crossing(g, layout_adam)
    print("Edge crossings 5th layout finished.")
    edge_crossing_torch = edge_crossing(g, layout_torch)
    print("Edge crossings 6th layout finished.")
    edge_crossing_fr = edge_crossing(g, layout_fr)
    print("Edge crossings 7th layout finished.")


    print("Determining causal path dispersion.")
    causal_path_dispersion_2 = causal_path_dispersion(g, layout_2, delta, steps=[3], runs=[200])
    print("Causal path dispersion first layout finished.")
    causal_path_dispersion_3 = causal_path_dispersion(g, layout_3, delta, steps=[3], runs=[200])
    print("Causal path dispersion second layout finished.")
    causal_path_dispersion_5 = causal_path_dispersion(g, layout_5, delta, steps=[3], runs=[200])
    print("Causal path dispersion third layout finished.")
    causal_path_dispersion_paper = causal_path_dispersion(g, layout_paper, delta, steps=[3], runs=[200])
    print("Causal path dispersion 4th layout finished.")
    causal_path_dispersion_adam = causal_path_dispersion(g, layout_adam, delta, steps=[3], runs=[200])
    print("Causal path dispersion 5th layout finished.")
    causal_path_dispersion_torch = causal_path_dispersion(g, layout_torch, delta, steps=[3], runs=[200])
    print("Causal path dispersion 6th layout finished.")
    causal_path_dispersion_fr = causal_path_dispersion(g, layout_fr, delta, steps=[3], runs=[200])
    print("Causal path dispersion 7th layout finished.")

    print("Determining closeness eccentricity.")
    closeness_eccentricity_2 = closeness_eccentricity(g, layout_2, delta, 0.1)
    print("Closeness eccentricity first layout finished.")
    closeness_eccentricity_3 = closeness_eccentricity(g, layout_3, delta, 0.1)
    print("Closeness eccentricity second layout finished.")
    closeness_eccentricity_5 = closeness_eccentricity(g, layout_5, delta, 0.1)
    print("Closeness eccentricity third layout finished.")
    closeness_eccentricity_paper = closeness_eccentricity(g, layout_paper, delta, 0.1)
    print("Closeness eccentricity 4th layout finished.")
    closeness_eccentricity_adam = closeness_eccentricity(g, layout_adam, delta, 0.1)
    print("Closeness eccentricity 5th layout finished.")
    closeness_eccentricity_torch = closeness_eccentricity(g, layout_torch, delta, 0.1)
    print("Closeness eccentricity 6th layout finished.")
    closeness_eccentricity_fr = closeness_eccentricity(g, layout_fr, delta, 0.1)
    print("Closeness eccentricity 7th layout finished.")

    if(len(clusters) > 0):
        print("Determining cluster distance ratio.")
        cluster_distance_ratio_2 = cluster_distance_ratio(g, clusters, layout_2)
        print("Cluster distance ratio first layout finished.")
        cluster_distance_ratio_3 = cluster_distance_ratio(g, clusters, layout_3)
        print("Cluster distance ratio second layout finished.")
        cluster_distance_ratio_5 = cluster_distance_ratio(g, clusters, layout_5)
        print("Cluster distance ratio third layout finished.")
        cluster_distance_ratio_paper = cluster_distance_ratio(g, clusters, layout_paper)
        print("Cluster distance ratio 4th layout finished.")
        cluster_distance_ratio_adam = cluster_distance_ratio(g, clusters, layout_adam)
        print("Cluster distance ratio 5th layout finished.")
        cluster_distance_ratio_torch = cluster_distance_ratio(g, clusters, layout_torch)
        print("Cluster distance ratio 6th layout finished.")
        cluster_distance_ratio_fr = cluster_distance_ratio(g, clusters, layout_fr)
        print("Cluster distance ratio 7th layout finished.")

    print("Determining shortest paths")
    if isinstance(g, pp.TemporalGraph):
        dist, _ = pp.algorithms.temporal_shortest_paths(g, delta)
    else:
        dist = shortest_paths_path_data(g)

    print("determining stress loss")
    tensor_layout_2 = tensor_from_layout(g, layout_2)
    tensor_layout_3 = tensor_from_layout(g, layout_3)
    tensor_layout_5 = tensor_from_layout(g, layout_5)
    tensor_layout_paper = tensor_from_layout(g, layout_paper)
    tensor_layout_adam = tensor_from_layout(g, layout_adam)
    tensor_layout_torch = tensor_from_layout(g, layout_torch)
    tensor_layout_fr = tensor_from_layout(g, layout_fr)

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
    stress_fr = stress_loss(tensor_layout_fr, dist)
    print("Stress loss 7th layout finished.")

    # write to file 

    if(len(clusters)>0):
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

            # fruchtermann Reingold Layout
            "Edge Crossing FR": edge_crossing_fr,
            "Causal Path Dispersion FR": causal_path_dispersion_fr,
            "Closeness Eccentricity FR": closeness_eccentricity_fr,
            "Cluster Distance Ratio FR": cluster_distance_ratio_fr,
            "Stress FR": stress_fr.item() if isinstance(stress_fr, torch.Tensor) else stress_fr,

            "DELTA": delta,
        }
    else:
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

            # fruchtermann Reingold Layout
            "Edge Crossing FR": edge_crossing_fr,
            "Causal Path Dispersion FR": causal_path_dispersion_fr,
            "Closeness Eccentricity FR": closeness_eccentricity_fr,
            "Stress FR": stress_fr.item() if isinstance(stress_fr, torch.Tensor) else stress_fr,

            "DELTA": delta,
    }

    return results

###################################### Styling ######################################################

style = {}
style['node_color'] = 'blue'
style['edge_color'] = 'grey'
style['edge_opacity'] = 0.3



###################################### Synthetic Graph ######################################################

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
layout_fr = pp.layout(synthetic_graph.to_static_graph(), layout='fr')
layout_fr = {key: value.tolist() for key, value in layout_fr.items()}
print("Layout FR created")

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
print("6th plot cerated")
pp.plot(graph, layout=layout_fr, backend='matplotlib', filename=FILENAME_PLOT + "layout_fr", **style_synthetic)
print("7th plot cerated")

results = eval(synthetic_graph, layout_2, layout_3, layout_5, layout_paper, layout_adam, layout_torch, layout_fr, DELTA, clusters)



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
DELTA = 62

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
layout_fr = pp.layout(highschool_graph.to_static_graph(), layout='fr')
layout_fr = {key: value.tolist() for key, value in layout_fr.items()}
print("Layout FR created")
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
pp.plot(graph, layout=layout_fr, backend='matplotlib', filename=FILENAME_PLOT + "layout_fr", **style_high_school)

results = eval(highschool_graph, layout_2, layout_3, layout_5, layout_paper, layout_adam, layout_torch, layout_fr, DELTA, clusters)

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
DELTA = 80

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
layout_fr = pp.layout(hospital_graph.to_static_graph(), layout='fr')
layout_fr = {key: value.tolist() for key, value in layout_fr.items()}
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
pp.plot(graph, layout=layout_fr, backend='matplotlib', filename=FILENAME_PLOT + "layout_fr", **style_hospital)


results = eval(hospital_graph, layout_2, layout_3, layout_5, layout_paper, layout_adam, layout_torch, layout_fr, DELTA, clusters)


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

'''
###################################### Office ######################################################

FILENAME_PLOT = "src/pathpyG/visualisations/Project_JS/evaluation/plots/office/office_"
DELTA = 50

# load graph
office_graph = pp.io.read_csv_temporal_graph('src/pathpyG/visualisations/Project_JS/graphs/office/network/edges.csv', is_undirected=True, timestamp_format='%S')
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
layout_fr = pp.layout(office_graph.to_static_graph(), layout='fr')
layout_fr = {key: value.tolist() for key, value in layout_fr.items()}
layout_paper, worked_paper = SGD_stress_paper(office_graph, iterations=30, delta=DELTA, learning_rate=0.01)
layout_adam, worked_adam = Adam_stress_torch(office_graph, iterations=500, delta=DELTA, learning_rate=0.5)
layout_torch, worked_torch = SGD_stress_torch(office_graph, iterations=200, delta=DELTA, learning_rate=0.001)

layout_2 = HotVis(office_graph, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
layout_3 = HotVis(office_graph, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
layout_5 = HotVis(office_graph, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)

#layout_fr = pp.visualisations.layout(net, layout='fr')


## plot
graph = office_graph.to_static_graph()
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style_office)
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style_office)
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style_office)
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style_office)
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style_office)
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style_office)
pp.plot(graph, layout=layout_fr, backend='matplotlib', filename=FILENAME_PLOT + "layout_fr", **style_office)

results = eval(office_graph, layout_2, layout_3, layout_5, layout_paper, layout_adam, layout_torch, layout_fr, DELTA, clusters)

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
graph = pp.MultiOrderModel.from_PathData(tube, max_order=1).layers[1]
print("Creating layouts.")
layout_fr = pp.layout(graph, layout='fr')
layout_fr = {key: value.tolist() for key, value in layout_fr.items()}
layout_paper, worked_paper = SGD_stress_paper(tube, iterations=30, delta=DELTA, learning_rate=0.01)
layout_adam, worked_adam = Adam_stress_torch(tube, iterations=500, delta=DELTA, learning_rate=0.5)
layout_torch, worked_torch = SGD_stress_torch(tube, iterations=200, delta=DELTA, learning_rate=0.001)

layout_2 = HotVis(tube, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
layout_3 =  HotVis(tube, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
layout_5 =  HotVis(tube, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)

## plot
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style)
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style)
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style)
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style)
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style)
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style)
pp.plot(graph, layout=layout_fr, backend='matplotlib', filename=FILENAME_PLOT + "layout_fr", **style)

results = eval(tube, layout_2, layout_3, layout_5, layout_paper, layout_adam, layout_torch, layout_fr, DELTA, clusters=[])

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
graph = pp.MultiOrderModel.from_PathData(wiki, max_order=1).layers[1]
layout_fr = pp.layout(graph, layout='fr')
layout_fr = {key: value.tolist() for key, value in layout_fr.items()}
layout_paper, worked_paper = SGD_stress_paper(wiki, iterations=30, delta=DELTA, learning_rate=0.01)
layout_adam, worked_adam = Adam_stress_torch(wiki, iterations=500, delta=DELTA, learning_rate=0.5)
layout_torch, worked_torch = SGD_stress_torch(wiki, iterations=200, delta=DELTA, learning_rate=0.001)

layout_2 = HotVis(wiki, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
layout_3 =  HotVis(wiki, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
layout_5 =  HotVis(wiki, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)

## plot
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style)
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style)
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style)
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style)
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style)
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style)
pp.plot(graph, layout=layout_fr, backend='matplotlib', filename=FILENAME_PLOT + "layout_fr", **style)

results = eval(wiki, layout_2, layout_3, layout_5, layout_paper, layout_adam, layout_torch, layout_fr, DELTA, clusters=[])


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
graph = pp.MultiOrderModel.from_PathData(flights, max_order=1).layers[1]
layout_fr = pp.layout(graph, layout='fr')
layout_fr = {key: value.tolist() for key, value in layout_fr.items()}
layout_paper, worked_paper = SGD_stress_paper(flights, iterations=30, delta=DELTA, learning_rate=0.01)
layout_adam, worked_adam = Adam_stress_torch(flights, iterations=500, delta=DELTA, learning_rate=0.5)
layout_torch, worked_torch = SGD_stress_torch(flights, iterations=200, delta=DELTA, learning_rate=0.001)

layout_2 = HotVis(flights, 2, 50000, DELTA, alpha=[1, 0.5], force=10)
layout_3 =  HotVis(flights, 3, 50000, DELTA, alpha=[1, 0.5, 0.3], force=10)
layout_5 =  HotVis(flights, 5, 50000, DELTA, alpha=[1, 0.5, 0.3, 0.25, 0.2], force=10)


## plot
pp.plot(graph, layout=layout_2, backend='matplotlib', filename=FILENAME_PLOT + "layout_2", **style)
pp.plot(graph, layout=layout_3, backend='matplotlib', filename=FILENAME_PLOT + "layout_3", **style)
pp.plot(graph, layout=layout_5, backend='matplotlib', filename=FILENAME_PLOT + "layout_5", **style)
pp.plot(graph, layout=layout_paper, backend='matplotlib', filename=FILENAME_PLOT + "layout_paper", **style)
pp.plot(graph, layout=layout_adam, backend='matplotlib', filename=FILENAME_PLOT + "layout_adam", **style)
pp.plot(graph, layout=layout_torch, backend='matplotlib', filename=FILENAME_PLOT + "layout_torch", **style)
pp.plot(graph, layout=layout_fr, backend='matplotlib', filename=FILENAME_PLOT + "layout_fr", **style)

results = eval(flights, layout_2, layout_3, layout_5, layout_paper, layout_adam, layout_torch, layout_fr, DELTA, clusters=[])


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
