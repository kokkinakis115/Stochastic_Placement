from geopy.distance import geodesic
import random
import numpy as np
import pandas as pd
from collections import deque

# def change_to_inf(G):
#     new_arr = np.zeros(G.shape, dtype=np.int32)
#     for i in range(G.shape[0]):
#         for j in range(G.shape[1]):
#             if i == j or G[i][j] != 0:
#                 new_arr[i][j] = G[i][j]
#             else:
#                 new_arr[i][j] = 10000
#     return new_arr

# def floyd_warshall(G):
#     G_inf = change_to_inf(G)
#     # distance = list(map(lambda i: list(map(lambda j: j, i)), G))

#     # Adding vertices individually
#     graph = G_inf
#     V = len(G)
#     for k in range(V):
#         for i in range(V):
#             for j in range(V):
#                 graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])
#     return graph        

def generate_network():
    real_nodes = pd.read_csv('../Topology.csv')
    real_nodes = real_nodes.iloc[::-1]
    
    NUM_OF_EDGE_NODES = 20
    NUM_OF_TOTAL_NODES = NUM_OF_EDGE_NODES + real_nodes.shape[0]
    
    SPEED_OF_LIGHT = 299792458  # meter per second
    PROPAGATION_FACTOR = 0.77  # https://en.wikipedia.org/wiki/Propagation_delay
    
    # Random point generation 
    
    # Area in attica 
    latitude_min = 37.92
    latitude_max = 38.08
    
    longitude_min = 23.67
    longitude_max = 23.86
    
    edge_points = deque([])
    for i in range(NUM_OF_EDGE_NODES):
        X = random.uniform(latitude_min, latitude_max)
        Y = random.uniform(longitude_min, longitude_max)
        edge_points.append([X, Y])
        
    # Find nearest fog nodes
    links = deque([])
    distances_deque = deque([])
    
    for index, [e_lat, e_long] in enumerate(edge_points):
        distances = []
        for i, dc in enumerate(real_nodes.to_numpy()):
            distance = geodesic((e_lat, e_long), (dc[2], dc[3]))
            distances.append(distance.km)
        distances = np.array(distances) # find shortest distances between real nodes and edge nodes
        sorted_distances = np.sort(distances)
        sorted_indices = np.argsort(distances)
        node_links = []
        link_distances = []
        for i in range(int(random.uniform(2,5))):
            node_links.append(sorted_indices[i])
            link_distances.append(sorted_distances[i])
        links.append(node_links)    
        distances_deque.append(link_distances)
        
    # Generate adjacency array
    
    adjacency_arr = np.zeros((NUM_OF_TOTAL_NODES,NUM_OF_TOTAL_NODES))
    for i in range(NUM_OF_EDGE_NODES):
        for j in range(len(links[i])):
            adjacency_arr[i][links[i][j]+NUM_OF_EDGE_NODES] = distances_deque[i][j]
            adjacency_arr[links[i][j]+NUM_OF_EDGE_NODES][i]
    for i, dc1 in enumerate(real_nodes.to_numpy()):
        for j, dc2 in enumerate(real_nodes.to_numpy()):
            if (i != j):
                distance = geodesic((dc1[2], dc1[3]), (dc2[2], dc2[3]))
                adjacency_arr[i+NUM_OF_EDGE_NODES][j+NUM_OF_EDGE_NODES] = distance.km
    
    compute_delay = lambda t: (t / SPEED_OF_LIGHT) * PROPAGATION_FACTOR * 10**9
    adjacency_arr = np.floor(compute_delay(adjacency_arr))
    adjacency_arr = adjacency_arr.astype(int)
    
    # Generate edge node
    for i in range(NUM_OF_EDGE_NODES):
        for j in range(NUM_OF_EDGE_NODES):
            if (i == j):
                adjacency_arr[i][j] = 0
            else:
                adjacency_arr[i][j] = random.choice([0, np.random.uniform(1,6)])
                adjacency_arr[j][i] = adjacency_arr[i][j]
    return adjacency_arr
        
