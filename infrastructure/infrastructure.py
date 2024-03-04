import numpy as np
import random
import networkx as nx

# from collections import deque
from Nodes import Node

class Infrastucture: #infrastructure class, contains list of all nodes and adjacency matrix

    def __init__(self, num_of_nodes, edge, fog, src):
        self.num_of_nodes = num_of_nodes
        self.edge = (0, edge)
        self.fog = (edge, edge+fog)
        self.cloud = edge+1+fog
        self.src = src

    def _create_adj_matrix(self, adj_matrix): #creates adjacency matrix for graph
        if (adj_matrix.size != 0 and adj_matrix.shape[0] == self.num_of_nodes):
            # generate matrix
            matrix = np.zeros((self.num_of_nodes, self.num_of_nodes), dtype=np.int8)
            for i in range(self.edge[0], self.edge[1]):
                for j in range(i, self.edge[1]):
                    if (i == j):
                        matrix[i][j] = 0
                    else:
                        matrix[i][j] = random.choice([0, np.random.uniform(1,6)])
                        matrix[j][i] = matrix[i][j]
                for j in range(self.edge[1], self.fog[1]):
                    matrix[i][j] = random.choice([0, np.random.uniform(20,31)])
                    matrix[j][i] = matrix[i][j]
            for i in range(self.fog[0], self.fog[1]):
                for j in range(i, self.fog[1]):
                    if (i == j):
                        matrix[i][j] = 0
                    else:
                        matrix[i][j] = random.choice([0, np.random.uniform(10,21)])
                        matrix[j][i] = matrix[i][j]                      
            
            for i in range(self.fog[0], self.fog[1]):
                for j in range(i, self.fog[1]):
                    if (i == j):
                        matrix[i][j] = 0
                    else:
                        matrix[i][j] = np.random.uniform(10,20)
                        matrix[j][i] = matrix[i][j]     
            for i in range(self.fog[0], self.fog[1]):
                matrix[i][self.num_of_nodes-1] = np.random.uniform(50,100)
                matrix[self.num_of_nodes-1][i] = matrix[i][self.num_of_nodes-1]
            self.adj_matrix = matrix
        else: 
            self.adj_matrix = adj_matrix

    def _create_infrastructure(self): # initializes nodes
       
        nodes_list = []
        for i in range(self.edge[1]):
            
            cpu_capacity = int(np.random.uniform(4,9))
            ram_capacity = int(np.random.uniform(4,17))
            monetary_cost = np.random.uniform(2,3)
            
            edge_node = Node(id = i, layer = 'Edge', cpu_capacity = cpu_capacity, ram_capacity = ram_capacity, monetary_cost = monetary_cost, src_nodes=self.src)
            nodes_list.append(edge_node)

        for i in range(self.fog[1]-self.edge[1]):

            cpu_capacity = int(np.random.uniform(80,121))
            ram_capacity = int(np.random.uniform(120,201))
            monetary_cost = np.random.uniform(1,1.5)
            
            fog_node = Node(id = i + self.edge[1], layer = 'Fog', cpu_capacity = cpu_capacity, ram_capacity = ram_capacity, monetary_cost = monetary_cost, src_nodes=self.src)
            nodes_list.append(fog_node)

        cpu_capacity = 500
        ram_capacity = 1000
        monetary_cost = 0.5
        
        cloud_node = Node(id = self.num_of_nodes-1, layer = 'Cloud', cpu_capacity = cpu_capacity, ram_capacity = ram_capacity, monetary_cost = monetary_cost, src_nodes=self.src)
        nodes_list.append(cloud_node)
        
        self.nodes_list = nodes_list

        
    def _produce_networkx(self):
        
        network = nx.from_numpy_array(self.adj_matrix)
        
        attributes_list = ['layer', 'total_cpu_capacity', 'total_ram_capacity', 'monetary_cost', 'emergency_allocation_cost']
        for attribute in attributes_list:
            attribute_dict = {node.id: getattr(node, attribute) for node in self.nodes_list}
            nx.set_node_attributes(network, attribute_dict, attribute)
        
        for i in range(self.num_of_nodes):
            for j in range(self.num_of_nodes):
                if self.adj_matrix[i][j] != 0 :
                    weight_dict = {(i, j): self.adj_matrix[i][j]}
                    nx.set_edge_attributes(network, weight_dict, 'delay')
        
        return network
            
            
            
            
        