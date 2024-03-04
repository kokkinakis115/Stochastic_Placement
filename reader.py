# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:03:56 2024

@author: PANAGIOTIS
"""

# import logging

import networkx as nx
import numpy as np
from network.links import Links
from network.nodes import Nodes
# import yaml
from geopy.distance import geodesic


def read_network(file, cpu=None, mem=None, dr=None):
    SPEED_OF_LIGHT = 299792458  # meter per second
    PROPAGATION_FACTOR = 0.77  # https://en.wikipedia.org/wiki/Propagation_delay

    if not file.endswith(".graphml"):
        raise ValueError("{} is not a GraphML file".format(file))
    network = nx.read_graphml(file, node_type=int)

    # Set nodes
    node_ids = ["pop{}".format(n) for n in network.nodes]  # add "pop" to node index (eg, 1 --> pop1)
    # node_ids_dict = [n: for n in network.nodes]
    # nx.set_node_attributes(network, node_cpu_dict, 'cpu')
    # if specified, use the provided uniform node capacities
    if cpu is not None and mem is not None:
        node_cpu = {"pop{}".format(n): cpu for n in network.nodes}
        node_mem = {"pop{}".format(n): mem for n in network.nodes}
        
        node_cpu_dict = {n: cpu for n in network.nodes}
        node_mem_dict = {n: mem for n in network.nodes}
        
        nx.set_node_attributes(network, node_cpu_dict, 'cpu')
        nx.set_node_attributes(network, node_mem_dict, 'mem')
    # Else try to read them from the the node attributes (ie, graphml)
    
    else:
        cpu = nx.get_node_attributes(network, 'cpu')
        mem = nx.get_node_attributes(network, 'mem')
        try:
            node_cpu = {"pop{}".format(n): cpu[n] for n in network.nodes}
            node_mem = {"pop{}".format(n): mem[n] for n in network.nodes}
            
            node_cpu_dict = {n: cpu for n in network.nodes}
            node_mem_dict = {n: mem for n in network.nodes}
            
            nx.set_node_attributes(network, node_cpu_dict, 'cpu')
            nx.set_node_attributes(network, node_mem_dict, 'mem')
        except KeyError:
            raise ValueError("No CPU or mem. specified for {} (as cmd argument or in graphml)".format(file))

    # Set links
    link_ids = [("pop{}".format(e[0]), "pop{}".format(e[1])) for e in network.edges]
    
    
    if dr is not None:
        link_dr = {("pop{}".format(e[0]), "pop{}".format(e[1])): dr for e in network.edges}
        link_dr_dict = {(e[0], e[1]): dr for e in network.edges}
        
        nx.set_edge_attributes(network, link_dr_dict, 'dr')
    else:
        dr = nx.get_edge_attributes(network, 'dr')
        
        try:
            link_dr = {("pop{}".format(e[0]), "pop{}".format(e[1])): dr[e] for e in network.edges}
            link_dr_dict = {(e[0], e[1]): dr for e in network.edges}
            
            nx.set_edge_attributes(network, link_dr_dict, 'dr')
        except KeyError:
            raise ValueError("No link data rate specified for {} (as cmd argument or in graphml)".format(file))

    # Calculate link delay based on geo positions of nodes; duplicate links for bidirectionality
    link_delay = {}
    link_delay_dict = {}
    for e in network.edges(data=True):
        delay = 0
        
        if e[2].get("LinkDelay"):
            delay = e[2]['LinkDelay']
        else:
            n1 = network.nodes(data=True)[e[0]]
            n2 = network.nodes(data=True)[e[1]]
            n1_lat, n1_long = n1.get("Latitude"), n1.get("Longitude")
            n2_lat, n2_long = n2.get("Latitude"), n2.get("Longitude")
            distance = geodesic((n1_lat, n1_long), (n2_lat, n2_long)).meters  # in meters
            delay = (distance / SPEED_OF_LIGHT * 1000) * PROPAGATION_FACTOR  # in milliseconds
        # round delay to int using np.around for consistency with emulator
        link_delay[("pop{}".format(e[0]), "pop{}".format(e[1]))] = int(np.around(delay))
        link_delay_dict[(e[0], e[1])] = int(np.around(delay))
    
    nx.set_edge_attributes(network, link_delay_dict, 'delay')
    
    # Add reversed links for bidirectionality
    for e in network.edges:
        e = ("pop{}".format(e[0]), "pop{}".format(e[1]))
        e_reversed = (e[1], e[0])
        link_ids.append(e_reversed)
        link_dr[e_reversed] = link_dr[e]
        link_delay[e_reversed] = link_delay[e]

    nodes = Nodes(node_ids, node_cpu, node_mem)
    links = Links(link_ids, link_dr, link_delay)
    return nodes, links, network