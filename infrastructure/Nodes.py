# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:26:18 2024

@author: PANAGIOTIS
"""

from collections import deque
import numpy as np
# from scipy.special import ndtri
# from statistics import NormalDist
# import functions


class Node: # node class, we assume each node contains 1 machine, 3 types of nodes (edge, fog, cioud)

    def __init__(self, id, layer, cpu_capacity, ram_capacity, monetary_cost, src_nodes):
        self.id = id
        self.layer = layer
        self.total_cpu_capacity = cpu_capacity
        self.remaining_cpu_capacity = cpu_capacity
        self.total_ram_capacity = ram_capacity
        self.remaining_ram_capacity = ram_capacity
        self.monetary_cost = monetary_cost
        self.emergency_allocation_cost = 10*monetary_cost
        self.src_nodes_max_delays = [int(np.random.uniform(3,7)) if layer == 'Edge' else 100 for i in range(src_nodes)]
        
        self.ms_stack = deque()

    def __repr__(self):
        return f"{self.layer} Node #{self.id}, CPU: {self.total_cpu_capacity}, RAM: {self.total_ram_capacity}, Cost: {self.monetary_cost}"

    def _get_specs(self): #returns remaining resources
        specs = dict(id = self.id, CPU_rem = self.remaining_cpu_capacity, RAM_rem = self.remaining_ram_capacity, Assigned_tasks = self.ms_stack)
        return specs
    
    def _assign_task(self, cpu_req, app_id, microservice_id): # function that assigns task to node if there are enough resources
        if (self.remaining_cpu_capacity >= cpu_req):
            self.remaining_cpu_capacity -= cpu_req
            self.ms_stack.append((app_id ,microservice_id))
            return True
        else:
            return False

    def _deassign_task(self, microservice):
        try:
            self.ms_stack.remove(microservice)
            self.remaining_cpu_capacity += microservice.cpu_req
            return True
        except ValueError:
            return False

    def _reset_node(self):
        self.ms_stack.clear()