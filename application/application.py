# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:46:02 2024

@author: PANAGIOTIS
"""

import numpy as np
import random
from collections import deque
from microservice import Microservice

class Application: # application/workload class, contains adjacency matrix of DAG 

    def __init__(self, app_id, num_of_ms, intensity, duration, src_node_id):
        self.app_id = app_id
        self.num_of_ms = num_of_ms
        self.intensity = intensity
        self.duration = duration
        self.ms_stack = []
        self.src_node_id = src_node_id
        
    def _create_DAG(self):
        dag_matrix = np.zeros((self.num_of_ms, self.num_of_ms), dtype=np.int8)
        for i in range(self.num_of_ms):
            for j in range(i, self.num_of_ms    ):
                if (i == j):
                    dag_matrix[i][j] = 0
                else:
                    dag_matrix[i][j] = random.choice([0, np.random.uniform(5, 10)])
        self.dag_matrix = dag_matrix    

    def _create_ms(self, workloads, vol_dict):
        ms_stack = deque()

        for id in range(self.num_of_ms):
            microservice = Microservice(app_id = self.app_id ,ms_id = id, intensity=self.intensity, volatility=random.choice(['low', 'medium', 'high']), duration = self.duration)
            microservice._create_usage2(workloads, vol_dict)
            ms_stack.append(microservice)

        self.ms_stack = ms_stack        