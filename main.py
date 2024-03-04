# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:02:34 2024

@author: PANAGIOTIS
"""

import numpy as np
import functions
import statistics
from simulation.simulation import Simulation
from models.predictor import BSM_predictor
from network_gen import generate_network

# Global Variables

workloads = np.loadtxt("./alibaba/workloads.csv", delimiter=',')
workloads = workloads/100
workload_volatility_array = [functions.calculate_volatility(workload) for workload in workloads]
workload_averages_array = [statistics.median(workload) for workload in workloads]

vol_dict = functions.produce_volatility_dict(workload_volatility_array)
starting_req_dict = {'low': 0.2, 'medium': 1.2 , 'high': 2.0}


def main():
    # microservice first time cpu requirements (before predictions) based on median usage

    # threshold_dict = {('low', 'low'): 0.15, ('medium', 'low'): 0.35, ('high', 'low'): 0.15, ('low', 'medium'): 0.25, ('medium', 'medium'): 0.5, ('high', 'medium'): 0.25, ('low', 'high'): 0.25, ('medium', 'high'): 0.7, ('high', 'high'): 0.4} 

    network_arr = generate_network()

    predictor = BSM_predictor(intr=0.00,
                              texp=6,
                              lookback_duration=6,
                              threshold=0.2)
    
    sim = Simulation(simulation_duration=144,
                     timeslot_duration=6,
                     lookforward_duration=6,
                     lookback_duration=6,
                     threshold=0.2,
                     number_of_applications=2,
                     number_of_edge_nodes=20,
                     number_of_fog_nodes=6,
                     number_of_src_nodes=10)
    sim._initialize_simulation(workloads, vol_dict, network=network_arr)
    # sim.infrastructure.nodes_list[1]._assign_task(cpu_req=1.0, app_id=0, microservice_id=0)
    print(sim._check_utilization())
    
    # print(sim.applications_stack[0].ms_stack)
    
if __name__ == "__main__":
    main()