# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:02:34 2024

"""

import numpy as np
import functions
import pandas as pd
import statistics
from simulation.simulation import Simulation
from models.predictor import BSM_predictor
from models.fixed_model import FixedModel
from network_gen import generate_network

# Global Variables

workloads = np.loadtxt("./alibaba/workloads.csv", delimiter=',')
workloads = workloads/100
workload_volatility_array = [functions.calculate_volatility(workload) for workload in workloads]
workload_averages_array = [statistics.median(workload) for workload in workloads]

vol_dict = functions.produce_volatility_dict(workload_volatility_array)
starting_req_dict = {'low': 0.5, 'medium': 1.5 , 'high': 2.5}


def experiment(predictor):
    # microservice first time cpu requirements (before predictions) based on median usage

    # threshold_dict = {('low', 'low'): 0.15, ('medium', 'low'): 0.35, ('high', 'low'): 0.15, ('low', 'medium'): 0.25, ('medium', 'medium'): 0.5, ('high', 'medium'): 0.25, ('low', 'high'): 0.25, ('medium', 'high'): 0.7, ('high', 'high'): 0.4} 

    network_arr = generate_network()
    # network_arr = None
    weights = {'cost': 2, 'latency': 1, 'relocation': 1}  # Example: prioritize minimizing relocations higher

    # predictor = BSM_predictor(intr=0.0,
    #                           texp=6,
    #                           lookback_duration=6,
    #                           threshold=0.1
    #                           )
    # predictor = FixedModel(2.5)

    sim = Simulation(simulation_duration=144,
                     timeslot_duration=6,
                     lookforward_duration=6,
                     lookback_duration=6,
                     threshold=0.1,
                     number_of_applications=100,
                     number_of_edge_nodes=20,
                     number_of_fog_nodes=6,
                     number_of_src_nodes=10,
                     weights = weights
                     )

    sim._initialize_simulation(workloads, vol_dict, network=network_arr)
    sim_overview, total_cost = sim._run_simulation(predictor, starting_req_dict)
    
    experiment_results = {'Total Number of Violations': sum(sim_overview['violations_in_time_slot']),
                          'Total Number of Relocations': sum(sim_overview['number_of_relocations']),
                          'Total Cost': total_cost,
                          'Average Resource Utilization Percentage': sum(sim_overview['total_utilization_percentage'])/len(sim_overview['total_utilization_percentage'])
                          }
    # print(f"Total Number of Violations: {sum(sim_overview['violations_in_time_slot'])}")
    # print(f"Total Number of Relocations: {sum(sim_overview['number_of_relocations'])}")
    # print(f"Total Cost: {total_cost}")    
    # print(f"Average Resource Utilization Percentage: {sum(sim_overview['total_utilization_percentage'])/len(sim_overview['total_utilization_percentage'])}")
    
    return experiment_results

if __name__ == "__main__":
    
    # Prediction Models
    fixed_predictor_15 = FixedModel(1.5)
    fixed_predictor_3 = FixedModel(3.0)
    
    bsm_predictor_005 = BSM_predictor(intr=0.0,
                              texp=6,
                              lookback_duration=6,
                              threshold=0.05
                              )
    bsm_predictor_010 = BSM_predictor(intr=0.0,
                              texp=6,
                              lookback_duration=6,
                              threshold=0.10
                              )
    
    #Experiments
    number_of_iterations = 10
    
    Fixed_results_15 = []
    Fixed_results_30 = []
    
    BSM_results_005 = []
    BSM_results_010 = []
    
    for i in range(number_of_iterations):
        print(f"Experiment #{i} for Fixed Model")
        Fixed_results_15.append(experiment(fixed_predictor_15))
        
    for i in range(number_of_iterations):
        print(f"Experiment #{i} for Fixed Model")
        Fixed_results_30.append(experiment(fixed_predictor_3))
        
    for i in range(number_of_iterations):
        print(f"Experiment #{i} for Black Scholes Model")
        BSM_results_005.append(experiment(bsm_predictor_005))

    for i in range(number_of_iterations):
        print(f"Experiment #{i} for Black Scholes Model")
        BSM_results_010.append(experiment(bsm_predictor_010))

    Final_Fixed_results_15 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in Fixed_results_15]),
                               "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in Fixed_results_15]),
                               "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in Fixed_results_15]),
                               "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in Fixed_results_15]),
                               "Average Cost": statistics.mean([result['Total Cost'] for result in Fixed_results_15]),
                               "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in Fixed_results_15]),
                               "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in Fixed_results_15]),
                               "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in Fixed_results_15])
                               }

    Final_Fixed_results_30 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in Fixed_results_30]),
                               "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in Fixed_results_30]),
                               "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in Fixed_results_30]),
                               "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in Fixed_results_30]),
                               "Average Cost": statistics.mean([result['Total Cost'] for result in Fixed_results_30]),
                               "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in Fixed_results_30]),
                               "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in Fixed_results_30]),
                               "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in Fixed_results_30])
                               }

    Final_BSM_results_005 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BSM_results_005]),
                             "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BSM_results_005]),
                             "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BSM_results_005]),
                             "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BSM_results_005]),
                             "Average Cost": statistics.mean([result['Total Cost'] for result in BSM_results_005]),
                             "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BSM_results_005]),
                             "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BSM_results_005]),
                             "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BSM_results_005])
                             }
    
    Final_BSM_results_010 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BSM_results_010]),
                             "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BSM_results_010]),
                             "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BSM_results_010]),
                             "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BSM_results_010]),
                             "Average Cost": statistics.mean([result['Total Cost'] for result in BSM_results_010]),
                             "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BSM_results_010]),
                             "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BSM_results_010]),
                             "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BSM_results_010])
                             }
    
    # df = pd.concat([pd.DataFrame(Final_Fixed_results_15, index=[0]), pd.DataFrame(Final_Fixed_results_30, index=[1]), pd.DataFrame(Final_BSM_results_005, index=[2]), pd.DataFrame(Final_BSM_results_010, index=[3])])
    # results_csv = df.to_csv('parameters/results.csv')
    
    
    
    
    
    