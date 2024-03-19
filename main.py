# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:02:34 2024

"""

import numpy as np
import functions
import pandas as pd
import random
import statistics
from simulation.simulation import Simulation
from models.predictor import BSM_predictor
from models.fixed_model import FixedModel
from network_gen import generate_network

# Global Variables

workloads = np.loadtxt("./parameters/workloads.csv", delimiter=',')
workloads = workloads/100
workload_volatility_array = [functions.calculate_volatility(workload) for workload in workloads]
workload_averages_array = [statistics.median(workload) for workload in workloads]

vol_dict = functions.produce_volatility_dict(workload_volatility_array)
starting_req_dict = {'low': 0.5, 'medium': 1.5 , 'high': 2.5}


def experiment(predictor, weights):
    
    # np.random.seed(10)
    # random.seed(10)
    
    # microservice first time cpu requirements (before predictions) based on median usage

    # threshold_dict = {('low', 'low'): 0.15, ('medium', 'low'): 0.35, ('high', 'low'): 0.15, ('low', 'medium'): 0.25, ('medium', 'medium'): 0.5, ('high', 'medium'): 0.25, ('low', 'high'): 0.25, ('medium', 'high'): 0.7, ('high', 'high'): 0.4} 

    # network_arr = generate_network()
    network_arr = None

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
                     number_of_applications=20,
                     number_of_edge_nodes=20,
                     number_of_fog_nodes=1,
                     number_of_src_nodes=10,
                     weights = weights
                     )

    sim._initialize_simulation(workloads, vol_dict, network=network_arr)
    sim_overview, total_cost, usage = sim._run_simulation(predictor, starting_req_dict)
    
    # print(sim_overview['total_latency'])
    
    experiment_results = {'Total Number of Violations': sum(sim_overview['violations_in_time_slot']),
                          'Total Number of Relocations': sum(sim_overview['number_of_relocations']),
                          'Total Cost': total_cost,
                          'Average Resource Utilization Percentage': sum(sim_overview['total_utilization_percentage'])/len(sim_overview['total_utilization_percentage']),
                          'Average latency': sum(sim_overview['average_latency'])/len(sim_overview['average_latency'])
                          }
    # print(f"Total Number of Violations: {sum(sim_overview['violations_in_time_slot'])}")
    # print(f"Total Number of Relocations: {sum(sim_overview['number_of_relocations'])}")
    # print(f"Total Cost: {total_cost}")    
    # print(f"Average Resource Utilization Percentage: {sum(sim_overview['total_utilization_percentage'])/len(sim_overview['total_utilization_percentage'])}")
    
    return experiment_results, usage

if __name__ == "__main__":
    
    # weights = {'cost': 1, 'latency': 1, 'relocation': 1}  # Example: prioritize minimizing relocations higher
    
    # # Prediction Models
    # fixed_predictor_15 = FixedModel(1.5)
    # fixed_predictor_3 = FixedModel(3.0)
    
    # bsm_predictor_005 = BSM_predictor(intr=0.0,
    #                           texp=6,
    #                           lookback_duration=6,
    #                           threshold=0.05
    #                           )
    # bsm_predictor_010 = BSM_predictor(intr=0.0,
    #                           texp=6,
    #                           lookback_duration=6,
    #                           threshold=0.10
    #                           )
    
    # # Experiments
    # ### Models
    # number_of_iterations = 10
    
    # Fixed_results_15 = []
    # Fixed_results_30 = []
    
    # BSM_results_005 = []
    # BSM_results_010 = []
    
    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for Fixed Model")
    #     res, _ = experiment(fixed_predictor_15, weights)
    #     Fixed_results_15.append(res)
        
    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for Fixed Model")
    #     res, _ = experiment(fixed_predictor_3, weights)
    #     Fixed_results_30.append(res)
    #     # Fixed_results_30.append(experiment(fixed_predictor_3))
        
    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for Black Scholes Model")
    #     res, _ = experiment(bsm_predictor_005, weights)
    #     BSM_results_005.append(res)

    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for Black Scholes Model")
    #     res, _ = experiment(bsm_predictor_010, weights)
    #     BSM_results_010.append(res)
    #     # BSM_results_010.append(experiment(bsm_predictor_010))
        
    # # Save Results

    # Final_Fixed_results_15 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in Fixed_results_15]),
    #                             "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in Fixed_results_15]),
    #                             "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in Fixed_results_15]),
    #                             "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in Fixed_results_15]),
    #                             "Average Cost": statistics.mean([result['Total Cost'] for result in Fixed_results_15]),
    #                             "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in Fixed_results_15]),
    #                             "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in Fixed_results_15]),
    #                             "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in Fixed_results_15]),
    #                             "Average Latency": statistics.mean([result['Average latency'] for result in Fixed_results_15]),
    #                             "Std Deviation in Latency": statistics.stdev([result['Average latency'] for result in Fixed_results_15]),
    #                             }

    # Final_Fixed_results_30 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in Fixed_results_30]),
    #                             "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in Fixed_results_30]),
    #                             "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in Fixed_results_30]),
    #                             "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in Fixed_results_30]),
    #                             "Average Cost": statistics.mean([result['Total Cost'] for result in Fixed_results_30]),
    #                             "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in Fixed_results_30]),
    #                             "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in Fixed_results_30]),
    #                             "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in Fixed_results_30]),
    #                             "Average Latency": statistics.mean([result['Average latency'] for result in Fixed_results_30]),
    #                             "Std Deviation in Latency": statistics.stdev([result['Average latency'] for result in Fixed_results_30]),
    #                             }

    # Final_BSM_results_005 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BSM_results_005]),
    #                           "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BSM_results_005]),
    #                           "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BSM_results_005]),
    #                           "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BSM_results_005]),
    #                           "Average Cost": statistics.mean([result['Total Cost'] for result in BSM_results_005]),
    #                           "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BSM_results_005]),
    #                           "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BSM_results_005]),
    #                           "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BSM_results_005]),
    #                           "Average Latency": statistics.mean([result['Average latency'] for result in BSM_results_005]),
    #                           "Std Deviation in Latency": statistics.stdev([result['Average latency'] for result in BSM_results_005]),
    #                           }
    
    # Final_BSM_results_010 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BSM_results_010]),
    #                           "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BSM_results_010]),
    #                           "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BSM_results_010]),
    #                           "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BSM_results_010]),
    #                           "Average Cost": statistics.mean([result['Total Cost'] for result in BSM_results_010]),
    #                           "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BSM_results_010]),
    #                           "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BSM_results_010]),
    #                           "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BSM_results_010]),
    #                           "Average Latency": statistics.mean([result['Average latency'] for result in BSM_results_010]),
    #                           "Std Deviation in Latency": statistics.stdev([result['Average latency'] for result in BSM_results_010]),
    #                           }
    
    # df = pd.concat([pd.DataFrame(Final_Fixed_results_15, index=[0]), pd.DataFrame(Final_Fixed_results_30, index=[1]), pd.DataFrame(Final_BSM_results_005, index=[2]), pd.DataFrame(Final_BSM_results_010, index=[3])])
    # results_csv = df.to_csv('parameters/results.csv')
    
    ### Usage 
    
    bsm_predictor = BSM_predictor(intr=0.0, texp=6, lookback_duration=6, threshold=0.10)
    weights = {'cost': 1, 'latency': 15, 'relocation': 1}
    _, usage_latency = experiment(bsm_predictor, weights)
    usage_latency_df = pd.DataFrame(usage_latency)
    usage_latency_df.to_csv("usage_latency.csv")
    
    weights = {'cost': 50, 'latency': 1, 'relocation': 1}
    _, usage_cost = experiment(bsm_predictor, weights)
    usage_cost_df = pd.DataFrame(usage_cost)
    usage_cost_df.to_csv("usage_cost.csv")
    
    weights = {'cost': 30, 'latency': 2, 'relocation': 1}
    _, usage_balanced = experiment(bsm_predictor, weights)
    usage_balanced_df = pd.DataFrame(usage_balanced)
    usage_balanced_df.to_csv("usage_balanced.csv")
    
    ### Objectives
    
    # Utilization
    
    # bsm_predictor = BSM_predictor(intr=0.0, texp=6, lookback_duration=6, threshold=0.10)
    # weights = {'cost': 1, 'latency': 100, 'relocation': 1}
    # _, usage = experiment(bsm_predictor, weights)
    # usage_df = pd.DataFrame(usage)
    # usage_df.to_csv("usage.csv")
    
    # number_of_iterations = 10
    # bsm_predictor = BSM_predictor(intr=0.0,
    #                           texp=6,
    #                           lookback_duration=6,
    #                           threshold=0.10
    #                           )
    
    # BS_model_cost = []
    # BS_model_latency = []
    
    # weights = {'cost': 15, 'latency': 1, 'relocation': 1}
    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for Cost Minimizing Model")
    #     results, _ = experiment(bsm_predictor, weights)
    #     BS_model_cost.append(results)
        
    # weights = {'cost': 1, 'latency': 15, 'relocation': 1}
    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for Relocations Minimizing Model")
    #     results, _ = experiment(bsm_predictor, weights)
    #     BS_model_latency.append(results)
        
    # Cost_minimizing_results = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BS_model_cost]),
    #                             "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BS_model_cost]),
    #                             "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BS_model_cost]),
    #                             "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BS_model_cost]),
    #                             "Average Cost": statistics.mean([result['Total Cost'] for result in BS_model_cost]),
    #                             "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BS_model_cost]),
    #                             "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BS_model_cost]),
    #                             "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BS_model_cost]),
    #                             "Average Latency": statistics.mean([result['Average latency'] for result in BS_model_cost]),
    #                             "Std Deviation in Latency": statistics.stdev([result['Average latency'] for result in BS_model_cost]),
    #                             }

    # Latency_minimizing_results = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BS_model_latency]),
    #                                 "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BS_model_latency]),
    #                                 "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BS_model_latency]),
    #                                 "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BS_model_latency]),
    #                                 "Average Cost": statistics.mean([result['Total Cost'] for result in BS_model_latency]),
    #                                 "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BS_model_latency]),
    #                                 "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BS_model_latency]),
    #                                 "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BS_model_latency]),
    #                                 "Average Latency": statistics.mean([result['Average latency'] for result in BS_model_latency]),
    #                                 "Std Deviation in Latency": statistics.stdev([result['Average latency'] for result in BS_model_latency]),
    #                                 }
        
    
    
    
    
    
    
    