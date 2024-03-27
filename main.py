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
    
    # network_arr = generate_network()
    network_arr = None


    sim = Simulation(simulation_duration=144,
                     timeslot_duration=6,
                     lookforward_duration=6,
                     lookback_duration=6,
                     threshold=0.1,
                     number_of_applications=50,
                     number_of_edge_nodes=40,
                     number_of_fog_nodes=6,
                     number_of_src_nodes=100,
                     weights = weights
                     )

    sim._initialize_simulation(workloads, vol_dict, network=network_arr)
    sim_overview, total_cost, usage = sim._run_simulation(predictor, starting_req_dict)
    
    # print(sim_overview['total_latency'])
    
    experiment_results = {'Total Number of Violations': sum(sim_overview['violations_in_time_slot']),
                          'Total Number of Relocations': sum(sim_overview['number_of_relocations']),
                          'Total Cost': total_cost,
                          'Average Resource Utilization Percentage': sum(sim_overview['total_utilization_percentage'])/len(sim_overview['total_utilization_percentage']),
                          'Average latency Src': sum(sim_overview['average_latency_src'])/len(sim_overview['average_latency_src']),
                          # 'Average latency MS': sum(sim_overview['average_latency_ms'])/len(sim_overview['average_latency_ms'])
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
    # fixed_predictor_3 = FixedModel(2.5)
    
    # bsm_predictor_005 = BSM_predictor(intr=0.0,
    #                           texp=6,
    #                           lookback_duration=6,
    #                           threshold=0.02
    #                           )
    # bsm_predictor_010 = BSM_predictor(intr=0.0,
    #                           texp=6,
    #                           lookback_duration=6,
    #                           threshold=0.10
    #                           )
    
    # # Experiments
    # ### Models
    # number_of_iterations = 20
    
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

    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for Black Scholes Model")
    #     res, _ = experiment(bsm_predictor_005, weights)
    #     BSM_results_005.append(res)

    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for Black Scholes Model")
    #     res, _ = experiment(bsm_predictor_010, weights)
    #     BSM_results_010.append(res)

    # # Save Results

    # Final_Fixed_results_15 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in Fixed_results_15]),
    #                             "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in Fixed_results_15]),
    #                             "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in Fixed_results_15]),
    #                             "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in Fixed_results_15]),
    #                             "Average Cost": statistics.mean([result['Total Cost'] for result in Fixed_results_15]),
    #                             "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in Fixed_results_15]),
    #                             "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in Fixed_results_15]),
    #                             "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in Fixed_results_15]),
    #                             "Average Latency": statistics.mean([result['Average latency Src'] for result in Fixed_results_15]),
    #                             "Std Deviation in Latency": statistics.stdev([result['Average latency Src'] for result in Fixed_results_15]),
    #                             # "Average Latency MS": statistics.mean([result['Average latency MS'] for result in Fixed_results_15]),
    #                             # "Std Deviation in Latency MS": statistics.stdev([result['Average latency MS'] for result in Fixed_results_15]),
    #                             }

    # Final_Fixed_results_30 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in Fixed_results_30]),
    #                             "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in Fixed_results_30]),
    #                             "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in Fixed_results_30]),
    #                             "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in Fixed_results_30]),
    #                             "Average Cost": statistics.mean([result['Total Cost'] for result in Fixed_results_30]),
    #                             "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in Fixed_results_30]),
    #                             "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in Fixed_results_30]),
    #                             "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in Fixed_results_30]),
    #                             "Average Latency": statistics.mean([result['Average latency Src'] for result in Fixed_results_30]),
    #                             "Std Deviation in Latency": statistics.stdev([result['Average latency Src'] for result in Fixed_results_30]),
    #                             # "Average Latency MS": statistics.mean([result['Average latency MS'] for result in Fixed_results_30]),
    #                             # "Std Deviation in Latency MS": statistics.stdev([result['Average latency MS'] for result in Fixed_results_30]),
    #                             }

    # Final_BSM_results_005 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BSM_results_005]),
    #                           "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BSM_results_005]),
    #                           "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BSM_results_005]),
    #                           "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BSM_results_005]),
    #                           "Average Cost": statistics.mean([result['Total Cost'] for result in BSM_results_005]),
    #                           "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BSM_results_005]),
    #                           "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BSM_results_005]),
    #                           "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BSM_results_005]),
    #                           "Average Latency": statistics.mean([result['Average latency Src'] for result in BSM_results_005]),
    #                           "Std Deviation in Latency ": statistics.stdev([result['Average latency Src'] for result in BSM_results_005]),
    #                           # "Average Latency MS": statistics.mean([result['Average latency MS'] for result in BSM_results_005]),
    #                           # "Std Deviation in Latency MS": statistics.stdev([result['Average latency MS'] for result in BSM_results_005]),
    #                           }
    
    # Final_BSM_results_010 = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BSM_results_010]),
    #                           "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BSM_results_010]),
    #                           "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BSM_results_010]),
    #                           "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BSM_results_010]),
    #                           "Average Cost": statistics.mean([result['Total Cost'] for result in BSM_results_010]),
    #                           "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BSM_results_010]),
    #                           "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BSM_results_010]),
    #                           "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BSM_results_010]),
    #                           "Average Latency": statistics.mean([result['Average latency Src'] for result in BSM_results_010]),
    #                           "Std Deviation in Latency": statistics.stdev([result['Average latency Src'] for result in BSM_results_010]),
    #                           # "Average Latency MS": statistics.mean([result['Average latency MS'] for result in BSM_results_010]),
    #                           # "Std Deviation in Latency MS": statistics.stdev([result['Average latency MS'] for result in BSM_results_010]),
    #                           }
    
    # df = pd.concat([pd.DataFrame(Final_Fixed_results_15, index=[0]), pd.DataFrame(Final_Fixed_results_30, index=[1]), pd.DataFrame(Final_BSM_results_010, index=[2]), pd.DataFrame(Final_BSM_results_005, index=[3])])
    # results_csv = df.to_csv('parameters/results.csv')
    
    # Usage 
    
    # fixed_predictor_15 = FixedModel(1.5)
    # fixed_predictor_25 = FixedModel(2.5)
    
    # bsm_predictor = BSM_predictor(intr=0.0,
    #                               texp=6, 
    #                               lookback_duration=6, 
    #                               threshold=0.05
    #                               )
    
    # weights = {'cost': 1, 'latency': 100, 'relocation': 1}
    # _, usage_latency = experiment(fixed_predictor_15, weights)
    # usage_latency_df = pd.DataFrame(usage_latency)
    # usage_latency_df.to_csv("parameters/usage_latency.csv")
    
    # weights = {'cost': 62, 'latency': 1, 'relocation': 1}
    # _, usage_cost = experiment(fixed_predictor_25, weights)
    # usage_cost_df = pd.DataFrame(usage_cost)
    # usage_cost_df.to_csv("parameters/usage_cost.csv")
    
    # weights = {'cost': 45, 'latency': 1, 'relocation': 1}
    # _, usage_balanced = experiment(bsm_predictor, weights)
    # usage_balanced_df = pd.DataFrame(usage_balanced)
    # usage_balanced_df.to_csv("parameters/usage_balanced.csv")
    
    
    ### Objectives
    
    # Utilization
    
    # bsm_predictor = BSM_predictor(intr=0.0,
    #                               texp=6, 
    #                               lookback_duration=6, 
    #                               threshold=0.10
    #                               )
    # weights = {'cost': 1, 'latency': 100, 'relocation': 1}
    # _, usage = experiment(bsm_predictor, weights)
    # usage_df = pd.DataFrame(usage)
    # usage_df.to_csv("usage.csv")
    
    # number_of_iterations = 10
    # bsm_predictor = BSM_predictor(intr=0.0,
    #                                 texp=6,
    #                                 lookback_duration=6,
    #                                 threshold=0.05
    #                                 )
    
    # BS_model_cost = []
    # BS_model_latency = []
    # BS_model_balanced = []
    
    # weights = {'cost': 50, 'latency': 1, 'relocation': 1}
    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for Cost Minimizing Model")
    #     results, _ = experiment(bsm_predictor, weights)
    #     BS_model_cost.append(results)
        
    # weights = {'cost': 1, 'latency': 15, 'relocation': 1}
    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for latency Minimizing Model")
    #     results, _ = experiment(bsm_predictor, weights)
    #     BS_model_latency.append(results)
        
    # weights = {'cost': 10, 'latency': 10, 'relocation': 1}
    # for i in range(number_of_iterations):
    #     print(f"Experiment #{i} for Balanced Model")
    #     results, _ = experiment(bsm_predictor, weights)
    #     BS_model_balanced.append(results)
        
    # Cost_minimizing_results = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BS_model_cost]),
    #                             "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BS_model_cost]),
    #                             "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BS_model_cost]),
    #                             "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BS_model_cost]),
    #                             "Average Cost": statistics.mean([result['Total Cost'] for result in BS_model_cost]),
    #                             "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BS_model_cost]),
    #                             "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BS_model_cost]),
    #                             "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BS_model_cost]),
    #                             "Average Latency Src": statistics.mean([result['Average latency Src'] for result in BS_model_cost]),
    #                             "Std Deviation in Latency Src": statistics.stdev([result['Average latency Src'] for result in BS_model_cost]),
    #                             "Average Latency MS": statistics.mean([result['Average latency MS'] for result in BS_model_cost]),
    #                             "Std Deviation in Latency MS": statistics.stdev([result['Average latency MS'] for result in BS_model_cost]),
    #                             }

    # Latency_minimizing_results = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BS_model_latency]),
    #                                 "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BS_model_latency]),
    #                                 "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BS_model_latency]),
    #                                 "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BS_model_latency]),
    #                                 "Average Cost": statistics.mean([result['Total Cost'] for result in BS_model_latency]),
    #                                 "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BS_model_latency]),
    #                                 "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BS_model_latency]),
    #                                 "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BS_model_latency]),
    #                                 "Average Latency Src": statistics.mean([result['Average latency Src'] for result in BS_model_latency]),
    #                                 "Std Deviation in Latency Src": statistics.stdev([result['Average latency Src'] for result in BS_model_latency]),
    #                                 "Average Latency MS": statistics.mean([result['Average latency MS'] for result in BS_model_latency]),
    #                                 "Std Deviation in Latency MS": statistics.stdev([result['Average latency MS'] for result in BS_model_latency]),
    #                                 }
    
    # Balanced_results = {"Average Number of Violations": statistics.mean([result['Total Number of Violations'] for result in BS_model_balanced]),
    #                             "Std Deviation in Number of Violations": statistics.stdev([result['Total Number of Violations'] for result in BS_model_balanced]),
    #                             "Average Number of Relocations": statistics.mean([result['Total Number of Relocations'] for result in BS_model_balanced]),
    #                             "Std Deviation in Number of Relocations": statistics.stdev([result['Total Number of Relocations'] for result in BS_model_balanced]),
    #                             "Average Cost": statistics.mean([result['Total Cost'] for result in BS_model_balanced]),
    #                             "Std Deviation in Cost": statistics.stdev([result['Total Cost'] for result in BS_model_balanced]),
    #                             "Average Resource Utilization Percentage": statistics.mean([result['Average Resource Utilization Percentage'] for result in BS_model_balanced]),
    #                             "Std Deviation in Resource Utilization Percentage": statistics.stdev([result['Average Resource Utilization Percentage'] for result in BS_model_balanced]),
    #                             "Average Latency Src": statistics.mean([result['Average latency Src'] for result in BS_model_balanced]),
    #                             "Std Deviation in Latency Src": statistics.stdev([result['Average latency Src'] for result in BS_model_balanced]),
    #                             "Average Latency MS": statistics.mean([result['Average latency MS'] for result in BS_model_balanced]),
    #                             "Std Deviation in Latency MS": statistics.stdev([result['Average latency MS'] for result in BS_model_balanced]),
    #                             }

    # df = pd.concat([pd.DataFrame(Cost_minimizing_results, index=[0]), pd.DataFrame(Balanced_results, index=[1]), pd.DataFrame(Latency_minimizing_results, index=[2])])
    # results_csv = df.to_csv('parameters/objectives_results.csv')
    
    bsm_predictor = BSM_predictor(intr=0.0,
                                    texp=6,
                                    lookback_duration=6,
                                    threshold=0.05
                                    )
    
    weights = {'cost': 62, 'latency': 1, 'relocation': 1}
    
    print("Experiment for Cost Minimizing Model.")
    cost_results, _ = experiment(bsm_predictor, weights)
    
    weights = {'cost': 1, 'latency': 50, 'relocation': 1}
    print("Experiment for latency Minimizing Model.")
    latency_results, _ = experiment(bsm_predictor, weights)
    
    weights = {'cost': 45 , 'latency': 1, 'relocation': 1}
    print("Experiment for Balanced Model.")
    balanced_results, _ = experiment(bsm_predictor, weights)
    
    Cost_minimizing_results = {"Average Number of Violations": cost_results['Total Number of Violations'],
                                "Average Number of Relocations": cost_results['Total Number of Relocations'],
                                "Average Cost": cost_results['Total Cost'],
                                "Average Resource Utilization Percentage": cost_results['Average Resource Utilization Percentage'],
                                "Average Latency Src": cost_results['Average latency Src'],
                                # "Average Latency MS": cost_results['Average latency MS'],
                                }

    Latency_minimizing_results = {"Average Number of Violations": latency_results['Total Number of Violations'],
                                    "Average Number of Relocations": latency_results['Total Number of Relocations'],
                                    "Average Cost": latency_results['Total Cost'],
                                    "Average Resource Utilization Percentage": latency_results['Average Resource Utilization Percentage'],
                                    "Average Latency Src": latency_results['Average latency Src'],
                                    # "Average Latency MS": latency_results['Average latency MS'],
                                    }
    
    Balanced_results = {"Average Number of Violations": balanced_results['Total Number of Violations'],
                        "Average Number of Relocations": balanced_results['Total Number of Relocations'],
                        "Average Cost": balanced_results['Total Cost'],
                        "Average Resource Utilization Percentage": balanced_results['Average Resource Utilization Percentage'],
                        "Average Latency Src": balanced_results['Average latency Src'],
                        # "Average Latency MS": balanced_results['Average latency MS'],
                        }
    
    df = pd.concat([pd.DataFrame(Cost_minimizing_results, index=[0]), pd.DataFrame(Balanced_results, index=[1]), pd.DataFrame(Latency_minimizing_results, index=[2])])
    results_csv = df.to_csv('parameters/objectives_results2.csv')
        
    
    
    
    
    
    
    