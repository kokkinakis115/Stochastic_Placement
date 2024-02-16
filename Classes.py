# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:50:00 2024

@author: PANAGIOTIS
"""

import numpy as np
# import pyfeng as pf
import matplotlib.pyplot as plt
import pandas as pd
import random
# import time
# import os
# import statistics
# from statistics import mean
# from scipy.stats import norm
# import seaborn as sns
# from warnings import filterwarnings
# import math
# import matplotlib.pyplot as plt
# import networkx as nx
from collections import deque
# from scipy.special import ndtri
# from statistics import NormalDist
import functions


class Node: # node class, we assume each node contains 1 machine, 3 types of nodes (edge, fog, cioud)

    def __init__(self, id, layer, cpu_capacity, ram_capacity, monetary_cost):
        self.id = id
        self.layer = layer
        self.total_cpu_capacity = cpu_capacity
        self.remaining_cpu_capacity = cpu_capacity
        self.total_ram_capacity = ram_capacity
        self.remaining_ram_capacity = ram_capacity
        self.monetary_cost = monetary_cost
        self.emergency_allocation_cost = 10*monetary_cost
        
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


class Infrastucture: #infrastructure class, contains list of all nodes and adjacency matrix

    def __init__(self, num_of_nodes, edge, fog):
        self.num_of_nodes = num_of_nodes
        self.edge = (0, edge)
        self.fog = (edge, edge+fog)
        self.cloud = edge+1+fog

    def _create_adj_matrix(self, adj_matrix): #creates adjacency matrix for graph
        if (not adj_matrix):
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
            
            edge_node = Node(id = i, layer = 'Edge', cpu_capacity = cpu_capacity, ram_capacity = ram_capacity, monetary_cost = monetary_cost)
            nodes_list.append(edge_node)

        for i in range(self.fog[1]-self.edge[1]):

            cpu_capacity = int(np.random.uniform(80,121))
            ram_capacity = int(np.random.uniform(120,201))
            monetary_cost = np.random.uniform(1,1.5)
            
            fog_node = Node(id = i + self.edge[1], layer = 'Fog', cpu_capacity = cpu_capacity, ram_capacity = ram_capacity, monetary_cost = monetary_cost)
            nodes_list.append(fog_node)

        cpu_capacity = 500
        ram_capacity = 1000
        monetary_cost = 0.5
        
        cloud_node = Node(id = self.num_of_nodes-1, layer = 'Cloud', cpu_capacity = cpu_capacity, ram_capacity = ram_capacity, monetary_cost = monetary_cost)
        nodes_list.append(cloud_node)
        
        self.nodes_list = nodes_list
          
        
class Microservice: #microservice class, time_series is created upon initialization based on given intensity of application

    def __init__(self, app_id, ms_id, intensity, volatility, duration):
        self.app_id = app_id
        self.ms_id = ms_id
        self.intensity = intensity
        self.volatility = volatility
        self.duration = duration
        self.assigned_to_node = None
        # self._create_usage1(duration_in_hours=duration+1, intensity=intensity, volatility=volatility)
        # self._create_usage2(intensity=intensity, volatility=volatility)
        
    def __repr__(self):
        return f"Microservice #{self.ms_id}, assigned to Node #{self.assigned_to_node}"

    def _assign_to_node(self, timeslot_duration, timeslot_nr, node):
        self.assigned_to_node = node.id
        return node._assign_task(self.usage_time_series[timeslot_duration*timeslot_nr], self.ms_id)
                
    def _create_usage1(self, duration_in_hours, intensity, volatility):
        
        sigma_dict = {'low': 15*60, 'medium': 20*60 , 'high': 30*60}
        mu_dict = {'low': 1800, 'medium': 3600 , 'high': 2*3600}
        lambd_dict = {'low': 50, 'medium': 20 , 'high': 10}
        
        sigma = sigma_dict[volatility]
        mu = mu_dict[intensity]
        lambd = lambd_dict[intensity]
        
        wl = functions.create_wl(sigma=sigma, mu=mu, lambd=lambd, users=200, duration_in_hours=duration_in_hours)
        
        sigma_dict_vol = {'low': 0.0005, 'medium': 0.001 , 'high': 0.002}
        sigma = sigma_dict_vol[volatility]
        mu = 0
        
        time_series = np.zeros(duration_in_hours*3600+1)
        for i in range(len(wl)):
            duration = wl.loc[i, 'end time']+1-wl.loc[i, 'start time']
            # time_series[wl.loc[i, 'start time']:wl.loc[i, 'end time']+1] += np.ones(duration)*wl.loc[i, 'normalized cpu requested']+(sigma * np.random.randn(duration) + mu) 
            
            time_series[wl.loc[i, 'start time']:wl.loc[i, 'end time']+1] += np.ones(duration)*wl.loc[i, 'normalized cpu requested']#+(sigma * np.random.randn(duration) + mu) 

            if (intensity != 'low'):
                time_series += sigma * np.random.randn(duration_in_hours*3600+1) + mu
            
            time_series = np.clip(time_series, 0.001, 1000)     
        
        self.usage_time_series = time_series[3600:]
        return
    
    def _create_usage2(self, workloads, vol_dict):
        
        self.usage_time_series = functions.produce_workload(intensity=self.intensity, vol=self.volatility, workloads=workloads, vol_dict=vol_dict)
        return

    def _plot_usage(self):
        dates = pd.date_range(start='00:00', periods=self.duration*3600+1, freq='1s')

        data = pd.DataFrame({'date': dates, 'value': self.usage_time_series})
        
        # Plot the time-series data
        plt.plot(data['date'], data['value'])
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.xticks(rotation = 45)
        plt.title("Usage Time Series")
        plt.show()


class Application: # application/workload class, contains adjacency matrix of DAG 

    def __init__(self, app_id, num_of_ms, intensity, duration):
        self.app_id = app_id
        self.num_of_ms = num_of_ms
        self.intensity = intensity
        self.duration = duration
        self.ms_stack = []
        
    def _create_DAG(self):
        dag_matrix = np.zeros((self.num_of_ms, self.num_of_ms), dtype=np.int8)
        for i in range(self.num_of_ms):
            for j in range(i, self.num_of_ms):
                if (i == j):
                    dag_matrix[i][j] = 0
                else:
                    dag_matrix[i][j] = random.choice([0, np.random.uniform(1,6)])
        self.dag_matrix = dag_matrix    

    def _create_ms(self, workloads, vol_dict):
        ms_stack = deque()

        for id in range(self.num_of_ms):
            microservice = Microservice(app_id = self.app_id ,ms_id = id, intensity=self.intensity, volatility=random.choice(['low', 'medium', 'high']), duration = self.duration)
            microservice._create_usage2(workloads, vol_dict)
            ms_stack.append(microservice)

        self.ms_stack = ms_stack        


class BSM_predictor:
    
    def __init__(self, intr = 0.0, texp = 3600, lookback_duration = 3600, threshold = 0.1):
        self.time_slot_duration = texp
        self.intr = intr
        self.lookback = lookback_duration
        # self.starting_strike = strike
        self.threshold = threshold
        return
        
    
    def _timeslot_volatility2(self, time_series, lookback):
        data = pd.DataFrame({'value': time_series})
        data['log'] = np.log(data['value'] / data['value'].shift(1))
        df = data['log'].rolling(window=lookback).std()
        return df
        
    def _compute_options(self, workload, starting_strike):
        texp = self.time_slot_duration #1 day in trading = 1 second in workload
        cp = 1

        num_of_time_slots = len(workload) // self.time_slot_duration
        predictions_array = [starting_strike]
        # max_array = []
        deltas_array = [0]
        volatility = self._timeslot_volatility2(workload, self.lookback)
        
        # strike_price = self.starting_strike
        
        for timeslot_num in range(num_of_time_slots): #compute loop for each timeslot (1h)
            # lookback_timeseries = workload[max((timeslot_num+1)*self.time_slot_duration-self.lookback, 0):min((timeslot_num+1)*self.time_slot_duration, len(workload)+1)]
            
            # timeslot_timeseries = workload[timeslot_num*self.time_slot_duration:min((timeslot_num+1)*self.time_slot_duration, len(workload)+1)]
            # spot_price = timeslot_timeseries[-1]
            
            spot_price = workload[(timeslot_num+1)*self.time_slot_duration]
            
            vol = volatility.loc[(timeslot_num+1)*self.time_slot_duration]
                
            prediction = min(2.5*spot_price, functions.compute_strike_price2(spot=spot_price, vol=vol, texp=texp, intr=self.intr, delta=self.threshold, cp=cp))
            
            predictions_array.append(prediction)
            # max_array.append(max(timeslot_timeseries))
            
        self.predictions = predictions_array[:-1]
        # self.max_array = max_array
        predictions = np.array([starting_strike])
        # maxima = np.array([])
        
        for timeslot_num in range(num_of_time_slots): 
            predictions = np.append(predictions, np.ones(self.time_slot_duration)*self.predictions[timeslot_num])
            # maxima = np.append(maxima, np.ones(self.time_slot_duration)*self.max_array[timeslot_num])
        
        self.predictions_time_series = predictions
        # self.maxima_time_series = maxima
        self.deltas_array = deltas_array

        return

    def _predict_next_timeslot(self, microservice, prediction_timeslot_num):

        timeslot_num = prediction_timeslot_num-1
        texp = self.time_slot_duration 
        cp = 1

        workload = microservice.usage_time_series

        volatility = self._timeslot_volatility2(workload, self.lookback)

        timeslot_timeseries = workload[timeslot_num*self.time_slot_duration:min((timeslot_num+1)*self.time_slot_duration, len(workload)+1)]
        spot_price = timeslot_timeseries[-1]

        vol = volatility.loc[(timeslot_num+1)*self.time_slot_duration]

        prediction = min(2*spot_price, functions.compute_strike_price2(spot=spot_price, vol=vol, texp=texp, intr=self.intr, delta=self.threshold, cp=cp))        

        return prediction


    def _plot_workload_and_options(self, workload):

        # num_of_time_slots = len(workload) // self.time_slot_duration

        dates1 = pd.date_range(start='00:00', periods=len(workload), freq='1s')
        dates2 = pd.date_range(start='00:00', periods=len(self.predictions_time_series), freq='1s')

        data = pd.DataFrame({'date': dates1, 'value': workload})

        # Set the 'date' column as the index
        data.set_index('date', inplace=True)

        # Plot the time-series data
        plt.plot(data.index, data['value'], label = "Workload")
        plt.plot(dates2, self.predictions_time_series, label = "Predictions")
        # plt.plot(data.index, self.maxima_time_series, label = "Time Slot Max")
        plt.xlabel('Time')
        plt.ylabel('Volume of Workloads')
        plt.xticks(rotation = 45)
        plt.gcf().autofmt_xdate()
        # plt.title(f'Workload Prediction for {workload.users} users in a timespan of {workload.size} minutes\n T = {self.time_slot_duration}, r = {self.intr}, M = {self.lookback}, t = {self.texp}')
        plt.legend()
        plt.show()

        return
    

class Simulation:

    def __init__(self, simulation_duration, timeslot_duration, lookforward_duration, lookback_duration, threshold, number_of_applications, number_of_edge_nodes, number_of_fog_nodes):

        self.simulation_duration = simulation_duration
        self.timeslot_duration = timeslot_duration
        self.lookforward_duration = lookforward_duration
        self.lookback_duration = lookback_duration
        self.threshold = threshold
        self.number_of_applications = number_of_applications
        self.number_of_edge_nodes = number_of_edge_nodes
        self.number_of_fog_nodes = number_of_fog_nodes
        self.num_of_total_nodes = number_of_edge_nodes + number_of_fog_nodes + 1

    def _initialize_simulation(self, workloads, vol_dict):

        total_nodes = self.number_of_edge_nodes + self.number_of_fog_nodes + 1

        infrastructure = Infrastucture(total_nodes, self.number_of_edge_nodes, self.number_of_fog_nodes)
        infrastructure._create_adj_matrix([])
        infrastructure._create_infrastructure()
        self.infrastructure = infrastructure

        applications_stack = deque()

        for app_id in range(self.number_of_applications):
            intensity = np.random.choice(['low', 'medium', 'high'], p = [0.4, 0.35, 0.25])
            number_of_ms = int(np.random.uniform(1,6))
            new_app = Application(app_id, number_of_ms, intensity, self.simulation_duration)
            new_app._create_DAG()
            new_app._create_ms(workloads, vol_dict)
            applications_stack.append(new_app)
    
        self.applications_stack = applications_stack
        return  


    def _check_utilization(self):

        nodes_utilization_list = deque()
        for node in self.infrastructure.nodes_list:
            node_utilization = (node.total_cpu_capacity - node.remaining_cpu_capacity)/node.total_cpu_capacity
            nodes_utilization_list.append(node_utilization)

        return nodes_utilization_list       


    def _check_violations(self, timeslot_nr):
        violation_counter = 0
        violations_stack = deque()

        for node_id in range(len(self.infrastructure.nodes_list)):
            node_usage = np.zeros(self.timeslot_duration)
            for app_id, ms_id in self.infrastructure.nodes_list[node_id].ms_stack:
                # print(app_id, ms_id)
                node_usage += self.applications_stack[app_id].ms_stack[ms_id].usage_time_series[timeslot_nr*self.timeslot_duration:(timeslot_nr+1)*self.timeslot_duration]
            if max(node_usage > self.infrastructure.nodes_list[node_id].total_cpu_capacity):
                violation_counter += 1
                if node_id not in violations_stack:
                    violations_stack.append(node_id)
        
        return violations_stack, violation_counter

    def _produce_initial_scheme(self, starting_req_dict):
        initial_scheme = []
        
        for app_id in range(self.number_of_applications):
            app_predictions= deque([])
            for ms in self.applications_stack[app_id].ms_stack:
                ms_prediction = starting_req_dict[ms.intensity]
                app_predictions.append(ms_prediction)
            initial_scheme.append(app_predictions)
        return initial_scheme
        
    
    def _produce_prediction_scheme(self, prediction_model, timeslot_num):
        prediction_scheme = []
        
        for app_id in range(self.number_of_applications):
            app_predictions= deque([])
            for ms in self.applications_stack[app_id].ms_stack:
                ms_prediction = prediction_model._predict_next_timeslot(microservice=ms, prediction_timeslot_num=timeslot_num)
                app_predictions.append(ms_prediction)
            prediction_scheme.append(app_predictions)
        return prediction_scheme
    
     
    def _assign_scheme_to_nodes(self, assignment_scheme, prediction_scheme):
        
        for app_id in range(self.number_of_applications):
            for (ms_id, node_id) in enumerate(assignment_scheme[app_id]):
                # print(node_id)
                if not self.infrastructure.nodes_list[node_id]._assign_task(cpu_req=prediction_scheme[app_id][ms_id], app_id=app_id, microservice_id=ms_id):
                    return (node_id, app_id, ms_id)
        return True
    
    def _calculate_cost(self):
        
        return
        
    def _run_simulation(self, prediction_model, starting_req_dict):
        
        timeslot = 0
        period = 0
        
        prediction_model = BSM_predictor(intr=0.00, texp=self.lookforward_duration, lookback_duration=self.lookback_duration, threshold=self.threshold)
        simulation_overview = {'utilization_per_timeslot': [], 'violations_in_time_slot': [], 'number_of_relocations': []}
        
        # initialize by assigning microservices to nodes (use starting requirements dictionary for cpu requirements)
        initial_assignment_scheme = self._produce_initial_scheme(starting_req_dict)
        
        # run allocation algorithm to produce initial_assignment_scheme # !!!
        
        initial_allocation_scheme = functions.create_random_scheme(self)
        self._assign_scheme_to_nodes(assignment_scheme=initial_allocation_scheme, prediction_scheme=initial_assignment_scheme)
        
        # check violations, cost and relocations required
        
        simulation_overview['utilization_per_timeslot'].append(self._check_utilization())
        simulation_overview['violations_in_time_slot'].append(0)
        simulation_overview['number_of_relocations'].append(0)

        number_of_relocations = 0
        violation_counter = 0

        # calculate initial cost # !!!
        # timeslot_cost = 0
        # total_cost = 0

        # make predictions for each microservice and run RA algorithm / RUN FOR EVERY TICK (5 minutes for alibaba)
        while (period < self.simulation_duration):

            period += 1
            if period % self.timeslot_duration == 0:
                timeslot += 1
                prediction_scheme = self._produce_prediction_scheme(prediction_model, timeslot)

                # run allocation algorithm to produce assignment_scheme # !!!
                # self._assign_scheme_to_nodes(assignment_scheme=assignment_scheme, prediction_scheme=prediction_scheme)

                simulation_overview['utilization_per_timeslot'].append(self._check_utilization())
                simulation_overview['violations_in_time_slot'].append(violation_counter)
                simulation_overview['number_of_relocations'].append(number_of_relocations)

                violation_counter = 0
                number_of_relocations = 0

            violations_stack = []
            for node_id in range(len(self.infrastructure.nodes_list)):
                node_usage = 0
                for app_id, ms_id in self.infrastructure.nodes_list[node_id].ms_stack:
                    node_usage += self.applications_stack[app_id].ms_stack[ms_id].usage_time_series[period]
                if node_usage > self.infrastructure.nodes_list[node_id].total_cpu_capacity:
                    violation_counter += 1
                    if node_id not in violations_stack:
                        violations_stack.append(node_id)

            # run relocation algorithm for each node in violations_stack # !!!
            # calculate new number_of_relocations # !!!

        # repeat

        return simulation_overview