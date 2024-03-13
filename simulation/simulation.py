import numpy as np
from collections import deque
from infrastructure.infrastructure import Infrastucture
from application.application import Application
from models.predictor import BSM_predictor
import models.relocations as ra

import functions
import networkx as nx

class Simulation:

    def __init__(self, simulation_duration, timeslot_duration, lookforward_duration, lookback_duration, threshold, number_of_applications, number_of_edge_nodes, number_of_fog_nodes, number_of_src_nodes, weights):

        self.simulation_duration = simulation_duration
        self.timeslot_duration = timeslot_duration
        self.lookforward_duration = lookforward_duration
        self.lookback_duration = lookback_duration
        self.threshold = threshold
        self.number_of_applications = number_of_applications
        self.number_of_edge_nodes = number_of_edge_nodes
        self.number_of_fog_nodes = number_of_fog_nodes
        self.number_of_infr_nodes = number_of_edge_nodes + number_of_fog_nodes + 1
        self.number_of_src_nodes = number_of_src_nodes
        self.weights = weights

    def _initialize_simulation(self, workloads, vol_dict, network = None):

        total_nodes = self.number_of_edge_nodes + self.number_of_fog_nodes + 1
        
        infrastructure = Infrastucture(total_nodes, self.number_of_edge_nodes, self.number_of_fog_nodes, self.number_of_src_nodes)
        if type(network) == type(None):
            network = []
        infrastructure._create_adj_matrix(network)
        infrastructure._create_infrastructure()
        self.infrastructure = infrastructure
    
        applications_stack = deque()

        for app_id in range(self.number_of_applications):
            intensity = np.random.choice(['low', 'medium', 'high'], p = [0.4, 0.35, 0.25])
            number_of_ms = int(np.random.uniform(1,7))
            new_app = Application(app_id, number_of_ms, intensity, self.simulation_duration, src_node_id=int(np.random.uniform(0, self.number_of_src_nodes)))
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
            
        initial_array = np.zeros((self.number_of_applications, 6))
        for app_id in range(self.number_of_applications):
            for (ms_id, cpu_req) in enumerate(initial_scheme[app_id]):
                initial_array[app_id][ms_id] = cpu_req
        return initial_array
        
    
    def _produce_prediction_scheme(self, prediction_model, timeslot_num):
        prediction_scheme = []
        
        for app_id in range(self.number_of_applications):
            app_predictions= deque([])
            for ms in self.applications_stack[app_id].ms_stack:
                ms_prediction = prediction_model._predict_next_timeslot(microservice=ms, prediction_timeslot_num=timeslot_num)
                app_predictions.append(ms_prediction)
            prediction_scheme.append(app_predictions)
            
        predictions_array = np.zeros((self.number_of_applications, 6))
        for app_id in range(self.number_of_applications):
            for (ms_id, cpu_req) in enumerate(prediction_scheme[app_id]):
                predictions_array[app_id][ms_id] = cpu_req
        return predictions_array
    
     
    def _assign_scheme_to_nodes(self, assignment_scheme, prediction_scheme):
        
        allocation_cost = 0
        for app_id in range(self.number_of_applications):
            for (ms_id, node_id) in enumerate(assignment_scheme[app_id]):
                # print(node_id)
                if prediction_scheme[app_id][ms_id] != 0:
                    if not self.infrastructure.nodes_list[node_id]._assign_task(cpu_req=prediction_scheme[app_id][ms_id], app_id=app_id, microservice_id=ms_id):
                        print(f"Failed to allocate Microservice #{ms_id} of Application #{app_id} to Node #{node_id}")
                        print(f"Node Usage: {self.infrastructure.nodes_list[node_id].ms_stack}")
                        return False
                allocation_cost += self.infrastructure.nodes_list[node_id].monetary_cost
        return allocation_cost
    
    
    def _get_resource_availability(self):
        node_resource_availability = {f"{node.layer} Node #{node.id}": float(node.remaining_cpu_capacity) for node in self.infrastructure.nodes_list}
        return node_resource_availability
    
    def _get_node_index(self):
        node_index = {f"{node.layer} Node #{node.id}": node.id for node in self.infrastructure.nodes_list}
        return node_index
    
    def _get_cpu_costs(self):
        cpuCosts = {f"{node.layer} Node #{node.id}": float(node.monetary_cost) for node in self.infrastructure.nodes_list}
        return cpuCosts
    
    def _get_node_latencies(self):
        latencies = {node.id: float(node.latency) for node in self.infrastructure.nodes_list}
        return latencies
    
    def _get_app_latencies(self):
        workload_latencies = [functions.add_arrays(np.zeros((6,6)), app.dag_matrix) for app in self.applications_stack]
        return workload_latencies
    
    def _clear_nodes(self):
        for node in self.infrastructure.nodes_list:
            node._reset_node()
            
    def _calculate_utilization_percentage(self):
        total_resources = 0 
        for node in self.infrastructure.nodes_list:
            total_resources += node.total_cpu_capacity
        used_resources = 0 
        for node in self.infrastructure.nodes_list:
            used_resources += node.total_cpu_capacity - node.remaining_cpu_capacity
        return used_resources/total_resources
        
    def _run_simulation(self, prediction_model, starting_req_dict):
        
        # print("App Demands: ", [app.intensity for app in self.applications_stack])
        # print("Number of microservices: ", [app.num_of_ms for app in self.applications_stack])
        timeslot = 0
        period = 0
        
        # prediction_model = BSM_predictor(intr=0.00, texp=self.lookforward_duration, lookback_duration=self.lookback_duration, threshold=self.threshold)
        simulation_overview = {'utilization_per_timeslot': [], 'violations_in_time_slot': [], 'number_of_relocations': [], 'total_utilization_percentage': []}
        
        # Initialize by assigning microservices to nodes (use starting requirements dictionary for cpu requirements)
        workloads = self._produce_initial_scheme(starting_req_dict)
        
        weights = [1, 1]  # Adjust based on requirements
        max_latency = 125
        node_index = self._get_node_index()
        cpuCosts = self._get_cpu_costs()
        latencies = self._get_node_latencies()
        node_resource_availability = self._get_resource_availability()
        # latencyMatrix = self.infrastructure.adj_matrix
        app_latencies = self._get_app_latencies()
        workloadAllocations = [[None for _ in range(6)] for _ in range(self.number_of_applications)]
        
        # sortedIndices = np.argsort(np.max(workloads, axis=1))
        # sortedWorkloads = workloads[sortedIndices, :]
        
        # print(workloads)
        # print(sortedWorkloads)
        
        # initial_allocation_scheme = functions.create_random_scheme(self)
        
        # Run the allocation using backtracking for each workload
        print("Initializing Allocation...")
        
        # for i in range(self.number_of_applications):
        #     node_resource_availability_copy = node_resource_availability.copy()  # Make a copy to not affect the original during backtracking
        #     if ra.backtrack_allocation(i, 0, node_resource_availability_copy, workloads, app_latencies, node_index, workloadAllocations, max_latency):
        #         print(f"Application {i} allocated successfully.")
        #     else:
        #         print(f"Failed to allocate all microservices for application {i} within constraints.")      
        # workload_Allocations_int = [[node_index[i] for i in row] for row in workloadAllocations]
        # for i in range(self.number_of_applications):
        #     for j in range(6):
        #         print(f"Final allocation: Application {i}, Microservice {j} is assigned to {workloadAllocations[i][j]}")
        
        for i in range(self.number_of_applications):
            for j in range(6):
                suitableNodes = ra.find_suitable_nodes(node_resource_availability, workloads[i, j])
                if suitableNodes:
                    min_cost, min_cost_node = ra.min_weighted_cost(suitableNodes, cpuCosts, latencies, weights, node_index)
                    node_resource_availability = ra.allocate_resources(node_resource_availability, min_cost_node, workloads[i, j])
                    workloadAllocations[i][j] = min_cost_node
                else:
                    print(f"No suitable node found for Workload {i}, Function {j}.")
        workload_Allocations_int = [[node_index[i] for i in row] for row in workloadAllocations]
        
        # for i in range(self.number_of_applications):
        #     for j in range(6):
        #         print(f"Initial allocation: Application {i}, Microservice {j} is assigned to {workloadAllocations[i][j]}")
        
        
        allocation_cost = self._assign_scheme_to_nodes(assignment_scheme=workload_Allocations_int, prediction_scheme=workloads)
        
        if not allocation_cost:
            print("Invalid Assignment Scheme")
            return simulation_overview
        
        # Check violations, cost and relocations required
        
        simulation_overview['utilization_per_timeslot'].append(self._check_utilization())
        simulation_overview['violations_in_time_slot'].append(0)
        simulation_overview['number_of_relocations'].append(0)
        simulation_overview['total_utilization_percentage'].append(self._calculate_utilization_percentage())

        historical_relocations = {}
        violation_counter = 0
        number_of_relocations = 0
        relocations = 0
        violations_stack = []
        total_cost = allocation_cost

        # Make predictions for each microservice and run RA algorithm / RUN FOR EVERY TICK (5 minutes for alibaba)
        while (period < self.simulation_duration):
            period += 1
            if period % self.timeslot_duration == 0:
                timeslot += 1
                
                self._clear_nodes()
                workloadAllocations = [[None for _ in range(6)] for _ in range(self.number_of_applications)]
                node_resource_availability = self._get_resource_availability()

                print(f"Initializing Allocation for timeslot #{timeslot}.")
                
                workloads = self._produce_prediction_scheme(prediction_model, timeslot)
                
                for i in range(self.number_of_applications):
                    for j in range(6):
                        suitableNodes = ra.find_suitable_nodes(node_resource_availability, workloads[i, j])
                        if suitableNodes:
                            min_cost, min_cost_node = ra.min_weighted_cost(suitableNodes, cpuCosts, latencies, weights, node_index)
                            node_resource_availability = ra.allocate_resources(node_resource_availability, min_cost_node, workloads[i, j])
                            workloadAllocations[i][j] = min_cost_node
                        else:
                            print(f"No suitable node found for Workload {i}, Function {j}.")
                workload_Allocations_int = [[node_index[i] for i in row] for row in workloadAllocations]

                allocation_cost = self._assign_scheme_to_nodes(assignment_scheme=workload_Allocations_int, prediction_scheme=workloads)

                if not allocation_cost:
                    print("Invalid Assignment Scheme")
                    return simulation_overview

                print(f"Total microservice relocations due to dynamic changes in timeslot #{timeslot}: {number_of_relocations}")

                simulation_overview['utilization_per_timeslot'].append(self._check_utilization())
                simulation_overview['violations_in_time_slot'].append(violation_counter)
                simulation_overview['number_of_relocations'].append(number_of_relocations)
                simulation_overview['total_utilization_percentage'].append(self._calculate_utilization_percentage())
                
                total_cost += allocation_cost
                historical_relocations = {}
                violation_counter = 0
                number_of_relocations = 0
   

            # Detect instances of underutilization
            violations_stack = []
            for node_id in range(len(self.infrastructure.nodes_list)):
                node_usage = 0
                for app_id, ms_id in self.infrastructure.nodes_list[node_id].ms_stack:
                    node_usage += self.applications_stack[app_id].ms_stack[ms_id].usage_time_series[period]
                if node_usage > self.infrastructure.nodes_list[node_id].total_cpu_capacity:
                    violation_counter += 1
                    if node_id not in violations_stack:
                        print(f"Detected insufficient resources for Node #{node_id} in Timeslot #{timeslot}")
                        violations_stack.append(node_id)

            # Produce dynamic changes for microservices in nodes with violations
            dynamicChanges = []
            for node_id in violations_stack:
                for ms in self.infrastructure.nodes_list[node_id].ms_stack:
                    dynamicChanges.append((ms[0], ms[1], self.applications_stack[ms[0]].ms_stack[ms[1]].usage_time_series[period]))
            
            # print(dynamicChanges)

            # Run relocation algorithm for each node in violations_stack
            node_resource_availability = self._get_resource_availability()
            relocations = ra.handle_dynamic_changes_with_latency(node_resource_availability, workloads, cpuCosts, latencies, self.weights, node_index, app_latencies, workloadAllocations, dynamicChanges, historical_relocations, max_latency)
            
            allocation_cost = 0
            for app_id, ms_id, _ in dynamicChanges:
                allocation_cost += self.infrastructure.nodes_list[node_index[workloadAllocations[app_id][ms_id]]].emergency_allocation_cost
            
            total_cost += allocation_cost
            
            self._clear_nodes()
            if not self._assign_scheme_to_nodes(assignment_scheme=workload_Allocations_int, prediction_scheme=workloads):
                print("Invalid Assignment Scheme")
                return simulation_overview

            # Calculate new number_of_relocations
            number_of_relocations += relocations

        # repeat

        return simulation_overview, total_cost
