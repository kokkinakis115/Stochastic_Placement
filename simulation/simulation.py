import numpy as np
from collections import deque
from infrastructure.infrastructure import Infrastucture
from application.application import Application
from models.predictor import BSM_predictor
import functions
import networkx as nx

class Simulation:

    def __init__(self, simulation_duration, timeslot_duration, lookforward_duration, lookback_duration, threshold, number_of_applications, number_of_edge_nodes, number_of_fog_nodes, number_of_src_nodes):

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

    def _initialize_simulation(self, workloads, vol_dict, network = None):

        if network == None:
            total_nodes = self.number_of_edge_nodes + self.number_of_fog_nodes + 1
            
            infrastructure = Infrastucture(total_nodes, self.number_of_edge_nodes, self.number_of_fog_nodes, self.number_of_src_nodes)
            infrastructure._create_adj_matrix([])
            infrastructure._create_infrastructure()
            self.infrastructure = infrastructure
        else:
            total_nodes = network.number_of_nodes()
            # TODO
            # infrastructure = Infrastucture(total_nodes, self.number_of_edge_nodes, self.number_of_fog_nodes)
            # infrastructure._create_adj_matrix(nx.adjacency_matrix(network, weight='delay'))
            # infrastructure._create_infrastructure()
    
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
