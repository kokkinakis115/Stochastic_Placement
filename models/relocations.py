import json
import numpy as np

def find_suitable_nodes(node_resource_availability, cpu_req):
    return [node for node, available_cpu in node_resource_availability.items() if available_cpu >= cpu_req]

def free_resources(node_resource_availability, node, cpu_req):
    node_resource_availability[node] += cpu_req
    return node_resource_availability

def min_weighted_cost(suitable_nodes, cpu_costs, latencies, weights, node_index):
    costs = weights[0] * np.array([cpu_costs[node] for node in suitable_nodes]) + \
            weights[1] * np.array([latencies[node_index[node]] for node in suitable_nodes])
    min_cost_index = np.argmin(costs)
    min_cost_node = suitable_nodes[min_cost_index]
    return costs[min_cost_index], min_cost_node

def allocate_resources(node_resource_availability, node, cpu_req):
    node_resource_availability[node] -= cpu_req
    return node_resource_availability

def min_weighted_cost_and_relocations(suitable_nodes, cpu_costs, latencies, weights, node_index, historical_relocations):
    # New objective: costs, latency, and relocations
    costs = weights['cost'] * np.array([cpu_costs[node] for node in suitable_nodes]) + \
            weights['latency'] * np.array([latencies[node_index[node]] for node in suitable_nodes]) + \
            weights['relocation'] * np.array([historical_relocations.get(node, 0) for node in suitable_nodes])
    min_cost_index = np.argmin(costs)
    min_cost_node = suitable_nodes[min_cost_index]
    return min_cost_node

def satisfies_latency_constraints(workload_index, function_index, node, allocations, workload_latencies, node_index, max_latency):
    cumulative_latency = 0
    for i in range(function_index):
        if allocations[i] is not None:
            cumulative_latency += workload_latencies[workload_index][i, function_index] + workload_latencies[workload_index][function_index, i]
    return cumulative_latency <= max_latency


def can_allocate(node_resource_availability, node, cpu_req):
    """Check if a node has enough resources to allocate a microservice."""
    return node_resource_availability[node] >= cpu_req

def satisfies_latency_constraints2(workload_index, function_index, node, allocations, workload_latencies, node_index, max_latency):
    cumulative_latency = 0
    allocated_node_index = node_index[node]  # Get the index of the node in the latency matrix

    for i in range(function_index):
        if allocations[i] is not None:
            allocated_node_i_index = node_index[allocations[i]]  # Index of the node allocated for the ith function
            cumulative_latency += workload_latencies[workload_index][allocated_node_i_index, allocated_node_index] + workload_latencies[workload_index][allocated_node_index, allocated_node_i_index]

    return cumulative_latency <= max_latency



def backtrack_allocation(workload_index, function_index, node_resource_availability, workloads, workload_latencies, node_index, allocations, max_latency):
    """
    Attempts to allocate all functions within a workload, backtracking if necessary.
    
    :param workload_index: The index of the current workload.
    :param function_index: The index of the current function within the workload to allocate.
    :param node_resource_availability: A dictionary tracking the available CPU resources for each node.
    :param workloads: A list of lists, where each sublist represents the CPU requirements of the functions in a workload.
    :param workload_latencies: A list of matrices, each matrix representing the latency constraints within a workload.
    :param node_index: A dictionary mapping node names to their indices in the latency matrices.
    :param allocations: A list of lists, where each sublist represents the current allocations for a workload.
    :param max_latency: The maximum allowed cumulative latency between functions in a workload.
    :return: True if allocation is successful for all functions, False otherwise.
    """
    if function_index == len(workloads[workload_index]):
        # Successfully allocated all functions in this workload
        return True

    for node in node_resource_availability.keys():
        # Check if this node can accommodate the current function's CPU requirement
        if can_allocate(node_resource_availability, node, workloads[workload_index][function_index]):
            # Temporarily allocate the current function to this node
            allocations[workload_index][function_index] = node
            temp_resources = node_resource_availability[node]
            node_resource_availability[node] -= workloads[workload_index][function_index]

            # Check if this allocation satisfies the latency constraints
            if satisfies_latency_constraints(workload_index, function_index, node, allocations[workload_index], workload_latencies, node_index, max_latency):
                # Recursively attempt to allocate the next function
                if backtrack_allocation(workload_index, function_index + 1, node_resource_availability, workloads, workload_latencies, node_index, allocations, max_latency):
                    return True  # Found a valid allocation for all subsequent functions

            # Backtrack: undo the current allocation and try the next node
            allocations[workload_index][function_index] = None
            node_resource_availability[node] = temp_resources

    return False  # Unable to find a valid allocation for this function within constraints


def handle_dynamic_changes_with_latency(node_resource_availability, workloads, cpuCosts, latencies, weights, node_index, workload_latencies, workloadAllocations, dynamicChanges, historical_relocations, max_latency):
    relocations = 0
    for change in dynamicChanges:
        time_slot, func_index, new_cpu_req = change
        current_allocation = workloadAllocations[time_slot]
        
        # Release current resources if allocated
        if current_allocation[func_index] is not None:
            node_resource_availability = free_resources(node_resource_availability, current_allocation[func_index], workloads[time_slot][func_index])
            current_allocation[func_index] = None
        
        # Find a new allocation that satisfies both CPU and latency requirements
        for node in find_suitable_nodes(node_resource_availability, new_cpu_req):
            # Temporarily allocate to check latency constraints
            current_allocation[func_index] = node
            if satisfies_latency_constraints(time_slot, func_index, node, current_allocation, workload_latencies, node_index, max_latency):
                # Commit the allocation if it satisfies latency constraints
                node_resource_availability = allocate_resources(node_resource_availability, node, new_cpu_req)
                workloadAllocations[time_slot][func_index] = node
                relocations += 1
                break
            else:
                # Undo temporary allocation if latency constraints are not satisfied
                current_allocation[func_index] = None
        
        if current_allocation[func_index] is None:
            # Failed to reallocate due to constraints; consider handling this scenario
            print(f"Failed to reallocate Function {func_index} in Workload {time_slot} within CPU and latency constraints.")
    
    return relocations


def handle_dynamic_changes(node_resource_availability, sortedWorkloads, cpuCosts, latencies, weights, node_index, latencyMatrix, workloadAllocations, dynamicChanges):
    relocations = 0
    for change in dynamicChanges:
        time_slot, func_index, new_cpu_req = change
        
        # Your existing logic to handle each change
        allocated_node = workloadAllocations[time_slot][func_index]
        if allocated_node and node_resource_availability[allocated_node] + sortedWorkloads[time_slot][func_index] < new_cpu_req:
            relocations += 1
            node_resource_availability = free_resources(node_resource_availability, allocated_node, sortedWorkloads[time_slot][func_index])
            suitableNodes = find_suitable_nodes(node_resource_availability, new_cpu_req)
            if suitableNodes:
                min_cost, min_cost_node = min_weighted_cost(suitableNodes, cpuCosts, latencies, weights, node_index)
                node_resource_availability = allocate_resources(node_resource_availability, min_cost_node, new_cpu_req)
                workloadAllocations[time_slot][func_index] = min_cost_node
            else:
                print(f"No suitable node found for function {func_index} after requirement change. Consider scaling resources.")
        elif allocated_node:
            # Adjust resource availability for new requirement
            additional_req = new_cpu_req - sortedWorkloads[time_slot][func_index]
            node_resource_availability = allocate_resources(node_resource_availability, allocated_node, additional_req)
        
        # Update the workload requirement
        sortedWorkloads[time_slot][func_index] = new_cpu_req

    return relocations



def handle_dynamic_changes_with_latency(node_resource_availability, workloads, cpuCosts, latencies, weights, node_index, workload_latencies, workloadAllocations, dynamicChanges, historical_relocations, max_latency):
    relocations = 0
    for change in dynamicChanges:
        time_slot, func_index, new_cpu_req = change
        current_allocation = workloadAllocations[time_slot]
        
        # Release current resources if allocated
        if current_allocation[func_index] is not None:
            node_resource_availability = free_resources(node_resource_availability, current_allocation[func_index], workloads[time_slot][func_index])
            current_allocation[func_index] = None
        
        # Find a new allocation that satisfies both CPU and latency requirements
        for node in find_suitable_nodes(node_resource_availability, new_cpu_req):
            # Temporarily allocate to check latency constraints
            current_allocation[func_index] = node
            if satisfies_latency_constraints(time_slot, func_index, node, current_allocation, workload_latencies, node_index, max_latency):
                # Commit the allocation if it satisfies latency constraints
                node_resource_availability = allocate_resources(node_resource_availability, node, new_cpu_req)
                workloadAllocations[time_slot][func_index] = node
                relocations += 1
                break
            else:
                # Undo temporary allocation if latency constraints are not satisfied
                current_allocation[func_index] = None
        
        if current_allocation[func_index] is None:
            # Failed to reallocate due to constraints; consider handling this scenario
            print(f"Failed to reallocate Function {func_index} in Workload {time_slot} within CPU and latency constraints.")
    
    return relocations

def handle_dynamic_changes_with_optimization(node_resource_availability, workloads, cpuCosts, latencies, weights, node_index, workload_latencies, workloadAllocations, dynamicChanges, historical_relocations, max_latency):
    relocations = 0

    for change in dynamicChanges:
        time_slot, func_index, new_cpu_req = change
        current_allocations = workloadAllocations[time_slot]
        
        # If the function is currently allocated, first try to adjust its resources without moving it
        if current_allocations[func_index] is not None:
            allocated_node = current_allocations[func_index]
            if node_resource_availability[allocated_node] + workloads[time_slot][func_index] >= new_cpu_req:
                node_resource_availability[allocated_node] += workloads[time_slot][func_index] - new_cpu_req
                workloads[time_slot][func_index] = new_cpu_req
                continue

        # If adjustment is not possible, or it wasn't allocated, look for a new node
        for node, available_cpu in node_resource_availability.items():
            if available_cpu >= new_cpu_req:
                # Check if reallocating to this node satisfies the latency constraints
                current_allocations[func_index] = node  # Temp allocation for checking
                if satisfies_latency_constraints2(time_slot, func_index, node, current_allocations, workload_latencies, node_index, max_latency):
                    # If the node satisfies the constraints, allocate it
                    if current_allocations[func_index] != node:  # If it was allocated somewhere else
                        node_resource_availability[current_allocations[func_index]] += workloads[time_slot][func_index]  # Free the old node
                    node_resource_availability[node] -= new_cpu_req
                    workloads[time_slot][func_index] = new_cpu_req
                    current_allocations[func_index] = node  # Confirm allocation
                    relocations += 1
                    break  # Stop looking for nodes
                else:
                    current_allocations[func_index] = None  # Undo temp allocation

        if current_allocations[func_index] is None:  # If no suitable node was found
            print(f"Could not find a suitable node for Function {func_index} in Workload {time_slot}.")

    return relocations





# # Json input with telemetry data
# json_input = '''
# {
#   "execution_plugin": "MarsalComputeNode",
#   "parameters": {
#     "telemetry": [
#       {"cpu": "15.54999999795109", "ip": "23.23.23.7", "latency": "1.112", "monetarycost": "1.5", "name": "py-k8s-master"},
#       {"cpu": "6.8500000005587935", "ip": "23.23.23.5", "latency": "1.08", "monetarycost": "1.5", "name": "py-k8s-worker-1"},
#       {"cpu": "6.399999998975545", "ip": "23.23.23.3", "latency": "1.08", "monetarycost": "1.5", "name": "py-k8s-worker-2"}
#     ]
#   }
# }
# '''

# data = json.loads(json_input)
# telemetry_data = data["parameters"]["telemetry"]

# # Process and prepare the telemetry data
# cpuCosts = {node["name"]: float(node["cpu"]) for node in telemetry_data}
# latencies = {i: float(node["latency"]) for i, node in enumerate(telemetry_data)}
# weights = [1, 1]  # Adjust based on requirements
# node_index = {node["name"]: i for i, node in enumerate(telemetry_data)}
# node_resource_availability = {node["name"]: float(node["cpu"]) for node in telemetry_data}


# # Initialize latency matrix and workloads (example data)
# np.random.seed(0)
# latencyMatrix = np.random.rand(len(telemetry_data), len(telemetry_data))
# np.fill_diagonal(latencyMatrix, 0)



# # Example workload requirements (CPU)
# # Let's assume each workload is represented by its CPU requirement
# # Introduce more workloads with their CPU requirements
# numWorkloads = 1  # Now, we have 3 workloads
# numFunctions = 2
# # Adjusted workloads to reflect different CPU requirements for each
# # workloads = np.array([
# #     [2, 3, 8, 2, 1],  # Workload 0
# #     [1, 2, 2, 8, 3],  # Workload 1
# #     [3, 1, 10, 3, 1]   # Workload 2
# # ])




# # # These are NxN matrices where N is the number of functions in a workload, and the value at [i, j] represents the latency between function i and j
# # workload_latencies = [
# #     np.array([[0, 1.2, 0.9, 1.5, 1.1], [1.2, 0, 1.0, 1.4, 1.2], [0.9, 1.0, 0, 1.3, 1.3], [1.5, 1.4, 1.3, 0, 0.8], [1.1, 1.2, 1.3, 0.8, 0]]),  # Workload 0
# #     np.array([[0, 1.1, 1.4, 1.2, 0.9], [1.1, 0, 1.3, 1.1, 1.0], [1.4, 1.3, 0, 1.5, 1.1], [1.2, 1.1, 1.5, 0, 1.2], [0.9, 1.0, 1.1, 1.2, 0]]),  # Workload 1
# #     np.array([[0, 1.0, 1.2, 1.3, 1.4], [1.0, 0, 1.1, 1.4, 1.5], [1.2, 1.1, 0, 1.0, 1.3], [1.3, 1.4, 1.0, 0, 1.1], [1.4, 1.5, 1.3, 1.1, 0]])   # Workload 2
# # ]

# workloads = np.array([
#     [11, 6]   # Workload 1
# ])




# # These are NxN matrices where N is the number of functions in a workload, and the value at [i, j] represents the latency between function i and j
# workload_latencies = [np.array([[0, 1.2], [1.2, 0]])]



# # Sort workloads (if needed, based on your specific criteria)

# # Initial Allocation Process (Unchanged)

# # Adapt the dynamic changes to affect different workloads
# dynamicChanges = [
#     (0, 0, 5), # Workload 0, Function 2 requires more CPU
# ]

# # Initialize variables for storing results
# workloadAllocations = np.zeros((numWorkloads, numFunctions), dtype=object)
# workloadLatencies = np.zeros(numWorkloads)

# # Sort the workloads in ascending order based on their maximum latency constraint (simplified here for example purposes)
# sortedIndices = np.argsort(np.max(workloads, axis=1))
# sortedWorkloads = workloads[sortedIndices, :]

# # Initial Allocation Process

# workloadAllocations = [[None for _ in range(numFunctions)] for _ in range(numWorkloads)]

# max_latency = 100

# # Run the allocation using backtracking for each workload
# for i in range(numWorkloads):
#     node_resource_availability_copy = node_resource_availability.copy()  # Make a copy to not affect the original during backtracking
#     if backtrack_allocation(i, 0, node_resource_availability_copy, workloads, workload_latencies, node_index, workloadAllocations, max_latency):
#         print(f"Workload {i} allocated successfully.")
#     else:
#         print(f"Failed to allocate all microservices for workload {i} within constraints.")

# # Output the final assignment
# for i in range(numWorkloads):
#     for j in range(numFunctions):
#         print(f"Final allocation: Workload {i}, Function {j} is assigned to {workloadAllocations[i][j]}")




# #for i in range(numWorkloads):
# #    for j in range(numFunctions):
# #        suitableNodes = find_suitable_nodes(node_resource_availability, sortedWorkloads[i, j])
# #        if suitableNodes:
# #            min_cost, min_cost_node = min_weighted_cost(suitableNodes, cpuCosts, latencies, weights, node_index)
# #            node_resource_availability = allocate_resources(node_resource_availability, min_cost_node, sortedWorkloads[i, j])
# #            workloadAllocations[i, j] = min_cost_node
# #        else:
# #            print(f"No suitable node found for Workload {i}, Function {j}.")
# #            break  # Break if we cannot find a node for a function
# #
# # Simulate dynamic changes and calculate relocations

# historical_relocations = {}
# # Weights for cost, latency, and relocations
# weights = {'cost': 1, 'latency': 1, 'relocation': 2}  # Example: prioritize minimizing relocations higher


# #relocations = handle_dynamic_changes(node_resource_availability, sortedWorkloads, cpuCosts, latencies, weights, node_index, latencyMatrix, workloadAllocations)
# #print(f"Total microservice relocations due to dynamic changes: {relocations}")

# relocations = handle_dynamic_changes_with_optimization(node_resource_availability, workloads, cpuCosts, latencies, weights, node_index, latencyMatrix, workloadAllocations, dynamicChanges, historical_relocations, max_latency)

# print(f"Total microservice relocations due to dynamic changes: {relocations}")
# for i in range(numWorkloads):
#     for j in range(numFunctions):
#         print(f"Final allocation: Workload {i}, Function {j} is assigned to {workloadAllocations[i][j]}")


