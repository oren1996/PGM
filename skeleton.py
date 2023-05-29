# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from pgmpy.inference import BeliefPropagation
import numpy as np
from pgmpy.readwrite import UAIReader
from pgmpy.factors import factor_product
import random
import time
import math

# imports from me
from pgmpy.models.MarkovNetwork import MarkovNetwork
import copy
from itertools import product

"""A helper function. You are free to use."""


def numberOfScopesPerVariable(scopes):
    # Initialize a dictionary to store the counts
    counts = {}
    # Iterate over each scope
    for scope in scopes:
        # Iterate over each variable in the scope
        for variable in scope:
            # Increment the count for the variable
            if variable in counts:
                counts[variable] += 1
            else:
                counts[variable] = 1
    # Sort the counts dictionary based on the frequency of variables in the scopes
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_counts


"""
You need to implement this function. It receives as input a junction tree object,
and a threshold value. It should return a set of variables such that in the 
junction tree that results from jt by removing these variables (from all bags), 
the size of the largest bag is threshold.
The heuristic used to remove RVs from the junction tree is: repeatedly remove the RV that appears in
the largest number of bags in the junction tree.
"""


def getCutset(jt, threshold):
    X = []
    largestCluster = findLargestCluster(jt.nodes)
    # Run until the largest cluster in the junction
    # Tree is smaller than the threshold w
    while largestCluster >= threshold:
        variablesAppearance = numberOfScopesPerVariable(jt.nodes)
        mostOccurVariable = max(variablesAppearance, key=lambda x: x[1])[0]
        X.append(mostOccurVariable)

        # edges to remove
        edges_to_remove, updated_edges = updateEdgesFromEdges(jt, mostOccurVariable)
        jt.remove_edges_from(edges_to_remove)
        jt.add_edges_from(updated_edges)

        # nodes to remove
        nodes_to_remove, updated_nodes = updateNodesFromNodes(jt, mostOccurVariable)
        jt.remove_nodes_from(nodes_to_remove)
        jt.add_nodes_from(updated_nodes)

        largestCluster = findLargestCluster(jt.nodes)
    return X, jt


def findLargestCluster(nodes):
    largestCluster = 0
    # Find the new largest cluster and
    # find the variable with the most appearance
    for cluster in nodes:
        if len(cluster) >= largestCluster:
            largestCluster = len(cluster)
    return largestCluster


def updateNodesFromNodes(jt, mostOccurVariable):
    nodes_to_remove = []
    updated_nodes = []
    for node in jt.nodes:
        if mostOccurVariable in node:
            updated_node = tuple(variable for variable in node if variable != mostOccurVariable)
            nodes_to_remove.append(node)
            updated_nodes.append(updated_node)
    return nodes_to_remove, updated_nodes


def updateEdgesFromEdges(jt, mostOccurVariable):
    edges_to_remove = []
    updated_edges = []
    for edge in jt.edges:
        flag = False
        for subedge in edge:
            if mostOccurVariable in subedge:
                flag = True
        if flag:
            updated_edge = tuple(
                tuple(variable for variable in factor if variable != mostOccurVariable) for factor in edge)
            edges_to_remove.append(edge)
            if set(updated_edge[0]).intersection(updated_edge[1]):
                updated_edges.append(updated_edge)
    return edges_to_remove, updated_edges


"""
You are provided with this function. It receives as input a junction tree object, the MarkovNetwork model,
and an evidence dictionary. It computes the partition function with this evidence.
"""


def computePartitionFunctionWithEvidence(jt, model, evidence):
    reducedFactors = []
    for factor in jt.factors:
        evidence_vars = []
        for var in factor.variables:
            if var in evidence:
                evidence_vars.append(var)
        if evidence_vars:
            reduce_vars = [(var, evidence[var]) for var in evidence_vars]
            new_factor = factor.reduce(reduce_vars, inplace=False)
            reducedFactors.append(new_factor)
        else:
            reducedFactors.append(factor.copy())

    totalfactor = factor_product(*[reducedFactors[i] for i in range(0, len(reducedFactors))])
    var_to_marg = (set(model.nodes()) - set(evidence.keys()))
    marg_prod = totalfactor.marginalize(var_to_marg, inplace=False)
    return marg_prod.values


"""This function implements the experiments where the sampling distribution is Q^{RB}"""


def ExperimentsDistributionQRB(path):
    pass


"""This function implements the experiments where the sampling distribution Q is uniform"""


def ExperimentsDistributionQUniform(path):
    pass


"""This function takes as input a markov network, a integer w and an integer N
w denotes the bound on the largest cluster of the junction tree
N denotes the number of samples
Returns Z/N (= an estimate of the partition function)
"""


def ComputePartitionFunction(markovNetwork, w, N, distribution):
    Z = 0
    T = MarkovNetwork.to_junction_tree(markovNetwork)
    jt_copy = copy.deepcopy(T)
    X, jt_copy = getCutset(jt_copy, w)

    # initializing Q and t_x
    Q = None
    t_x = 0
    print(X)

    if distribution == 'q1':
        # 1) Q is a uniform distribution
        Q = generateUniformDistribution(X)
        print(Q)

    elif distribution == 'q2':
        # 2) QRB distribution
        Q = generateRBDistribution(X, T)
        print(Q)

    for i in range(N):
        x, randomAssignment = GenerateSample(X, Q)
        part_x = computePartitionFunctionWithEvidence(jt_copy, markovNetwork, x)
        if Q[randomAssignment] != 0:
            t_x = part_x / Q[randomAssignment]
        else:
            t_x = t_x
        Z = Z + t_x

    return Z / N


def generateUniformDistribution(X):
    # Generate a uniform distribution over the variables
    # Assign equal probabilities to all possible assignments
    n = len(X)
    Q = {}
    for assignment in product([0, 1], repeat=n):
        Q[assignment] = 1 / (2 ** n)
    return Q


def GenerateSample(X, Q):
    # Extract the assignments and probabilities as separate lists
    Q_list, Q_probability = zip(*Q.items())

    # Choose an assignment based on the probabilities
    randomAssignment = random.choices(Q_list, weights=Q_probability)[0]
    x = {}
    i = 0
    for variable in X:
        x[variable] = randomAssignment[i]
        i += 1
    return x, tuple(randomAssignment)


def generateRBDistribution(X, firstJT):
    belief_propagation = BeliefPropagation(firstJT)
    query_result = belief_propagation.query(X)
    n = len(X)
    Q = {}
    for assignment in product([0, 1], repeat=n):
        currentAssignment = dict(zip(X, assignment))
        Q[assignment] = query_result.get_value(**currentAssignment)
    return Q


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = '/Users/orenelbazis/PycharmProjects/PGM_test/grid4x4.uai'
    reader = UAIReader(path)

    # Part 1 + Part 2
    markovNetwork = reader.get_model()
    w = int(input('Enter the bound on the largest cluster of the junction tree: '))
    N = int(input('the number of samples: '))
    distribution = str(input('Choose the distribution Q that you would like to use over X\n'
                             '- Enter q1 for an uniform distribution\n'
                             '- Enter q2 for the real probability (QRB): '))
    partitionFunction = ComputePartitionFunction(markovNetwork, w=w, N=N, distribution=distribution)
    print(partitionFunction)

    # print("grid4x4 Experiments:")
    # ExperimentsDistributionQRB(path)
    # ExperimentsDistributionQUniform(path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
