# PGM

This repository contains a Python script that implements functions related to Markov networks and junction trees. The script includes various utility functions and experiments for estimating the partition function of a Markov network.

# Function 
### getCutset(jt, threshold)

This function takes a junction tree object (`jt`) and a threshold value as input. It removes variables from the junction tree such that the size of the largest bag is less than or equal to the threshold. It returns the set of variables removed and the modified junction tree.

### computePartitionFunctionWithEvidence(jt, model, evidence)

This function computes the partition function of a junction tree (`jt`) with evidence. It takes the junction tree object, the Markov network model, and an evidence dictionary as input. It returns the partition function value.

### ComputePartitionFunction(markovNetwork, w, N, distribution)

This function computes an estimate of the partition function of a Markov network. It takes the Markov network model, an integer `w` denoting the bound on the largest cluster of the junction tree, an integer `N` denoting the number of samples, and a `distribution` parameter specifying the type of sampling distribution to use. The function returns the estimated partition function value.

To run the script, execute it in your preferred development environment. The script prompts for input values such as the path to the input file, the bound on the largest cluster, the number of samples, and the distribution type.

## Experiments

The script includes placeholders for conducting experiments with different sampling distributions. Two functions, `ExperimentsDistributionQRB` and `ExperimentsDistributionQUniform`, are provided for implementing experiments using different distributions. You can modify these functions to conduct your experiments based on the requirements.
