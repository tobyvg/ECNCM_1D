# Energy conserving neural closure models 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tobyvg.github.io/ECNCM_1D.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tobyvg.github.io/ECNCM_1D.jl/dev/)
[![Build Status](https://github.com/tobyvg/ECNCM_1D.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tobyvg/ECNCM_1D.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tobyvg/ECNCM_1D.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tobyvg/ECNCM_1D.jl)

## Abstract 

In turbulence modeling, we are concerned with finding closure models that represent the effect of the subgrid
scales on the resolved scales. Recent approaches gravitate towards machine learning techniques to construct
such models. However, the stability of machine-learned closure models and their abidance by physical
structure (e.g. symmetries, conservation laws) are still open problems. To tackle both issues, we take the
‘discretize first, filter next’ approach. In this approach we apply a spatial averaging filter to existing fine-grid
discretizations. The main novelty is that we introduce an additional set of equations which dynamically
model the energy of the subgrid scales. Having an estimate of the energy of the subgrid scales, we can
use the concept of energy conservation to derive stability. The subgrid energy containing variables are
determined via a data-driven technique. The closure model is used to model the interaction between the
filtered quantities and the subgrid energy. Therefore the total energy should be conserved. Abiding by
this conservation law yields guaranteed stability of the system. In this work, we propose a novel skew-
symmetric convolutional neural network architecture that satisfies this law. The result is that stability is
guaranteed, independent of the weights and biases of the network. Importantly, as our framework allows
for energy exchange between resolved and subgrid scales it can model backscatter. To model dissipative
systems (e.g. viscous flows), the framework is extended with a diffusive component. The introduced neural
network architecture is constructed such that it also satisfies momentum conservation. We apply the new
methodology to both the viscous Burgers’ equation and the Korteweg-De Vries equation in 1D. The novel
architecture displays superior stability properties when compared to a vanilla convolutional neural network.

## Required Julia (v1.9.3) packages

- Plots
- JLD
- LaTeXStrings
- ProgressBars
- LinearAlgebra
- SparseArrays
- Random
- Distributions
- Flux
- Zygote

The Tutorial.ipynb can be run using Jupyter Notebook. Corresponding paper can be found at https://arxiv.org/abs/2301.13770.
