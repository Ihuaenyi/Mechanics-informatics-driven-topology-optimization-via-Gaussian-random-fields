# Mechanics-informatics-driven-topology-optimization-via-Gaussian-random-fields
Computational framework for designing optimally informative mechanical test specimens using Gaussian random fields and Bayesian optimization. Maximizes stress state entropy to enable single-test material characterization. Features: topology optimization via GRFs, ICNN-based constitutive learning, and automated FE integration.

# Overview
Traditional material characterization requires multiple standardized tests, each targeting a distinct loading state. This framework reimagines material characterization as an information utilization problem. Instead of adapting specimen design to multiple simple tests, we design a single complex, informative specimen that maximizes stress state entropy - a quantitative measure of mechanical information content.
Key Achievement: Single uniaxial tension test replaces multiple conventional tests, achieving order-of-magnitude improvement in parameter identification accuracy (from 21.47% to 1.02% average error) compared to low-information specimens.

# Features
Stress State Entropy Maximization: Quantitative metric ensuring specimens activate specific stress states required for accurate characterization

Gaussian Random Field Topology Optimization: Karhunen-Loève expansion with eigenvalue-based design control

Bayesian Optimization: Tree-structured Parzen Estimator for efficient design space exploration

Physics-Informed Learning: Input Convex Neural Networks for constitutive law discovery

Multi-Material Compatibility: Works with orthotropic elasticity and anisotropic plasticity

