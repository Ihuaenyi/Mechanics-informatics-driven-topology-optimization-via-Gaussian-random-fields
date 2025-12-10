# Mechanics-informatics-driven-topology-optimization-via-Gaussian-random-fields
Computational framework for designing optimally informative mechanical test specimens using Gaussian random fields and Bayesian optimization. Maximizes stress state entropy to enable single-test material characterization. Features: topology optimization via GRFs, ICNN-based constitutive learning, and automated FE integration.

## Overview
Traditional material characterization requires multiple standardized tests, each targeting a distinct loading state. This framework reimagines material characterization as an information utilization problem. Instead of adapting specimen design to multiple simple tests, we design a single complex, informative specimen that maximizes stress state entropy - a quantitative measure of mechanical information content.
Key Achievement: Single uniaxial tension test replaces multiple conventional tests, achieving order-of-magnitude improvement in parameter identification accuracy (from 21.47% to 1.02% average error) compared to low-information specimens.

# Features
Stress State Entropy Maximization: Quantitative metric ensuring specimens activate specific stress states required for accurate characterization

Gaussian Random Field Topology Optimization: Karhunen-Loève expansion with eigenvalue-based design control

Bayesian Optimization: Tree-structured Parzen Estimator for efficient design space exploration

Physics-Informed Learning: Input Convex Neural Networks for constitutive law discovery

Multi-Material Compatibility: Works with orthotropic elasticity and anisotropic plasticity

## Requirements
Python 3.8+, ABAQUS 2020+ (or compatible FE solver), CUDA GPU (recommended)

# Methodology
Design Pipeline

GRF Generation: Karhunen-Loève expansion with eigenvalue control

Level-Set Thresholding: Binary geometry from continuous random fields

Morphological Regularization: Remove sharp features and stress concentrations

FE Analysis: Stress field computation via ABAQUS

Entropy Quantification: Stress state distribution analysis

Bayesian Optimization: Iterative eigenvalue refinement using TPE

## Key Innovation
Traditional density-based topology optimization relies on material interpolation schemes fundamentally incompatible with inelastic constitutive laws. Our GRF-based approach uses level-set thresholding to generate binary geometries without interpolation, enabling application to diverse material behaviors including anisotropic plasticity.
Applications

Orthotropic Linear Elasticity: Four independent elastic constants (C₁₁, C₂₂, C₁₂, C₆₆)
Anisotropic Plasticity: Hill48 yield criterion with six material constants (K, σ_y0, n, F, G, N)
Extensible Framework: Applicable to rate-dependent, asymmetric, and 3D constitutive models

## Citation
If you use this code in your research, please cite:
Paper: Mechanics informatics-driven topology optimization for test specimen design and single-test material characterization: A Gaussian random field approach 
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5800046

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Contact
Royal C. Ihuaenyi - r.ihuaenyi@northeastern.edu
Juner Zhu - j.zhu@northeastern.edu

Department of Mechanical and Industrial Engineering
Northeastern University
Boston, MA 02115, USA
Acknowledgments
This research was supported by the U.S. National Science Foundation through grant CMMI-2450006.
