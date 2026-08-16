# TCbFWI

**TensorCore-based Full Waveform Inversion**


TCbFWI is an experimental framework for Full Waveform Inversion (FWI) that explores the use of NVIDIA GPU computing and Tensor Core acceleration for computationally intensive seismic wave-propagation and inversion processes.

This repository is intended for research and experimental development in GPU-accelerated seismic inversion, numerical wave propagation, Tensor Core computing, and high-performance computational geophysics.

**DOI**: 

---

## Overview

TCbFWI combines several major computational paradigms:

1. **Full Waveform Inversion (FWI)** for seismic model reconstruction through iterative waveform matching.
2. **CUDA GPU Computing** for massively parallel seismic wave-propagation and inversion operations.
3. **NVIDIA Tensor Cores** for accelerating selected matrix-based numerical computations through the CUDA WMMA API.
4. **Finite-Difference Wave Propagation** for numerical simulation of seismic wavefields.
5. **Adjoint-State Method** for calculating the FWI gradient from forward and adjoint wavefields.
6. **Anisotropic Elastic Wave Propagation** for modeling more complex seismic wave behavior.
7. **PML Boundary Conditions** for reducing artificial reflections at computational boundaries.
8. **Adaptive Gradient Sparsification** for processing and controlling inversion gradients.
9. **2D and 3D GPU Kernels** for experimental implementation of FWI computations in different spatial dimensions.

The Tensor Core implementation is based on CUDA FP16 and WMMA operations, with `16 × 16 × 16` WMMA tiles defined in the framework's CUDA header.


---

## Key Features

- Tensor Core-based computational kernels using CUDA WMMA;
- GPU-accelerated 2D and 3D FWI operations;
- Finite-difference seismic wave propagation;
- Forward and adjoint wavefield computation;
- Anisotropic elastic P-wave modeling;
- Perfectly Matched Layer (PML) absorbing boundaries;
- Tensor Core-oriented misfit and residual calculations;
- GPU-based gradient computation;
- Adaptive sparsity-based gradient processing;
- Parallel residual-norm reduction;
- Model-update direction calculation;
- CUDA FP16 support for Tensor Core operations;
- CUDA runtime, cuBLAS, CURAND, and WMMA integration.


---

## Possible Extensions

The framework can be extended for:

- Multi-GPU Full Waveform Inversion;
- Large-scale 3D seismic inversion;
- Multi-source and simultaneous-source FWI;
- Frequency-domain or multi-frequency inversion strategies;
- Advanced Tensor Core kernel optimization;
- Mixed-precision FWI;
- GPU memory optimization and wavefield checkpointing;
- Distributed seismic inversion;
- Advanced gradient preconditioning;
- Performance benchmarking across NVIDIA GPU architectures;
- Integration with seismic imaging and inversion workflows.


---

## Project Status

TCbFWI is an **experimental research and prototype framework currently under development**.

It is currently best suited for:

- Concept validation;
- Experimental Tensor Core acceleration;
- CUDA kernel development;
- Numerical experiments in Full Waveform Inversion;
- GPU-accelerated seismic wave propagation;
- Research in high-performance computational geophysics;
- Investigation of Tensor Core-based seismic inversion.

> **Note:** TCbFWI is experimental software. Numerical accuracy, convergence, stability, and computational performance should be independently validated for each experimental configuration and GPU architecture before using the results for scientific or production purposes.


---

## License

TCbFWI is distributed under the terms and conditions specified in the `LICENSE` file included in this repository.


---

## Citation

If TCbFWI is used in academic research, please cite the corresponding software release or publication associated with this repository.

A citation entry can be added here after the repository is formally released through GitHub and Zenodo.

> **Citation**: 







