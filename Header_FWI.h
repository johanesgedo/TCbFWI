
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <math_constants.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <memory>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cerrno>
#include "helper_cuda.h"
#include "helper_string.h"

using namespace nvcuda::wmma;

/// #include "safe_memory.cu"

/// Lame's parameters for Elastic coefficients.
#define lambda_coef  1.064
#define mu_coef     0.821
#define rho_coef    2.73

#define TILE_DIM 8                  /// Tile size for shared memory
#define BLOCK_SIZE (TILE_DIM * TILE_DIM * TILE_DIM) /// Large block size
#define TILE_DIM_TC 16              /// Optimal tile size for Tensor Core
#define BLOCK_SIZE_2D (TILE_DIM_TC * TILE_DIM_TC)   /// Medium block size
#define dumping_factor  1e-3f       /// Dumping factor for adjoint wavefield
#define sparsity_threshold 1e-4     /// Adaptive sparsification threshold
#define maximum_iteration 3         /// Maximum Iteration for LSQR

/*=================================================================================
  The tile (sub-matrix) size used in the Warp Matrix Multiply-Accumulate (WMMA) API
  in CUDA to take advantage of Tensor Cores
===================================================================================*/
#define WMMA_M  16  /// Number of rows in the output matrix tile
#define WMMA_N  16  /// Number of columns in the output matrix tile
#define WMMA_K  16  /// Number of elements used for the dot-product operation in one WMMA operation

/// Variables for learning rate
#define ALPHA 0.001f            /// Learning rate default
#define BETA1 0.9f              /// Momentum decay
#define BETA2 0.999f            /// RMSprop decay
#define EPSILON 1e-8f           /// Stabilizer
#define CONV_THRESHOLD  1e-6f   /// Convergence threshold

/// Assumptions for the Munchausen_DQN_Update kernel: number of actions no more than 32 (for softmax local buffer)
#define MAX_ACTIONS 32
#define MAX_ENSEMBLE 16

/// Defines the embedding dimensions and the number of actions (Feudal RL Kernel)
#define STATE_EMBED_DIM 128
#define GOAL_EMBED_DIM 64
#define NUM_ACTIONS 10

/// Hyperparameter GrandNorm
#define MU 0.5f /// 0.12 - 0.75

/// Tunable parameters
constexpr int THREADS = 256;        /// threads per block
constexpr int VECTOR_WIDTH = 4;     /// elements per thread (float4)

/// Global parameters for Fused_Eligibility kernel
__constant__ float discount = 0.99f;
__constant__ float lambda_decay = 0.9f;

/// Declare the second differentiation matrix as constant memory (must be initialized from the host) for the PDE_Residual_WMMA_3D kernel.
__constant__ half d2T[WMMA_M * WMMA_K];  /// For ∂²/∂t²
__constant__ half d2X[WMMA_M * WMMA_K];  /// For ∂²/∂x²
__constant__ half d2Y[WMMA_M * WMMA_K];  /// For ∂²/∂y²
__constant__ half d2Z[WMMA_M * WMMA_K];  /// For ∂²/∂z²








