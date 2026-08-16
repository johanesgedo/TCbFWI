# TensorCore-based Full Waveform Inversion
# Copyright (C) 2026 Johanes Gedo Sea
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


#include "Header_FWI.h"

extern "C" {
__global__ void Convert_float2_to_float(
    int nx,
    int nz,
    float2* input,
    float* output
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nx * nz) {
        output[2 * idx] = input[idx].x;
        output[2 * idx + 1] = input[idx].y;
    }
}
}


extern "C" {
__global__ Compute_SQRT_Half_3D(
    size_t nx,
    size_t ny,
    size_t nz,
    float* variable
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_elements = nx * ny * nz;
    if (idx < num_elements) {
        variable[idx] = sqrtf(variable[idx]);
    }
}
}


extern "C" {
__global__ void Init_RNG_States(
    size_t total_states,
    unsigned long seed,
    curandStatePhilox4_32_10_t* states
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_states) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}
}


__device__ __forceinline__ float Ricker_Wavelet(
    float dt,
    float f0
)
{
    float tmp = PI * f0 * (dt - 1.0f / f0);
    return (1.0f - 2.0f * tmp * tmp) * expf(-tmp * tmp);
}


__device__ __forceinline__ bool Is_Receiver_3D(
    int ix,
    int iy,
    int iz,
    int start_x,
    int start_y,
    int receiver_depth_z,
    int spacing_x,
    int spacing_y
)
{
    return (iz == receiver_depth_z) &&
            ((ix - start_x) % spacing_y == 0) &&
            ((iy - start_y) % spacing_y == 0);
}


__device__ __forceinline__ void Pressure_From_3D_Velocity(
    float* __restrict__ velocity,
    size_t nx,
    size_t ny,
    size_t nz,
    float dx,
    float dy,
    float dz,
    float dt,
    float* __restrict__ pressure
)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    size_t iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix < 1 || ix >= nx-1 ||
        iy < 1 || iy >= ny-1 ||
        iz < 1 || iz >= nz-1) return;

    size_t N = nx * ny * nz;
    size_t idx = iz * (nx * ny) + iy * nx + ix;

    float* velocity_x = velocity;
    float* velocity_y = velocity + N;
    float* velocity_z = velocity + 2 * N;

    __shared__ float sh_vx[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];
    __shared__ float sh_vy[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];
    __shared__ float sh_vz[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];

    size_t lx = threadIdx.x + 1;
    size_t ly = threadIdx.y + 1;
    size_t lz = threadIdx.z + 1;

    sh_vx[lz][ly][lx] = velocity_x[idx];
    sh_vy[lz][ly][lx] = velocity_y[idx];
    sh_vz[lz][ly][lx] = velocity_z[idx];

    /// Halo X
    if (threadIdx.x == 0) {
        sh_vx[lz][ly][0] = velocity_x[idx - 1];
        sh_vy[lz][ly][0] = velocity_y[idx - 1];
        sh_vz[lz][ly][0] = velocity_z[idx - 1];
    }
    if (threadIdx.x == blockDIm.x - 1) {
        sh_vx[lz][ly][TILE_DIM + 1] = velocity_x[idx + 1];
        sh_vy[lz][ly][TILE_DIM + 1] = velocity_y[idx + 1];
        sh_vz[lz][ly][TILE_DIM + 1] = velocity_z[idx + 1];
    }

    /// Halo Y
    if (threadIdx.y == 0) {
        sh_vx[lz][0][ix] = velocity_x[idx - nx];
        sh_vy[lz][0][lx] = velocity_y[idx - nx];
        sh_vz[lz][0][lx] = velocity_z[idx - nx];
    }
    if (threadIdx.y == blockDim.y - 1) {
        sh_vx[lz][TILE_DIM + 1][lx] = velocity_x[idx + nx];
        sh_vy[lz][TILE_DIM + 1][lx] = velocity_y[idx + nx];
        sh_vz[lz][TILE_DIM + 1][lx] = velocity_z[idx + nx];
    }

    /// Halo Z
    if (threadIdx.z == 0) {
        sh_vx[0][ly][lx] = velocity_x[idx - nx * ny];
        sh_vy[0][ly][lx] = velocity_y[idx - nx * ny];
        sh_vz[0][ly][lx] = velocity_z[idx - nx * ny];
    }
    if (threadIdx.z == blockDim.z - 1) {
        sh_vx[TILE_DIM + 1][ly][lx] = velocity_x[idx + nx * ny];
        sh_vy[TILE_DIM + 1][ly][lx] = velocity_y[idx + nx * ny];
        sh_vz[TILE_DIM + 1][ly][lx] = velocity_z[idx + nx * ny];
    }

    __syncthreads();

    float dpdx = (sh_vx[lz][ly][lx+1] - sh_vx[lz][ly][lx-1]) / (2.0f * dx);
    float dpdy = (sh_vy[lz][ly+1][lx] - sh_vy[lz][ly-1][lx]) / (2.0f * dy);
    float dpdz = (sh_vz[lz+1][ly][lx] - sh_vz[lz-1][ly][lx]) / (2.0f * dz);

    pressure[idx] = -dt * (dpdx + dpdy + dpdz);
}


__device__ __forceinline__ void Velocity_3D_From_Pressure(
    float* __restrict__ pressure,
    size_t nx,
    size_t ny,
    size_t nz,
    float dx,
    float dy,
    float dz,
    float dt,
    float* __restrict__ velocity_x_axis,
    float* __restrict__ velocity_y_axis,
    float* __restrict__ velocity_z_axis
)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    size_t iz = blockIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    size_t idx = iz * (nx * ny) + iy * nx + ix;

    __shared__ half sh_p[WMMA_M + 2][WMMA_N + 2];
    size_t lx = threadIdx.x + 1;
    size_t ly = threadIdx.y + 1;

    sh_p[ly][lx] = pressure[idx];

    /// Halo X
    if (threadIdx.x == 0 && ix > 0) {
        sh_p[ly][0] = pressure[idx - 1];
    }
    if (threadIdx.x == WMMA_M - 1 && ix + 1 < nx) {
        sh_p[ly][WMMA_M + 1] = pressure[idx + 1];
    }

    /// Halo Y
    if (threadIdx.y == 0 && iy > 0) {
        sh_p[0][lx] = pressure[idx - nx];
    }
    if (threadIdx.y == WMMA_N - 1 && iy + 1 < ny) {
        sh_p[WMMA_N + 1][lx] = pressure[idx + nx];
    }

    __syncthreads();

    float p_left = __half2float(sh_p[ly][lx - 1]);
    float p_right = __half2float(sh_p[ly][lx + 1]);
    float dpdx = (p_right - p_left) / (2.0f * dx);

    float p_back = __half2float(sh_p[ly - 1][lx]);
    float p_front = __half2float(sh_p[ly + 1][lx]);
    float dpdy = (p_back - p_front) / (2.0f * dy);

    float p_down = __half2float(pressure[(iz-1) * (nx*ny) + iy * nx + ix]);
    float p_up = __half2float(pressure[(iz+1) * (nx*ny) + iy * nx + ix]);
    float dpdz = (p_down - p_up) / (2.0f * dz);

    /// Load vp into WMMA fragment for vector multiply
    /// Use WMMA on a 1x4 matrix to multiply four values at once (dpdx, dpdy, dpdz, 0)
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_dp;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_coef;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out;

    frag_dp.x[0] = dpdx;
    frag_dp.x[1] = dpdy;
    frag_dp.x[2] = dpdz;
    frag_dp.x[3] = 0.0f;

    /// vp coefficient fragment: diagonal of size 4
    float vp = (lambda_coef + 2.0f * mu_coef) / rho_coef;
    for (int i = 0; i < frag_coef.num_elements; ++i) {
        int r = i / WMMA_M;
        int c = i % WMMA_N;
        frag_coef.x[i] = (r == c && r < 3) ? vp : 0.0f;
    }

    fill_fragment(frag_out, 0.0f);
    mma_sync(frag_out, frag_dp, frag_coef, frag_out);
    __syncthreads();

    float vx_val = frag_out.x[0];
    float vy_val = frag_out.x[1];
    float vz_val = frag_out.x[2];

    velocity_x_axis[idx] = vx_val;
    velocity_y_axis[idx] = vy_val;
    velocity_z_axis[idx] = vz_val;
}


extern "C" {
__global__ void FFT_3D_From_Velocity_Model(
    float* __restrict__ velocity,
    size_t nx,
    size_t ny,
    size_t nz,
    float dx,
    float dy,
    float dz,
    float dt,
    float2* __restrict__ frequency
)
{
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t tz = threadIdx.z;
    size_t ix = blockIdx.x * blockDim.x + tx;
    size_t iy = blockIdx.y * blockDim.y + ty;
    size_t iz = blockIdx.z * blockDim.z + tz;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    __shared__ float2 tile[TILE_DIM][TILE_DIM][TILE_DIM];

    float2 val = make_float2(0.0f, 0.0f);
    if (ix < nx && iy < ny && iz < nz) {
        size_t global_idx = iz * nx * ny + iy * nx + ix;
        val.x = velocity[global_idx];
        val.y = 0.0f;
    }

    if (tx < TILE_DIM && ty < TILE_DIM && tz < TILE_DIM) {
        tile[tz][ty][tx] = val;
    }
    __syncthreads();

    /// 1D FFT along X
    for (int stride = 1; stride < TILE_DIM; stride <<= 1) {
        float2 temp = tile[tz][ty][tx];
        int pos = tx % (2 * stride);
        float angle = -2.0f * M_PI * pos / (2.0f * stride);
        float2 tw = make_float2(cosf(angle), sinf(angle));
        if (pos < stride) {
            float2 a = temp;
            if (tx + stride < TILE_DIM) {
                float2 b = tile[tz][ty][tx + stride];
                float2 t;
                t.x = b.x * tw.x - b.y * tw.y;
                t.y = b.x * tw.y + b.y * tw.x;
                tile[tz][ty][tx] = make_float2(a.x + t.x, a.y + t.y);
            }
        }
        __syncthreads();
    }

    /// 1D FFT along Y
    for (int stride = 1; stride < TILE_DIM; stride <<= 1) {
        float2 temp = tile[tz][ty][tx];
        int pos = ty % (2 * stride);
        float angle = -2.0f * M_PI * pos / (2.0f * stride);
        float2 tw = make_float2(cosf(angle), sinf(angle));
        if (pos < stride) {
            float2 a = temp;
            if (ty + stride < TILE_DIM) {
                float2 b = tile[tz][ty + stride][tx];
                float2 t;
                t.x = b.x * tw.x - b.y * tw.y;
                t.y = b.x * tw.y + b.y * tw.x;
                tile[tz][ty][tx] = make_float2(a.x + t.x, a.y + t.y);
            }
        }
        __syncthreads();
    }

    /// 1D FFT along Z
    for (int stride = 1; stride < TILE_DIM; stride <<= 1) {
        float2 temp = tile[tz][ty][tx];
        int pos = tz % (2 * stride);
        float angle = -2.0f * M_PI * pos / (2.0f * stride);
        float2 tw = make_float2(cosf(angle), sinf(angle));
        if (pos < stride) {
            float2 a = temp;
            if (tz + stride < TILE_DIM) {
                float2 b = tile[tz + stride][ty][tx];
                float2 t;
                t.x = b.x * tw.x - b.y * tw.y;
                t.y = b.x * tw.y + b.y * tw.x;
                tile[tz][ty][tx] = make_float2(a.x + t.x, a.y + t.y);
            }
        }
        __syncthreads();
    }

    if (ix < nx && iy < ny && iz < nz) {
        size_t global_idx = iz * nx * ny + iy * nx + ix;
        frequency[global_idx] = tile[tz][ty][tx];
    }
}
}


///===========================================================
/// Fast Fourier Transform from Velocity Model
///===========================================================
__device__ __forceinline__ unsigned Bit_Reversal(
    unsigned x,
    unsigned log2n
)
{
    unsigned y = 0;
    for (unsigned i = 0; i < log2n; ++i) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

__device__ __forceinline__ float2 Complex_Multiplication(
    const float2 &a,
    const float2 &b
)
{
    return make_float2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}


extern "C" {
__global__ void FDTD_3D_Forward_Propagation_PML_Anisotropy_TensorCore(
    float* __restrict__ velocity_rms,
    float* __restrict__ velocity_x,
    float* __restrict__ velocity_y,
    float* __restrict__ velocity_z,
    float* __restrict__ pressure,
    float2* __restrict__ frequency,
    float* min_velocity,
    size_t nx,
    size_t ny,
    size_t nz,
    float dx,
    float dy,
    float dz,
    float dt,
    float* __restrict__ data_output
)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    size_t iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    size_t idx = ix + iy * nx + iz * nx * ny;

    __shared__ __align__(128) float tile_p[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];
    __shared__ __align__(128) float tile_vx[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];
    __shared__ __align__(128) float tile_vy[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];
    __shared__ __align__(128) float tile_vz[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];
    __shared__ __align__(128) half tileData_shared[WMMA_M * WMMA_N];

    int max_distance = sqrtf(powf(nx*dx,2) + powf(ny*dy,2) + powf(nz*dz,2));

    /// Calculate maximum time
    /// min_velocity is calculated from reduction kernel
    int safety_factor = 1.2;
    int max_time = safety_factor * max_distance / *min_velocity;

    /// Taking real frequecy
    float real_frequency = frequency[idx].x;

    /// Grid points per wavelength
    int N_lambda_x = __float2half_rn(*min_velocity / (real_frequency * dx));
    int N_lambda_y = __float2half_rn(*min_velocity / (real_frequency * dy));
    int N_lambda_z = __float2half_rn(*min_velocity / (real_frequency * dz));

    /// Perfectly Matched Layer (PML)
    int PML_THICKNESS_X = ceil(N_lambda_x * (*min_velocity) / (real_frequency * dx));
    int PML_THICKNESS_Y = ceil(N_lambda_y * (*min_velocity) / (real_frequency * dy));
    int PML_THICKNESS_Z = ceil(N_lambda_z * (*min_velocity) / (real_frequency * dz));

    /// Initial position of the receiver horizontally (x,y)
    int start_x = PML_THICKNESS_X;
    int start_y = PML_THICKNESS_Y;

    /// Vertical position where the receiver is placed
    int receiver_depth = PML_THICKNESS_Z + 5;

    /// Distance between receivers in grid units
    int spacing_x = 2;
    int spacing_y = 2;

    /// Calculate maximum shots
    int max_shots_x = (nx - 2 * PML_THICKNESS_X) / spacing_x;
    int max_shots_y = (ny - 2 * PML_THICKNESS_Y) / spacing_y;
    int total_states = max_shots_x * max_shots_y;
    if (blockIdx.x >= total_shots) return;

    /// Depth offset to determine the depth of the wave source position
    int fixed_depth_offset = 0.5 * N_lambda_z;

    /// Simultaneous multi-shot simulations (anisotropic PML assumption)
    int sx = PML_THICKNESS_X + blockIdx.x * spacing_x;
    int sy = PML_THICKNESS_Y + blockIdx.y * spacing_y;
    int sz = PML_THICKNESS_Z + fixed_depth_offset;
    if (sx >= nx - PML_THICKNESS_X || sy >= ny - PML_THICKNESS_Y || sz >= nz - PML_THICKNESS_Z) return;

    int local_i = threadIdx.x + 1;
    int local_j = threadIdx.y + 1;
    int local_k = threadIdx.z + 1;

    float vel, p, vx, vy, vz;
    vel = velocity_rms[threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x];
    p = pressure[idx];
    vx = velocity_x[idx];
    vy = velocity_y[idx];
    vz = velocity_z[idx];

    float sigma_x = 0.0f;
    float sigma_y = 0.0f;
    float sigma_z = 0.0f;

    if (ix < PML_THICKNESS_X) {
        sigma_x = (float)(PML_THICKNESS_X - ix) / PML_THICKNESS_X;
    }
    else if (ix > nx - PML_THICKNESS_X) {
        sigma_x = (float)(ix - (nx - PML_THICKNESS_X)) / PML_THICKNESS_X;
    }
    if (iy < PML_THICKNESS_Y) {
        sigma_y = (float)(PML_THICKNESS_Y - iy) / PML_THICKNESS_Y;
    }
    else if (iy > ny - PML_THICKNESS_Y) {
        sigma_y = (float)(iy - (ny - PML_THICKNESS_Y)) / PML_THICKNESS_Y;
    }
    if (iz < PML_THICKNESS_Z) {
        sigma_z = (float)(PML_THICKNESS_Z - iz) / PML_THICKNESS_Z;
    }
    else if (iz > nz - PML_THICKNESS_Z) {
        sigma_z = (float)(iz - (nz - PML_THICKNESS_Z)) / PML_THICKNESS_Z;
    }

    float dumping = expf(-(sigma_x + sigma_y + sigma_z) * dt);

    for (int t = 0; t < max_time; t++) {
        tile_p[local_i][local_j][local_k] = p;
        tile_vx[local_i][local_j][local_k] = vx;
        tile_vz[local_i][local_j][local_k] = vz;

        /// Load halo in X
        if (threadIdx.x == 0 && ix > 0) {
            tile_p[local_i - 1][local_j][local_k] = pressure[idx - 1];
        }
        if (threadIdx.x == blockDim.x - 1 && ix < nx - 1) {
            tile_p[local_i + 1][local_j][local_k] = pressure[idx + 1];
        }

        /// Load halo in Y
        if (threadIdx.y == 0 && iy > 0) {
            tile_p[local_i][local_j - 1][local_k] = pressure[idx - nx];
        }
        if (threadIdx.y == blockDim.y - 1 && iy < ny - 1) {
            tile_p[local_i][local_j + 1][local_k] = pressure[idx + nx];
        }

        /// Load halo in Z
        if (threadIdx.z == 0 && iz > 0) {
            tile_p[local_i][local_j][local_k-1] = pressure[idx - nx * ny];
        }
        if (threadIdx.z == 0 && iz > 0) {
            tile_p[local_i][local_j][local_k + 1] = pressure[idx + nx * ny];
        }

        __syncthreads();

        /// 3D Finite difference
        float dpdx = (tile_p[local_i+1][local_j][local_k] - tile_p[local_i-1][local_j][local_k]) / (2.0f * dx);
        float dpdy = (tile_p[local_i][local_j+1][local_k] - tile_p[local_j][local_j-1][local_k]) / (2.0f * dy);
        float dpdz = (tile_p[local_i][local_j][local_k+1] - tile_p[local_i][local_j][local_k-1]) / (2.0f * dz);

        vx += dt * dpdx;
        vy += dt * dpdy;
        vz += dt * dpdz;

        ///---------------------------
        /// Running 3D Tensor Core
        ///---------------------------

        /// XY-plane
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA_xy;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> fragB_xy;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC_xy;
        fill_fragment(fragC_xy, 0.0f);

        for (int k = 0; k < WMMA_M * WMMA_N; ++k) {
            int li = k % WMMA_M;
            int ti = max(0, min(local_i + li, TILE_DIM_TC + 1));

            tileData_shared[k] = __float2half((tile_vx[ti][local_j][local_k] + tile_vz[ti][local_j][local_k]) * 0.5f);
        }

        load_matrix_sync(fragA_xy, tileData_shared, WMMA_K);

        for (int i = 0; i < fragB_xy.num_elements; ++i) {
            fragB_xy.x[i] = __float2half(1.0f);
        }
        mma_sync(fragC_xy, fragA_xy, fragB_xy, fragC_xy);
        __syncthreads();


        /// YZ-plane
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA_yz;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> fragB_yz;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC_yz;
        fill_fragment(fragC_yz, 0.0f);

        for (int k = 0; k < WMMA_N * WMMA_K; ++k) {
            int lj = k % WMMA_N;
            int tj = max(0, min(local_j + lj, TILE_DIM_TC + 1));

            tileData_shared[k] = __float2half((tile_vy[local_i][tj][local_k] + tile_vx[local_i][tj][local_k]) * 0.5f);
        }

        load_matrix_sync(fragA_yz, tileData_shared, WMMA_K);

        for (int i = 0; i < fragB_yz.num_elements; ++i) {
            fragB_yz.x[i] = __float2half(1.0f);
        }
        mma_sync(fragC_yz, fragA_yz, fragB_yz, fragC_yz);
        __syncthreads();


        /// ZX-plane
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA_zx;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> fragB_zx;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC_zx;
        fill_fragment(fragC_zx, 0.0f);

        for (int k = 0; k < WMMA_M * WMMA_K; ++k) {
            int lk = k % WMMA_M;
            int tk = max(0, min(local_j + lk, TILE_DIM_TC + 1));

            tileData_shared[k] = __float2half((tile_vz[local_i][local_j][tk] + tile_vy[local_i][local_j][tk]) * 0.5f);
        }

        load_matrix_sync(fragA_zx, tileData_shared, WMMA_K);

        for (int i = 0; i < fragB_zx.num_elements; ++i) {
            fragB_zx.x[i] = __float2half(1.0f);
        }
        mma_sync(fragC_zx, fragA_zx, fragB_zx, fragC_zx);
        __syncthreads();

        /// Combine the three results
        float div_xy = 0.0f;
        float div_yz = 0.0f;
        float div_zx = 0.0f;

        for (int i = 0; i < fragC_xy.num_elements; ++i) {
            div_xy += fragC_xy.x[i];
            div_yz += fragC_yz.x[i];
            div_zx += fragC_zx.x[i];
        }
        float divergence = div_xy + div_yz + div_zx;

        p -= vel * vel * dt * divergence;
        vx *= dumping;
        vy *= dumping;
        vz *= dumping;

        if (ix == sx && iy == sy && iz == sz) {
            p += Ricker_Wavelet(t * dt, real_frequency);
        }
        __syncthreads();

        pressure[idx] = p;
        velocity_x[idx] = vx;
        velocity_y[idx] = vy;
        velocity_z[idx] = vz;

        if (Is_Receiver_3D(ix, iy, iz, start_x, start_y, receiver_depth, spacing_x, spacing_y)) {
            int rx = (ix - start_x) / spacing_x;
            int ry = (iy - start_y) / spacing_y;
            int r = ry * total_shots + rx;
            data_output[r * max_time + t] = p;
        }
        __syncthreads();
    }
}
}


extern "C" {
__global__ void Elastic_PWave_Anisotropy_3D_TensorCore(
    size_t nx,
    size_t ny,
    size_t nz,
    float dx,
    float dy,
    float dz,
    float dt,
    float* __restrict__ d_VelocityModel,
    float* __restrict__ d_Data,
    float* __restrict__ d_Wavefield
)
{
    size_t ix = blockIdx.x * WMMA_M + threadIdx.x;
    size_t iy = blockIdx.y * WMMA_N + threadIdx.y;
    size_t iz = blockIdx.z * WMMA_K;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    __shared__ __align__(128) half sh_Data[WMMA_M * WMMA_N];
    __shared__ __align__(128) float temp[WMMA_M * WMMA_N];

    float lame = (lambda_coef + 2.0f * mu_coef) / rho_coef;
    float xscale = (__fdividef(dx,dt) * lame) * __fdividef((dt*dt), (dx*dx));
    float yscale = (__fdividef(dy,dt) * lame) * __fdividef((dt*dt), (dy*dy));
    float zscale = (__fdividef(dz,dt) * lame) * __fdividef((dt*dt), (dz*dz));

    sh_Data[threadIdx.y * WMMA_M + threadIdx.x] = d_Data[idx];
    __syncthreads();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_Data;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_Stencil;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_Laplacian;

    load_matrix_sync(frag_Data, sh_Data, WMMA_N);

    for (int i = 0; i < frag_Stencil.num_elements; i++) {
        frag_Stencil.x[i] = (i == 0) ? __float2half(zscale) :
                            (i == 1) ? __float2half(yscale) :
                            (i == 2) ? __float2half(xscale) : __float2half(0.0f);
    }

    fill_fragment(frag_Laplacian, 0.0f);
    mma_sync(frag_Laplacian, frag_Data, frag_Stencil, frag_Laplacian);
    __syncthreads();

    float factorial = 1.0f;
    float value_high_order = 0.0f;

    for (int order = 2; order <= 6; order += 2) {
        factorial *= (order - 1) * order;
        float coeff = __fdividef(1.0f , factorial);

        value_high_order += coeff * (zscale * (d_Data[idx + order * nx * ny] +
                                               d_Data[idx - order * nx * ny]) +
                                     yscale * (d_Data[idx + order * nx] +
                                               d_Data[idx - order * nx]) +
                                     xscale * (d_Data[idx + order] +
                                               d_Data[idx - order]));
    }

    store_matrix_sync(temp, frag_Laplacian, WMMA_N, mem_row_major);
    __syncthreads();

    for (int i = 0; i < WMMA_M * WMMA_N; i++) {
        sh_Data[i] = temp[i];
    }
    __syncthreads();

    if (ix < nx && iy < ny && iz < nz) {
        d_Wavefield[idx] = 2.0f * d_Data[idx] - d_Wavefield[idx] +
                           d_VelocityModel[idx] * (__half2float(sh_Data[threadIdx.y * 16 + threadIdx.x]) +
                                                   value_high_order);
    }
}
}


extern "C" {
__global__ void Gaussian_Smoothing_3D_TensorCore(
    float* __restrict__ d_Data,
    float* __restrict__ d_Wavefield,
    int nx,
    int ny,
    int nz,
    float* __restrict__ d_Result
)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int ix = blockIdx.x * WMMA_M + tx;
    int iy = blockIdx.y * WMMA_N + ty;
    int iz = blockIdx.z * WMMA_K + tz;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    int slice_stride = nx * ny;

    extern __shared__ float shm[];

    float* sh_Data = shm;
    float* sh_wavefield = sh_Data + (WMMA_M+2)*(WMMA_N+2)*(WMMA_K+2);

    int sx = tx + 1;
    int sy = ty + 1;
    int sz = tz + 1;
    int pitch_y = WMMA_M + 2;
    int pitch_z = pitch_y * (WMMA_N + 2);

    for (int dz = -1; dz <= 1; dz++) {
        int gz = iz + dz;
        for (int dy = -1; dy <= 1; dy++) {
            int gy = iy + dy;
            for (int dx = -1; dx <= 1; dx++) {
                int gx = ix + dx;
                bool inside = (gx >= 0 && gx < nx && gy >= 0 && gy < ny && gz >= 0 && gz < nz);
                float vD = inside ? d_Data[gz*slice_stride + gy*nx + gx] : 0.0f;
                float vW = inside ? d_Wavefield[gz*slice_stride + gy*nx + gx] : 0.0f;
                int lz = sz + dz;
                int ly = sy + dy;
                int lx = sx + dx;
                sh_Data[lz*pitch_z + ly*pitch_y + lx] = vD;
                sh_Wavefield[lz*pitch_z + ly*pitch_y + lx] = vW;
            }
        }
    }
    __syncthreads();

    __shared__ half mid1[WMMA_M][WMMA_N];
    __shared__ half mid2[WMMA_M][WMMA_N];

    const float sigma = 1.0f;
    const float norm3D = 1.0f / (powf(2.0f*M_PI*sigma*sigma, 1.5f));

    float sum1 = 0;
    float acc1 = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float w = norm3D * expf(-(dx*dx+dy*dy)/(2*sigma*sigma));
            sum1 += w;
            float v = sh_Data[sz*pitch_z + (sy+dy)*pitch_y + (sx+dx)];
            acc1 += w * v;
        }
    }
    mid1[ty][tx] = __float2half(acc1/sum1);

    float sum2 = 0;
    float acc2 = 0;
    for (int dz2 = -1; dz2 <= 1; ++dz2) {
        for (int dx2 = -1; dx2 <= 1; ++dx2) {
            float w = norm3D * expf(-(dx2*dx2+dz2*dz2)/(2*sigma*sigma));
            sum2 += w;
            float v = sh_Data[(sz+dz2)*pitch_z + sy*pitch_y + (sx+dx2)];
            acc2 += w * v;
        }
    }
    mid2[ty][tx] = __float2half(acc2/sum2);
    __syncthreads();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    fill_fragment(c_frag, 0.0f);
    load_matrix_sync(a_frag, &mid1[0][0], WMMA_N);
    load_matrix_sync(b_frag, &mid2[0][0], WMMA_N);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();

    float out_val = c_frag.x[ty*WMMA_N + tx];

    int out_idx = iz*slice_stride + iy*nx + ix;
    if (ix < nx && iy < ny && iz < nz) {
        d_Result[out_idx] = out_val;
    }
}
}


extern "C" {
__global__ void Absorbing_Boundary_Condition_3D(
    int nx,
    int ny,
    int nz,
    float dx,
    float dy,
    float dz,
    float dt,
    float* __restrict__ d_VelocityModel,
    float* __restrict__ d_Data,
    float* __restrict__ d_Wavefield
)
{
    float ov, ovs, cosa, beta, gamma, dpdx, dpdy, dpdz, dpdt;
    float dpdxs, dpdys, dpdzs, dpdts, loss_pde, loss_bc, total_loss;

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    __shared__ float shared_pn[TILE_DIM][TILE_DIM][TILE_DIM];
    __shared__ float shared_v[TILE_DIM][TILE_DIM][TILE_DIM];

    int index = iz * nx * ny * iy * nx + ix;
    shared_pn[threadIdx.z][threadIdx.y][threadIdx.x] = d_Data[index];
    shared_v[threadIdx.z][threadIdx.y][threadIdx.x] = d_VelocityModel[index];

    __syncthreads();

    ovs = __fdividef(1.0f, shared_v[threadIdx.z][threadIdx.y][threadIdx.x]);
    ov = sqrtf(ovs);

    int idx_forward = min(ix + 1, nx-1);
    int idx_backward = max(ix-1, 0);
    int idy_forward = min(iy+1, ny-1);
    int idy_backward = max(iy-1, 0);
    int idz_forward = min(iz+1, nz-1);
    int idz_backward = max(iz-1, 0);

    dpdx = __fdividef(d_Data[iz*nx*ny + iy*nx+idx_forward] - d_Data[iz*nx*ny + iy*nx+idx_backward], 2.0f * dx);
    dpdy = __fdividef(d_Data[iz*nx*ny + idy_forward*nx + ix] - d_Data[iz*nx*ny + idy_backward*nx + ix], 2.0f * dy);
    dpdz = __fdividef(d_Data[idz_forward*nx*ny + iy*nx + ix] - d_Data[idz_backward*nx*ny + iy*nx + ix], 2.0f * dz);
    dpdt = __fdividef(d_Data[index] - shared_pn[threadIdx.z][threadIdx.y][threadIdx.x], dt);

    dpdxs = __fmul_rn(dpdx, dpdx);
    dpdy
}
}


extern "C" {
__global__ void Combine_Misfit_Function_TensorCore_3D(
    float* __restrict__ d_dataObserved,
    float* __restrict__ d_dataSynthetic,
    size_t nx,
    size_t ny,
    size_t nz,
    float* __restrict__ d_misfit
)
{
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tile_z = blockIdx.z;

    int x0 = tile_x * WMMA_M;
    int y0 = tile_y * WMMA_N;
    int z0 = tile_z * WMMA_K;

    if (x0 + WMMA_M > nx || y0 + WMMA_N > ny || z0 + WMMA_K > nz) return;

    __shared__ half tile_obs[WMMA_M * WMMA_K];
    __shared__ half tile_syn[WMMA_K * WMMA_N];

    int tx = threadIdx.x;
    int tz = threadIdx.y;
    if (tx < WMMA_M && tz < WMMA_K) {
        int i = tx;
        int k = tz;
        size_t idxA = (z0 + k) * (nx*ny) + (y0 + 0) * nx + (x0 + i);
        tile_obs[k * WMMA_M + i] = __float2half(d_dataObserved[idxA]);
    }
    if (tz < WMMA_K && tx < WMMA_N) {
        int k = tx;
        int j = tz;
        size_t idxB = (z0 + k) * (nx*ny) + (y0 + j) * nx + (x0 + 0);
        tile_syn[k * WMMA_N + j] = __float2half(d_dataSynthetic[idxB]);
    }
    __syncthreads();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> fragB;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;

    load_matrix_sync(fragA, tile_obs, WMMA_K);
    load_matrix_sync(fragB, tile_syn, WMMA_N);
    fill_fragment(fragC, 0.0f);

    mma_sync(fragC, fragA, fragB, fragC);
    __syncthreads();

    float tmp[WMMA_M * WMMA_N];
    store_matrix_sync(tmp, fragC, WMMA_N, mem_row_major);
    __syncthreads();

    float norm_obs = 0.0f;
    float norm_syn = 0.0f;
    float transport = 0.0f;
    float dot = 0.0f;

    for (int idx = 0; idx < WMMA_M * WMMA_N; ++idx) {
        float o = __half2float(tile_obs[idx % (WMMA_M * WMMA_K)]);
        float s = __half2float(tile_syn[idx % (WMMA_K * WMMA_N)]);
        norm_obs += o*o;
        norm_syn += s*s;
        transport += fabsf(o-s);
        dot += tmp[idx];
    }
    float norm_prod = sqrtf(norm_obs) * sqrtf(norm_syn) + 1e-6f;
    float cross_corr = dot / norm_prod;
    float combined = 0.5f * (1.0f - cross_corr) + 0.5f * (transport / (WMMA_M * WMMA_N));
    int tile_id = tile_z * (gridDim.y * gridDim.x) + tile_y * gridDim.x + tile_x;

    d_misfit[tile_id] = combined;
}
}


extern "C" {
__global__ void Adjoint_Wavefield_TensorCore_3D(
    float* __restrict__ d_wavefield,
    float* __restrict__ d_misfit,
    size_t nx,
    size_t ny,
    size_t nz,
    float* __restrict__ d_adjoint
)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_DIM_TC;
    int by = blockIdx.y * TILE_DIM_TC;
    int bz = blockIdx.z;

    int ix = bx + tx;
    int iy = by + ty;
    int iz = bz;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    size_t idx = ix + iy * nx + iz * nx * ny;

    /// Flattening
    __shared__ half s_wavefield[TILE_DIM_TC * TILE_DIM_TC];
    __shared__ half s_misfit[TILE_DIM_TC * TILE_DIM_TC];
    __shared__ float s_adjoint[TILE_DIM_TC * TILE_DIM_TC];

    int tid = ty * TILE_DIM_TC + tx;

    s_wavefield[tid] = __float2half(d_wavefield[idx]);
    s_misfit[tid] = __float2half(d_misfit[idx]);
    __syncthreads();

    fragment<matrix_a, TILE_DIM_TC, TILE_DIM_TC, TILE_DIM_TC, half, row_major> frag_wave;
    fragment<matrix_b, TILE_DIM_TC, TILE_DIM_TC, TILE_DIM_TC, half, row_major> frag_misfit;
    fragment<accumulator, TILE_DIM_TC, TILE_DIM_TC, TILE_DIM_TC, float> frag_out;

    load_matrix_sync(frag_wave, s_wavefield, TILE_DIM_TC);
    load_matrix_sync(frag_misfit, s_misfit, TILE_DIM_TC);

    fill_fragment(frag_out, 0.0f);
    mma_sync(frag_out, frag_wave, frag_misfit, frag_out);

    store_matrix_sync(s_adjoint, frag_out, TILE_DIM_TC, mem_row_major);
    __syncthreads();

    float val = s_adjoint[tid];
    val = val / (sqrtf(val*val) + dumping_factor);
    d_adjoint[idx] = (fabsf(val) > 1e-4f) ? val : 0.0f;
}
}


extern "C" {
__global__ void Adaptive_Sparsity_Based_Gradient_TensorCore_3D(
    float* __restrict__ d_wavefield,
    float* __restrict__ d_adjoint,
    size_t nx,
    size_t ny,
    size_t nz,
    float* __restrict__ d_gradient
)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    size_t ix = bx * WMMA_M + tx;
    size_t iy = by * WMMA_N + ty;
    size_t iz_base = bz * WMMA_K;

    int thread_linear = ty * WMMA_M + tx;
    int linearTid = thread_linear;
    int warpId = linearTid >> 5;
    int laneId = linearTid & 31;

    __shared__ half s_wavefield[WMMA_M * WMMA_N];
    __shared__ half s_adjoint[WMMA_M * WMMA_N];
    __shared__ float s_grad_accum;

    if (tx == 0 && ty == 0) {
        s_grad_accum = 0.0f;
    }
    __syncthreads();

    const int num_slices = WMMA_K;

    /// Loop over slices in K direction
    for (int k = 0; k < num_slices; ++k) {
        size_t iz = iz_base + k;
        size_t idx = ix + iy * nx + iz * nx * ny;

        float wave_val = 0.0f;
        float adj_val = 0.0f;
        if (ix < nx && iy < ny && iz < nz) {
            wave_val = d_wavefield[idx];
            adj_val = d_adjoint[idx];
        }

        s_wavefield[thread_linear] = __float2half(wave_val);
        s_adjoint[thread_linear] = __float2half(adj_val);
        __syncthreads();

        /// Only warp 0 executes WMMA
        if (warpId == 0) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> fragB;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;

            load_matrix_sync(fragA, s_wavefield, WMMA_M);
            load_matrix_sync(fragB, s_adjoint, WMMA_M);
            fill_fragment(fragC, 0.0f);
            mma_sync(fragC, fragA, fragB, fragC);

            float local_sum = 0.0f;

            for (int i = 0; i < fragC.num_elements; ++i) {
                local_sum += fragC.x[i];
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                local_sum += __shfl_down_sync(0xffffffff, local_sum, offset)
            }

            if (laneId == 0) {
                atomicAdd(&s_grad_accum, local_sum);
            }
        }
        __syncthreads();
    }

    if (ix < nx && iy < ny) {
        float grad = s_grad_accum / float(num_slices);
        grad = (fabsf(grad) > sparsity_threshold) ? grad : 0.0f;

        size_t iz_mid = iz_base + (WMMA_K / 2);
        if (iz_mid < nz) {
            size_t out_idx = ix + iy * nx + iz_mid * nx * ny;
            d_gradient[out_idx] = grad;
        }
    }
}
}


extern "C" {
__global__ void Compute_Residual_Tiled_3D(
    float* __restrict__ d_gradient,
    float* __restrict__ d_wavefield,
    size_t nx,
    size_t ny,
    size_t nz,
    float* __restrict__ d_residue
)
{
    size_t tile_x = blockIdx.x * TILE_DIM;
    size_t tile_y = blockIdx.y * TILE_DIM;
    size_t tile_z = blockIdx.z * TILE_DIM;

    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t tz = threadIdx.z;

    size_t gx = tile_x + tx;
    size_t gy = tile_y + ty;
    size_t gz = tile_z + tz;

    size_t idx = gx + gy * nx + gz * nx * ny;

    __shared__ float sh_gradient[TILE_DIM][TILE_DIM][TILE_DIM];
    __shared__ float sh_wavefield[TILE_DIM][TILE_DIM][TILE_DIM];

    if (gx < nx && gy < ny && gz < nz) {
        sh_gradient[tx][ty][tz] = d_gradient[idx];
        sh_wavefield[tx][ty][tz] = d_wavefield[idx];
    }
    else {
        sh_gradient[tx][ty][tz] = 0.0f;
        sh_wavefield[tx][ty][tz] = 0.0f;
    }
    __syncthreads();

    if (gx < nx && gy < ny && gz < nz) {
        float g_val = sh_gradient[tx][ty][tz];
        float Ap_val = sh_wavefield[tx][ty][tz];
        float denom = Ap_val + dumping_factor;
        denom = fabsf(denom) > 1e-6f ? denom : 1e-6f;

        float reciprocal = __frcp_rn(denom);
        reciprocal *= (2.0f - denom * reciprocal);

        float residual = fmaf(reciprocal, (g_val - Ap_val), 0.0f);
        d_residue[idx] = r;
    }
}
}


extern "C" {
__global__ void Parallel_Reduction_for_Residual_Norm_TensorCore_3D(
    float* __restrict__ d_residue,
    size_t nx,
    size_t ny,
    size_t nz,
    float* __restrict__ d_norm
)
{
    const size_t total_size = nx * ny * nz;
    const size_t tile_size = WMMA_M * WMMA_K;
    const size_t tileId = blockIdx.x;
    const size_t = baseIdx = tileId * tile_size;

    extern __shared__ half shTile[];

    int tid = threadIdx.x;

    for (int i = tid; i < tile_size; i += blockDim.x) {
        size_t idx = baseIdx + i;
        float v = (idx < total_size) ? d_residue[idx] : 0.0f;
        shTile[i] = __float2half(v*v);
    }
    __syncthreas();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> aFrag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> bFrag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    fill_fragment(cFrag, 0.0f);
    fill_fragment(bFrag, __float2half(1.0f));

    load_matrix_sync(aFrag, shTile, WMMA_K);
    mma_sync(cFrag, aFrag, bFrag, cFrag);

    float sum = 0.0f;
    for (int i = 0; i < cFrag.num_elements; ++i) {
        sum += cFrag.x[i];
    }

    if (tid == 0) {
        d_norm[tileId] = sum;
    }
}
}


extern "C" {
__global__ void Update_Direction_Vector_3D(
    float* __restrict__ d_residue,
    float* __restrict__ beta,
    size_t nx,
    size_t ny,
    size_t nz,
    float* __restrict__ d_direction_vector
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_size = nx * ny * nz;

    if (idx < total_size) {
        float r = d_residue[idx];
        float p_old = d_direction_vector[idx];
        float b = beta[idx];

        d_direction_vector[idx] = r + b * p_old;
    }
}
}


extern "C" {
__global__ void Pearson_Correlation_Coefficient(
    float* __restrict__ Data1,
    float* __restrict__ Data2,
    size_t size_Data1,
    size_t size_Data2,
    float* Output_reduction
)
{
    if (size_Data1 != size_Data2) return;
    int N = size_Data1;

    extern __shared__ float sdata[];
    float* s_sumx = sdata;
    float* s_sumy = s_sumx + blockDim.x;
    float* s_sumxy = s_sumy + blockDim.x;
    float* s_sumx2 = s_sumxy + blockDim.x;
    float* s_sumy2 = s_sumx2 + blockDim.x;

    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float sumx = 0.0f;
    float sumy = 0.0f;
    float sumxy = 0.0f;
    float sumx2 = 0.0f;
    float sumy2 = 0.0f;

    for (int i = idx; i < N; i += stride) {
        float x = Data1[i];
        float y = Data2[i];
        sumx += x;
        sumy += y;
        sumxy += x * y;
        sumx2 += x * x;
        sumy2 += y * y;
    }

    s_sumx[tid] = sumx;
    s_sumy[tid] = sumy;
    s_sumxy[tid] = sumxy;
    s_sumx2[tid] = sumx2;
    s_sumy2[tid] = sumy2;
    __syncthreads();

    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sumx[tid] += s_sumx[tid + offset];
            s_sumy[tid] += s_sumy[tid + offset];
            s_sumxy[tid] += s_sumxy[tid + offset];
            s_sumx2[tid] += s_sumx2[tid + offset];
            s_sumy2[tid] += s_sumy2[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean_x = s_sumx[0] / N;
        float mean_y = s_sumy[0] / N;
        float cov_xy = s_sumxy[0] - N * mean_x * mean_y;
        float var_x = s_sumx2[0] - N * mean_x * mean_x;
        float var_y = s_sumy2[0] - N * mean_y * mean_y;
        float corr = cov_xy / (sqrtf(var_x) * sqrtf(var_y));
        Output_reduction[0] = corr;
    }
}
}











