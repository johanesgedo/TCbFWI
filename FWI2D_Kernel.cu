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
__global__ void Compute_SQRT_Half_2D(
    size_t nx,
    size_t nz,
    float* __restrict__ variable
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_elements = nx * nz;
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


__device__ __forceinline__ bool Is_Receiver_2D(
    int ix,
    int iz,
    int start_x,
    int receiver_depth,
    int spacing
)
{
    return (iz == receiver_depth) && ((ix - start_x) % spacing == 0);
}


__device__ __forceinline__ void Pressure_From_2D_Velocity(
    float* __restrict__ velocity,
    size_t nx,
    size_t nz,
    float dx,
    float dz,
    float dt,
    float* __restrict__ pressure
)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < 1 || ix >= nx - 1 || iz < 1 || iz >= nz - 1) return;

    int idx = ix + iz * nx;

    __shared__ __align__(128) half shared_velocity_x[WMMA_M][WMMA_N];
    __shared__ __align__(128) half shared_velocity_z[WMMA_M][WMMA_N];

    size_t idx_vx = 2 * (ix + iz * nx);
    size_t idx_vz = 2 * (ix + iz * nx) + 1;

    if (ix < nx && iz < nz) {
        shared_velocity_x[threadIdx.x][threadIdx.y] = velocity[idx_vx];
        shared_velocity_z[threadIdx.x][threadIdx.y] = velocity[idx_vz];
    }
    else {
        shared_velocity_x[threadIdx.x][threadIdx.y] = __float2half(0.0f);
        shared_velocity_z[threadIdx.x][threadIdx.y] = __float2half(0.0f);
    }
    __syncthreads();

    float v_x_left = __half2float(shared_velocity_x[threadIdx.x - 1][threadIdx.y]);
    float v_x_right = __half2float(shared_velocity_x[threadIdx.x + 1][threadIdx.y]);
    float dpdx = (v_x_right - v_x_left) / (2.0f * dx);

    float v_z_down = __half2float(shared_velocity_z[threadIdx.x][threadIdx.y - 1]);
    float v_z_up = __half2float(shared_velocity_z[threadIdx.x][threadIdx.y + 1]);
    float dpdz = (v_z_up - v_z_down) / (2.0f * dz);

    float pressure_val = dt * (dpdx + dpdz);

    pressure[idx] = __float2half(pressure_val);
}


__device__ __forceinline__ void Velocity_2D_Horizontal(
    float* __restrict__ pressure,
    size_t nx,
    size_t nz,
    float dx,
    float* __restrict__ velocity_horizontal
)
{
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t idx = blockIdx.x * blockDim.x + tx;
    size_t idz = blockIdx.y * blockDim.y + ty;

    __shared__ float s_pressure[WMMA_M][WMMA_N + 2];

    if (idx < nx && idz < nz && tx < WMMA_M && ty < WMMA_N) {
        size_t g_idx = idx + idz * nx;
        s_pressure[ty][tx + 1] = pressure[g_idx];

        if (tx == 0 && idx > 0) {
            s_pressure[ty][0] = pressure[(idx - 1) + idz * nx];
        }

        if (tx == WMMA_M - 1 && idx < nx - 1) {
            s_pressure[ty][WMMA_M + 1] = pressure[(idx + 1) + idz * nx];
        }
    }
    __syncthreads();

    if (idx <= 0 || idx >= nx - 1 || idz >= nz) return;

    if (tx < WMMA_M && ty < WMMA_N) {
        float p_left = s_pressure[ty][tx];
        float p_right = s_pressure[ty][tx + 2];
        float dpdx = (p_right - p_left) / (2.0f * dx);
        float vp = (lambda_coef + 2.0f * mu_coef) / rho_coef;
        float v_x = vp * dpdx;
        size_t grid_idx = idx + idz * nx;
        velocity_horizontal[grid_idx] = __float2half(v_x);
    }
}


__device__ __forceinline__ void Velocity_2D_Vertical(
    float* __restrict__ pressure,
    size_t nx,
    size_t nz,
    float dz,
    float* __restrict__ velocity_vertical
)
{
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t ix = blockIdx.x * blockDim.x + tx;
    size_t iz = blockIdx.y * blockDim.y + ty;

    __shared__ float s_pressure[WMMA_M + 2][WMMA_N];

    /// Load global pressure to shared memory
    if (ix < nx && iz < nz && tx < WMMA_M && ty < WMMA_N) {
        size_t g_idx = idx + idz * nz;
        s_pressure[ty][tx + 1] = pressure[g_idx];

        /// Load halo left (thread 0)
        if (tx == 0 && idx > 0) {
            s_pressure[ty][0] = pressure[(idx - 1) + idz * nx];
        }

        /// Load halo right (thread N-1)
        if (tx == WMMA_M - 1 && idx < nx - 1) {
            s_pressure[ty][WMMA_M + 1] = pressure[(idx + 1) + idz * nx];
        }
    }
    __syncthreads();

    /// Checking boundary
    if (idx <= 0 || idx >= nx - 1 || idz >= nz) return;

    if (tx < WMMA_M && ty < WMMA_N) {
        float p_left = s_pressure[ty][tx];
        float p_right = s_pressure[ty][tx + 2];
        float dpdx = (p_right - p_left) / (2.0f * dx);
        float vp = (lambda_coef + 2.0f * mu_coef) / rho_coef;
        float v_x = vp * dpdx;
        size_t grid_idx = idx + idz * nx;
        velocity_horizontal[grid_idx] = __float2half(v_x);
    }
}


__device__ __forceinline__ void FFT_2D_From_Velocity_Model(
    float* __restrict__ velocity,
    size_t nx,
    size_t nz,
    /// float dx,
    /// float dz,
    /// float dt,
    float2* __restrict__ frequency
)
{
    /// Constants for FFT
    size_t tidx = threadIdx.x;
    size_t tidy = threadIdx.y;
    size_t ix = blockIdx.x * blockDim.x + tidx;
    size_t iz = blockIdx.y * blockDim.y + tidy;

    if (ix >= nx || iz >= nz) return;

    __shared__ float2 shared_tile_x[WMMA_M][WMMA_N];
    __shared__ float2 shared_tile_z[WMMA_M][WMMA_N];

    /// 1) Load real data from velocity (convert from half to float2 real + imag=0)
    if (ix < nx && iz < nz) {
        size_t idx = ix + iz * nx;
        float v_real = velocity[idx];
        shared_tile_x[tidy][tidx] = make_float2(v_real, 0.0f);
    }
    else {
        shared_tile_x[tidy][tidx] = make_float2(0.0f, 0.0f);
    }
    __syncthreads();

    /// 2) FFT 1D along X direction (row-wise)
    for (int stride = 1; stride < 32; stride *= 2) {
        float2 temp = shared_tile_x[tidy][tidx];
        int twiddle_factor_idx = (tidx % (2 * stride));
        float angle = -2.0f * M_PI * twiddle_factor_idx / (2.0f * stride);
        float2 twiddle = make_float2(cosf(angle), sinf(angle));

        if ((tidx % (2 * stride)) < stride && (tidx + stride) < WMMA_N) {
            float2 a = temp;
            float2 b = shared_tile_x[tidy][tidx + stride];
            float2 t = make_float2(b.x * twiddle.x - b.y * twiddle.y,
                                   b.x * twiddle.y + b.y * twiddle.x);
            shared_tile_x[tidy][tidx] = make_float2(a.x + t.x, a.y + t.y);
        }
        __syncthreads();
    }

    /// 3) Transpose to shared_tile_z (prepare for FFT along Z)
    shared_tile_z[tidx][tidy] = shared_tile_x[tidy][tidx];
    __syncthreads();

    /// 4) FFT 1D along Z direction (column-wise)
    for (int stride = 1; stride < WMMA_M; stride *= 2) {
        float2 temp = shared_tile_z[tidy][tidx];
        int twiddle_factor_idx = (tidy % (2 * stride));
        float angle = -2.0f * M_PI * twiddle_factor_idx / (2.0f * stride);
        float2 twiddle = make_float2(cosf(angle), sinf(angle));

        if ((tidy % (2 * stride)) < stride && (tidy + stride) < WMMA_M) {
            float2 a = temp;
            float2 b = shared_tile_z[tidy + stride][tidx];
            float2 t = make_float2(b.x * twiddle.x - b.y * twiddle.y,
                                   b.x * twiddle.y + b.y * twiddle.x);
            shared_tile_z[tidy][tidx] = make_float2(a.x + t.x, a.y + t.y);
        }
        __syncthreads();
    }

    /// 5) Write result to global memory
    if (ix < nx && iz < nz) {
        size_t global_idx = ix + iz * nx;
        frequency[global_idx] = shared_tile_z[tidx][tidy];
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

__device__ void FFT2D_Row(
    float2 *data,
    int n,
    int log2n
)
{
    /// Bit-reversal reorder
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned j = bit_reverse(i, log2n);
        if (j > (unsigned)i) {
            float2 tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
    __syncthreads();

    /// Butterfly stages
    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        float theta = -CUDART_PI_F / m2;
        float2 w_m = make_float2(cosf(theta), sinf(theta));

        for (int k = threadIdx.x; k < n; k += blockDim.x) {
            int j = k & (m - 1);
            int base = k - j;

            if (j < m2) {
                float angle = thata * j;
                float2 w = make_float2(cosf(angle), sinf(angle));
                float2 u = data[base + j];
                float2 t = complex_mul(w, data[base + j + m2]);
                data[base + j] = make_float2(u.x + t.x, u.y + t.y);
                data[base + j + m2] = make_float2(u.x - t.x, u.y - t.y);
            }
        }
        __syncthreads();
    }
}


extern "C" {
__global__ void FDTD_2D_Forward_Propagation_PML_Anisotropy_TensorCore(
    float* __restrict__ velocity_rms,
    float* __restrict__ velocity_x,
    float* __restrict__ velocity_z,
    float* __restrict__ pressure,
    float2* __restrict__ frequency,
    float min_velocity,
    size_t nx,
    size_t nz,
    float dx,
    float dz,
    float dt,
    float* __restrict__ data_output
)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= nx || iz >= nz) return;

    size_t idx = ix + iz * nx;

    __shared__ float tile_p[WMMA_M + 2][WMMA_N + 2];
    __shared__ float tile_vx[WMMA_M + 2][WMMA_N + 2];
    __shared__ float tile_vz[WMMA_M + 2][WMMA_N + 2];
    __shared__ half tileData_shared[WMMA_M * WMMA_N];

    /// Maximum distance of velocity model
    int max_distance = sqrtf(powf(nx*dx,2) + powf(nz*dz,2));

    const float safety_factor = 1.2f;
    int max_time = int(ceilf(safety_factor * max_distance / min_velocity));

    /// Real frequency
    float real_frequency = frequency[idx].x;
    if (real_frequency <= 0.0f) real_frequency = 10.0f;

    /// Calculate grid points per wavelength
    float min_v = min_velocity;
    int N_lambda_x = max(1, int(floorf(min_v / (real_frequency * dx))));
    int N_lambda_z = max(1, int(floorf(min_v / (real_frequency * dz))));

    /// Calculate perfectly matched layer (PML)
    int PML_THICKNESS_X = int(ceilf(N_lambda_x * min_v / (real_frequency * dx)));
    int PML_THICKNESS_Z = int(ceilf(N_lambda_z * min_v / (real_frequency * dz)));

    /// Initial position of the receiver horizontally
    int start_x = PML_THICKNESS_X;

    /// Calculate vertical position of the receiver horizontally
    int receiver_depth = PML_THICKNESS_Z + 5;

    /// Distance between receivers in grid units
    int spacing = 2;

    /// Calculate maximum shots
    int max_shots = (nx - 2 * PML_THICKNESS_X) / spacing;
    if (blockIdx.x >= max_shots) return;

    /// Fixed depth offset to determine the depth of the wave source position
    int fixed_depth_offset = N_lambda_z / 2;

    /// Calculate simultaneous multi-shot simulations (anisotropic PNL assumption)
    int sx = PML_THICKNESS_X + blockIdx.x * spacing;
    int sz = PML_THICKNESS_Z + fixed_depth_offset;
    if (sz >= nz - PML_THICKNESS_Z) return;

    int local_i = threadIdx.x + 1;
    int local_j = threadIdx.y + 1;

    float vel = velocity_rms[idx];
    float p = pressure[idx];
    float vx = velocity_x[idx];
    float vz = velocity_z[idx];

    float sigma_x = 0.0f;
    float sigma_z = 0.0f;

    if (ix <  PML_THICKNESS_X) {
        sigma_x = (PML_THICKNESS_X - ix) / (float)PML_THICKNESS_X;
    }
    else if (ix > nx - PML_THICKNESS_X) {
        sigma_x = (ix - (nx - PML_THICKNESS_X)) / (float)PML_THICKNESS_X;
    }

    if (iz < PML_THICKNESS_Z) {
        sigma_z = (PML_THICKNESS_Z - iz) / (float)PML_THICKNESS_Z;
    }
    else if (iz > nz - PML_THICKNESS_Z) {
        sigma_z = (iz - (nz - PML_THICKNESS_Z)) / (float)PML_THICKNESS_Z;
    }

    float dumping = expf(-(sigma_x + sigma_z) * dt);

    for (int t = 0; t < max_time; ++t) {
        tile_p[local_j][local_i] = p;
        tile_vx[local_j][local_i] = vx;
        tile_vz[local_j][local_i] = vz;

        if (threadIdx.x == 0 && ix > 0) {
            tile_p[local_j][0] = pressure[idx - 1];
        }
        if (threadIdx.x == blockDim.x - 1 && ix < nx - 1) {
            tile_p[local_j][TILE_DIM_TC + 1] = pressure[idx + 1];
        }
        if (threadIdx.y == 0 && iz > 0) {
            tile_p[0][local_i] = pressure[idx - nx];
        }
        if (threadIdx.y == blockDim.y - 1 && iz < nz - 1) {
            tile_p[TILE_DIM_TC + 1][local_i] = pressure[idx + nx];
        }

        /// Fill vx/vz halo identical
        if (threadIdx.x == 0 && ix > 0) {
            tile_vx[local_j][0] = velocity_x[idx - 1];
            tile_vz[local_j][0] = velocity_z[idx - 1];
        }
        if (threadIdx.x == blockDim.x - 1 && ix < nx - 1) {
            tile_vx[local_j][WMMA_N + 1] = velocity_x[idx + 1];
            tile_vz[local_j][WMMA_N + 1] = velocity_z[idx + 1];
        }
        if (threadIdx.y == 0 && iz > 0) {
            tile_vx[0][local_i] = velocity_x[idx - nx];
            tile_vz[0][local_i] = velocity_z[idx - nx];
        }
        if (threadIdx.y == blockDim.y - 1 && iz < nz - 1) {
            tile_vx[WMMA_M + 1][local_i] = velocity_x[idx + nx];
            tile_vz[WMMA_M + 1][local_i] = velocity_z[idx + nx];
        }

        __syncthreads();

        float dpdx = (tile_p[local_j][local_i+1] - tile_p[local_j][local_i-1]) / (2.0f * dx);
        float dpdz = (tile_p[local_j][local_i] - tile_p[local_j-1][local_i]) / (2.0f * dz);

        vx += dt * dpdx;
        vz += dt * dpdz;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA;
        fragment<natrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> fragB;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
        fill_fragment(fragC, __float2half(0.0f));

        for (int k = 0; k < WMMA_M * WMMA_K; ++k) {
            int li = k % WMMA_M;
            int lj = k / WMMA_M;
            int ti = li + local_i - (WMMA_M / 2);
            int tj = lj + local_j - (WMMA_K / 2);

            ti = max(0, min(ti, TILE_DIM_TC + 1));
            tj = max(0, min(tj, TILE_DIM_TC + 1));

            tileData_shared[k] = __float2half((tile_vx[tj][ti] + tile_vz[tj][ti]) * 0.5f);
        }

        load_matrix_sync(fragA, tileData_shared, WMMA_K);

        for (int i = 0; i < fragB.num_elements; i++) {
            fragB.x[i] = __float2half(1.0f);
        }
        mma_sync(fragC, fragA, fragB, fragC);
        __syncthreads();

        float divergence = 0.0f;
        for (int k = 0; k < fragC.num_elements; ++k) {
            divergence += fragC.x[k];
        }

        p -= vel * vel * dt * divergence;
        vx *= dumping;
        vz *= dumping;
        if (ix == sx && iz == sz) {
            p += Ricker_Wavelet(t * dt, real_frequency);
        }
        __syncthreads();

        pressure[idx] = p;
        velocity_x[idx] = vx;
        velocity_z[idx] = vz;

        if (Is_Receiver_2D(ix, iz, start_x, receiver_depth, spacing)) {
            int r = (ix - start_x) / spacing;
            data_output[r * max_time + t] = p;
        }
        __syncthreads();
    }
}
}


extern "C" {
__global__ void FDTD_2D_Backward_Propagation_PML_Anisotropy_TensorCore(
    float* __restrict__ velocity_rms,
    float* __restrict__ velocity_x,
    float* __restrict__ velocity_z,
    float* __restrict__ pressure,
    float2* __restrict__ frequency,
    float* min_velocity,
    size_t nx,
    size_t nz,
    float dx,
    float dz,
    float dt,
    float* __restrict__ data_output
)
{
    size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= nx || iz >= nz) return;

    size_t idx = ix + iz * nx;

    __shared__ float tile_p[WMMA_M + 2][WMMA_N + 2];
    __shared__ float tile_vx[WMMA_M + 2][WMMA_N + 2];
    __shared__ float tile_vz[WMMA_M + 2][WMMA_N + 2];
    __shared__ float half tileData_shared[WMMA_M * WMMA_N];

    /// Maximum distance of velocity model
    int max_distance = sqrtf(powf(nx*dx,2) + powf(nz*dz,2));

    /// Calculate maximum time
    /// min_velocity is calculated from reduction kernel
    const float safety_factor = 1.2f;
    int max_time = int(ceilf(safety_factor * max_distance / (*min_velocity)));

    float real_frequency = frequency[idx].x;
    if (real_frequency <= 0.0f) real_frequency = 10.0f;

    /// Grid points per wavelength
    float min_v = *min_velocity;
    int N_lambda_x = max(1, int(floorf(min_v / (real_frequency * dx))));
    int N_lambda_z = max(1, int(floorf(min_v / (real_frequency * dz))));

    /// Perfectly Matched Layer (PML)
    int PML_THICKNESS_X = int(ceilf(N_lambda_x * min_v / (real_frequency * dx)));
    int PML_THICKNESS_Z = int(ceilf(N_lambda_z * min_v / (real_frequency * dz)));

    /// Initial position of the receiver horizontally
    int start_x = PML_THICKNESS_X;

    /// Vertical position where the receiver is placed
    int receiver_depth = PML_THICKNESS_Z + 5;

    /// Distance between receivers in grid units
    int spacing = 2;

    /// Maximum shots
    int max_shots = (nx - 2 * PML_THICKNESS_X) / spacing;
    if (blockIdx.x >= max_shots) return;

    /// Depth offset to determine the depth of the wave source position
    int fixed_depth_offset = N_lambda_z / 2;

    int sx = PML_THICKNESS_X + blockIdx.x * spacing;
    int sz = PML_THICKNESS_Z + fixed_depth_offset;
    if (sz >= nz - PML_THICKNESS_Z) return;

    int local_i = threadIdx.x + 1;
    int local_j = threadIdx.y + 1;

    float vel = velocity_rms[idx];
    float p = pressure[idx];
    float vx = velocity_x[idx];
    float vz = velocity_z[idx];

    float sigma_x = 0.0f;
    float sigma_z = 0.0f;

    if (ix < PML_THICKNESS_X) {
        sigma_x = (PML_THICKNESS_X - ix) / (float)PML_THICKNESS_X;
    }
    else if (ix > nx - PML_THICKNESS_X) {
        sigma_x = (ix - (nx - PML_THICKNESS_X)) / (float)PML_THICKNESS_X;
    }

    if (iz < PML_THICKNESS_Z) {
        sigma_z = (PML_THICKNESS_Z - iz) / (float)PML_THICKNESS_Z;
    }
    else if (iz > nz - PML_THICKNESS_Z) {
        sigma_z = (iz - (nz - PML_THICKNESS_Z)) / (float)PML_THICKNESS_Z;
    }

    float dumping = expf(-(sigma_x + sigma_z) * dt);

    for (int t = max_time-1; t >= 0; --t) {
        tile_p[local_j][local_i] = p;
        tile_vx[local_j][local_i] = vx;
        tile_vz[local_j][local_i] = vz;

        if (threadIdx.x == 0 && ix > 0) {
            tile_p[local_j][0] = pressure[idx - 1];
        }
        if (threadIdx.x == blockDim.x - 1 && ix < nx - 1) {
            tile_p[local_j][TILE_DIM_TC + 1] = pressure[idx + 1];
        }
        if (threadIdx.y == 0 && iz > 0) {
            tile_p[0][local_i] = pressure[idx - nx];
        }
        if (threadIdx.y == blockDim.y - 1 && iz < nz - 1) {
            tile_p[TILE_DIM_TC + 1][local_i] = pressure[idx + nx];
        }

        if (threadIdx.x == 0 && ix > 0) {
            tile_vx[local_j][0] = velocity_x[idx - 1];
            tile_vz[local_j][0] = velocity_z[idx - 1];
        }
        if (threadIdx.x == blockDim.x - 1 && ix < nx - 1) {
            tile_vx[local_j][WMMA_N + 1] = velocity_x[idx + 1];
            tile_vz[local_j][WMMA_N + 1] = velocity_z[idx + 1];
        }
        if (threadIdx.y == 0 && iz > 0) {
            tile_vx[0][local_i] = velocity_x[idx - nx];
            tile_vz[0][local_i] = velocity_z[idx - nx];
        }
        if (threadIdx.y == blockDim.y - 1 && iz < nz - 1) {
            tile_vx[WMMA_M + 1][local_i] = velocity_x[idx + nx];
            tile_vz[WMMA_M + 1][local_i] = velocity_z[idx + nx];
        }

        __syncthreads();

        float dpdx = (tile_p[local_j][local_i + 1] - tile_p[local_j][local_i - 1]) / (2.0f * dx);
        float dpdz = (tile_p[local_j + 1][local_i] - tile_p[local_j - 1][local_i]) / (2.0f * dz);

        vx += dt * dpdx;
        vz += dt * dpdz;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> fragB;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
        fill_fragment(fragC, __float2half(0.0f));

        for (int k = 0; k < WMMA_M * WMMA_K; ++k) {
            int li = k % WMMA_M;
            int lj = k / WMMA_M;
            int ti = li + local_i - (WMMA_M / 2);
            int tj = lj + local_j - (WMMA_K / 2);

            ti = max(0, min(ti, TILE_DIM_TC + 1));
            tj = max(0, min(tj, TILE_DIM_TC + 1));

            tileData_shared[k] = __float2half((tile_vx[tj][ti] + tile_vz[tj][ti]) * 0.5f);
        }

        load_matrix_sync(fragA, tileData_shared, WMMA_K);

        for (int i = 0; i < fragB.num_elements; i++) {
            fragB.x[i] = __float2half(1.0f);
        }
        mma_sync(fragC, fragA, fragB, fragC);
        __syncthreads();

        float divergence = 0.0f;

        for (int k = 0; k < fragC.num_elements; ++k) {
            divergence += fragC.x[k];
        }

        p -= vel * vel * dt * divergence;
        vx *= dumping;
        vz *= dumping;
        if (ix == sx && iz == sz) {
            p += Ricker_Wavelet(t * dt, real_frequency);
        }
        __syncthreads();

        pressure[idx] = p;
        velocity_x[idx] = vx;
        velocity_z[idx] = vz;

        if (Is_Receiver_2D(ix, iz, start_x, receiver_depth, spacing)) {
            int r = (ix - start_x) / spacing;
            data_output[r * max_time + t] = p;
        }
        __syncthreads();
    }
}
}


extern "C" {
__global__ void Elastic_PWave_Anisotropy_2D_TensorCore(
    size_t nx,
    size_t nz,
    float dx,
    float dz,
    float dt,
    float* __restrict__ d_VelocityModel,
    float* __restrict__ d_Data,
    float* __restrict__ d_mu,
    float* __restrict__ d_lambda,
    float* __restrict__ d_velocity_x,
    float* __restrict__ d_velocity_z,
    float* __restrict__ d_pressure,
    float* __restrict__ d_wavefield
)
{
    const int tx = threadIdx.x;
    const int tz = threadIdx.y;
    const int ix = blockIdx.x * WMMA_M + tx;
    const int iz = blockIdx.y * WMMA_N + tz;

    if (ix >= nx || iz >= nz) return;
    if (blockDim.x != WMMA_M || blockDim.y != WMMA_N) return;

    const int idx = iz * nx + ix;

    __shared__ half sh_Data[WMMA_M * WMMA_N];
    sh_Data[tz * WMMA_M + tx] = __float2half(d_Data[idx]);
    __syncthreads();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_Data;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_Kernel;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_Output;

    load_matrix_sync(frag_Data, sh_Data, WMMA_N);
    fill_fragment(frag_Kernel, __float2half(0.0f));
    fill_fragment(frag_Output, 0.0f);

    float lame = d_lambda[idx] + 2.0f * d_mu[idx];
    float xscale = (d_velocity_x[idx] * lame * dt * dt) / (dx * dx);
    float zscale = (d_velocity_z[idx] * lame * dt * dt) / (dz * dz);

    frag_Kernel.x[0] = __float2half(zscale);
    frag_Kernel.x[1] = __float2half(xscale);

    mma_sync(frag_Output, frag_Data, frag_Kernel, frag_Output);

    float lap_val = frag_Output.x[tz * WMMA_M + tx];
    float high_order = 0.0f;
    float factorial = 1.0f;

    for (int order = 2; order <= 6; order += 2) {
        factorial *= (order - 1) * order;
        float coeff = 1.0f / factorial;
        if (ix >= order && ix < nx - order && iz >= order && iz < nz - order) {
            high_order += coeff * (zscale * (d_Data[idx + order * nx] + d_Data[idx - order * nx]) +
                                   xscale * (d_Data[idx + order] + d_Data[idx - order]));
        }
    }

    d_Wavefield[idx] = 2.0f * d_Data[idx] - d_Wavefield[idx] + d_VelocityModel[idx] * (lap_val + high_order);
}
}


extern "C" {
__global__ void Gaussian_Smoothing_2D_TensorCore(
    float* __restrict__ d_Data,
    float* __restrict__ d_Wavefield,
    int nx,
    int ny,
    float* __restrict__ d_Result
)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ix = blockIdx.x * WMMA_M + tx;
    int iy = blockIdx.y * WMMA_N + ty;

    if (ix >= nx || iy >= ny) return;

    int lin = iy * nx + ix;

    extern __shared__ float shm[];
    float* sh_D = shm;
    float* sh_W = sh_D + (WMMA_M+2)*(WMMA_N+2);

    int sx = tx + 1;
    int sy = ty + 1;
    int pitch = WMMA_M + 2;

    for (int dy = -1; dy <= 1; ++dy) {
        int y = iy + dy;
        bool v_y = (y >= 0 && y < ny);
        for (int dx = -1; dx <= 1; ++dx) {
            int x = ix + dx;
            bool v_x = (x >= 0 && x < nx);
            float vD = (v_x && v_y) ? d_Data[y*nx + x] : 0.0f;
            float vW = (v_x && v_y) ? d_Wavefield[y*nx + x] : 0.0f;
            sh_D[(sy+dy)*pitch + (sx+dx)] = vD;
            sh_W[(sy+dy)*pitch + (sx+dx)] = vW;
        }
    }
    __syncthreads();

    const float sigma = 1.0f;
    const float norm2D = 1.0f / (2.0f * M_PI * sigma * sigma);

    __shared__ half mid1[WMMA_M][WMMA_N];
    __shared__ half mid2[WMMA_M][WMMA_N];

    {
        float sum = 0.0f;
        float accD = 0.0f;
        float accW = 0.0f;
        for (int dx = -1; dx <= 1; ++dx) {
            float w =  norm2D * expf(-dx*dx/(2*sigma*sigma));
            sum += w;
            float vD = sh_D[sy*pitch + (sx+dx)];
            float vW = sh_W[sy*pitch + (sx+dx)];
            accD += w*vD;
            accW += w*vW;
        }
        float out = ((accD/sum) + (accW/sum)) * 0.5f;
        mid1[ty][tx] = __float2half(out);
    }

    {
        float sum = 0.0f;
        float accD = 0.0f;
        float accW = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {
            float w = norm2D * expf(-dy*dy/(2*sigma*sigma));
            sum += w;
            float vD = sh_D[(sy+dy)*pitch + sx];
            float vW = sh_W[(sy+dy)*pitch + sx];
            accD += w*vD;
            accW += w*vW;
        }
        float out = ((accD/sum) + (accW/sum)) * 0.5f;
        mid2[ty][tx] = __float2half(out);
    }

    __syncthreads();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c;
    fill_fragment(c, 0.0f);
    load_matrix_sync(a, &mid1[0][0], WMMA_K);
    load_matrix_sync(b, &mid2[0][0], WMMA_N);
    mma_sync(c, a, b, c);

    float out = c.x[ty*WMMA_N + tx];
    d_Result[lin] = out;
}
}


extern "C" {
__global__ void Combine_Misfit_Function_TensorCore_2D(
    float* __restrict__ d_dataObserved,
    float* __restrict__ d_dataSynthetic,
    size_t nx,
    size_t ny,
    float* __restrict__ d_misfit
)
{
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int ix = tile_x * WMMA_M;
    int iy = tile_y * WMMA_N;

    if (ix + WMMA_M > nx || iy + WMMA_N > ny) return;

    __shared__ half tile_obs[WMMA_M * WMMA_K];
    __shared__ half tile_syn[WMMA_K * WMMA_N];

    for (int i = 0; i < WMMA_M; i++) {
        for (int j = 0; j < WMMA_N; j++) {
            int idx = (iy + j) * nx + (ix + i);
            half h_obs = __float2half(d_dataObserved[idx]);
            half h_syn = __float2half(d_dataSynthetic[idx]);
            tile_obs[j * WMMA_M + i] = h_obs;
            tile_syn[j * WMMA_M + i] = h_syn;
        }
    }
    __syncthreads();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_obs;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_syn;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_dot;

    load_matrix_sync(frag_obs, tile_obs, WMMA_M);
    load_matrix_sync(frag_syn, tile_syn, WMMA_M);
    fill_fragment(frag_dot, 0.0f);

    mma_sync(frag_dot, frag_obs, frag_syn, frag_dot);

    float dot_result[WMMA_M * WMMA_N];
    store_matrix_sync(dot_result, frag_dot, WMMA_M, mem_row_major);
    __syncthreads();

    float norm_obs = 0.0f;
    float norm_syn = 0.0f;
    float transport = 0.0f;
    float dot_total = 0.0f;

    for (int i = 0; i < WMMA_M * WMMA_N; i++) {
        float o = __half2float(tile_obs[i]);
        float s = __half2float(tile_syn[i]);
        norm_obs += o * o;
        norm_syn += s * s;
        transport += fabsf(o - s);
        dot_total += dot_result[i];
    }

    float norm_product = sqrtf(norm_obs) * sqrtf(norm_syn) + 1e-6f;
    float cross_corr = dot_total / norm_product;
    float combined = 0.5f * (1.0f - cross_corr) + 0.5f * (transport / (WMMA_M * WMMA_N));
    int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
    d_misfit[tile_id] = combined;
}
}


extern "C" {
__global__ void Adjoint_Wavefield_TensorCore_2D(
    float* __restrict__ d_wavefield,
    float* __restrict__ d_misfit,
    size_t nx,
    size_t nz,
    float* __restrict__ d_adjoint
)
{
    int bx = blockIdx.x * TILE_DIM_TC;
    int by = blockIdx.y * TILE_DIM_TC;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ix = bx + tx;
    int iy = by + ty;

    if (ix >= nx || iy >= ny) return;

    size_t idx = iy * nx + ix;

    int tid = ty * TILE_DIM_TC + tx;

    /// Flattening
    __shared__ half s_wavefield[TILE_DIM_TC * TILE_DIM_TC];
    __shared__ half s_misfit[TILE_DIM_TC * TILE_DIM_TC];
    __shared__ float s_adjoint[TILE_DIM_TC * TILE_DIM_TC];

    s_wavefield[tid] = __float2half(d_wavefield[idx]);
    s_misfit[tid] = __float2half(d_misfit[idx]);

    fragment<matrix_a, TILE_DIM_TC, TILE_DIM_TC, TILE_DIM_TC, half, row_major> A;
    fragment<matrix_b, TILE_DIM_TC, TILE_DIM_TC, TILE_DIM_TC, half, row_major> B;
    fragment<accumulator, TILE_DIM_TC, TILE_DIM_TC, TILE_DIM_TC, float> C;

    load_matrix_sync(A, s_wavefield, TILE_DIM_TC);
    load_matrix_sync(B, s_misfit, TILE_DIM_TC);

    fill_fragment(C, 0.0f);
    mma_sync(C, A, B, C);

    store_matrix_sync(s_adjoint, C, TILE_DIM_TC, mem_row_major);
    __syncthreads();

    float val = s_adjoint[tid];
    val = val / (sqrtf(val*val) + dumping_factor);
    d_adjoint[idx] = (fabsf(val) > 1e-4f) ? val : 0.0f;
}
}


extern "C" {
__global__ void Adaptive_Sparsity_Based_Gradient_TensorCore_2D(
    float* __restrict__ d_wavefield,
    float* __restrict__ d_adjoint,
    size_t nx,
    size_t nz,
    float* __restrict__ d_gradient
)
{
    int tx = threadIdx.x;
    int tz = threadIdx.y;
    int bx = blockIdx.x;
    int bz = blockIdx.y;

    size_t ix = bx * WMMA_M + tx;
    size_t iz = bz * WMMA_N + tz;

    int thread_linear = tz * WMMA_M + tx;
    int warpId = thread_linear >> 5;
    int laneId = thread_linear & 31;

    constexpr int MAX_WARPS = (WMMA_M * WMMA_N) / 32;

    __shared__ half s_wavefield[WMMA_M * WMMA_N];
    __shared__ half s_adjoint[WMMA_M * WMMA_N];
    __shared__ float warp_partial_sum[MAX_WARPS];

    if (thread_linear < MAX_WARPS * 32 && laneId == 0) {
        warp_partial_sum[warpId] = 0.0f;
    }
    __syncthreads();

    float wave_val = 0.0f;
    float adj_val = 0.0f;
    if (ix < nx && iz < nz) {
        size_t idx = ix + iz * nx;
        wave_val = d_wavefield[idx];
        adj_val = d_adjoint[idx];
    }

    s_wavefield[thread_linear] = __float2half(wave_val);
    s_adjoint[thread_linear] = __float2half(adj_val);
    __syncthreads();

    if (warpId < MAX_WARPS) {
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
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }

        if (laneId == 0) {
            warp_partial_sum[warpId] = local_sum;
        }
    }
    __syncthreads();

    float final_sum = 0.0f;
    if (tx == 0 && tz == 0) {
        for (int w = 0; w < MAX_WARPS; ++w) {
            final_sum += warp_partial_sum[w];
        }
    }
    __syncthreads();

    if (ix < nx && iz < nz && tx == 0 && tz == 0) {
        float grad = final_sum / float(WMMA_M * WMMA_N);
        size_t out_idx = ix + iz * nx;
        grad = (fabsf(grad) > sparsity_threshold) ? grad : 0.0f;
        d_gradient[out_idx] = grad;
    }
}
}


extern "C" {
__global__ void Compute_Residual_Tiled_2D(
    float* __restrict__ d_gradient,
    float* __restrict__ d_wavefield,
    size_t nx,
    size_t nz,
    float* __restrict__ d_residue
)
{
    size_t tile_x = blockIdx.x * TILE_DIM_TC;
    size_t tile_y = blockIdx.y * TILE_DIM_TC;

    __shared__ float sh_gradient[TILE_DIM_TC][TILE_DIM_TC];
    __shared__ float sh_wavefield[TILE_DIM_TC][TILE_DIM_TC];

    size_t local_x = threadIdx.x;
    size_t local_y = threadIdx.y;

    size_t global_x = tile_x + local_x;
    size_t global_y = tile_y + local_y;
    size_T idx = global_x + global_y * nx;

    if (global_x < nx && global_y < ny) {
        sh_gradient[local_y][local_x] = d_gradient[idx];
        sh_wavefield[local_y][local_x] = d_wavefield[idx];
    }
    else {
        sh_gradient[local_y][local_x] = 0.0f;
        sh_wavefield[local_y][local_x] = 0.0f;
    }
    __syncthreads();

    if (global_x < nx && global_y < ny) {
        float gradient_val = sh_gradient[local_y][local_x];
        float wavefield_val = sh_wavefield[local_y][local_x];
        float denom = wavefield_val + dumping_factor;
        float reciprocal = __frcp_rn(denom);
        reciprocal = reciprocal * (2.0f - denom * reciprocal);

        float res = (gradient_val - wavefield_val) * reciprocal;
        d_residue[idx] = res;
    }
}
}


extern "C" {
__global__ void Parallel_Reduction_for_Residual_Norm_TensorCore_2D(
    float* __restrict__ d_residue,
    size_t nx,
    size_t ny,
    float* __restrict__ d_norm
)
{
    const size_t total_size = nx * ny;
    const size_t tile_size = WMMA_M * WMMA_K;
    const size_t tileId = blockIdx.x;
    const size_t baseIdx = tileId * tile_size;

    extern __shared__ half shTile[];

    int tid = threadIdx.x;

    for (int i = tid; i < tile_size; i += blockDim.x) {
        size_t idx = baseIdx + i;
        float v = (idx < total_size) ? d_residue[idx] : 0.0f;
        shTile[i] = __float2half(v*v);
    }
    __syncthreads();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> aFrag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> bFrag;
    fragement<accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

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
__global__ void Update_Direction_Vector_2D(
    float* __restrict__ d_residue,
    float* __restrict__ beta,
    size_t nx,
    size_t nz,
    float* __restrict__ d_direction_vector
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_size = nx * nz;

    if (idx < total_size) {
        float r = d_residue[idx];
        float p_old = d_direction_vector[idx];
        float b = beta[idx];

        d_direction_vector[idx] = r + b * p_old;
    }
}
}


extern "C" {
__global__ void PINNs_Integration_TensorCore_3D(
    float* __restrict__ d_gradient,
    float* __restrict__ d_correction,
    float* __restrict__ d_direction_vector,
    float* __restrict__ weight,
    float* __restrict__ alpha,
    size_t nx,
    size_t ny,
    size_t nz,
    float* __restrict__ d_model_update
)
{
    size_t total_size = nx * ny * nz;
    size_t tile_id = blockIdx.x;
    size_t base_idx = tile_id * WMMA_M * WMMA_N;

    if (base_idx >= total_size) return;

    __shared__ half sh_G[WMMA_M * WMMA_K];
    __shared__ half sh_C[WMMA_M * WMMA_K];
    __shared__ half sh_W[WMMA_K * WMMA_N];

    __shared__ float sh_output[WMMA_M * WMMA_N];
    __shared__ float sh_alpha[WMMA_M * WMMA_N];

    int tid = threadIdx.x;

    for (int i = tid; i < WMMA_M * WMMA_K; i += blockDim.x) {
        int idx = base_idx + i;
        sh_G[i] = (idx < total_size) ? __float2half(d_gradient[idx]) : __float2half(0.0f);
        sh_C[i] = (idx < total_size) ? __float2half(d_correction[idx]) : __float2half(0.0f);
    }

    for (int i = tid; i < WMMA_K * WMMA_N; i += blockDim.x) {
        int idx = base_idx + i;
        sh_W[i] = (idx < total_size) ? __float2half(weight[idx]) : __float2half(0.0f);
    }

    for (int i = tid; i < WMMA_M * WMMA_N; i += blockDim.x) {
        int idx = base_idx + i;
        sh_alpha[i] = (idx < total_size) ? alpha[idx] : 0.0f;
    }

    __syncthreads();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_G, frag_C;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_W;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_GW, frag_CW;

    fill_fragment(frag_GW, 0.0f);
    fill_fragment(frag_CW, 0.0f);

    load_matrix_sync(frag_G, sh_G, WMMA_K);
    load_matrix_sync(frag_C, sh_C, WMMA_K);
    load_matrix_sync(frag_W, sh_W, WMMA_K);

    mma_sync(frag_GW, frag_G, frag_W, frag_GW);
    mma_sync(frag_CW, frag_C, frag_W, frag_CW);

    for (int i = tid; i < WMMA_M * WMMA_N; i += blockDim.x) {
        sh_output[i] = (frag_GW.x[i] + frag_CW.x[i]) * sh_alpha[i];
    }
    __syncthreads();

    for (int i = tid; i < WMMA_M * WMMA_N; i += blockDim.x) {
        int idx = base_idx + i;
        if (idx < total_size) {
            d_model_update[idx] = sh_output[i];
        }
    }
}
}


extern "C" {
__global__ void PINNs_Integration_TensorCore_2D(
    float* __restrict__ d_gradient,
    float* __restrict__ d_correction,
    float* __restrict__ d_direction_vector,
    float* __restrict__ weight,
    float* __restrict__ alpha,
    size_t nx,
    size_t nz,
    float* __restrict__ d_model_update
)
{
    size_t total_size = nx * nz;
    size_t tile_id = blockIdx.x;
    size_t base_idx = tile_id * WMMA_M * WMMA_N;

    if (base_idx >= total_size) return;

    __shared__ half sh_G[WMMA_M * WMMA_K];
    __shared__ half sh_C[WMMA_M * WMMA_K];
    __shared__ half sh_W[WMMA_K * WMMA_N];

    __shared__ float sh_output[WMMA_M * WMMA_N];
    __shared__ float sh_alpha[WMMA_M * WMMA_N];

    int tid = threadIdx.x;

    for (int i = tid; i < WMMA_M * WMMA_K; i += blockDim.x) {
        int idx = base_idx + i;
        sh_G[i] = (idx < total_size) ? __float2half(d_gradient[idx]) : __float2half(0.0f);
        sh_C[i] = (idx < total_size) ? __float2half(d_correction[idx]) : __float2half(0.0f);
    }

    for (int i = tid; i < WMMA_K * WMMA_N; i += blockDim.x) {
        int idx = base_idx + i;
        sh_W[i] = (idx < total_size) ? __float2half(weight[idx]) : __float2half(0.0f);
    }

    for (int i = tid; i < WMMA_M * WMMA_N; i += blockDim.x) {
        int idx = base_idx + i;
        sh_alpha[i] = (idx < total_size) ? alpha[idx] : 0.0f;
    }

    __syncthreads();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_G, frag_C;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> frag_W;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_GW, frag_CW;

    fill_fragment(frag_GW, 0.0f);
    fill_fragment(frag_CW, 0.0f);

    load_matrix_sync(frag_G, sh_G, WMMA_K);
    load_matrix_sync(frag_C, sh_C, WMMA_K);
    load_matrix_sync(frag_W, sh_W, WMMA_K);

    mma_sync(frag_GW, frag_G, frag_W, frag_GW);
    mma_sync(frag_CW, frag_C, frag_W, frag_CW);

    for (int i = tid; i < WMMA_M * WMMA_N; i += blockDim.x) {
        sh_output[i] = (frag_GW.x[i] + frag_CW.x[i]) * sh_alpha;
    }
    __syncthreads();

    for (int i = tid; i < WMMA_M * WMMA_N;  i += blockDim.x) {
        int idx = base_idx + i;
        if (idx < total_size) {
            d_model_update[idx] = sh_output[i];
        }
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





















