#include "kernels.cuh"

__constant__ OuterBcConfig d_outer_bc;

void upload_outer_bc(const OuterBcConfig& cfg)
{
    cudaMemcpyToSymbol(d_outer_bc, &cfg, sizeof(OuterBcConfig));
}

__device__ void zou_he_west(float* f_in, int y, float rho)
{
    float f0 = f_in[F_IDX(y, 0, 0)];
    float f2 = f_in[F_IDX(y, 0, 2)];
    float f3 = f_in[F_IDX(y, 0, 3)];
    float f4 = f_in[F_IDX(y, 0, 4)];
    float f6 = f_in[F_IDX(y, 0, 6)];
    float f7 = f_in[F_IDX(y, 0, 7)];
    float ux  = 1.f - (f0 + f2 + f4 + 2.f * (f3 + f6 + f7)) / rho;
    float uy  = (f3 + f6 - f4 - f7) / rho;
    f_in[F_IDX(y, 0, 1)] = f3 + (2.f / 3.f) * rho * ux;
    f_in[F_IDX(y, 0, 5)] = f7 + 0.5f * (f4 - f2) + (1.f / 6.f) * rho * ux + 0.5f * rho * uy;
    f_in[F_IDX(y, 0, 8)] = f6 + 0.5f * (f2 - f4) + (1.f / 6.f) * rho * ux - 0.5f * rho * uy;
}

__device__ void zou_he_east(float* f_in, int y, float rho)
{
    int xm = WIDTH - 1;
    float f0 = f_in[F_IDX(y, xm, 0)];
    float f1 = f_in[F_IDX(y, xm, 1)];
    float f3 = f_in[F_IDX(y, xm, 3)];
    float f4 = f_in[F_IDX(y, xm, 4)];
    float f5 = f_in[F_IDX(y, xm, 5)];
    float f8 = f_in[F_IDX(y, xm, 8)];
    float ux  = -1.f + (f0 + f1 + f3 + 2.f * (f4 + f5 + f8)) / rho;
    float uy  = (f4 + f5 - f3 - f8) / rho;
    f_in[F_IDX(y, xm, 2)] = f4 - (2.f / 3.f) * rho * ux;
    f_in[F_IDX(y, xm, 6)] = f8 + 0.5f * (f3 - f1) - (1.f / 6.f) * rho * ux + 0.5f * rho * uy;
    f_in[F_IDX(y, xm, 7)] = f5 + 0.5f * (f1 - f3) - (1.f / 6.f) * rho * ux - 0.5f * rho * uy;
}

// ── D2Q9 lattice constants in GPU constant memory ─────────────────────────────
//
// Dir  label  cx   cy
//  0   rest    0    0
//  1   E      +1    0
//  2   W      -1    0
//  3   S       0   +1   (cy>0 → higher row index, downward on screen)
//  4   N       0   -1
//  5   SE     +1   +1
//  6   SW     -1   +1
//  7   NW     -1   -1
//  8   NE     +1   -1

__constant__ int   d_cx [Q] = { 0,  1, -1,  0,  0,  1, -1, -1,  1 };
__constant__ int   d_cy [Q] = { 0,  0,  0,  1, -1,  1,  1, -1, -1 };
__constant__ float d_w  [Q] = { 4.f/9.f,
                                 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,
                                 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f };
__constant__ int   d_opp[Q] = { 0, 2, 1, 4, 3, 7, 8, 5, 6 };

// ── Kernels ───────────────────────────────────────────────────────────────────

__global__ void k_macroscopic(const float* __restrict__ f_in,
                               float* __restrict__ rho,
                               float* __restrict__ ux,
                               float* __restrict__ uy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float r = 0.f, vx = 0.f, vy = 0.f;
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        float fi = f_in[F_IDX(y, x, i)];
        r  += fi;
        vx += (float)d_cx[i] * fi;
        vy += (float)d_cy[i] * fi;
    }
    float inv_r       = (r > 1e-10f) ? 1.f / r : 0.f;
    rho[S_IDX(y, x)]  = r;
    ux [S_IDX(y, x)]  = vx * inv_r;
    uy [S_IDX(y, x)]  = vy * inv_r;
}

__global__ void k_collision(const float* __restrict__ f_in,
                             float* __restrict__ f_out,
                             const float* __restrict__ rho,
                             const float* __restrict__ ux,
                             const float* __restrict__ uy,
                             const bool* __restrict__ wall)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;
    if (wall[S_IDX(y, x)]) return;

    float r  = rho[S_IDX(y, x)];
    float vx = ux [S_IDX(y, x)];
    float vy = uy [S_IDX(y, x)];
    float u2 = vx * vx + vy * vy;

    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        float cu  = (float)d_cx[i] * vx + (float)d_cy[i] * vy;
        float feq = d_w[i] * r * (1.f + 3.f*cu + 4.5f*cu*cu - 1.5f*u2);
        f_out[F_IDX(y, x, i)] = f_in[F_IDX(y, x, i)] + (1.f / TAU) * (feq - f_in[F_IDX(y, x, i)]);
    }
}

__global__ void k_streaming(float* __restrict__ f_in,
                             const float* __restrict__ f_out,
                             const bool* __restrict__ wall)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        int sx = x - d_cx[i];
        int sy = y - d_cy[i];
        if (sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT
            && !wall[S_IDX(sy, sx)])
            f_in[F_IDX(y, x, i)] = f_out[F_IDX(sy, sx, i)];
    }
}

// Reflect links on fluid nodes that point into a wall (fixes internal barrier leak).
__global__ void k_wall_link_bounce_back(float* __restrict__ f_in,
                                         const float* __restrict__ f_out,
                                         const bool* __restrict__ wall)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;
    if (wall[S_IDX(y, x)]) return;

    #pragma unroll
    for (int i = 1; i < Q; ++i) {
        int sx = x - d_cx[i];
        int sy = y - d_cy[i];
        if (sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT
            && wall[S_IDX(sy, sx)])
            f_in[F_IDX(y, x, i)] = f_out[F_IDX(y, x, d_opp[i])];
    }
}

__global__ void k_wall_bounce_back(float* __restrict__ f_in,
                                    const float* __restrict__ f_out,
                                    const bool* __restrict__ wall)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;
    if (!wall[S_IDX(y, x)]) return;

    #pragma unroll
    for (int i = 1; i < Q; ++i)
        f_in[F_IDX(y, x, i)] = f_out[F_IDX(y, x, d_opp[i])];
}

// Outer BC driven by d_outer_bc (per-edge bounce-back or Zou-He density).
__global__ void k_outer_boundary(float* __restrict__ f_in,
                                  const float* __restrict__ f_out,
                                  const bool* __restrict__ wall)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    if (wall[S_IDX(y, x)]) return;

    const bool on_west  = (x == 0);
    const bool on_east  = (x == WIDTH - 1);
    const bool on_south = (y == 0);
    const bool on_north = (y == HEIGHT - 1);
    if (!on_west && !on_east && !on_south && !on_north) return;

    const bool zou_west = on_west
        && d_outer_bc.edge[0] == EdgeBcType::ZouHeRho;
    const bool zou_east = on_east
        && d_outer_bc.edge[1] == EdgeBcType::ZouHeRho;

    if (zou_west)
        zou_he_west(f_in, y, d_outer_bc.edge_rho[0]);
    if (zou_east)
        zou_he_east(f_in, y, d_outer_bc.edge_rho[1]);

    #pragma unroll
    for (int i = 1; i < Q; ++i) {
        if (zou_west && (i == 1 || i == 5 || i == 8)) continue;
        if (zou_east && (i == 2 || i == 6 || i == 7)) continue;
        int sx = x - d_cx[i];
        int sy = y - d_cy[i];
        if (sx < 0 || sx >= WIDTH || sy < 0 || sy >= HEIGHT)
            f_in[F_IDX(y, x, i)] = f_out[F_IDX(y, x, d_opp[i])];
    }
}
