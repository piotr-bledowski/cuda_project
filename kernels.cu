#include "kernels.cuh"
#include <cstring>

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
                             const float* __restrict__ uy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

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

// ── Fused macroscopic + BGK collision ─────────────────────────────────────────
// Replaces the separate k_macroscopic + k_collision pair for the inner loop.
// ρ and u are computed entirely in registers — no global write of scalar fields.
__global__ void k_collide(const float* __restrict__ f_in,
                           float* __restrict__ f_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float fi[Q];
    float r = 0.f, vx = 0.f, vy = 0.f;
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        fi[i] = f_in[F_IDX(y, x, i)];
        r  += fi[i];
        vx += (float)d_cx[i] * fi[i];
        vy += (float)d_cy[i] * fi[i];
    }
    float inv_r = (r > 1e-10f) ? 1.f / r : 0.f;
    vx *= inv_r;
    vy *= inv_r;
    float u2    = vx * vx + vy * vy;
    float omega = 1.f / TAU;
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        float cu  = (float)d_cx[i] * vx + (float)d_cy[i] * vy;
        float feq = d_w[i] * r * (1.f + 3.f*cu + 4.5f*cu*cu - 1.5f*u2);
        f_out[F_IDX(y, x, i)] = fi[i] + omega * (feq - fi[i]);
    }
}

// ── Pull streaming with shared-memory tile ────────────────────────────────────
// Each block cooperatively loads a (TILE_DIM × TILE_DIM) patch of f_out
// (post-collision) into shared memory, then every thread pulls all 9 directions
// from shared memory instead of making 9 separate global reads.
//
// For a 16×16 block the tile is 18×18.  With D2Q9 offsets ∈ {-1,0,+1}², the
// source of every direction always falls inside the 18×18 tile, so no global
// fallback is needed for interior cells.  Out-of-domain directions are skipped
// (same behaviour as k_streaming); k_outer_boundary fills them afterwards.
//
// Shared memory: 18 × 18 × 9 × 4 B = 11,664 B ≈ 11.4 KB per block.
// Bank access: stride between consecutive threads = Q = 9, which is coprime
// with 32 → no bank conflicts in either half-warp.
__global__ void k_streaming_shmem(float* __restrict__ f_in,
                                   const float* __restrict__ f_out)
{
    const int tx = threadIdx.x;   // 0 .. BLOCK_DIM-1
    const int ty = threadIdx.y;
    const int x  = blockIdx.x * BLOCK_DIM + tx;
    const int y  = blockIdx.y * BLOCK_DIM + ty;

    // Tile origin in global coords (one cell to the top-left of the block).
    const int x_base = blockIdx.x * BLOCK_DIM - 1;
    const int y_base = blockIdx.y * BLOCK_DIM - 1;

    __shared__ float s[TILE_DIM][TILE_DIM][Q];  // 18 × 18 × 9

    // ── Cooperative tile load ────────────────────────────────────────────────
    // 256 threads fill 18*18 = 324 cells; each thread handles 1 or 2 cells.
    const int tid    = ty * BLOCK_DIM + tx;
    const int n_cell = TILE_DIM * TILE_DIM;   // 324
    for (int idx = tid; idx < n_cell; idx += BLOCK_DIM * BLOCK_DIM) {
        int hy = idx / TILE_DIM;
        int hx = idx % TILE_DIM;
        int gx = x_base + hx;
        int gy = y_base + hy;
        if (gx >= 0 && gx < WIDTH && gy >= 0 && gy < HEIGHT) {
            #pragma unroll
            for (int i = 0; i < Q; ++i)
                s[hy][hx][i] = f_out[F_IDX(gy, gx, i)];
        } else {
            #pragma unroll
            for (int i = 0; i < Q; ++i)
                s[hy][hx][i] = 0.f;
        }
    }
    __syncthreads();

    if (x >= WIDTH || y >= HEIGHT) return;

    // ── Pull streaming from shared memory ────────────────────────────────────
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        int gsx = x - d_cx[i];
        int gsy = y - d_cy[i];
        if (gsx < 0 || gsx >= WIDTH || gsy < 0 || gsy >= HEIGHT)
            continue;   // out-of-domain: k_outer_boundary handles this direction
        // Source local coords in the tile (always 0..TILE_DIM-1 for BLOCK_DIM=16)
        f_in[F_IDX(y, x, i)] = s[ty + 1 - d_cy[i]][tx + 1 - d_cx[i]][i];
    }
}

__global__ void k_streaming(float* __restrict__ f_in,
                             const float* __restrict__ f_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        int sx = x - d_cx[i];
        int sy = y - d_cy[i];
        // Non-periodic: only pull from interior neighbors; boundary
        // directions are filled by k_outer_boundary (prevents left↔right leak).
        if (sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT)
            f_in[F_IDX(y, x, i)] = f_out[F_IDX(sy, sx, i)];
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

// Outer BC: closed container uses bounce-back for every direction that
// would stream in from outside the domain (handles edges and corners uniformly).
// Open mode keeps Zou-He reservoirs on west/east.
__global__ void k_outer_boundary(float* __restrict__ f_in,
                                  const float* __restrict__ f_out,
                                  const bool* __restrict__ wall,
                                  int bc_mode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    const bool is_wall = wall[S_IDX(y, x)];
    if (is_wall) return;

    const bool on_west  = (x == 0);
    const bool on_east  = (x == WIDTH - 1);
    const bool on_south = (y == 0);
    const bool on_north = (y == HEIGHT - 1);
    if (!on_west && !on_east && !on_south && !on_north) return;

    if (bc_mode == BC_OPEN && (on_west || on_east)) {
        if (on_west) {
            float f0 = f_in[F_IDX(y, 0, 0)];
            float f2 = f_in[F_IDX(y, 0, 2)];
            float f3 = f_in[F_IDX(y, 0, 3)];
            float f4 = f_in[F_IDX(y, 0, 4)];
            float f6 = f_in[F_IDX(y, 0, 6)];
            float f7 = f_in[F_IDX(y, 0, 7)];
            float rho = RHO_LEFT;
            float ux  = 1.f - (f0 + f2 + f4 + 2.f * (f3 + f6 + f7)) / rho;
            float uy  = (f3 + f6 - f4 - f7) / rho;
            f_in[F_IDX(y, 0, 1)] = f3 + (2.f / 3.f) * rho * ux;
            f_in[F_IDX(y, 0, 5)] = f7 + 0.5f * (f4 - f2) + (1.f / 6.f) * rho * ux + 0.5f * rho * uy;
            f_in[F_IDX(y, 0, 8)] = f6 + 0.5f * (f2 - f4) + (1.f / 6.f) * rho * ux - 0.5f * rho * uy;
        }
        if (on_east) {
            int xm = WIDTH - 1;
            float f0 = f_in[F_IDX(y, xm, 0)];
            float f1 = f_in[F_IDX(y, xm, 1)];
            float f3 = f_in[F_IDX(y, xm, 3)];
            float f4 = f_in[F_IDX(y, xm, 4)];
            float f5 = f_in[F_IDX(y, xm, 5)];
            float f8 = f_in[F_IDX(y, xm, 8)];
            float rho = RHO_RIGHT;
            float ux  = -1.f + (f0 + f1 + f3 + 2.f * (f4 + f5 + f8)) / rho;
            float uy  = (f4 + f5 - f3 - f8) / rho;
            f_in[F_IDX(y, xm, 2)] = f4 - (2.f / 3.f) * rho * ux;
            f_in[F_IDX(y, xm, 6)] = f8 + 0.5f * (f3 - f1) - (1.f / 6.f) * rho * ux + 0.5f * rho * uy;
            f_in[F_IDX(y, xm, 7)] = f5 + 0.5f * (f1 - f3) - (1.f / 6.f) * rho * ux - 0.5f * rho * uy;
        }
        // Bounce-back for remaining exterior directions (corners, north/south)
        #pragma unroll
        for (int i = 1; i < Q; ++i) {
            if (on_west  && (i == 1 || i == 5 || i == 8)) continue;
            if (on_east  && (i == 2 || i == 6 || i == 7)) continue;
            int sx = x - d_cx[i];
            int sy = y - d_cy[i];
            if (sx < 0 || sx >= WIDTH || sy < 0 || sy >= HEIGHT)
                f_in[F_IDX(y, x, i)] = f_out[F_IDX(y, x, d_opp[i])];
        }
        return;
    }

    // Closed container: bounce-back every population arriving from outside
    #pragma unroll
    for (int i = 1; i < Q; ++i) {
        int sx = x - d_cx[i];
        int sy = y - d_cy[i];
        if (sx < 0 || sx >= WIDTH || sy < 0 || sy >= HEIGHT)
            f_in[F_IDX(y, x, i)] = f_out[F_IDX(y, x, d_opp[i])];
    }
}

// ── CPU helper ────────────────────────────────────────────────────────────────

void rebuild_wall(bool* h_wall, bool* d_wall, bool hole_open)
{
    memset(h_wall, 0, HEIGHT * WIDTH * sizeof(bool));
    for (int y = 0; y < HEIGHT; ++y)
        h_wall[S_IDX(y, WALL_X)] = true;
    if (hole_open)
        for (int y = HOLE_START; y <= HOLE_END; ++y)
            h_wall[S_IDX(y, WALL_X)] = false;
    cudaMemcpy(d_wall, h_wall, HEIGHT * WIDTH * sizeof(bool), cudaMemcpyHostToDevice);
}
