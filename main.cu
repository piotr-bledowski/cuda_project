/*
 * LBM D2Q9 Fluid Simulation
 *
 * Two chambers separated by a vertical wall with a central hole.
 * Left chamber starts denser (rho=1.5), right is lighter (rho=0.7).
 * Press SPACE to open/close the hole.
 *
 * Build: see CMakeLists.txt
 */

#include <cuda_runtime.h>
#include <SDL3/SDL.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── Simulation parameters ────────────────────────────────────────────────────

#define WIDTH 100
#define HEIGHT 100
#define Q 9
#define CELL_SIZE 5
#define TAU 1.0f
#define RHO_LEFT 1.5f
#define RHO_RIGHT 0.5f
#define WALL_X (WIDTH / 4)
#define HOLE_START (HEIGHT / 3)
#define HOLE_END (2 * HEIGHT / 3)
#define TARGET_FPS 60

// Flat-array index macros
#define F_IDX(y, x, i) ((y) * WIDTH * Q + (x) * Q + (i))
#define S_IDX(y, x) ((y) * WIDTH + (x))

// ── D2Q9 lattice constants (GPU constant memory) ──────────────────────────────
//
// Dir  label  cx   cy
//  0   rest    0    0
//  1   E      +1    0
//  2   W      -1    0
//  3   S       0   +1   (cy>0 → higher row index, i.e. downward on screen)
//  4   N       0   -1
//  5   SE     +1   +1
//  6   SW     -1   +1
//  7   NW     -1   -1
//  8   NE     +1   -1

__constant__ int d_cx[Q] = {0, 1, -1, 0, 0, 1, -1, -1, 1};
__constant__ int d_cy[Q] = {0, 0, 0, 1, -1, 1, 1, -1, -1};
__constant__ float d_w[Q] = {4.f / 9.f,
                             1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f,
                             1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f};
__constant__ int d_opp[Q] = {0, 2, 1, 4, 3, 7, 8, 5, 6};

// ── CUDA kernels ──────────────────────────────────────────────────────────────

// Step 1 — compute macroscopic density ρ and velocity (ux, uy) from f
__global__ void k_macroscopic(const float *__restrict__ f_in,
                              float *__restrict__ rho,
                              float *__restrict__ ux,
                              float *__restrict__ uy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;

    float r = 0.f, vx = 0.f, vy = 0.f;
#pragma unroll
    for (int i = 0; i < Q; ++i)
    {
        float fi = f_in[F_IDX(y, x, i)];
        r += fi;
        vx += (float)d_cx[i] * fi;
        vy += (float)d_cy[i] * fi;
    }
    float inv_r = (r > 1e-10f) ? 1.f / r : 0.f;
    rho[S_IDX(y, x)] = r;
    ux[S_IDX(y, x)] = vx * inv_r;
    uy[S_IDX(y, x)] = vy * inv_r;
}

// Step 2 — BGK single-relaxation-time collision
__global__ void k_collision(const float *__restrict__ f_in,
                            float *__restrict__ f_out,
                            const float *__restrict__ rho,
                            const float *__restrict__ ux,
                            const float *__restrict__ uy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;

    float r = rho[S_IDX(y, x)];
    float vx = ux[S_IDX(y, x)];
    float vy = uy[S_IDX(y, x)];
    float u2 = vx * vx + vy * vy;

#pragma unroll
    for (int i = 0; i < Q; ++i)
    {
        float cu = (float)d_cx[i] * vx + (float)d_cy[i] * vy;
        float feq = d_w[i] * r * (1.f + 3.f * cu + 4.5f * cu * cu - 1.5f * u2);
        float fi = f_in[F_IDX(y, x, i)];
        f_out[F_IDX(y, x, i)] = fi + (1.f / TAU) * (feq - fi);
    }
}

// Step 3 — streaming: pull each f_i from the upstream neighbour
__global__ void k_streaming(float *__restrict__ f_in,
                            const float *__restrict__ f_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;

#pragma unroll
    for (int i = 0; i < Q; ++i)
    {
        int sx = (x - d_cx[i] + WIDTH) % WIDTH;
        int sy = (y - d_cy[i] + HEIGHT) % HEIGHT;
        f_in[F_IDX(y, x, i)] = f_out[F_IDX(sy, sx, i)];
    }
}

// Step 4 — full bounce-back on interior wall nodes
__global__ void k_wall_bounce_back(float *__restrict__ f_in,
                                   const float *__restrict__ f_out,
                                   const bool *__restrict__ wall)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;
    if (!wall[S_IDX(y, x)])
        return;

#pragma unroll
    for (int i = 1; i < Q; ++i)
        f_in[F_IDX(y, x, i)] = f_out[F_IDX(y, x, d_opp[i])];
}

// Step 5 — outer domain boundary conditions, matching project2.py exactly:
//   Left / Right walls: full bounce-back (both velocity components reverse)
//   Top  / Bottom walls: straight direction → full BB;
//                        diagonals → specular (only the normal cy component reverses)
//
// Note: at corner cells a single thread writes conflicting updates for the shared
// diagonal direction; the last if-branch wins (top/bottom beats left/right),
// identical to what the sequential Python code does.
__global__ void k_outer_boundary(float *__restrict__ f_in,
                                 const float *__restrict__ f_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;

    if (x == 0)
    {                                                 // left wall
        f_in[F_IDX(y, 0, 1)] = f_out[F_IDX(y, 0, 2)]; // E ← W
        f_in[F_IDX(y, 0, 5)] = f_out[F_IDX(y, 0, 7)]; // SE ← NW
        f_in[F_IDX(y, 0, 8)] = f_out[F_IDX(y, 0, 6)]; // NE ← SW
    }
    if (x == WIDTH - 1)
    { // right wall
        f_in[F_IDX(y, WIDTH - 1, 2)] = f_out[F_IDX(y, WIDTH - 1, 1)];
        f_in[F_IDX(y, WIDTH - 1, 7)] = f_out[F_IDX(y, WIDTH - 1, 5)];
        f_in[F_IDX(y, WIDTH - 1, 6)] = f_out[F_IDX(y, WIDTH - 1, 8)];
    }
    if (y == 0)
    {                                                 // top wall
        f_in[F_IDX(0, x, 3)] = f_out[F_IDX(0, x, 4)]; // S ← N
        f_in[F_IDX(0, x, 5)] = f_out[F_IDX(0, x, 8)]; // SE ← NE (specular)
        f_in[F_IDX(0, x, 6)] = f_out[F_IDX(0, x, 7)]; // SW ← NW (specular)
    }
    if (y == HEIGHT - 1)
    { // bottom wall
        f_in[F_IDX(HEIGHT - 1, x, 4)] = f_out[F_IDX(HEIGHT - 1, x, 3)];
        f_in[F_IDX(HEIGHT - 1, x, 8)] = f_out[F_IDX(HEIGHT - 1, x, 5)];
        f_in[F_IDX(HEIGHT - 1, x, 7)] = f_out[F_IDX(HEIGHT - 1, x, 6)];
    }
}

// Step 6 — reservoir: cells adjacent to the closed part of the barrier are
// kept at fixed equilibrium distributions to maintain the density gradient.
__global__ void k_reservoir(float *__restrict__ f_in,
                            float *__restrict__ rho,
                            bool hole_open)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= HEIGHT)
        return;

    bool in_hole = hole_open && (y >= HOLE_START && y <= HOLE_END);
    if (in_hole)
        return;

    if (WALL_X > 0)
    {
        int x = WALL_X - 1;
        rho[S_IDX(y, x)] = RHO_LEFT;
#pragma unroll
        for (int i = 0; i < Q; ++i)
            f_in[F_IDX(y, x, i)] = d_w[i] * RHO_LEFT;
    }
    if (WALL_X < WIDTH - 1)
    {
        int x = WALL_X + 1;
        rho[S_IDX(y, x)] = RHO_RIGHT;
#pragma unroll
        for (int i = 0; i < Q; ++i)
            f_in[F_IDX(y, x, i)] = d_w[i] * RHO_RIGHT;
    }
}

// ── CPU helpers ───────────────────────────────────────────────────────────────

static void rebuild_wall(bool *h_wall, bool *d_wall, bool hole_open)
{
    memset(h_wall, 0, HEIGHT * WIDTH * sizeof(bool));
    for (int y = 0; y < HEIGHT; ++y)
        h_wall[S_IDX(y, WALL_X)] = true;
    if (hole_open)
        for (int y = HOLE_START; y <= HOLE_END; ++y)
            h_wall[S_IDX(y, WALL_X)] = false;
    cudaMemcpy(d_wall, h_wall, HEIGHT * WIDTH * sizeof(bool),
               cudaMemcpyHostToDevice);
}

static inline Uint32 pack_argb(Uint8 r, Uint8 g, Uint8 b)
{
    return (0xFFu << 24) | ((Uint32)r << 16) | ((Uint32)g << 8) | b;
}

static Uint32 rho_color(float r)
{
    float t = (r - RHO_RIGHT) / (RHO_LEFT - RHO_RIGHT);
    if (t < 0.f)
        t = 0.f;
    else if (t > 1.f)
        t = 1.f;
    return pack_argb((Uint8)(255 * t), 0, (Uint8)(255 * (1.f - t)));
}

static Uint32 vel_color(float v)
{
    float n = v / 0.07f;
    if (n < -1.f)
        n = -1.f;
    else if (n > 1.f)
        n = 1.f;
    if (n >= 0.f)
        return pack_argb((Uint8)(200 * n), 0, 0);
    return pack_argb(0, 0, (Uint8)(200 * (-n)));
}

static void draw_frame(SDL_Renderer *ren, SDL_Texture *tex,
                       const float *h_rho,
                       const float *h_ux,
                       const float *h_uy,
                       const bool *h_wall)
{
    void *pixels;
    int pitch;
    SDL_LockTexture(tex, nullptr, &pixels, &pitch);
    int stride = pitch / (int)sizeof(Uint32);
    Uint32 *px = (Uint32 *)pixels;

    const Uint32 wall_c = pack_argb(255, 255, 255);

    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            bool is_wall = h_wall[S_IDX(y, x)];
            px[y * stride + x] = is_wall ? wall_c : rho_color(h_rho[S_IDX(y, x)]);
            px[y * stride + WIDTH + x] = is_wall ? wall_c : vel_color(h_ux[S_IDX(y, x)]);
            px[y * stride + 2 * WIDTH + x] = is_wall ? wall_c : vel_color(h_uy[S_IDX(y, x)]);
        }
    }
    SDL_UnlockTexture(tex);

    SDL_FRect dst = {0.f, 0.f, (float)(WIDTH * CELL_SIZE * 3), (float)(HEIGHT * CELL_SIZE)};
    SDL_RenderClear(ren);
    SDL_RenderTexture(ren, tex, nullptr, &dst);
    SDL_RenderPresent(ren);
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int /*argc*/, char * /*argv*/[])
{
    // ── SDL window setup ──────────────────────────────────────────────────────
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        fprintf(stderr, "SDL_Init error: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window *window = SDL_CreateWindow(
        "LBM CUDA  |  [SPACE] toggle hole  |  Density  |  Vel X  |  Vel Y",
        WIDTH * CELL_SIZE * 3, HEIGHT * CELL_SIZE, 0);
    SDL_Renderer *ren = SDL_CreateRenderer(window, NULL);
    SDL_SetRenderVSync(ren, 1);
    // Texture is WIDTH*3 wide: panel 0=density, 1=ux, 2=uy
    SDL_Texture *tex = SDL_CreateTexture(ren,
                                         SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
                                         WIDTH * 3, HEIGHT);

    // ── Device allocations ────────────────────────────────────────────────────
    const size_t f_sz = (size_t)HEIGHT * WIDTH * Q * sizeof(float);
    const size_t sc_sz = (size_t)HEIGHT * WIDTH * sizeof(float);
    const size_t msk_sz = (size_t)HEIGHT * WIDTH * sizeof(bool);

    float *d_f_in, *d_f_out, *d_rho, *d_ux, *d_uy;
    bool *d_wall;
    cudaMalloc(&d_f_in, f_sz);
    cudaMalloc(&d_f_out, f_sz);
    cudaMalloc(&d_rho, sc_sz);
    cudaMalloc(&d_ux, sc_sz);
    cudaMalloc(&d_uy, sc_sz);
    cudaMalloc(&d_wall, msk_sz);

    // ── Host allocations ──────────────────────────────────────────────────────
    std::vector<float> h_f(HEIGHT * WIDTH * Q, 0.f);
    std::vector<float> h_rho(HEIGHT * WIDTH, 0.f);
    std::vector<float> h_ux(HEIGHT * WIDTH, 0.f);
    std::vector<float> h_uy(HEIGHT * WIDTH, 0.f);
    bool *h_wall = new bool[HEIGHT * WIDTH]();

    // ── Initial conditions ────────────────────────────────────────────────────
    const float w0[Q] = {4.f / 9.f,
                         1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f,
                         1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f};
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
        {
            float r = (x < WALL_X) ? RHO_LEFT : RHO_RIGHT;
            h_rho[S_IDX(y, x)] = r;
            for (int i = 0; i < Q; ++i)
                h_f[F_IDX(y, x, i)] = w0[i] * r;
        }

    cudaMemcpy(d_f_in, h_f.data(), f_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, h_rho.data(), sc_sz, cudaMemcpyHostToDevice);
    cudaMemset(d_ux, 0, sc_sz);
    cudaMemset(d_uy, 0, sc_sz);

    bool hole_open = true;
    rebuild_wall(h_wall, d_wall, hole_open);

    // ── Launch configuration ──────────────────────────────────────────────────
    const dim3 block2d(16, 16);
    const dim3 grid2d((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    const dim3 block1d(256);
    const dim3 grid1d((HEIGHT + 255) / 256);

    // ── Main loop ─────────────────────────────────────────────────────────────
    const Uint64 ms_per_frame = 1000u / TARGET_FPS;
    bool running = true;

    while (running)
    {
        Uint64 t0 = SDL_GetTicks();

        SDL_Event ev;
        while (SDL_PollEvent(&ev))
        {
            if (ev.type == SDL_EVENT_QUIT)
                running = false;
            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_SPACE)
            {
                hole_open = !hole_open;
                rebuild_wall(h_wall, d_wall, hole_open);
            }
        }

        // Simulation step (mirrors project2.py loop order)
        k_macroscopic<<<grid2d, block2d>>>(d_f_in, d_rho, d_ux, d_uy);
        k_collision<<<grid2d, block2d>>>(d_f_in, d_f_out, d_rho, d_ux, d_uy);
        k_streaming<<<grid2d, block2d>>>(d_f_in, d_f_out);
        k_wall_bounce_back<<<grid2d, block2d>>>(d_f_in, d_f_out, d_wall);
        k_outer_boundary<<<grid2d, block2d>>>(d_f_in, d_f_out);
        k_reservoir<<<grid1d, block1d>>>(d_f_in, d_rho, hole_open);
        cudaDeviceSynchronize();

        // Copy scalars to host for rendering
        cudaMemcpy(h_rho.data(), d_rho, sc_sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ux.data(), d_ux, sc_sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uy.data(), d_uy, sc_sz, cudaMemcpyDeviceToHost);

        draw_frame(ren, tex, h_rho.data(), h_ux.data(), h_uy.data(), h_wall);

        Uint64 elapsed = SDL_GetTicks() - t0;
        if (elapsed < ms_per_frame)
            SDL_Delay((Uint32)(ms_per_frame - elapsed));
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_f_in);
    cudaFree(d_f_out);
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_wall);
    delete[] h_wall;
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
