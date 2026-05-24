/*
 * LBM D2Q9 CUDA Fluid Simulation
 *
 * Two chambers separated by a vertical wall with a central hole.
 * Left chamber starts denser (RHO_LEFT), right is lighter (RHO_RIGHT).
 * Press SPACE to open/close the hole.
 *
 * File layout:
 *   sim_params.h  — grid constants, index macros
 *   kernels.cuh/cu — CUDA kernels + rebuild_wall
 *   ui.h/cpp      — SDL window, rendering, stats overlay
 *   main.cu       — simulation loop (this file)
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <vector>
#include <algorithm>
#include "sim_params.h"
#include "kernels.cuh"
#include "ui.h"

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int /*argc*/, char * /*argv*/[])
{
    // ── SDL setup ─────────────────────────────────────────────────────────────
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window *window = ui_create_window();
    SDL_Renderer *ren = ui_create_renderer(window);
    SDL_Texture *tex = ui_create_sim_texture(ren);

    // ── GPU info ──────────────────────────────────────────────────────────────
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

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
    std::vector<float> h_rho(HEIGHT * WIDTH, 0.f);
    std::vector<float> h_ux(HEIGHT * WIDTH, 0.f);
    std::vector<float> h_uy(HEIGHT * WIDTH, 0.f);
    bool *h_wall = new bool[HEIGHT * WIDTH]();

    // ── Initial conditions ────────────────────────────────────────────────────
    const float w0[Q] = {4.f / 9.f,
                         1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f,
                         1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f};
    {
        std::vector<float> h_f(HEIGHT * WIDTH * Q);
        for (int y = 0; y < HEIGHT; ++y)
            for (int x = 0; x < WIDTH; ++x)
            {
                float r = (x < WALL_X) ? RHO_LEFT : RHO_RIGHT;
                h_rho[S_IDX(y, x)] = r;
                for (int i = 0; i < Q; ++i)
                    h_f[F_IDX(y, x, i)] = w0[i] * r;
            }
        cudaMemcpy(d_f_in, h_f.data(), f_sz, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_rho, h_rho.data(), sc_sz, cudaMemcpyHostToDevice);
    cudaMemset(d_ux, 0, sc_sz);
    cudaMemset(d_uy, 0, sc_sz);

    bool hole_open = true;
    rebuild_wall(h_wall, d_wall, hole_open);

    // ── Kernel launch config ──────────────────────────────────────────────────
    const dim3 block2d(16, 16);
    const dim3 grid2d((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // Theoretical occupancy for the most compute-heavy kernel
    int maxActiveBlocks = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, k_macroscopic, block2d.x * block2d.y, 0);
    float occupancy = 100.f * (maxActiveBlocks * block2d.x * block2d.y) / (float)devProp.maxThreadsPerMultiProcessor;

    // ── Stats struct ──────────────────────────────────────────────────────────
    FrameStats stats = {};
    stats.fps = (float)TARGET_FPS;
    stats.grid_x = (int)grid2d.x;
    stats.grid_y = (int)grid2d.y;
    stats.block_x = (int)block2d.x;
    stats.block_y = (int)block2d.y;
    stats.sm_count = devProp.multiProcessorCount;
    stats.occupancy_pct = occupancy;
    stats.nu = (TAU - 0.5f) / 3.f;
    snprintf(stats.gpu_name, sizeof(stats.gpu_name), "%s", devProp.name);

    // ── Main loop ─────────────────────────────────────────────────────────────
    const Uint64 ms_per_frame = 1000u / TARGET_FPS;
    Uint64 prev_ticks = SDL_GetTicks();
    bool running = true;

    while (running)
    {
        // Events
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

        // Simulation step
        k_macroscopic<<<grid2d, block2d>>>(d_f_in, d_rho, d_ux, d_uy);
        k_clamp_velocity<<<grid2d, block2d>>>(d_ux, d_uy); // keep |u| ≤ U_MAX
        k_collision<<<grid2d, block2d>>>(d_f_in, d_f_out, d_rho, d_ux, d_uy);
        k_streaming<<<grid2d, block2d>>>(d_f_in, d_f_out);
        k_wall_bounce_back<<<grid2d, block2d>>>(d_f_in, d_f_out, d_wall);
        k_outer_boundary<<<grid2d, block2d>>>(d_f_in, d_f_out);
        cudaDeviceSynchronize();

        // Copy scalars to host
        cudaMemcpy(h_rho.data(), d_rho, sc_sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ux.data(), d_ux, sc_sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uy.data(), d_uy, sc_sz, cudaMemcpyDeviceToHost);

        // Per-field min/max (skip wall cells)
        float rho_min = FLT_MAX, rho_max = -FLT_MAX;
        float ux_min = FLT_MAX, ux_max = -FLT_MAX;
        float uy_min = FLT_MAX, uy_max = -FLT_MAX;
        for (int i = 0; i < HEIGHT * WIDTH; ++i)
        {
            if (h_wall[i])
                continue;
            rho_min = std::min(rho_min, h_rho[i]);
            rho_max = std::max(rho_max, h_rho[i]);
            ux_min = std::min(ux_min, h_ux[i]);
            ux_max = std::max(ux_max, h_ux[i]);
            uy_min = std::min(uy_min, h_uy[i]);
            uy_max = std::max(uy_max, h_uy[i]);
        }
        stats.rho_min = rho_min;
        stats.rho_max = rho_max;
        stats.ux_min = ux_min;
        stats.ux_max = ux_max;
        stats.uy_min = uy_min;
        stats.uy_max = uy_max;
        stats.hole_open = hole_open;
        stats.step++;

        // FPS (exponential moving average)
        Uint64 now = SDL_GetTicks();
        Uint64 elapsed = now - prev_ticks;
        if (elapsed > 0)
            stats.fps = stats.fps * 0.9f + 0.1f * (1000.f / (float)elapsed);
        prev_ticks = now;

        ui_draw_frame(ren, tex, h_rho.data(), h_ux.data(), h_uy.data(),
                      h_wall, stats);

        Uint64 frame_ms = SDL_GetTicks() - now + elapsed;
        if (frame_ms < ms_per_frame)
            SDL_Delay((Uint32)(ms_per_frame - frame_ms));
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
