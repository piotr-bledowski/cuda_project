/*
 * LBM D2Q9 CUDA Fluid Simulation
 *
 * Scenario-driven initial conditions, walls, and outer BC.
 * Tab — cycle scenario | R — reset | SPACE — scenario action
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <vector>
#include <algorithm>
#include "sim_params.h"
#include "sim_config.h"
#include "kernels.cuh"
#include "scenarios.h"
#include "ui.h"


static void capture_viz_bounds(const float* h_rho,
                               const float* h_ux,
                               const float* h_uy,
                               const bool*  h_wall,
                               FrameStats&  stats)
{
    float rho_min = FLT_MAX, rho_max = -FLT_MAX;
    float vel_max = 0.f;
    for (int i = 0; i < HEIGHT * WIDTH; ++i)
    {
        if (h_wall[i])
            continue;
        rho_min = std::min(rho_min, h_rho[i]);
        rho_max = std::max(rho_max, h_rho[i]);
        vel_max = std::max(vel_max, std::fabs(h_ux[i]));
        vel_max = std::max(vel_max, std::fabs(h_uy[i]));
    }
    stats.viz_rho_min = rho_min;
    stats.viz_rho_max = rho_max;
    stats.viz_vel_scale = std::max(vel_max, U_MAX * 0.05f);
}

static void apply_scenario(SimScenario& scenario,
                           float* h_rho, float* h_ux, float* h_uy, float* h_f,
                           bool* h_wall,
                           float* d_f_in, float* d_rho, float* d_ux, float* d_uy,
                           bool* d_wall,
                           size_t f_sz, size_t sc_sz, size_t msk_sz,
                           FrameStats& stats)
{
    scenario.reset_walls(h_wall, true);
    scenario.reset_fields(h_rho, h_ux, h_uy, h_f);
    upload_outer_bc(scenario.outer_bc());

    cudaMemcpy(d_f_in, h_f, f_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, h_rho, sc_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ux, h_ux, sc_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy, sc_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wall, h_wall, msk_sz, cudaMemcpyHostToDevice);

    snprintf(stats.scenario_name, sizeof(stats.scenario_name), "%s", scenario.name());
    stats.rho_ref_high = scenario.display_rho_high();
    stats.rho_ref_low  = scenario.display_rho_low();
    stats.hole_open = scenario.supports_internal_wall() && scenario.chambers.hole_open;
    stats.scenario_has_partition = scenario.supports_internal_wall();
    stats.bc_closed = true;
    capture_viz_bounds(h_rho, h_ux, h_uy, h_wall, stats);
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int /*argc*/, char* /*argv*/[])
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window* window = ui_create_window();
    SDL_Renderer* ren = ui_create_renderer(window);
    SDL_Texture* tex = ui_create_sim_texture(ren);

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    const size_t f_sz = (size_t)HEIGHT * WIDTH * Q * sizeof(float);
    const size_t sc_sz = (size_t)HEIGHT * WIDTH * sizeof(float);
    const size_t msk_sz = (size_t)HEIGHT * WIDTH * sizeof(bool);

    float *d_f_in, *d_f_out, *d_rho, *d_ux, *d_uy;
    bool* d_wall;
    cudaMalloc(&d_f_in, f_sz);
    cudaMalloc(&d_f_out, f_sz);
    cudaMalloc(&d_rho, sc_sz);
    cudaMalloc(&d_ux, sc_sz);
    cudaMalloc(&d_uy, sc_sz);
    cudaMalloc(&d_wall, msk_sz);

    std::vector<float> h_rho(HEIGHT * WIDTH, 0.f);
    std::vector<float> h_ux(HEIGHT * WIDTH, 0.f);
    std::vector<float> h_uy(HEIGHT * WIDTH, 0.f);
    std::vector<float> h_f(HEIGHT * WIDTH * Q);
    bool* h_wall = new bool[HEIGHT * WIDTH]();

    SimScenario scenario;
    scenario_set_id(scenario, ScenarioId::Chambers);

    FrameStats stats = {};
    stats.fps = (float)TARGET_FPS;

    apply_scenario(scenario, h_rho.data(), h_ux.data(), h_uy.data(), h_f.data(),
                   h_wall, d_f_in, d_rho, d_ux, d_uy, d_wall,
                   f_sz, sc_sz, msk_sz, stats);

    const dim3 block2d(16, 16);
    const dim3 grid2d((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    int maxActiveBlocks = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, k_macroscopic, block2d.x * block2d.y, 0);
    stats.grid_x = (int)grid2d.x;
    stats.grid_y = (int)grid2d.y;
    stats.block_x = (int)block2d.x;
    stats.block_y = (int)block2d.y;
    stats.sm_count = devProp.multiProcessorCount;
    stats.occupancy_pct = 100.f * (maxActiveBlocks * block2d.x * block2d.y)
        / (float)devProp.maxThreadsPerMultiProcessor;
    stats.nu = (TAU - 0.5f) / 3.f;
    snprintf(stats.gpu_name, sizeof(stats.gpu_name), "%s", devProp.name);

    const Uint64 ms_per_frame = 1000u / TARGET_FPS;
    Uint64 prev_ticks = SDL_GetTicks();
    bool running = true;
    bool is_drawing = false;
    bool is_erasing = false;

    while (running)
    {
        bool wall_updated = false;
        bool fields_updated = false;

        SDL_Event ev;
        while (SDL_PollEvent(&ev))
        {
            if (ev.type == SDL_EVENT_QUIT)
                running = false;

            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_R)
            {
                scenario.bubble.burst = false;
                apply_scenario(scenario, h_rho.data(), h_ux.data(), h_uy.data(),
                               h_f.data(), h_wall, d_f_in, d_rho, d_ux, d_uy, d_wall,
                               f_sz, sc_sz, msk_sz, stats);
            }

            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_TAB)
            {
                scenario_cycle(scenario);
                apply_scenario(scenario, h_rho.data(), h_ux.data(), h_uy.data(),
                               h_f.data(), h_wall, d_f_in, d_rho, d_ux, d_uy, d_wall,
                               f_sz, sc_sz, msk_sz, stats);
            }

            if (ev.type == SDL_EVENT_KEY_DOWN)
            {
                if (scenario.on_key(ev.key.key, h_wall))
                {
                    if (scenario.id == ScenarioId::Bubble && scenario.bubble.burst)
                    {
                        scenario.reset_fields(h_rho.data(), h_ux.data(), h_uy.data(),
                                              h_f.data());
                        cudaMemcpy(d_f_in, h_f.data(), f_sz, cudaMemcpyHostToDevice);
                        cudaMemcpy(d_rho, h_rho.data(), sc_sz, cudaMemcpyHostToDevice);
                        cudaMemset(d_ux, 0, sc_sz);
                        cudaMemset(d_uy, 0, sc_sz);
                        fields_updated = true;
                    }
                    else
                        wall_updated = true;
                    stats.hole_open = scenario.supports_internal_wall()
                        && scenario.chambers.hole_open;
                }
            }

            if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN)
            {
                if (ev.button.button == SDL_BUTTON_LEFT) is_drawing = true;
                if (ev.button.button == SDL_BUTTON_RIGHT) is_erasing = true;
            }
            else if (ev.type == SDL_EVENT_MOUSE_BUTTON_UP)
            {
                if (ev.button.button == SDL_BUTTON_LEFT) is_drawing = false;
                if (ev.button.button == SDL_BUTTON_RIGHT) is_erasing = false;
            }
        }

        if (is_drawing || is_erasing)
        {
            float mouse_x, mouse_y;
            SDL_GetMouseState(&mouse_x, &mouse_y);
            int grid_x = (int)mouse_x / CELL_SIZE;
            int grid_y = (int)mouse_y / CELL_SIZE;

            if (grid_x >= 0 && grid_x < WIDTH && grid_y >= 0 && grid_y < HEIGHT)
            {
                int idx = S_IDX(grid_y, grid_x);
                if (is_drawing && !h_wall[idx])
                {
                    h_wall[idx] = true;
                    wall_updated = true;
                }
                else if (is_erasing && h_wall[idx])
                {
                    h_wall[idx] = false;
                    wall_updated = true;
                }
            }
        }

        if (wall_updated)
            cudaMemcpy(d_wall, h_wall, msk_sz, cudaMemcpyHostToDevice);

        if (fields_updated)
        {
            cudaMemcpy(d_ux, h_ux.data(), sc_sz, cudaMemcpyHostToDevice);
            cudaMemcpy(d_uy, h_uy.data(), sc_sz, cudaMemcpyHostToDevice);
        }

        k_macroscopic<<<grid2d, block2d>>>(d_f_in, d_rho, d_ux, d_uy);
        k_collision<<<grid2d, block2d>>>(d_f_in, d_f_out, d_rho, d_ux, d_uy, d_wall);
        k_streaming<<<grid2d, block2d>>>(d_f_in, d_f_out, d_wall);
        k_wall_link_bounce_back<<<grid2d, block2d>>>(d_f_in, d_f_out, d_wall);
        k_wall_bounce_back<<<grid2d, block2d>>>(d_f_in, d_f_out, d_wall);
        k_outer_boundary<<<grid2d, block2d>>>(d_f_in, d_f_out, d_wall);
        cudaDeviceSynchronize();

        cudaMemcpy(h_rho.data(), d_rho, sc_sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ux.data(), d_ux, sc_sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_uy.data(), d_uy, sc_sz, cudaMemcpyDeviceToHost);

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
        stats.step++;

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
