/*
 * LBM D2Q9 CUDA Fluid Simulation
 *
 * Two chambers separated by a vertical wall with a central hole.
 * Left chamber starts denser (RHO_LEFT), right is lighter (RHO_RIGHT).
 * Press SPACE to open/close the hole.
 *
 * File layout:
 * sim_params.h   — grid constants, index macros
 * kernels.cuh/cu — CUDA kernels + rebuild_wall
 * ui.h/cpp       — SDL window, rendering, stats overlay
 * main.cu        — simulation loop, CLI (-n steps per frame)
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
#include "sim_params.h"
#include "kernels.cuh"
#include "ui.h"

static void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s [options] [steps]\n"
        "\n"
        "  -n, --steps N   LBM substeps per rendered frame (default %d)\n"
        "  -h, --help      Show this help\n"
        "\n"
        "Examples:\n"
        "  %s              %d steps per frame\n"
        "  %s 32           32 steps per frame\n"
        "  %s -n 64\n"
        "\n",
        prog, DEFAULT_STEPS_PER_FRAME,
        prog, DEFAULT_STEPS_PER_FRAME,
        prog, prog);
}

static int parse_steps_per_frame(int argc, char** argv)
{
    int steps = DEFAULT_STEPS_PER_FRAME;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--steps") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", argv[i]);
                print_usage(argv[0]);
                std::exit(1);
            }
            char* end = nullptr;
            long v = std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 1) {
                fprintf(stderr, "error: invalid step count '%s'\n", argv[i]);
                std::exit(1);
            }
            steps = (int)v;
            continue;
        }
        if (argv[i][0] == '-') {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            std::exit(1);
        }
        char* end = nullptr;
        long v = std::strtol(argv[i], &end, 10);
        if (end == argv[i] || *end != '\0' || v < 1) {
            fprintf(stderr, "error: invalid step count '%s'\n", argv[i]);
            print_usage(argv[0]);
            std::exit(1);
        }
        steps = (int)v;
    }
    return steps;
}


static void capture_viz_bounds(const float* h_rho,
    const float* h_ux,
    const float* h_uy,
    const bool* h_wall,
    FrameStats& stats)
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

    if (rho_max - rho_min < 0.01f) {
        rho_min = RHO_RIGHT; 
        rho_max = RHO_LEFT;
    }

    stats.viz_rho_min = rho_min;
    stats.viz_rho_max = rho_max;
    stats.viz_vel_scale = std::max(vel_max, U_MAX * 0.05f);
}

void update_central_wall(bool* h_wall, bool* d_wall, int num_holes, bool is_open, bool wipe_board) {

    if (wipe_board) {
        std::fill(h_wall, h_wall + (HEIGHT * WIDTH), false);
    }

    for (int y = 0; y < HEIGHT; ++y) {
        h_wall[S_IDX(y, WALL_X)] = true;
    }

    if (is_open) {
        // Scale: 1 = 30px, 2 = 15px, 3 = 10px, 4 = 7px
        int hole_size = 30 / num_holes;
        int total_hole_space = num_holes * hole_size;
        int remaining_wall = HEIGHT - total_hole_space;
        int spacing = remaining_wall / (num_holes + 1);

        int current_y = spacing;
        for (int i = 0; i < num_holes; ++i) {

            for (int j = 0; j < hole_size && current_y < HEIGHT; ++j) {
                h_wall[S_IDX(current_y, WALL_X)] = false;
                current_y++;
            }
            current_y += spacing;
        }
    }

    size_t msk_sz = (size_t)HEIGHT * WIDTH * sizeof(bool);
    cudaMemcpy(d_wall, h_wall, msk_sz, cudaMemcpyHostToDevice);
}

void load_scenario(int scenario, float* h_rho, bool* h_wall, float* d_f_in, float* d_f_out, float* d_rho, float* d_ux, float* d_uy, bool* d_wall, bool reset_walls) {
    const size_t f_sz = (size_t)HEIGHT * WIDTH * Q * sizeof(float);
    const size_t sc_sz = (size_t)HEIGHT * WIDTH * sizeof(float);
    const size_t msk_sz = (size_t)HEIGHT * WIDTH * sizeof(bool);

    std::vector<float> h_f(HEIGHT * WIDTH * Q);
    const float w0[Q] = { 4.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f };

    if (reset_walls) {
        std::fill(h_wall, h_wall + (HEIGHT * WIDTH), false);
    }

    int cx = WIDTH / 2;
    int cy = HEIGHT / 2;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            float r = RHO_RIGHT;

            if (scenario == 1) {
                r = (x < WALL_X) ? RHO_LEFT : RHO_RIGHT;
            }
            else if (scenario == 2) {
                int dx = x - cx;
                int dy = y - cy;
                int dist_sq = dx * dx + dy * dy;
                int radius = HEIGHT / 4;

                if (dist_sq < radius * radius) {
                    r = RHO_LEFT;
                }
            }
            else if (scenario == 3) {
                if ((x < cx && y < cy) || (x > cx && y > cy)) {
                    r = RHO_LEFT;
                }
            }
            else if (scenario == 4) {

            }

            // Aplikowanie gęstości
            h_rho[S_IDX(y, x)] = r;
            for (int i = 0; i < Q; ++i) {
                h_f[F_IDX(y, x, i)] = w0[i] * r;
            }
        }
    }

    // Wysyłanie wszystkiego na GPU
    cudaMemcpy(d_f_in, h_f.data(), f_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_out, h_f.data(), f_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, h_rho, sc_sz, cudaMemcpyHostToDevice);
    cudaMemset(d_ux, 0, sc_sz);
    cudaMemset(d_uy, 0, sc_sz);
    cudaMemcpy(d_wall, h_wall, msk_sz, cudaMemcpyHostToDevice);
}

void flood_fill_density(int start_x, int start_y, float target_rho, float* h_rho, bool* h_wall, float* d_f_in, float* d_f_out, float* d_rho) {
    if (h_wall[S_IDX(start_y, start_x)]) return; // Nie wylewamy wody wewnątrz betonu!

    float original_rho = h_rho[S_IDX(start_y, start_x)];
    // Zabezpieczenie, by nie wlewać tego samego do tego samego (uniknięcie pętli)
    if (std::abs(original_rho - target_rho) < 0.01f) return;

    std::queue<std::pair<int, int>> q;
    q.push({ start_x, start_y });

    int dx[] = { 0, 0, -1, 1 };
    int dy[] = { -1, 1, 0, 0 };

    // Pobieramy cały obecny płyn z GPU do RAMu, żeby nie skasować fal na zewnątrz naszego naczynia
    const size_t f_sz = (size_t)HEIGHT * WIDTH * Q * sizeof(float);
    std::vector<float> h_f(HEIGHT * WIDTH * Q);
    cudaMemcpy(h_f.data(), d_f_in, f_sz, cudaMemcpyDeviceToHost);

    const float w0[Q] = { 4.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f };

    std::vector<bool> visited(HEIGHT * WIDTH, false);
    visited[S_IDX(start_y, start_x)] = true;

    // Klasyczne rozlewanie (BFS)
    while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.pop();

        int idx = S_IDX(cy, cx);
        h_rho[idx] = target_rho; // Zmieniamy gęstość na nową

        // Restartujemy fizykę cząsteczek TYLKO w tym jednym zamalowanym pikselu (prędkość = 0)
        for (int i = 0; i < Q; ++i) {
            h_f[F_IDX(cy, cx, i)] = w0[i] * target_rho;
        }

        // Sprawdzamy 4 sąsiadów
        for (int dir = 0; dir < 4; ++dir) {
            int nx = cx + dx[dir];
            int ny = cy + dy[dir];

            if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT) {
                int n_idx = S_IDX(ny, nx);
                // Jeśli to nie mur, nie byliśmy tu i ma starą gęstość - lejemy dalej!
                if (!h_wall[n_idx] && !visited[n_idx] && std::abs(h_rho[n_idx] - original_rho) < 0.01f) {
                    visited[n_idx] = true;
                    q.push({ nx, ny });
                }
            }
        }
    }

    // Odsyłamy zamalowany płyn z powrotem na kartę graficzną
    cudaMemcpy(d_f_in, h_f.data(), f_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_out, h_f.data(), f_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, h_rho, (size_t)HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
}


// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    const int steps_per_frame = parse_steps_per_frame(argc, argv);
    fprintf(stderr, "LBM: %d step(s) per frame (use -h for help)\n", steps_per_frame);

    // ── SDL setup ─────────────────────────────────────────────────────────────
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window* window = ui_create_window();
    SDL_Renderer* ren = ui_create_renderer(window);
    SDL_Texture* tex = ui_create_sim_texture(ren);

    // ── GPU info ──────────────────────────────────────────────────────────────
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    // ── Device allocations ────────────────────────────────────────────────────
    const size_t f_sz = (size_t)HEIGHT * WIDTH * Q * sizeof(float);
    const size_t sc_sz = (size_t)HEIGHT * WIDTH * sizeof(float);
    const size_t msk_sz = (size_t)HEIGHT * WIDTH * sizeof(bool);

    float* d_f_in, * d_f_out, * d_rho, * d_ux, * d_uy;
    bool* d_wall;
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
    bool* h_wall = new bool[HEIGHT * WIDTH]();

    // ── Initial conditions ────────────────────────────────────────────────────
    bool hole_open = true;
    int  bc_mode = BC_CLOSED;
    int current_holes = 1;
    int current_scenario = 1; // 1=Komory, 2=Kropla, 3=Szachownica, 4=Sandbox

    load_scenario(current_scenario, h_rho.data(), h_wall, d_f_in, d_f_out, d_rho, d_ux, d_uy, d_wall, true);
    if (current_scenario == 1) {
        update_central_wall(h_wall, d_wall, current_holes, hole_open, true);
    }

    // ── Kernel launch config ──────────────────────────────────────────────────
    const dim3 block2d(16, 16);
    const dim3 grid2d((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // Theoretical occupancy — query k_streaming_shmem as it uses the most shmem
    int maxActiveBlocks = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, k_streaming_shmem, block2d.x * block2d.y, 0);
    float occupancy = 100.f * (maxActiveBlocks * block2d.x * block2d.y) / (float)devProp.maxThreadsPerMultiProcessor;

    // ── Stats struct ──────────────────────────────────────────────────────────
    FrameStats stats = {};
    stats.fps = (float)TARGET_FPS;
    stats.sps = (float)(TARGET_FPS * steps_per_frame);
    stats.steps_per_frame = steps_per_frame;
    stats.grid_x = (int)grid2d.x;
    stats.grid_y = (int)grid2d.y;
    stats.block_x = (int)block2d.x;
    stats.block_y = (int)block2d.y;
    stats.sm_count = devProp.multiProcessorCount;
    stats.occupancy_pct = occupancy;
    stats.nu = (TAU - 0.5f) / 3.f;
    snprintf(stats.gpu_name, sizeof(stats.gpu_name), "%s", devProp.name);
    capture_viz_bounds(h_rho.data(), h_ux.data(), h_uy.data(), h_wall, stats);

    // ── Main loop ─────────────────────────────────────────────────────────────
    const Uint64 ms_per_frame = 1000u / TARGET_FPS;
    Uint64 prev_ticks = SDL_GetTicks();
    bool running = true;

    Uint64 resume_time = SDL_GetTicks() + 1000;

    bool is_drawing = false;
    bool is_erasing = false;

    while (running)
    {
        bool wall_updated = false;

        // Events
        SDL_Event ev;
        while (SDL_PollEvent(&ev))
        {
            if (ev.type == SDL_EVENT_QUIT)
                running = false;

            if (ev.type == SDL_EVENT_KEY_DOWN) {
                int new_scenario = -1;

                if (ev.key.key == SDLK_Q) new_scenario = 1;
                if (ev.key.key == SDLK_W) new_scenario = 2;
                if (ev.key.key == SDLK_E) new_scenario = 3;
                if (ev.key.key == SDLK_D) new_scenario = 4; // NOWY TRYB: D = Draw / Sandbox

                if (new_scenario != -1 && new_scenario != current_scenario) {
                    current_scenario = new_scenario;

                    // TRUE: Zmieniamy tryb, więc usuwamy wszelkie bazgroły myszką
                    load_scenario(current_scenario, h_rho.data(), h_wall, d_f_in, d_f_out, d_rho, d_ux, d_uy, d_wall, true);
                    if (current_scenario == 1) {
                        update_central_wall(h_wall, d_wall, current_holes, hole_open, true);
                    }
                    capture_viz_bounds(h_rho.data(), h_ux.data(), h_uy.data(), h_wall, stats);
                    resume_time = SDL_GetTicks() + 1000;
                }
            }

            // R - Reset
            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_R) {
                load_scenario(current_scenario, h_rho.data(), h_wall, d_f_in, d_f_out, d_rho, d_ux, d_uy, d_wall, false);
                if (current_scenario == 1) {
                    update_central_wall(h_wall, d_wall, current_holes, hole_open, false);
                }
                capture_viz_bounds(h_rho.data(), h_ux.data(), h_uy.data(), h_wall, stats);
                resume_time = SDL_GetTicks() + 1000;
            }

            // C - Wyczyszczenie ścian
            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_C) {
                if (current_scenario == 1) {
                    update_central_wall(h_wall, d_wall, current_holes, hole_open, true);
                }
                else {
                    std::fill(h_wall, h_wall + (HEIGHT * WIDTH), false);
                    cudaMemcpy(d_wall, h_wall, (size_t)HEIGHT * WIDTH * sizeof(bool), cudaMemcpyHostToDevice);
                }
            }

            // B - toggle closed/open boundary mode
            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_B)
                bc_mode = (bc_mode == BC_CLOSED) ? BC_OPEN : BC_CLOSED;

            // SPACE - open/close holes
            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_SPACE) {
                if (current_scenario == 1) {
                    hole_open = !hole_open;
                    update_central_wall(h_wall, d_wall, current_holes, hole_open, false);
                }
            }

            // 1-4 number of holes
            if (ev.type == SDL_EVENT_KEY_DOWN) {
                if (current_scenario == 1) {
                    int new_holes = -1;
                    if (ev.key.key == SDLK_1) new_holes = 1;
                    if (ev.key.key == SDLK_2) new_holes = 2;
                    if (ev.key.key == SDLK_3) new_holes = 3;
                    if (ev.key.key == SDLK_4) new_holes = 4;

                    if (new_holes != -1 && new_holes != current_holes) {
                        current_holes = new_holes;
                        update_central_wall(h_wall, d_wall, current_holes, hole_open, false);
                    }
                }
            }

            if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                if (ev.button.button == SDL_BUTTON_LEFT) is_drawing = true;
                if (ev.button.button == SDL_BUTTON_RIGHT) is_erasing = true;

                if (ev.button.button == SDL_BUTTON_MIDDLE) {
                    float mouse_x, mouse_y;
                    SDL_GetMouseState(&mouse_x, &mouse_y);
                    int grid_x = (int)mouse_x / CELL_SIZE;
                    int grid_y = (int)mouse_y / CELL_SIZE;

                    if (grid_x >= 0 && grid_x < WIDTH && grid_y >= 0 && grid_y < HEIGHT) {
                        float target_density = (SDL_GetModState() & SDL_KMOD_SHIFT) ? RHO_RIGHT : RHO_LEFT;

                        flood_fill_density(grid_x, grid_y, target_density, h_rho.data(), h_wall, d_f_in, d_f_out, d_rho);

                        std::fill(h_wall, h_wall + (HEIGHT * WIDTH), false);
                        cudaMemcpy(d_wall, h_wall, (size_t)HEIGHT * WIDTH * sizeof(bool), cudaMemcpyHostToDevice);
                        wall_updated = true;

                        resume_time = SDL_GetTicks() + 500;
                    }
                }
            }
            else if (ev.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                if (ev.button.button == SDL_BUTTON_LEFT) is_drawing = false;
                if (ev.button.button == SDL_BUTTON_RIGHT) is_erasing = false;
            }
        }

        if (is_drawing || is_erasing) {
            float mouse_x, mouse_y;
            SDL_GetMouseState(&mouse_x, &mouse_y);

            // Przeliczenie pikseli na indeks siatki LBM
            int grid_x = (int)mouse_x / CELL_SIZE;
            int grid_y = (int)mouse_y / CELL_SIZE;

            // Zabezpieczenie przed wyjściem myszką poza ekran
            if (grid_x >= 0 && grid_x < WIDTH && grid_y >= 0 && grid_y < HEIGHT) {
                int idx = S_IDX(grid_y, grid_x);

                // Rysujemy
                if (is_drawing && !h_wall[idx]) {
                    h_wall[idx] = true;
                    wall_updated = true;
                }
                // Ścieramy
                else if (is_erasing && h_wall[idx]) {
                    h_wall[idx] = false;
                    wall_updated = true;
                }
            }
        }

        if (wall_updated) {
            cudaMemcpy(d_wall, h_wall, msk_sz, cudaMemcpyHostToDevice);
        }

        // ── Simulation: steps_per_frame substeps before each render ──────────
        // k_collide (fused macro+collision) + k_streaming_shmem (tiled pull)
        // run back-to-back with no device sync between steps.
        // Wall bounce-back and outer BC follow each streaming step.
        if (SDL_GetTicks() >= resume_time) {
            for (int sub = 0; sub < steps_per_frame; ++sub) {
                k_collide << <grid2d, block2d >> > (d_f_in, d_f_out);
                k_streaming_shmem << <grid2d, block2d >> > (d_f_in, d_f_out);
                k_wall_bounce_back << <grid2d, block2d >> > (d_f_in, d_f_out, d_wall);
                k_outer_boundary << <grid2d, block2d >> > (d_f_in, d_f_out, d_wall, bc_mode);
            }
        }
        // Extract macroscopic fields once per frame for visualisation
        k_macroscopic << <grid2d, block2d >> > (d_f_in, d_rho, d_ux, d_uy);
        cudaDeviceSynchronize();

        // Copy scalars to host (single D2H per frame)
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
        stats.bc_closed = (bc_mode == BC_CLOSED);
        stats.step += steps_per_frame;

        // FPS and SPS — exponential moving average over frames
        Uint64 now = SDL_GetTicks();
        Uint64 elapsed = now - prev_ticks;
        if (elapsed > 0) {
            float measured_fps = 1000.f / (float)elapsed;
            stats.fps = stats.fps * 0.9f + 0.1f * measured_fps;
            stats.sps = stats.sps * 0.9f + 0.1f * (steps_per_frame * measured_fps);
        }
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