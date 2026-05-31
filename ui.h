#pragma once
#include <SDL3/SDL.h>
#include "sim_params.h"

// ── Window layout ─────────────────────────────────────────────────────────────

static constexpr int DIVIDER_W = 4;               // px between panels
static constexpr int PANEL_W   = WIDTH  * CELL_SIZE;
static constexpr int PANEL_H   = HEIGHT * CELL_SIZE;
static constexpr int STATS_H   = 125;             // stats bar height
static constexpr int WIN_W     = 3 * PANEL_W + 2 * DIVIDER_W;
static constexpr int WIN_H     = PANEL_H + STATS_H;

// ── Stats passed from the simulation loop ─────────────────────────────────────

struct FrameStats {
    float fps;
    float rho_min, rho_max;
    float ux_min,  ux_max;
    float uy_min,  uy_max;
    float viz_rho_min, viz_rho_max; // fixed colormap bounds (captured at reset)
    float viz_vel_scale;            // symmetric |u| scale for velocity panels
    int   grid_x,  grid_y;    // kernel launch grid (in blocks)
    int   block_x, block_y;   // threads per block
    int   sm_count;
    float occupancy_pct;
    char  gpu_name[256];
    bool  hole_open;
    bool  bc_closed;           // true = closed mixing box, false = open reservoirs
    int   step;
    float nu;                  // kinematic viscosity = (TAU-0.5)/3
};

// ── Interface ─────────────────────────────────────────────────────────────────

SDL_Window*   ui_create_window();
SDL_Renderer* ui_create_renderer(SDL_Window* win);
SDL_Texture*  ui_create_sim_texture(SDL_Renderer* ren);

void ui_draw_frame(SDL_Renderer* ren, SDL_Texture* tex,
                   const float*  h_rho,
                   const float*  h_ux,
                   const float*  h_uy,
                   const bool*   h_wall,
                   const FrameStats& stats);
