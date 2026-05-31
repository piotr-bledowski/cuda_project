#include "ui.h"
#include <cstdio>
#include <algorithm>

// ── Layout constants ──────────────────────────────────────────────────────────

// Text is rendered at 1.5× — debug chars appear as 12×12 px instead of 8×8.
static constexpr float TEXT_SCALE = 1.5f;
static constexpr int LINE_H = 18; // px between successive text lines

// x-offsets (screen px) of the left edge of each simulation panel
static constexpr int PANEL_X[3] = {
    0,
    PANEL_W + DIVIDER_W,
    2 * PANEL_W + 2 * DIVIDER_W};

// x-coordinate where the vertical divider inside the stats bar sits
static constexpr int VSTATS_X = 458;

// ── Color helpers ─────────────────────────────────────────────────────────────

static Uint32 pack_argb(Uint8 r, Uint8 g, Uint8 b)
{
    return (0xFFu << 24) | ((Uint32)r << 16) | ((Uint32)g << 8) | b;
}

// Blue (low density) → Red (high density), mapped to fixed viz bounds
static Uint32 rho_color(float r, float rho_min, float rho_max)
{
    float span = rho_max - rho_min;
    float t = (span > 1e-8f) ? (r - rho_min) / span : 0.5f;
    t = std::max(0.f, std::min(1.f, t));
    return pack_argb((Uint8)(255 * t), 0, (Uint8)(255 * (1.f - t)));
}

// Red = positive, Blue = negative, Black = zero; symmetric vel_scale
static Uint32 vel_color(float v, float vel_scale)
{
    float n = (vel_scale > 1e-8f) ? v / vel_scale : 0.f;
    n = std::max(-1.f, std::min(1.f, n));
    if (n >= 0.f)
        return pack_argb((Uint8)(255 * n), 0, 0);
    return pack_argb(0, 0, (Uint8)(255 * (-n)));
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

static void fill_rect(SDL_Renderer *ren, int x, int y, int w, int h,
                      Uint8 r, Uint8 g, Uint8 b, Uint8 a = 255)
{
    SDL_SetRenderDrawColor(ren, r, g, b, a);
    SDL_FRect rc = {(float)x, (float)y, (float)w, (float)h};
    SDL_RenderFillRect(ren, &rc);
}

// ── Text helper ───────────────────────────────────────────────────────────────
// Called while render scale is set to TEXT_SCALE.
// sx, sy are screen-pixel coordinates; they are divided by TEXT_SCALE
// internally so the text appears at the intended screen position.

static void txt(SDL_Renderer *ren, float sx, float sy, const char *s,
                Uint8 r = 210, Uint8 g = 210, Uint8 b = 210)
{
    SDL_SetRenderDrawColor(ren, r, g, b, 255);
    SDL_RenderDebugText(ren, sx / TEXT_SCALE, sy / TEXT_SCALE, s);
}

// ── Simulation texture ────────────────────────────────────────────────────────

static void fill_sim_texture(SDL_Texture *tex,
                             const float *h_rho,
                             const float *h_ux,
                             const float *h_uy,
                             const bool *h_wall,
                             const FrameStats &stats)
{
    void *pixels;
    int pitch;
    SDL_LockTexture(tex, nullptr, &pixels, &pitch);
    int stride = pitch / (int)sizeof(Uint32);
    Uint32 *px = (Uint32 *)pixels;

    const Uint32 wall_px = pack_argb(220, 220, 220);
    for (int y = 0; y < HEIGHT; ++y)
    {
        for (int x = 0; x < WIDTH; ++x)
        {
            bool w = h_wall[S_IDX(y, x)];
            px[y * stride + x] = w ? wall_px : rho_color(h_rho[S_IDX(y, x)], stats.viz_rho_min, stats.viz_rho_max);
            px[y * stride + WIDTH + x] = w ? wall_px : vel_color(h_ux[S_IDX(y, x)], stats.viz_vel_scale);
            px[y * stride + 2 * WIDTH + x] = w ? wall_px : vel_color(h_uy[S_IDX(y, x)], stats.viz_vel_scale);
        }
    }
    SDL_UnlockTexture(tex);
}

// ── Public interface ──────────────────────────────────────────────────────────

SDL_Window *ui_create_window()
{
    return SDL_CreateWindow(
        "LBM CUDA  |  D2Q9 Fluid Simulation  |  [SPACE] hole  [B] BC  [R] reset",
        WIN_W, WIN_H, 0);
}

SDL_Renderer *ui_create_renderer(SDL_Window *win)
{
    SDL_Renderer *ren = SDL_CreateRenderer(win, NULL);
    SDL_SetRenderVSync(ren, 1);
    return ren;
}

SDL_Texture *ui_create_sim_texture(SDL_Renderer *ren)
{
    return SDL_CreateTexture(ren,
                             SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
                             WIDTH * 3, HEIGHT);
}

void ui_draw_frame(SDL_Renderer *ren, SDL_Texture *tex,
                   const float *h_rho, const float *h_ux, const float *h_uy,
                   const bool *h_wall, const FrameStats &stats)
{
    // ── Background ────────────────────────────────────────────────────────────
    SDL_SetRenderDrawColor(ren, 22, 22, 26, 255);
    SDL_RenderClear(ren);

    // ── Simulation panels ─────────────────────────────────────────────────────
    fill_sim_texture(tex, h_rho, h_ux, h_uy, h_wall, stats);

    for (int p = 0; p < 3; ++p)
    {
        SDL_FRect src = {(float)(p * WIDTH), 0.f, (float)WIDTH, (float)HEIGHT};
        SDL_FRect dst = {(float)PANEL_X[p], 0.f, (float)PANEL_W, (float)PANEL_H};
        SDL_RenderTexture(ren, tex, &src, &dst);
    }

    // ── Panel dividers ────────────────────────────────────────────────────────
    fill_rect(ren, PANEL_W, 0, DIVIDER_W, PANEL_H, 55, 55, 60);
    fill_rect(ren, 2 * PANEL_W + DIVIDER_W, 0, DIVIDER_W, PANEL_H, 55, 55, 60);

    // ── Panel header bars (semi-transparent overlay at top of each panel) ─────
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    for (int p = 0; p < 3; ++p)
    {
        SDL_SetRenderDrawColor(ren, 12, 12, 40, 215);
        SDL_FRect hbar = {(float)PANEL_X[p], 0.f, (float)PANEL_W, 26.f};
        SDL_RenderFillRect(ren, &hbar);
    }
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_NONE);

    // ── Stats area ────────────────────────────────────────────────────────────
    fill_rect(ren, 0, PANEL_H, WIN_W, 2, 60, 60, 65);               // separator line
    fill_rect(ren, 0, PANEL_H + 2, WIN_W, STATS_H - 2, 15, 15, 20); // background

    // Vertical divider between general and per-panel info
    fill_rect(ren, VSTATS_X, PANEL_H + 2, 2, STATS_H - 2, 48, 48, 55);

    // ── All text rendered at TEXT_SCALE ───────────────────────────────────────
    SDL_SetRenderScale(ren, TEXT_SCALE, TEXT_SCALE);

    // Panel header labels
    const char *hlabels[3] = {"DENSITY", "VEL X", "VEL Y"};
    for (int p = 0; p < 3; ++p)
        txt(ren, (float)(PANEL_X[p] + 8), 8.f, hlabels[p], 170, 170, 255);

    // ── General stats (left of stats bar) ────────────────────────────────────
    char buf[256];
    float sy = (float)(PANEL_H + 10);

    snprintf(buf, sizeof(buf), "FPS: %.1f     Step: %d", stats.fps, stats.step);
    txt(ren, 8.f, sy, buf, 230, 230, 230);
    sy += LINE_H;

    snprintf(buf, sizeof(buf), "Grid: %dx%d   TAU: %.2f   nu: %.4f",
             WIDTH, HEIGHT, (double)TAU, (double)stats.nu);
    txt(ren, 8.f, sy, buf);
    sy += LINE_H;

    snprintf(buf, sizeof(buf), "rho_L: %.3f   rho_R: %.3f   Hole: %s   BC: %s",
             (double)RHO_LEFT, (double)RHO_RIGHT,
             stats.hole_open ? "OPEN  " : "CLOSED",
             stats.bc_closed ? "CLOSED" : "OPEN  ");
    if (stats.hole_open)
        txt(ren, 8.f, sy, buf, 255, 205, 80);
    else
        txt(ren, 8.f, sy, buf, 160, 160, 160);
    sy += LINE_H;

    snprintf(buf, sizeof(buf), "GPU: %s   SMs: %d", stats.gpu_name, stats.sm_count);
    txt(ren, 8.f, sy, buf, 140, 220, 140);
    sy += LINE_H;

    snprintf(buf, sizeof(buf), "Kernels: %dx%d blk x %dx%d thr  =  %d threads",
             stats.grid_x, stats.grid_y, stats.block_x, stats.block_y,
             stats.grid_x * stats.grid_y * stats.block_x * stats.block_y);
    txt(ren, 8.f, sy, buf);
    sy += LINE_H;

    snprintf(buf, sizeof(buf), "Theoretical occupancy: %.1f%%", (double)stats.occupancy_pct);
    txt(ren, 8.f, sy, buf);

    // ── Per-panel stats (right of stats bar) ─────────────────────────────────
    const int RX = VSTATS_X + 10;
    const int COL_W = (WIN_W - RX) / 3;

    const char *pnames[3] = {"DENSITY (LBM)", "VEL X (lu/ts)", "VEL Y (lu/ts)"};
    float mins[3] = {stats.rho_min, stats.ux_min, stats.uy_min};
    float maxs[3] = {stats.rho_max, stats.ux_max, stats.uy_max};

    // Vertical alignment guides in stats area (subtle, aligned with panel centres)
    // (drawn after texture, before text — kept as text overlay for simplicity)

    sy = (float)(PANEL_H + 10);

    for (int p = 0; p < 3; ++p)
        txt(ren, (float)(RX + p * COL_W), sy, pnames[p], 175, 175, 255);
    sy += LINE_H;

    for (int p = 0; p < 3; ++p)
    {
        snprintf(buf, sizeof(buf), "min: %+.4f", (double)mins[p]);
        txt(ren, (float)(RX + p * COL_W), sy, buf, 110, 190, 255);
    }
    sy += LINE_H;

    for (int p = 0; p < 3; ++p)
    {
        snprintf(buf, sizeof(buf), "max: %+.4f", (double)maxs[p]);
        txt(ren, (float)(RX + p * COL_W), sy, buf, 255, 140, 110);
    }
    sy += LINE_H;

    snprintf(buf, sizeof(buf), "map: %.3f..%.3f", (double)stats.viz_rho_min, (double)stats.viz_rho_max);
    txt(ren, (float)RX, sy, buf, 140, 140, 160);
    snprintf(buf, sizeof(buf), "scale: +/-%.4f", (double)stats.viz_vel_scale);
    txt(ren, (float)(RX + COL_W), sy, buf, 140, 140, 160);
    txt(ren, (float)(RX + 2 * COL_W), sy, buf, 140, 140, 160);

    SDL_SetRenderScale(ren, 1.0f, 1.0f);

    SDL_RenderPresent(ren);
}
