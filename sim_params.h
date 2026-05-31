#pragma once

// ── Grid ─────────────────────────────────────────────────────────────────────
#define WIDTH 100
#define HEIGHT 100
#define Q 9

// ── Display ──────────────────────────────────────────────────────────────────
#define CELL_SIZE 5
#define TARGET_FPS 60

// ── Simulation defaults (override at runtime via CLI: -n N or positional N) ─
#define DEFAULT_STEPS_PER_FRAME 16

// ── Kernel launch geometry ───────────────────────────────────────────────────
// Must match dim3 block2d(BLOCK_DIM, BLOCK_DIM) in main.cu.
#define BLOCK_DIM 16
#define TILE_DIM  (BLOCK_DIM + 2)   // shared-memory tile including 1-cell halo

// ── Physics ──────────────────────────────────────────────────────────────────
#define TAU 1.0f
#define RHO_LEFT 1.15f // ≤ 30% contrast keeps D2Q9 in the low-Mach regime
#define RHO_RIGHT 0.85f
#define U_MAX 0.1f // reference velocity scale for viz floor

// ── Boundary modes ───────────────────────────────────────────────────────────
enum BcMode : int {
    BC_CLOSED = 0, // bounce-back on all outer walls — density equalizes
    BC_OPEN   = 1  // Zou-He fixed-density reservoirs on west/east
};

// ── Geometry ─────────────────────────────────────────────────────────────────
#define WALL_X (WIDTH / 4)
#define HOLE_START (HEIGHT / 3)
#define HOLE_END (2 * HEIGHT / 3)

// ── Flat-array indexing ───────────────────────────────────────────────────────
#define F_IDX(y, x, i) ((y) * WIDTH * Q + (x) * Q + (i))
#define S_IDX(y, x) ((y) * WIDTH + (x))
