#pragma once

// ── Grid ─────────────────────────────────────────────────────────────────────
#define WIDTH 100
#define HEIGHT 100
#define Q 9

// ── Display ──────────────────────────────────────────────────────────────────
#define CELL_SIZE 5
#define TARGET_FPS 60

// ── Physics ──────────────────────────────────────────────────────────────────
#define TAU 1.0f
#define U_MAX 0.1f // reference velocity scale for viz floor

// ── Flat-array indexing ───────────────────────────────────────────────────────
#define F_IDX(y, x, i) ((y) * WIDTH * Q + (x) * Q + (i))
#define S_IDX(y, x) ((y) * WIDTH + (x))
