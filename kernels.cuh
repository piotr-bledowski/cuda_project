#pragma once
#include "sim_params.h"
#include <cuda_runtime.h>

// ── D2Q9 kernel declarations ──────────────────────────────────────────────────
//
// Viz extraction — macroscopic fields from distribution functions (once/frame)
__global__ void k_macroscopic(const float* __restrict__ f_in,
                               float* __restrict__ rho,
                               float* __restrict__ ux,
                               float* __restrict__ uy);

// ── Optimised inner-loop kernels (called N times per frame; N from CLI) ───────
//
// Step 1 — fused macroscopic + BGK collision
//   Computes ρ and u in registers (no global write), outputs post-collision f_out.
__global__ void k_collide(const float* __restrict__ f_in,
                           float* __restrict__ f_out);

// Step 2 — pull streaming with shared-memory tile
//   Loads a (TILE_DIM × TILE_DIM) tile of f_out (post-collision) into shared
//   memory and pulls all 9 directions from it, avoiding repeated global reads.
__global__ void k_streaming_shmem(float* __restrict__ f_in,
                                   const float* __restrict__ f_out);

// Step 3 — bounce-back on interior wall nodes
__global__ void k_wall_bounce_back(float* __restrict__ f_in,
                                    const float* __restrict__ f_out,
                                    const bool* __restrict__ wall);

// Step 4 — outer BC: bounce-back (closed) or Zou-He (open) on west/east
__global__ void k_outer_boundary(float* __restrict__ f_in,
                                  const float* __restrict__ f_out,
                                  const bool* __restrict__ wall,
                                  int bc_mode);

// ── CPU helper ────────────────────────────────────────────────────────────────
void rebuild_wall(bool* h_wall, bool* d_wall, bool hole_open);
