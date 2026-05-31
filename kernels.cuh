#pragma once
#include "sim_params.h"
#include <cuda_runtime.h>

// ── D2Q9 kernel declarations ──────────────────────────────────────────────────
//
// Step 1 — macroscopic fields from distribution functions
__global__ void k_macroscopic(const float* __restrict__ f_in,
                               float* __restrict__ rho,
                               float* __restrict__ ux,
                               float* __restrict__ uy);

// Step 2 — BGK collision
__global__ void k_collision(const float* __restrict__ f_in,
                             float* __restrict__ f_out,
                             const float* __restrict__ rho,
                             const float* __restrict__ ux,
                             const float* __restrict__ uy);

// Step 3 — streaming (pull scheme)
__global__ void k_streaming(float* __restrict__ f_in,
                             const float* __restrict__ f_out);

// Step 4 — bounce-back on interior wall nodes
__global__ void k_wall_bounce_back(float* __restrict__ f_in,
                                    const float* __restrict__ f_out,
                                    const bool* __restrict__ wall);

// Step 5 — outer BC: bounce-back (closed) or Zou-He (open) on west/east
__global__ void k_outer_boundary(float* __restrict__ f_in,
                                  const float* __restrict__ f_out,
                                  const bool* __restrict__ wall,
                                  int bc_mode);

// ── CPU helper ────────────────────────────────────────────────────────────────
void rebuild_wall(bool* h_wall, bool* d_wall, bool hole_open);
