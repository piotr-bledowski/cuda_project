#pragma once
#include "sim_params.h"
#include "sim_config.h"
#include <cuda_runtime.h>

// ── D2Q9 kernel declarations ──────────────────────────────────────────────────

__global__ void k_macroscopic(const float* __restrict__ f_in,
                               float* __restrict__ rho,
                               float* __restrict__ ux,
                               float* __restrict__ uy);

__global__ void k_collision(const float* __restrict__ f_in,
                             float* __restrict__ f_out,
                             const float* __restrict__ rho,
                             const float* __restrict__ ux,
                             const float* __restrict__ uy,
                             const bool* __restrict__ wall);

__global__ void k_streaming(float* __restrict__ f_in,
                             const float* __restrict__ f_out,
                             const bool* __restrict__ wall);

__global__ void k_wall_link_bounce_back(float* __restrict__ f_in,
                                         const float* __restrict__ f_out,
                                         const bool* __restrict__ wall);

__global__ void k_wall_bounce_back(float* __restrict__ f_in,
                                    const float* __restrict__ f_out,
                                    const bool* __restrict__ wall);

__global__ void k_outer_boundary(float* __restrict__ f_in,
                                  const float* __restrict__ f_out,
                                  const bool* __restrict__ wall);

void upload_outer_bc(const OuterBcConfig& cfg);
