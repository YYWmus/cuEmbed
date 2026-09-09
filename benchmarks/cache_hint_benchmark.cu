// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// Keep this benchmark behavior-identical to manual_benchmark apart from the
// optional PTX cache-eviction-hint controls compiled in below.
#define CUEMBED_CACHE_HINT_BENCHMARK 1
#include "manual_benchmark.cu"
