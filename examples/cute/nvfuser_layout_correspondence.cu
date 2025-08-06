/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

// Example demonstrating correspondence between CUTE Layouts and nvFuser IterDomain transforms
//
// This example shows how various layout transformations in CUTE correspond to
// domain transformations in nvFuser's IterDomain system:
//
// 1. Split transformation: Layout split corresponds to IterDomain split
// 2. Merge transformation: Layout merge corresponds to IterDomain merge  
// 3. Reorder transformation: Layout reorder corresponds to IterDomain reorder
// 4. Tile transformation: Layout tiling corresponds to IterDomain tiling
// 5. Flatten transformation: Layout flatten corresponds to IterDomain flatten

template <class Layout>
void print_layout_info(const char* name, Layout const& layout) {
  print(name); print(": "); print(layout); print("\n");
}

// Demonstrate split transformation
// CUTE: layout.split(factor) -> (outer, inner)
// nvFuser: IterDomain.split(factor) -> (outer, inner)
void demonstrate_split() {
  print("=== Split Transformation ===\n");
  
  // Original layout: 64 elements
  auto original = make_layout(Int<64>{});
  print_layout_info("Original", original);
  
  // Split by factor 8: 64 -> (8, 8)
  auto split_layout = logical_divide(original, Int<8>{});
  print_layout_info("After split(8)", split_layout);
  
  // This corresponds to nvFuser:
  // IterDomain* original_id = new IterDomain(0, 64);
  // auto [outer_id, inner_id] = IterDomain::split(original_id, 8, true);
}

// Demonstrate merge transformation  
// CUTE: layout.merge() -> combined
// nvFuser: IterDomain.merge(outer, inner) -> combined
void demonstrate_merge() {
  print("=== Merge Transformation ===\n");
  
  // Original layout: (8, 8)
  auto original = make_layout(Shape<Int<8>, Int<8>>{});
  print_layout_info("Original", original);
  
  // Merge: (8, 8) -> 64
  auto merged = make_layout(flatten(original.shape()));
  print_layout_info("After merge", merged);
  
  // This corresponds to nvFuser:
  // auto [outer_id, inner_id] = IterDomain::split(original_id, 8, true);
  // IterDomain* merged_id = IterDomain::merge(outer_id, inner_id);
}

// Demonstrate reorder transformation
// CUTE: layout.reorder(new_order) -> reordered
// nvFuser: TensorDomain.reorder(old2new_map) -> reordered
void demonstrate_reorder() {
  print("=== Reorder Transformation ===\n");
  
  // Original layout: (M, N, K) = (32, 16, 8)
  auto original = make_layout(Shape<Int<32>, Int<16>, Int<8>>{});
  print_layout_info("Original (M,N,K)", original);
  
  // Reorder to (K, M, N): (8, 32, 16)
  auto reordered = make_layout(Shape<Int<8>, Int<32>, Int<16>>{});
  print_layout_info("After reorder (K,M,N)", reordered);
  
  // This corresponds to nvFuser:
  // std::unordered_map<int64_t, int64_t> old2new = {{0,1}, {1,2}, {2,0}};
  // tensor_domain->reorder(old2new);
}

// Demonstrate tile transformation
// CUTE: layout.tile(tiler) -> tiled
// nvFuser: TensorDomain.split() for tiling -> tiled
void demonstrate_tile() {
  print("=== Tile Transformation ===\n");
  
  // Original layout: 128 elements
  auto original = make_layout(Int<128>{});
  print_layout_info("Original", original);
  
  // Tile by (16, 8): 128 -> ((16, 8), (8, 16))
  auto tiled = zipped_divide(original, make_layout(Shape<Int<16>, Int<8>>{}));
  print_layout_info("After tile(16,8)", tiled);
  
  // This corresponds to nvFuser:
  // auto [block_id, thread_id] = IterDomain::split(original_id, 16, false);
  // auto [thread_x, thread_y] = IterDomain::split(thread_id, 8, false);
}

// Demonstrate flatten transformation
// CUTE: layout.flatten() -> flattened
// nvFuser: TensorDomain.flatten(start_dim, end_dim) -> flattened
void demonstrate_flatten() {
  print("=== Flatten Transformation ===\n");
  
  // Original layout: (4, 8, 16)
  auto original = make_layout(Shape<Int<4>, Int<8>, Int<16>>{});
  print_layout_info("Original", original);
  
  // Flatten: (4, 8, 16) -> 512
  auto flattened = make_layout(flatten(original.shape()));
  print_layout_info("After flatten", flattened);
  
  // This corresponds to nvFuser:
  // tensor_domain->flatten(0, 2); // Flatten dimensions 0 to 2
}

// Demonstrate composition (function composition)
// CUTE: layout1.compose(layout2) -> composed
// nvFuser: Multiple transformations applied sequentially
void demonstrate_composition() {
  print("=== Composition Transformation ===\n");
  
  // Layout 1: (8, 8)
  auto layout1 = make_layout(Shape<Int<8>, Int<8>>{});
  print_layout_info("Layout 1", layout1);
  
  // Layout 2: (4, 4)
  auto layout2 = make_layout(Shape<Int<4>, Int<4>>{});
  print_layout_info("Layout 2", layout2);
  
  // Compose: layout1 o layout2
  auto composed = composition(layout1, layout2);
  print_layout_info("Composed (layout1 o layout2)", composed);
  
  // This corresponds to nvFuser applying multiple transformations:
  // tensor_domain->split(0, 4, true);  // First transformation
  // tensor_domain->split(1, 4, true);  // Second transformation
}

// Demonstrate stride transformations
// CUTE: Different stride patterns
// nvFuser: Different memory layouts and access patterns
void demonstrate_strides() {
  print("=== Stride Transformations ===\n");
  
  // Row-major layout: (4, 8) with row-major strides
  auto row_major = make_layout(Shape<Int<4>, Int<8>>{}, LayoutLeft{});
  print_layout_info("Row-major", row_major);
  
  // Column-major layout: (4, 8) with column-major strides  
  auto col_major = make_layout(Shape<Int<4>, Int<8>>{}, LayoutRight{});
  print_layout_info("Column-major", col_major);
  
  // This corresponds to nvFuser memory layouts:
  // Row-major: Contiguous in first dimension
  // Column-major: Contiguous in last dimension
}

int main() {
  print("CUTE Layout and nvFuser IterDomain Correspondence Example\n");
  print("========================================================\n\n");
  
  demonstrate_split();
  demonstrate_merge();
  demonstrate_reorder();
  demonstrate_tile();
  demonstrate_flatten();
  demonstrate_composition();
  demonstrate_strides();
  
  print("Example completed successfully!\n");
  return 0;
} 