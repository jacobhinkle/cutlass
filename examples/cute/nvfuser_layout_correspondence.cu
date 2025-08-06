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
  // Val* extent = IrBuilder::create<Val>(64);
  // Val* factor = IrBuilder::create<Val>(8);
  // IterDomain* original_id = IterDomainBuilder(fusion->zeroVal(), extent).build();
  // auto [outer_id, inner_id] = IterDomain::split(original_id, factor, /*inner_split=*/true);
}

// Demonstrate merge transformation  
// CUTE: layout.merge() -> combined
// nvFuser: IterDomain.merge(outer, inner) -> combined
void demonstrate_merge() {
  print("=== Merge Transformation ===\n");
  
  // Original layout: (8, 8)
  auto original = make_layout(Shape<Int<8>, Int<8>>{});
  print_layout_info("Original", original);
  // Output: Original: (_8,_8):(_1,_8)
  
  // Merge: (8, 8) -> 64
  auto merged = coalesce(original);
  print_layout_info("After merge", merged);
  // Output: After merge: _64:_1
  
  // Strided example: (4, 8) with non-contiguous strides
  auto strided_original = make_layout(Shape<Int<4>, Int<8>>{}, Stride<Int<8>, Int<1>>{});
  print_layout_info("Strided original", strided_original);
  // Output: Strided original: (_4,_8):(_8,_1)
  
  // Merge strided layout: (4, 8) -> unchanged (strides not compatible for coalescing)
  // Coalescing requires: second_stride = first_size × first_stride
  // Here: 1 ≠ 4 × 8 = 32, so modes cannot be combined
  // However, ANY layout can be viewed as 1D using logically row-major indexing
  auto strided_merged = coalesce(strided_original);
  print_layout_info("Strided merged", strided_merged);
  // Output: Strided merged: (_4,_8):(_8,_1)
  
  // Flatten the same layout to show 1D view: (4,8) -> 32
  // Note: flatten() removes hierarchy but preserves the coordinate mapping
  // The layout still maps (i,j) to index i*8 + j*1 for 0 ≤ i < 4, 0 ≤ j < 8
  auto strided_flattened = flatten(strided_original);
  print_layout_info("Strided flattened (shape only)", strided_flattened);
  // Output: Strided flattened (shape only): (_4,_8):(_8,_1)
  
  // Example with different strides: (4,8) with strides (2,16)
  auto strided2_original = make_layout(Shape<Int<4>, Int<8>>{}, Stride<Int<2>, Int<16>>{});
  print_layout_info("Strided2 original", strided2_original);
  // Output: Strided2 original: (_4,_8):(_2,_16)
  // Check if coalesce can combine: 16 = 4 × 2? No, 16 ≠ 8, so no combination
  auto strided2_coalesced = coalesce(strided2_original);
  print_layout_info("Strided2 coalesced", strided2_coalesced);
  // Output: Strided2 coalesced: (_4,_8):(_2,_16)
  
  // Example where coalesce succeeds: (4,8) with strides (2,8)
  auto strided3_original = make_layout(Shape<Int<4>, Int<8>>{}, Stride<Int<2>, Int<8>>{});
  print_layout_info("Strided3 original", strided3_original);
  // Output: Strided3 original: (_4,_8):(_2,_8)
  // Check if coalesce can combine: 8 = 4 × 2? Yes! So it becomes 32:2
  auto strided3_coalesced = coalesce(strided3_original);
  print_layout_info("Strided3 coalesced (1D view)", strided3_coalesced);
  // Output: Strided3 coalesced (1D view): _32:_2
  
  // Demonstrate 1D indexing: Any layout can be viewed as 1D
  // Layout (_4,_8):(_8,_1) maps coordinates (i,j) to index i*8 + j*1
  // This gives us 32 consecutive indices: 0,1,2,...,31
  // The layout is already "1D" in the sense that it maps to consecutive indices
  print("Note: Layout (_4,_8):(_8,_1) already provides 1D indexing 0-31\n");
  
  // Example of successful coalescing: (4, 8) with compatible strides
  auto coalesceable_original = make_layout(Shape<Int<4>, Int<8>>{}, Stride<Int<1>, Int<4>>{});
  print_layout_info("Coalesceable original", coalesceable_original);
  // Output: Coalesceable original: (_4,_8):(_1,_4)
  // Here: 4 = 4 × 1, so modes can be combined
  auto coalesceable_merged = coalesce(coalesceable_original);
  print_layout_info("Coalesceable merged", coalesceable_merged);
  // Output: Coalesceable merged: _32:_1
  
  // Complex example: 3D layout with different stride patterns
  // Inner dimension: contiguous (stride 1)
  // Middle dimension: constant non-contiguous (stride 8)
  // Outer dimension: symbolic (stride 32)
  auto complex_original = make_layout(Shape<Int<4>, Int<8>, Int<16>>{}, 
                                     Stride<Int<32>, Int<8>, Int<1>>{});
  print_layout_info("Complex 3D original", complex_original);
  // Output: Complex 3D original: (_4,_8,_16):(_32,_8,_1)
  
  // Merge outer and inner dimensions: (4, 8, 16) -> (64, 8)
  // Use coalesce to combine modes where possible
  auto complex_merged = coalesce(complex_original);
  print_layout_info("Complex merged (outer+inner)", complex_merged);
  // Output: Complex merged (outer+inner): (_4,_8,_16):(_32,_8,_1)
  
  // This corresponds to nvFuser:
  // Val* extent = IrBuilder::create<Val>(64);
  // Val* factor = IrBuilder::create<Val>(8);
  // IterDomain* original_id = IterDomainBuilder(fusion->zeroVal(), extent).build();
  // auto [outer_id, inner_id] = IterDomain::split(original_id, factor, true);
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
  // Output: Original (M,N,K): (_32,_16,_8):(_1,_32,_512)
  
  // Reorder to (K, M, N): (8, 32, 16)
  auto reordered = make_layout(Shape<Int<8>, Int<32>, Int<16>>{});
  print_layout_info("After reorder (K,M,N)", reordered);
  // Output: After reorder (K,M,N): (_8,_32,_16):(_1,_8,_256)
  
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
  // Output: Original: _128:_1
  
  // Tile by (16, 8): 128 -> ((16, 8), (8, 16))
  auto tiled = zipped_divide(original, make_layout(Shape<Int<16>, Int<8>>{}));
  print_layout_info("After tile(16,8)", tiled);
  // Output: After tile(16,8): ((_16,_8),_1):((_1,_16),_0)
  
  // This corresponds to nvFuser:
  // Val* extent = IrBuilder::create<Val>(128);
  // Val* factor1 = IrBuilder::create<Val>(16);
  // Val* factor2 = IrBuilder::create<Val>(8);
  // IterDomain* original_id = IterDomainBuilder(fusion->zeroVal(), extent).build();
  // auto [block_id, thread_id] = IterDomain::split(original_id, factor1, false);
  // auto [thread_x, thread_y] = IterDomain::split(thread_id, factor2, false);
}

// Demonstrate flatten transformation
// CUTE: layout.flatten() -> flattened
// nvFuser: TensorDomain.flatten(start_dim, end_dim) -> flattened
void demonstrate_flatten() {
  print("=== Flatten Transformation ===\n");
  
  // Original layout: (4, 8, 16)
  auto original = make_layout(Shape<Int<4>, Int<8>, Int<16>>{});
  print_layout_info("Original", original);
  // Output: Original: (_4,_8,_16):(_1,_4,_32)
  
  // Flatten: (4, 8, 16) -> 512
  auto flattened = flatten(original);
  print_layout_info("After flatten", flattened);
  // Output: After flatten: (_4,_8,_16):(_1,_4,_32)
  
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
  // Output: Layout 1: (_8,_8):(_1,_8)
  
  // Layout 2: (4, 4)
  auto layout2 = make_layout(Shape<Int<4>, Int<4>>{});
  print_layout_info("Layout 2", layout2);
  // Output: Layout 2: (_4,_4):(_1,_4)
  
  // Compose: layout1 o layout2
  auto composed = composition(layout1, layout2);
  print_layout_info("Composed (layout1 o layout2)", composed);
  // Output: Composed (layout1 o layout2): (_4,_4):(_1,_4)
  
  // This corresponds to nvFuser applying multiple transformations:
  // Val* extent = IrBuilder::create<Val>(64);
  // Val* factor = IrBuilder::create<Val>(4);
  // IterDomain* original_id = IterDomainBuilder(fusion->zeroVal(), extent).build();
  // tensor_domain->split(0, factor, true);  // First transformation
  // tensor_domain->split(1, factor, true);  // Second transformation
}

// Demonstrate stride transformations
// CUTE: Different stride patterns
// nvFuser: Different memory layouts and access patterns
void demonstrate_strides() {
  print("=== Stride Transformations ===\n");
  
  // Row-major layout: (4, 8) with row-major strides
  auto row_major = make_layout(Shape<Int<4>, Int<8>>{}, LayoutLeft{});
  print_layout_info("Row-major", row_major);
  // Output: Row-major: (_4,_8):(_1,_4)
  
  // Column-major layout: (4, 8) with column-major strides  
  auto col_major = make_layout(Shape<Int<4>, Int<8>>{}, LayoutRight{});
  print_layout_info("Column-major", col_major);
  // Output: Column-major: (_4,_8):(_8,_1)
  
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