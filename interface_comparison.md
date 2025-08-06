# nvFuser, CUTE, and Cutlass Interface Comparison

This document provides a comprehensive comparison of common features across the three interfaces: nvFuser, CUTE, and Cutlass. Each entry includes a link to the relevant GitHub source code.

## Table of Contents

1. [1. Definition](#1-definition)
   - [1.1 Core Domain Transformation Features](#11-core-domain-transformation-features)
   - [1.2 Advanced Layout Operations](#12-advanced-layout-operations)
   - [1.3 Memory Layout and Access Patterns](#13-memory-layout-and-access-patterns)
   - [1.4 Tensor Operations](#14-tensor-operations)

2. [2. Scheduling](#2-scheduling)
   - [2.1 Parallelization and Execution](#21-parallelization-and-execution)
   - [2.2 Tensor Core MMA Operations](#22-tensor-core-mma-operations)
   - [2.3 Mathematical Operations](#23-mathematical-operations)

3. [3. Syncing](#3-syncing)
   - [3.1 Asynchronous Memory Operations](#31-asynchronous-memory-operations)

4. [4. Epilogue Fusion](#4-epilogue-fusion)
   - [4.1 Bias and Activation Operations](#41-bias-and-activation-operations)
   - [4.2 Fusion Operations](#42-fusion-operations)

5. [5. Key Differences and Design Philosophies](#5-key-differences-and-design-philosophies)

6. [6. Example Correspondence](#6-example-correspondence)

---

## 1. Definition

### 1.1 Core Domain Transformation Features

| Feature | nvFuser | CUTE | Cutlass |
|---------|---------|------|---------|
| **Split Domain** | [`IterDomain::split()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L133) | [`logical_divide()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L1575) | Layout operations via CUTE |
| **Merge Domain** | [`IterDomain::merge()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L124) | [`flatten()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L530) | Layout operations via CUTE |
| **Reorder Domain** | [`TensorDomain::reorder()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L698) | [`composition()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L1135) | Layout operations via CUTE |
| **Flatten Domain** | [`TensorDomain::flatten()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L719) | [`flatten()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L530) | Layout operations via CUTE |

### 1.2 Advanced Layout Operations

| Feature | nvFuser | CUTE | Cutlass |
|---------|---------|------|---------|
| **Tile/Divide** | [`IterDomain::split()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L133) | [`zipped_divide()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L1625) | Layout operations via CUTE |
| **Composition** | Multiple transformations | [`composition()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L1135) | Layout operations via CUTE |
| **Swizzle** | [`IterDomain::swizzle()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L355) | [`Swizzle` layouts](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/swizzle_layout.hpp#L71) | Layout operations via CUTE |
| **Resize/Expand** | [`IterDomain::resize()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L140) | [`composition()` with padding](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L1135) | Layout operations via CUTE |

### 1.3 Memory Layout and Access Patterns

| Feature | nvFuser | CUTE | Cutlass |
|---------|---------|------|---------|
| **Row-Major Layout** | Implicit in IterDomain | [`LayoutLeft`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L355) | Layout operations via CUTE |
| **Column-Major Layout** | Implicit in IterDomain | [`LayoutRight`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L363) | Layout operations via CUTE |
| **Stride Patterns** | [`IterDomain` stride info](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L82) | [`stride()` accessors](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L143) | Layout operations via CUTE |
| **Broadcast** | [`IterDomain::isBroadcast()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L175) | Zero-stride layouts | Layout operations via CUTE |

### 1.4 Tensor Operations

| Feature | nvFuser | CUTE | Cutlass |
|---------|---------|------|---------|
| **Tensor Creation** | [`TensorView`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L415) | [`make_tensor()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/tensor_impl.hpp#L566) | [`TensorView`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/tensor_view.h#L128) |
| **Tensor Slicing** | [`slice()` operations](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L688) | [`slice()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L688) | [`subview()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/tensor_view.h#L221) |
| **Tensor Reshape** | Multiple transformations | [`unflatten()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L543) | Layout operations via CUTE |

---

## 2. Scheduling

### 2.1 Parallelization and Execution

| Feature | nvFuser | CUTE | Cutlass |
|---------|---------|------|---------|
| **Thread Mapping** | [`ParallelType`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L227) | [`make_layout()` thread layouts](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L335) | GEMM kernel mapping |
| **Block Mapping** | [`isBlockDim()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L200) | [`blocked_product()` layouts](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L1749) | GEMM kernel mapping |
| **Grid Mapping** | [`isThreadDim()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L205) | [`domain_distribute()` layouts](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L1448) | GEMM kernel mapping |

### 2.2 Tensor Core MMA Operations

| Feature | nvFuser | CUTE | Cutlass |
|---------|---------|------|---------|
| **MMA Instructions** | [`isMma()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L344) | [`mma_atom`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/atom/mma_atom.hpp#L257) | [`Mma` operations](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp#L678) |
| **Tensor Core Layouts** | Instruction loops | [`mma_traits`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/atom/mma_traits_sm90_gmma.hpp#L241) | [`Mma` layouts](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp#L678) |
| **MMA Swizzling** | [`SwizzleType`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L355) | [`Swizzle` layouts](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/swizzle_layout.hpp#L71) | [`Mma` swizzle](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp#L678) |
| **Tensor Core Tiling** | [`IterDomain::split()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L133) | [`logical_divide()`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/layout.hpp#L1575) | [`Mma` tiling](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp#L678) |

### 2.3 Mathematical Operations

| Feature | nvFuser | CUTE | Cutlass |
|---------|---------|------|---------|
| **Reduction** | [`IterType::Reduction`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L170) | Reduction layouts | GEMM operations |
| **Broadcast** | [`IterType::Broadcast`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L180) | Zero-stride layouts | GEMM operations |
| **Gather/Scatter** | [`IterType::GatherScatter`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L185) | Strided layouts | GEMM operations |

---

## 3. Syncing

### 3.1 Asynchronous Memory Operations

| Feature | nvFuser | CUTE | Cutlass |
|---------|---------|------|---------|
| **TMA (Tensor Memory Accelerator)** | [`LoadStoreOpType::CpAsyncBulk`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L415) | [`copy_traits_sm90_tma`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/atom/copy_traits_sm90_tma.hpp#L1368) | [`Tma` operations](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp#L678) |
| **Asynchronous Copy** | [`TensorView` operations](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ir/internal_base_nodes.h#L415) | [`copy` operations](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cute/algorithm/copy.hpp#L398) | [`Copy` operations](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp#L678) |


---

## 4. Epilogue Fusion

### 4.1 Bias and Activation Operations

| Feature | nvFuser | CUTE | Cutlass |
|---------|---------|------|---------|
| **Bias Addition** | [`biasEpilogue()`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/tests/cpp/utils.cpp#L624) | Layout operations | [`EpilogueFusionParams`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/util/reference/host/conv.hpp#L90) |
| **Activation Functions** | [`LinearOp` with bias](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ops/composite.cpp#L201) | Layout operations | [`ActivationFunctor`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/epilogue/fusion/operations.hpp#L37) |
| **GELU Activation** | [`biasGeluFwd`](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/tests/cpp/test_gpu2.cpp#L1110) | Layout operations | [`ScaledGELU_taylor`](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/util/reference/host/gett.hpp#L647) |
| **ReLU Activation** | [`ReLU` operations](https://github.com/NVIDIA/Fuser/blob/24f20ed739ec7ab054299d4bd7d8abf981169be4/csrc/ops/composite.cpp#L201) | Layout operations | [`Clamp` as ReLU](https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/include/cutlass/util/reference/host/gett.hpp#L647) |



---

## 5. Key Differences and Design Philosophies

### nvFuser
- **Focus**: Dynamic tensor operations and fusion optimization with TMA support
- **Domain Model**: IterDomain-based with explicit transformation tracking and LoadStoreOp
- **Execution**: Runtime fusion and optimization with hardware acceleration
- **Key Strength**: Automatic kernel fusion and optimization with Tensor Core and TMA support

### CUTE
- **Focus**: Compile-time layout algebra and tensor operations with Tensor Core support
- **Domain Model**: Layout-based with mathematical composition and TMA integration
- **Execution**: Compile-time optimization with hardware acceleration
- **Key Strength**: Expressive layout algebra, compile-time optimization, and Tensor Core layouts

### Cutlass
- **Focus**: High-performance linear algebra kernels with Tensor Core support
- **Domain Model**: TensorView-based with BLAS-like operations and TMA support
- **Execution**: Optimized GEMM, Tensor Core MMA, and TMA operations
- **Key Strength**: Highly optimized matrix operations with hardware acceleration

---

## 6. Example Correspondence

The file `cutlass/examples/cute/nvfuser_layout_correspondence.cu` demonstrates the correspondence between these interfaces:

- **Split**: `IterDomain::split()` ↔ `logical_divide()` ↔ Layout operations via CUTE
- **Merge**: `IterDomain::merge()` ↔ `flatten()` ↔ Layout operations via CUTE
- **Reorder**: `TensorDomain::reorder()` ↔ `composition()` ↔ Layout operations via CUTE
- **Tile**: `IterDomain::split()` ↔ `zipped_divide()` ↔ Layout operations via CUTE

This comparison shows how these three interfaces provide different approaches to the same fundamental tensor operations, each optimized for their specific use cases. 