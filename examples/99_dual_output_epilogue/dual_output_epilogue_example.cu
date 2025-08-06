/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file dual_output_epilogue_example.cu
 * 
 * This example demonstrates a GEMM kernel with a dual-output epilogue that computes:
 *   output1 = alpha * acc + beta1 * bias1
 *   output2 = alpha * acc + gamma * bias2
 * 
 * This corresponds to section 4.3 "Multiple Aux Inputs and Outputs Example" in
 * NVFuserEVTTranslation.md
 */

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tensor_op.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_evt.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_io.h"
#include "cutlass/util/tensor_view_io.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define the dual-output computation operation
template<typename ElementCompute>
struct DualOutputLinearCombination {
    struct Arguments {
        ElementCompute alpha;
        ElementCompute beta1;
        ElementCompute gamma;
    };
    
    Arguments args_;
    
    template<typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, 
          Array<ElementCompute, FragmentSize> const& frg_bias1,
          Array<ElementCompute, FragmentSize> const& frg_bias2,
          int epi_v, int epi_m, int epi_n) {
        
        // Create output fragments
        Array<ElementCompute, FragmentSize> result1;
        Array<ElementCompute, FragmentSize> result2;
        
        // Perform computations for both outputs
        for (int i = 0; i < FragmentSize; ++i) {
            // output1 = alpha * acc + beta1 * bias1
            result1[i] = args_.alpha * frg_acc[i] + args_.beta1 * frg_bias1[i];
            // output2 = alpha * acc + gamma * bias2
            result2[i] = args_.alpha * frg_acc[i] + args_.gamma * frg_bias2[i];
        }
        
        // Return tuple of both outputs
        return cute::make_tuple(result1, result2);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define the GEMM kernel with dual-output epilogue
template<typename ElementA, typename ElementB, typename ElementC, typename ElementD, typename ElementCompute>
struct DualOutputGemm {
    
    using ElementAccumulator = ElementCompute;
    
    // Define the MMA operation
    using Mma = cutlass::gemm::collective::CollectiveMma<
        cutlass::gemm::collective::KernelScheduleAuto,
        cutlass::gemm::collective::MmaAuto,
        cutlass::gemm::collective::TileShape_64x128x64,
        cutlass::gemm::collective::ClusterShape_1x2x1,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelTmaWarpSpecialized,
        ElementA, cutlass::layout::RowMajor,
        ElementB, cutlass::layout::ColumnMajor,
        ElementAccumulator, cutlass::layout::RowMajor,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm90
    >;
    
    // Define the epilogue tile
    using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
    
    // Define auxiliary load operations for both bias tensors
    using AuxLoadBias1 = cutlass::epilogue::fusion::Sm90AuxLoad<
        Mma::NumStages, EpilogueTile, ElementCompute, 
        cutlass::layout::RowMajor, cutlass::gemm::collective::SmemLayoutAtomAuto,
        cutlass::epilogue::thread::LinearCombination<ElementCompute, ElementCompute, ElementCompute>>;
    
    using AuxLoadBias2 = cutlass::epilogue::fusion::Sm90AuxLoad<
        Mma::NumStages, EpilogueTile, ElementCompute, 
        cutlass::layout::RowMajor, cutlass::gemm::collective::SmemLayoutAtomAuto,
        cutlass::epilogue::thread::LinearCombination<ElementCompute, ElementCompute, ElementCompute>>;
    
    // Define the dual-output computation
    using DualOutputCompute = DualOutputLinearCombination<ElementCompute>;
    
    // Compose the complete EVT with multiple aux loads
    using EVTOp = cutlass::epilogue::fusion::Sm90EVT<
        DualOutputCompute,
        AuxLoadBias1,
        AuxLoadBias2
    >;
    
    // Define the epilogue
    using Epilogue = cutlass::epilogue::collective::CollectiveEpilogue<
        cutlass::gemm::collective::EpilogueScheduleAuto,
        cutlass::gemm::collective::EpilogueTileAuto,
        cutlass::epilogue::thread::LinearCombination<ElementCompute, ElementCompute, ElementCompute>,
        cutlass::epilogue::thread::LinearCombination<ElementCompute, ElementCompute, ElementCompute>,
        cutlass::epilogue::collective::EpilogueTmaWarpSpecialized,
        EVTOp
    >;
    
    // Define the GEMM kernel
    using GemmKernel = cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, cutlass::layout::RowMajor,
        ElementB, cutlass::layout::ColumnMajor,
        ElementAccumulator, cutlass::layout::RowMajor,
        ElementD, cutlass::layout::RowMajor,
        cutlass::epilogue::collective::EpilogueTileAuto,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelTmaWarpSpecialized,
        cutlass::epilogue::collective::EpilogueTmaWarpSpecialized,
        EVTOp
    >;
    
    using GemmKernelType = typename GemmKernel::CollectiveOp;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Host function to run the dual-output GEMM
template<typename ElementA, typename ElementB, typename ElementC, typename ElementD, typename ElementCompute>
int run_dual_output_gemm(
    int M, int N, int K,
    ElementCompute alpha, ElementCompute beta1, ElementCompute gamma,
    ElementA const* A, int lda,
    ElementB const* B, int ldb,
    ElementC const* bias1, int ldbias1,
    ElementC const* bias2, int ldbias2,
    ElementD* output1, int ldoutput1,
    ElementD* output2, int ldoutput2) {
    
    using GemmKernel = typename DualOutputGemm<ElementA, ElementB, ElementC, ElementD, ElementCompute>::GemmKernelType;
    
    // Define the epilogue arguments
    typename GemmKernel::EpilogueOutputOp::Arguments epilogue_args{
        {alpha, beta1, gamma},  // DualOutputCompute arguments
        {bias1, ldbias1},       // AuxLoadBias1 arguments
        {bias2, ldbias2}        // AuxLoadBias2 arguments
    };
    
    // Define the GEMM arguments
    typename GemmKernel::Arguments args{
        {M, N, K},
        {A, lda},
        {B, ldb},
        {output1, ldoutput1, output2, ldoutput2},
        epilogue_args
    };
    
    // Create the GEMM kernel
    GemmKernel gemm_kernel;
    
    // Get the workspace size
    size_t workspace_size = gemm_kernel.get_workspace_size(args);
    
    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    
    // Initialize the kernel
    cutlass::Status status = gemm_kernel.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize GEMM kernel" << std::endl;
        if (workspace) cudaFree(workspace);
        return -1;
    }
    
    // Run the kernel
    status = gemm_kernel.run();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to run GEMM kernel" << std::endl;
        if (workspace) cudaFree(workspace);
        return -1;
    }
    
    // Cleanup
    if (workspace) cudaFree(workspace);
    
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Main function demonstrating the dual-output epilogue
int main() {
    
    // Problem size
    int M = 1024;
    int N = 1024;
    int K = 1024;
    
    // Scalar parameters
    float alpha = 1.0f;
    float beta1 = 0.5f;
    float gamma = 0.25f;
    
    // Allocate host memory
    cutlass::HostTensor<float, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<float, cutlass::layout::ColumnMajor> B({K, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> bias1({M, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> bias2({M, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> output1({M, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> output2({M, N});
    
    // Initialize tensors
    cutlass::reference::host::TensorFillRandomGaussian(A.host_view(), 0, 1.0f);
    cutlass::reference::host::TensorFillRandomGaussian(B.host_view(), 0, 1.0f);
    cutlass::reference::host::TensorFillRandomGaussian(bias1.host_view(), 0, 1.0f);
    cutlass::reference::host::TensorFillRandomGaussian(bias2.host_view(), 0, 1.0f);
    
    // Copy to device
    A.sync_device();
    B.sync_device();
    bias1.sync_device();
    bias2.sync_device();
    output1.sync_device();
    output2.sync_device();
    
    // Run the dual-output GEMM
    int result = run_dual_output_gemm<float, float, float, float, float>(
        M, N, K,
        alpha, beta1, gamma,
        A.device_data(), A.layout().stride(0),
        B.device_data(), B.layout().stride(0),
        bias1.device_data(), bias1.layout().stride(0),
        bias2.device_data(), bias2.layout().stride(0),
        output1.device_data(), output1.layout().stride(0),
        output2.device_data(), output2.layout().stride(0)
    );
    
    if (result == 0) {
        std::cout << "Dual-output GEMM completed successfully!" << std::endl;
        std::cout << "Computed: output1 = " << alpha << " * acc + " << beta1 << " * bias1" << std::endl;
        std::cout << "Computed: output2 = " << alpha << " * acc + " << gamma << " * bias2" << std::endl;
    } else {
        std::cerr << "Dual-output GEMM failed!" << std::endl;
        return -1;
    }
    
    return 0;
} 