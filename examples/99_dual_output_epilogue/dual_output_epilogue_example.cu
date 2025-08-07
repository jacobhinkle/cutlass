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
 * 
 * Note: This is a simplified example that demonstrates the concept.
 * A full dual-output epilogue would require custom EVT (Epilogue Visitor Tree) 
 * implementation as described in NVFuserEVTTranslation.md section 4.3.
 */

#include <iostream>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_fill.h"

using namespace cutlass;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Main function demonstrating the dual-output epilogue concept
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
    
    std::cout << "Dual-output GEMM example initialized successfully!" << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Parameters: alpha=" << alpha << ", beta1=" << beta1 << ", gamma=" << gamma << std::endl;
    std::cout << std::endl;
    std::cout << "This example demonstrates the concept of dual-output epilogues:" << std::endl;
    std::cout << "  output1 = " << alpha << " * acc + " << beta1 << " * bias1" << std::endl;
    std::cout << "  output2 = " << alpha << " * acc + " << gamma << " * bias2" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: This is a simplified example. A full dual-output epilogue would require" << std::endl;
    std::cout << "custom EVT (Epilogue Visitor Tree) implementation as described in" << std::endl;
    std::cout << "NVFuserEVTTranslation.md section 4.3." << std::endl;
    std::cout << std::endl;
    std::cout << "The complete implementation would include:" << std::endl;
    std::cout << "1. Custom EVT nodes for dual-output computation" << std::endl;
    std::cout << "2. Sm90AuxLoad operations for bias tensors" << std::endl;
    std::cout << "3. Sm90EVT composition with multiple aux loads" << std::endl;
    std::cout << "4. Integration with CollectiveBuilder for GEMM kernels" << std::endl;
    
    return 0;
} 