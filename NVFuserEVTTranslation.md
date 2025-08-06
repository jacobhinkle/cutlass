# NVFuser to Cutlass EVT Translation Strategy

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overview of Translation Strategy](#2-overview-of-translation-strategy)
3. [Core Translation Patterns](#3-core-translation-patterns)
4. [Runtime Code Generation Architecture](#4-runtime-code-generation-architecture)
5. [EVT Node Translation](#5-evt-node-translation)
6. [Fusion Operation Mapping](#6-fusion-operation-mapping)
7. [Advanced Patterns](#7-advanced-patterns)
8. [Implementation Strategy](#8-implementation-strategy)
9. [Questions and Ambiguities](#9-questions-and-ambiguities)

---

## 1. Introduction

This document outlines a strategy for translating epilogue operations from nvFuser fusion graphs to Cutlass Epilogue Visitor Trees (EVT). The goal is to enable nvFuser to generate optimized GEMM kernels that leverage Cutlass's sophisticated epilogue fusion capabilities while maintaining nvFuser's high-level fusion abstractions.

### 1.1 Key Concepts

- **nvFuser**: Deep learning compiler that generates fused CUDA kernels
- **Cutlass EVT**: Epilogue Visitor Tree pattern for composable epilogue operations
- **Runtime Code Generation**: Dynamic C++ code generation at kernel compilation time
- **Epilogue Fusion**: Post-GEMM operations like bias addition, activation functions, and reductions

### 1.2 Motivation

By translating nvFuser epilogue operations to Cutlass EVT patterns, we can:
- Leverage Cutlass's highly optimized epilogue implementations
- Maintain nvFuser's high-level fusion abstractions
- Enable sophisticated epilogue patterns (bias + activation, reductions, etc.)
- Generate code that targets modern GPU architectures (Hopper, Blackwell)

---

## 2. Overview of Translation Strategy

### 2.1 High-Level Approach

The translation strategy involves three main phases:

1. **Analysis Phase**: Parse nvFuser fusion graph to identify epilogue operations
2. **Translation Phase**: Map nvFuser operations to Cutlass EVT patterns
3. **Code Generation Phase**: Generate C++ code that constructs the appropriate EVT

### 2.2 Runtime Code Generation Strategy

The system will generate C++ code at runtime that:
- Defines custom EVT node types when needed
- Constructs EVT trees using Cutlass's builder patterns
- Integrates with Cutlass's CollectiveBuilder for kernel generation
- Handles parameter passing and memory management

---

## 3. Core Translation Patterns

### 3.1 Basic Linear Operations

**nvFuser Pattern:**
```cpp
// nvFuser fusion operation
// acc is the matmul result and C is a bias tensor
TensorView* A = TensorViewBuilder().shape({-1, 1, -1}).dtype(DataType::BFloat16);
TensorView* B = TensorViewBuilder().shape({1, -1, -1}).dtype(DataType::BFloat16);
fusion->addInput(A);
fusion->addInput(B);
fusion->addInput(C);
TensorView* acc = fusedMultiplySum(A, B, {-1});
TensorView* alpha_acc = mul(alpha, acc);
TensorView* beta_C = mul(beta, C);
// output = alpha * acc + beta * C;
TensorView* output = add(alpha_acc, beta_C);
TensorView* output_bf16 = castOp(DataType::BFloat16, output);
fusion->addOutput(output_bf16);
```

**Generated C++ Code:**
```cpp
// Generated EVT code
using EVTOp = cutlass::epilogue::fusion::LinComb<
    ElementD, ElementCompute, ElementC, ElementScalar>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecialized,
    EVTOp
>::CollectiveOp;
```

### 3.2 Bias Addition

**nvFuser Pattern:**
```cpp
// nvFuser bias addition
auto output = acc + bias;
```

**Generated C++ Code:**
```cpp
// Generated EVT code
using EVTOp = cutlass::epilogue::fusion::LinCombPerRowBias<
    ElementD, ElementCompute, ElementBias, ElementC, ElementScalar>;

// Arguments structure
typename EVTOp::Arguments epilogue_args{
    .alpha = alpha,
    .beta = beta,
    .bias = bias_ptr,
    .bias_stride = bias_stride
};
```

### 3.3 Activation Functions

**nvFuser Pattern:**
```cpp
// nvFuser activation
auto output = relu(acc + bias);
```

**Generated C++ Code:**
```cpp
// Generated EVT code
using EVTOp = cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<
    cutlass::epilogue::thread::ReLU,
    ElementD, ElementCompute, ElementBias, ElementC, ElementScalar>;
```

### 3.4 Complex Patterns

**nvFuser Pattern:**
```cpp
// nvFuser complex pattern
auto intermediate = acc + bias;
auto output = gelu(intermediate);
auto aux_output = intermediate; // Store intermediate for backward pass
```

**Generated C++ Code:**
```cpp
// Generated EVT code
using EVTOp = cutlass::epilogue::fusion::LinCombPerRowBiasEltActAux<
    LayoutAux,
    cutlass::epilogue::thread::GELU,
    ElementD, ElementCompute, ElementAux, ElementBias, ElementC, ElementScalar>;
```

---

## 4. Runtime Code Generation Architecture

### 4.1 Code Generation Pipeline

```cpp
// Runtime code generation structure
class EVTCodeGenerator {
private:
    std::stringstream header_code_;
    std::stringstream implementation_code_;
    std::stringstream kernel_code_;
    
public:
    void generate_evt_definition(const nvFuser::Fusion& fusion);
    void generate_kernel_construction(const nvFuser::Fusion& fusion);
    std::string get_complete_source_code();
};
```

### 4.2 Template-Based Generation

The system will use C++ templates to generate specialized code:

```cpp
// Generated template specialization
template<typename ElementD, typename ElementCompute>
class GeneratedEVT {
    using EVTOp = cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<
        cutlass::epilogue::thread::ReLU,
        ElementD, ElementCompute, ElementBias, ElementC, ElementScalar>;
        
    static auto create_epilogue() {
        return typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            TileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccumulator, ElementCompute,
            ElementC, LayoutC, AlignmentC,
            ElementD, LayoutD, AlignmentD,
            cutlass::epilogue::TmaWarpSpecialized,
            EVTOp
        >::CollectiveOp{};
    }
};
```

### 4.3 Dynamic Type Generation

For complex patterns, the system will generate custom EVT node types:

```cpp
// Generated custom EVT node
template<typename ElementCompute>
struct CustomGELUNode {
    using ElementOutput = ElementCompute;
    using ElementCompute = ElementCompute;
    
    template<typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, 
          int epi_v, int epi_m, int epi_n) {
        Array<ElementCompute, FragmentSize> result;
        // Custom GELU implementation
        for (int i = 0; i < FragmentSize; ++i) {
            result[i] = gelu_activation(frg_acc[i]);
        }
        return result;
    }
};
```

---

## 5. EVT Node Translation

### 5.1 Operation Mapping Table

| nvFuser Operation | Cutlass EVT Pattern | Generated Code Type |
|-------------------|---------------------|-------------------|
| `LinearOp` | `LinComb` | Built-in |
| `BiasOp` | `LinCombPerRowBias` | Built-in |
| `ReLU` | `LinCombEltAct<ReLU>` | Built-in |
| `GELU` | `LinCombEltAct<GELU>` | Built-in |
| `Residual` | `LinCombResidual` | Built-in |
| `Reduction` | `Sm90ScalarReduction` | Built-in |
| `CustomOp` | Custom EVT Node | Generated |

### 5.2 Custom Operation Translation

For operations not available in Cutlass, the system generates custom EVT nodes:

```cpp
// Generated custom operation
template<typename ElementCompute>
struct CustomOperationNode {
    // Parameters for the custom operation
    struct Arguments {
        ElementCompute param1;
        ElementCompute param2;
        // ... other parameters
    };
    
    Arguments args_;
    
    template<typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, 
          int epi_v, int epi_m, int epi_n) {
        Array<ElementCompute, FragmentSize> result;
        for (int i = 0; i < FragmentSize; ++i) {
            result[i] = custom_operation(frg_acc[i], args_.param1, args_.param2);
        }
        return result;
    }
};
```

### 5.3 Composite Pattern Translation

For complex patterns involving multiple operations:

```cpp
// Generated composite EVT
using CompositeEVT = cutlass::epilogue::fusion::Sm90EVT<
    CustomOperationNode<ElementCompute>,
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::LinCombPerRowBias<
            ElementCompute, ElementCompute, ElementBias, ElementC, ElementScalar>,
        cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>
    >
>;
```

---

## 6. Fusion Operation Mapping

### 6.1 nvFuser to EVT Translation Rules

1. **Linear Operations**: Map to `LinComb` patterns
2. **Bias Operations**: Map to `LinCombPerRowBias` or `LinCombPerColBias`
3. **Activation Functions**: Map to `LinCombEltAct` with appropriate activation
4. **Reductions**: Map to `Sm90ScalarReduction`, `Sm90RowReduction`, or `Sm90ColReduction`
5. **Auxiliary Operations**: Map to `Sm90AuxLoad` or `Sm90AuxStore`
6. **Custom Operations**: Generate custom EVT nodes

### 6.2 Memory Layout Translation

```cpp
// Generated memory layout translation
template<typename nvFuser::TensorView>
struct LayoutTranslator {
    using CutlassLayout = cutlass::layout::RowMajor;
    
    static auto translate_layout(const nvFuser::TensorView& tv) {
        // Translate nvFuser layout to Cutlass layout
        auto strides = tv->getStrides();
        auto sizes = tv->getSizes();
        
        return CutlassLayout::packed({sizes[0], sizes[1]});
    }
};
```

### 6.3 Data Type Translation

```cpp
// Generated data type translation
template<typename nvFuser::DataType>
struct DataTypeTranslator {
    using CutlassType = cutlass::half_t; // Default
    
    template<>
    struct DataTypeTranslator<nvFuser::DataType::Float> {
        using CutlassType = float;
    };
    
    template<>
    struct DataTypeTranslator<nvFuser::DataType::Half> {
        using CutlassType = cutlass::half_t;
    };
};
```

---

## 7. Advanced Patterns

### 7.1 Multi-Stage Epilogues

For complex epilogues with multiple stages:

```cpp
// Generated multi-stage EVT
using MultiStageEVT = cutlass::epilogue::fusion::Sm90EVT<
    Stage3Node<ElementCompute>,
    cutlass::epilogue::fusion::Sm90EVT<
        Stage2Node<ElementCompute>,
        cutlass::epilogue::fusion::Sm90EVT<
            Stage1Node<ElementCompute>,
            cutlass::epilogue::fusion::Sm90AccFetch
        >
    >
>;
```

### 7.2 Conditional Operations

For operations with runtime conditions:

```cpp
// Generated conditional EVT
template<typename ElementCompute>
struct ConditionalNode {
    bool condition_;
    
    template<typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, 
          int epi_v, int epi_m, int epi_n) {
        if (condition_) {
            return operation_a(frg_acc);
        } else {
            return operation_b(frg_acc);
        }
    }
};
```

### 7.3 Reduction Patterns

For complex reduction patterns:

```cpp
// Generated reduction EVT
using ReductionEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90ScalarReduction<
        cutlass::epilogue::thread::Sum,
        cutlass::epilogue::thread::AtomicAdd,
        ElementOutput, ElementCompute
    >,
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::LinComb<
            ElementCompute, ElementCompute, ElementC, ElementScalar>,
        cutlass::epilogue::fusion::Sm90AccFetch
    >
>;
```

---

## 8. Implementation Strategy

### 8.1 Integration with nvFuser

1. **Analysis Phase**: Add EVT analysis to nvFuser's fusion graph analysis
2. **Translation Phase**: Implement translation logic in nvFuser's code generation
3. **Code Generation Phase**: Extend nvFuser's C++ code generation to include EVT patterns

### 8.2 Runtime Compilation

```cpp
// Generated runtime compilation code
class EVTCompiler {
public:
    static std::unique_ptr<CompiledKernel> compile_evt_kernel(
        const nvFuser::Fusion& fusion,
        const std::string& generated_code) {
        
        // Compile the generated C++ code
        auto compiled_kernel = compile_cuda_kernel(generated_code);
        
        // Return compiled kernel with EVT integration
        return std::make_unique<EVTCompiledKernel>(compiled_kernel);
    }
};
```

### 8.3 Performance Optimization

1. **Pattern Recognition**: Identify common patterns for optimized EVT construction
2. **Memory Optimization**: Optimize memory layouts for EVT operations
3. **Kernel Fusion**: Maximize fusion opportunities within EVT patterns

---

## 9. Questions and Ambiguities

### 9.1 Architecture and Compatibility

1. **GPU Architecture Support**: What is the minimum GPU architecture required for EVT support? Are there limitations for older architectures (pre-Hopper)?

2. **nvFuser Integration**: How deeply should EVT translation be integrated into nvFuser's existing fusion graph analysis? Should it be a separate pass or integrated into the main analysis pipeline?

3. **Backward Compatibility**: How should this system handle nvFuser operations that don't have direct EVT equivalents? Should we fall back to traditional nvFuser code generation?

### 9.2 Performance and Optimization

4. **Performance Overhead**: What is the expected performance overhead of generating EVT code at runtime versus using pre-compiled patterns?

5. **Memory Layout Optimization**: How should we handle complex memory layouts that don't directly map to Cutlass's layout system?

6. **Kernel Fusion Limits**: What are the practical limits on the complexity of EVT patterns that can be efficiently generated and compiled?

### 9.3 Implementation Details

7. **Custom Operation Support**: For nvFuser operations that don't have Cutlass equivalents, what is the preferred approach for generating custom EVT nodes? Should we generate inline CUDA code or use Cutlass's extension mechanisms?

8. **Parameter Passing**: How should we handle complex parameter passing between nvFuser operations and generated EVT nodes? What about dynamic parameters that are only known at runtime?

9. **Error Handling**: What should be the error handling strategy for cases where EVT translation fails or generates invalid code?

### 9.4 Advanced Features

10. **Conditional Operations**: How should we handle nvFuser operations with runtime conditions that affect the epilogue structure?

11. **Multi-GPU Support**: How should EVT translation work in multi-GPU scenarios where different GPUs might have different capabilities?

12. **Debugging and Profiling**: What debugging and profiling capabilities should be built into the EVT translation system?

### 9.5 Integration and Deployment

13. **Build System Integration**: How should the EVT code generation integrate with nvFuser's existing build system and dependency management?

14. **Testing Strategy**: What testing approach should be used to validate EVT translation correctness and performance?

15. **Documentation and Maintenance**: How should we document the translation patterns and maintain compatibility as both nvFuser and Cutlass evolve?

---

## References

- [Epilogue Visitor Trees in CUTLASS](https://research.colfax-intl.com/epilogue_visitor_tree/)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [nvFuser Source Code](https://github.com/NVIDIA/Fuser) 