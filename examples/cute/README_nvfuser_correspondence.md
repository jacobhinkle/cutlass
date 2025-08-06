# CUTE Layout and nvFuser IterDomain Correspondence Example

This example demonstrates the correspondence between CUTE Layout transformations and nvFuser IterDomain transformations.

## Overview

The example shows how various layout transformations in CUTE correspond to domain transformations in nvFuser's IterDomain system:

### 1. Split Transformation
- **CUTE**: `logical_divide(layout, factor)` 
- **nvFuser**: `IterDomain::split(domain, factor, inner_split)`
- **Purpose**: Divides a layout into outer and inner components

### 2. Merge Transformation  
- **CUTE**: `flatten(layout.shape())`
- **nvFuser**: `IterDomain::merge(outer_domain, inner_domain)`
- **Purpose**: Combines multiple dimensions into a single dimension

### 3. Reorder Transformation
- **CUTE**: `make_layout(new_shape, new_stride)`
- **nvFuser**: `TensorDomain::reorder(old2new_map)`
- **Purpose**: Changes the order of dimensions

### 4. Tile Transformation
- **CUTE**: `zipped_divide(layout, tiler)`
- **nvFuser**: Multiple `IterDomain::split()` calls for tiling
- **Purpose**: Divides a layout into tiles and residuals

### 5. Flatten Transformation
- **CUTE**: `flatten(layout.shape())`
- **nvFuser**: `TensorDomain::flatten(start_dim, end_dim)`
- **Purpose**: Flattens multiple dimensions into one

### 6. Composition Transformation
- **CUTE**: `composition(layout1, layout2)`
- **nvFuser**: Sequential application of multiple transformations
- **Purpose**: Combines multiple layout transformations

### 7. Stride Transformations
- **CUTE**: Different stride patterns (row-major vs column-major)
- **nvFuser**: Different memory layouts and access patterns
- **Purpose**: Demonstrates different memory access patterns

## Building and Running

```bash
# Build the example
ninja cute_nvfuser_layout_correspondence

# Run the example
./examples/cute/cute_nvfuser_layout_correspondence
```

## Key Concepts

### CUTE Layout System
- **Shape**: Defines the dimensions of the layout
- **Stride**: Defines how to traverse the layout
- **Size**: Total number of elements
- **Rank**: Number of dimensions
- **Depth**: Nesting level of the layout

### nvFuser IterDomain System
- **IterDomain**: Represents a single iterable dimension
- **TensorDomain**: Collection of IterDomains
- **Transformations**: Operations that modify domains
- **Parallelization**: How domains are mapped to hardware

## Correspondence Examples

The example output shows how:
- A 64-element layout `_64:_1` split by 8 becomes `(_8,_8):(_1,_8)`
- A (32,16,8) layout `(_32,_16,_8):(_1,_32,_512)` reordered becomes `(_8,_32,_16):(_1,_8,_256)`
- A 128-element layout `_128:_1` tiled by (16,8) becomes `((_16,_8),_1):((_1,_16),_0)` with nested structure
- Row-major `(_4,_8):(_1,_4)` and column-major `(_4,_8):(_8,_1)` layouts have different stride patterns

This demonstrates the fundamental relationship between CUTE's compile-time layout system and nvFuser's runtime domain transformation system. 