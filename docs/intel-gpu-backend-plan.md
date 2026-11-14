# Intel GPU backend feasibility and roadmap

## Goal

Evaluate whether ghost-infer can support Intel GPUs via a SYCL / oneAPI backend while preserving the current CUDA path and overall architecture.

## Why this matters

The project is currently optimized for NVIDIA GPUs. The repository documentation and code make that constraint explicit: the build and runtime are centered on CUDA, NVRTC, and NVIDIA-specific kernels, and the README states that Windows + NVIDIA is the supported baseline.

This proposal keeps the current engine intact and explores how to add a second backend for Intel GPUs without forcing a full rewrite of the CPU/server/app layers.

## Current repository context

Key observations from the codebase:

- The root binary is a single Rust program and the project is mostly Rust.
- GPU execution is isolated behind a `cuda` module and a `Backend` enum in `src/main.rs`.
- The engine can run either on CPU or on the GPU, and the CUDA backend is a first-class runtime path.
- The project’s publicly documented support scope is currently Windows + NVIDIA GPUs.
- The repo includes a desktop app, server API, and GGUF model pipeline that are mostly independent of the GPU vendor.

This gives us a clean backend boundary: the engine, tokenizer, GGUF handling, server, app, and orchestration do not need to be rewritten to add Intel support.

## Feasibility

Adding Intel GPU support is feasible, but it is a backend port, not a tiny compatibility tweak.

The most realistic approach is:

1. Keep the existing CUDA backend as the default and reference implementation.
2. Add a new Intel backend behind the same abstraction layer used for CPU vs. CUDA.
3. Use SYCL / oneAPI for Intel GPU kernels and runtime integration.
4. Start with a narrow, dense-model prototype before expanding to MoE and advanced optimizations.

This is a substantial engineering effort because the project’s performance path depends on vendor-specific acceleration and custom kernels, not just generic matrix math.

## Main technical risks

### 1. CUDA-specific kernel design

The current engine relies on optimized CUDA operations, including tensor-core-heavy paths and specialized kernels for generation, attention, attention decode, and MoE routing. A SYCL backend will need equivalent operations and likely different scheduling strategies.

### 2. Runtime and compilation model

The current build path expects NVRTC and NVIDIA runtime discovery. Intel support requires a parallel oneAPI / DPC++ toolchain, Level Zero runtime support, and separate GPU discovery code.

### 3. Feature parity

Full compatibility with the repository’s advanced features is a bigger challenge than a simple dense model prototype. MoE routing, expert caches, speculative decoding, long-context handling, and quantization support all need Intel-aware implementations.

### 4. Performance expectations

The repo is explicitly tuned for a specific GPU class and optimized to push throughput. Intel support should not be measured against the CUDA backend immediately; it should be evaluated with a realistic baseline on the target Intel hardware and model class.

## Recommended approach

### Phase 1: feasibility prototype

Target a single dense model with a supported quantization and validate the flow end-to-end:

- build on Intel oneAPI / DPC++ toolchain
- confirm Level Zero GPU discovery and context creation
- run the model on a single Intel GPU
- verify logits and token generation against the CPU reference or known outputs
- benchmark prompt and generation throughput

This should be treated as a prototype gate, not a full feature release.

### Phase 2: minimal Intel backend

Once the prototype works, add the backend behind the existing engine abstraction:

- CPU backend remains unchanged
- CUDA backend remains unchanged
- Intel backend becomes a third backend option
- Model selection logic is extended to detect Intel devices and choose the backend

### Phase 3: broader parity

After the minimal backend is stable, expand scope to:

- supported quantized formats
- prompt processing / context management
- KV cache support
- speculative decoding or equivalent performance optimizations
- expert routing and fused paths for MoE models

This phase should only proceed once the prototype demonstrates correctness and stable performance.

## Intel toolchain recommendation

The practical path for Intel GPU support is SYCL via the Intel oneAPI ecosystem:

- DPC++ / C++ compiler
- oneAPI Base / HPC toolkits
- Level Zero runtime for Intel GPUs
- SYCL Graph support where appropriate for reducing launch overhead

This is the most mature route for Intel GPU programming and aligns with the broader ecosystem used by Intel GPU workloads.

## Scope statement for the PR

This PR is intentionally a feasibility and planning document rather than a production backend implementation. It makes the following explicit:

- the repo is currently NVIDIA-first
- adding Intel GPU support is possible
- the right architecture is a parallel SYCL-based backend behind the existing abstraction
- the correct path is a staged prototype before broad feature parity

## Success criteria

A meaningful first success milestone would be:

- a single Intel GPU device is detected and selectable
- a supported dense GGUF model runs on that GPU
- output matches a known-good CPU or CUDA reference for a small workload
- generation throughput is acceptable for the target device

Once this succeeds, the project can decide whether to continue with a broader Intel backend effort.

## Proposed PR title

Add Intel GPU backend feasibility and roadmap

## Proposed PR summary

This PR documents the feasibility of adding an Intel GPU backend to ghost-infer using SYCL / oneAPI while preserving the current CUDA path. It explains the current NVIDIA-first architecture, identifies the main technical risks, and proposes a staged implementation plan starting with a dense-model prototype before broader parity work.

## Notes

This is a documentation-first PR meant to establish a realistic engineering direction without overcommitting to a full Intel port in one step.

## Related files

- `README.md`
- `docs/building.md`
- `src/main.rs`
- future backend abstraction and Intel-specific implementation files

This proposal intentionally keeps the project’s existing user-facing architecture intact and focuses on the backend decision point.

---

This document is intended as a project-planning artifact and may be refined as the Intel backend prototype becomes more concrete.
