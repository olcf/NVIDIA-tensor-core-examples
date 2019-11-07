NVIDIA Tensor Core Examples
===========================

This repository collects multiple examples for using NVIDIA Tensor Cores.
Please see individual examples for their licensing requirements.


Examples
--------

* [cudaTensorCoreGemm](cudaTensorCoreGemm/readme.txt) - Implements a GEMM operation using WMMA instructions
* [simpleCUBLASEx](simpleCUBLASEx/readme.txt) - Demonstrates an SGEMM using Tensor Cores via the cublasGemmEx
  API
* [simpleCUBLASHgemm](simpleCUBLASHgemm/readme.txt) - Demonstrates calling HGEMM directly from cuBLAS
* [simpleCUBLASSgemm](simpleCUBLASSgemm) - Demonstrates using Tensor Cores implicitly from SGEMM
* [CUTLASS WMMA GEMM](https://github.com/NVIDIA/cutlass/tree/master/examples/05_wmma_gemm) - Using WMMA instructions from the CUTLASS
  framework.
* [pictc](https://github.com/vishalmehta1991/pictc/README.md) - Implements a simple Particle-In-Cell pusher using Tensor Cores
* [DCGAN](https://github.com/NVIDIA/apex/tree/master/examples/dcgan) - Illustrates using Automatic Mixed Precision
  (AMP) within PyTorch using the DCGAN network.
* [ImageNet](https://github.com/NVIDIA/apex/tree/master/examples/imagenet) - Illustrates using Automatic Mixed
  Precision (AMP) with imagenet.

Instructions
------------

Some examples are stored in git submodules. It is necessary to call 
`git submodule init` after cloning or clone with the `--recursive-submodules`
option.
