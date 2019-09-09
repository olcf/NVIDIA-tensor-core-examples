NVIDIA Tensor Core Examples
===========================

This repository collects multiple examples for using NVIDIA Tensor Cores.
Please see individual examples for their licensing requirements.


Examples
--------

* cudaTensorCoreGemm - Implements a GEMM operation using WMMA instructions
* simpleCUBLASEx - Demonstrates an SGEMM using Tensor Cores via the cublasGemmEx
  API
* simpleCUBLASHgemm - Demonstrates calling HGEMM directly from cuBLAS
* simpleCUBLASSgemm - Demonstrates using Tensor Cores implicitly from SGEMM
* cutlass/examples/05\_wmma\_gemm - Using WMMA instructions from the CUTLASS
  framework.
* pictc - Implements a simple Particle-In-Cell pusher using Tensor Cores
* apex/tree/master/examples/dcgan - Illustrates using Automatic Mixed Precision
  (AMP) within PyTorch using the DCGAN network.
* apex/tree/master/examples/imagenet - Illustrates using Automatic Mixed
  Precision (AMP) with imagenet.

Instructions
------------

Some examples are stored in git submodules. It is necessary to call 
`git submodule init` after cloning or clone with the `--recursive-submodules`
option.
