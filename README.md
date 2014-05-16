gls
===

GPU Lattice Siever

License: LGPL

This software performs the sieving step of the Number Field Sieve algorithm, offloading a large portion of the computationally or memory access intensive routines to any available OpenCL compatible processor.

This being said, this is an initial implementation to evaluate the different algorithmic components involved in lattice sieving in order to determine viability, bottlenecks, future directions, etc.  Currently the overall runtime is not as fast as say CADO's lattice siever.  The code will likely be heavily re-factored with the exception of the cofactorization.  Sieving and refactoring of small primes is currently done in a very inefficient way on the CPU.  Sieving by vectors currently doesn't offload the computationally heavy portion of its routines due to memory access patterns.  In a future rewrite we will test a merge sort based approach on the GPU to relieve this. 

The code has been tested and works on nVidia 600 and 700 GPUs and Intel and AMD CPUs.  It appears there are a number of issues with AMD/ATI's implementation of OpenCL (saving a long rant).

Portions of this code were based on routines found in CADO, msieve, and Bob Silverman's lattice siever.

