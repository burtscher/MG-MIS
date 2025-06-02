# MG-MIS v1.0

MG-MIS is a fast single-node multi-GPU (CUDA) implementation for computing maximal independent sets (MIS) in large undirected graphs. The algorithm is particularly optimized for graphs that do not fit in the global memory of a single GPU, but fit in the global memory of all the GPUs in the node combined. It operates on graphs stored in binary CSR format. The inputs used in the paper are generated using Graph500 and Indigo3. 

## Input generation

### To generate input using Graph500:
  Download the code from `https://github.com/graph500/graph500`
  
  Navigate to the `src` directory and compile the code:
  
  ```
  cd src
  make
  ``` 
  
  Ensure that `mpicc` is installed and the path to `openmpi` is specified in `PATH` and `LD_LIBRARY_PATH`
  
  Run the code:
      
  ```
  ./graph500_reference_bfs <scale> <edgefactor> graph.txt
  ```
 
#### To clean the generated graph (by removing self-loops and duplicate edges):

Compile the code: 
```
g++ -O3 clean_graph500.cpp -o convert
```

Run the code: 
```
./clean graph.txt graph.egr
```

### To generate input using Indigo3

Follow the instructions at: `https://github.com/burtscher/Indigo3Suite`

## Using MG-MIS

The MG-MIS CUDA code consists of the source files MG-MIS_10.cu and ECLgraph.h, located in the root directory of this repository. See the paper listed below for a description of MG-MIS. Note that MG-MIS is protected by the 3-Clause BSD license.

The MG-MIS code can be compiled as follows:
```
nvcc -O3 -arch=sm_70 -Xcompiler -fopenmp MG-MIS_10.cu -o mis
```

To compute an MIS of the input file graph.egr, enter:
```
./mis graph.egr <number of GPUs>
```

## Publication

Anju Mongandampulath Akathoott, Benila Virgin Jerald Xavier and Martin Burtscher. “A Multi-GPU Algorithm for Computing Maximal Independent Sets in Large Graphs.” Proceedings of the 2025 ACM International Conference on Supercomputing. June 2025. [pdf](https://userweb.cs.txstate.edu/~mb92/papers/ics25.pdf)



