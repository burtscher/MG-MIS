# MG-MIS v1.0

MG-MIS is a fast single-node multi-GPU (CUDA) implementation for computing maximal independent sets (MIS) in large undirected graphs. The algorithm is particularly optimized for graphs that do not fit in the global memory of a single GPU, but fit in the global memory of all the GPUs in the node combined. It operates on graphs stored in binary CSR format. 
