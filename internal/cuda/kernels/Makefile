NVCC ?= nvcc
# Default sm_75 (Turing T4). Override for other GPUs:
#   make CUDA_ARCH=sm_80   # Ampere A100
#   make CUDA_ARCH=sm_89   # Ada Lovelace L4/L40
#   make CUDA_ARCH=sm_121  # Blackwell GB10 (DGX Spark)
CUDA_ARCH ?= sm_75
NVCC_FLAGS ?= -O2 -arch=$(CUDA_ARCH)

SRCS = elementwise.cu flash_attention.cu gemm_int8.cu gemm_int4.cu
OBJS = $(SRCS:.cu=.o)
LIB  = libkernels.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB)
