TORCH_DIR = $(shell python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
PYTHON_INCLUDE_DIR = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB_DIR = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
PYTHON_VERSION = $(shell python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

NVCC = nvcc
NVCCFLAGS = -std=c++17 -O2 \
	-I$(TORCH_DIR)/include \
	-I$(TORCH_DIR)/include/torch/csrc/api/include \
	-I$(PYTHON_INCLUDE_DIR)

ifeq ($(DEBUG), 1)
	NVCCFLAGS += -g -G
endif

LDFLAGS = -L$(TORCH_DIR)/lib \
	-ltorch -ltorch_cpu -lc10 \
	-Xlinker -rpath -Xlinker $(TORCH_DIR)/lib \
	-L$(PYTHON_LIB_DIR)/../ -Xlinker -rpath -Xlinker $(PYTHON_LIB_DIR)/../ \
	-lpthread -ldl -lcudart \
	-lpython$(PYTHON_VERSION)

CUDA_PATH      ?= /usr/local/cuda
SANITIZER_PATH ?= $(CUDA_PATH)/compute-sanitizer

all: libop_callback_uvm.so

libop_callback_uvm.so: op_callback_uvm.cpp
	$(NVCC) $(NVCCFLAGS) -I$(SANITIZER_PATH)/include -Xcompiler -fPIC -shared op_callback_uvm.cpp -o libop_callback_uvm.so $(LDFLAGS) -L$(SANITIZER_PATH) -lsanitizer-public

clean:
	rm -f libop_callback_uvm.so

