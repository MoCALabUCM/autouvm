# autouvm

## Profiler

profiler is built on the PASTA framework.

Installation:

requirement: NVIDIA GPU with CUDA >= 11.0, pytorch > 2.0

install command:
```shell
cd profiler
bash ./bin/utils/check_build_env.sh
./bin/build
```

code mapping: 

profiler/sanalyzer/src/tools/uvm_advisor.cpp: uvm analyzer. 

profiler/sanalyzer/src/tools/roofline_flops.cpp: per-kernel flop counting. 

profiler/sanalyzer/src/tools/roofline_time.cpp: per-kernel time measurement. 


## Prefetcher

Installation:

require the same build env with the profiler.

install command:
```
cd prefetcher
make -j
cd uvm_helper && make
```

code mapping:

prefetcher/op_callback_uvm.cpp: the uvm prefetch code. 

prefetcher/uvm_helper/ctrl_gddr_size.cu: the help process that controls the memory oversubscription ratio.
