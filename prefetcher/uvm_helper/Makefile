# Find all .cu files
CU_FILES := $(wildcard *.cu)
# Convert .cu files to executable names *.cu -> *.out
EXECUTABLES := $(CU_FILES:.cu=.out)

all: $(EXECUTABLES)

# Pattern rule to compile .cu files
%.out: %.cu
	nvcc $< -o $@

clean:
	rm -f $(EXECUTABLES)
