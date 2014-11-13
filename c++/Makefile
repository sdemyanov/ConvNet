CC        := g++
CCFLAGS   := -std=c++11 -m64 -O3 -shared -fPIC

kMexFolder  := .
kCppInclude := $(kMexFolder)/include
kCppSources := $(kMexFolder)/sources
CPU_HEADERS := $(wildcard $(kCppInclude)/*.h)
CPU_SOURCES := $(wildcard $(kCppSources)/*.cpp)
INCLUDES    := -I$(kCppInclude)

kCudaFolder  := $(kMexFolder)/cuda
kCudaInclude := $(kCudaFolder)/include
kCudaSources := $(kCudaFolder)/sources
CUDA_HEADERS := $(wildcard $(kCudaInclude)/*.h)
CUDA_SOURCES := $(wildcard $(kCudaSources)/*.cu)
INCLUDES     += -I$(kCudaInclude)

MATLABDIR := /usr/local/MATLAB/R2014b
MATLABINC := $(MATLABDIR)/extern/include
MATLABLIB := $(MATLABDIR)/bin/glnxa64
LIBRARIES := -L$(MATLABLIB) -lut
INCLUDES  += -I$(MATLABINC)

######### COMPILE CPP OBJFILES #########

kObjFolder   := $(kMexFolder)/obj
OBJ_FILES    := $(addprefix $(kObjFolder)/, $(notdir $(CPU_SOURCES:.cpp=.o)))
kBuildFolder := $(kMexFolder)/build/gpu

$(kObjFolder)/%.o: $(kCppSources)/%.cpp $(CPU_HEADERS) $(CUDA_HEADERS)
	$(CC) -c $< $(CCFLAGS) $(INCLUDES) -o $@

######### COMPILE CUDA OBJFILES #########

CUDADIR   := /usr/local/cuda
CUDABIN   := $(CUDADIR)/bin
CUDAINC   := $(CUDADIR)/include
CUDALIB   := $(CUDADIR)/lib64
INCLUDES  += -I$(CUDAINC)
LIBRARIES += -L$(CUDALIB) -lcudart -lcurand -lcublas

NVCC           := $(CUDABIN)/nvcc
GENCODE_SM35   := -gencode arch=compute_35,code=sm_35
NVCCFLAGS      := $(GENCODE_SM35) -m64 -O3 -std=c++11 --compiler-options '-fPIC'

kCudaObjFolder   := $(kCudaFolder)/obj
CUDA_OBJ_FILES   := $(addprefix $(kCudaObjFolder)/, $(notdir $(CUDA_SOURCES:.cu=.cu.o)))

$(kCudaObjFolder)/%.cu.o: $(kCudaSources)/%.cu $(CUDA_HEADERS)
	$(NVCC) -c $< $(NVCCFLAGS) $(INCLUDES) -o $@

######### COMPILE MEXFILES #########

MEX    := $(MATLABDIR)/bin/mex
MEXEXT := .$(shell $(MATLABDIR)/bin/mexext)

%_mex.cpp: $(OBJ_FILES) $(CUDA_OBJ_FILES)
	$(CC) $@ $^ $(CCFLAGS) $(INCLUDES) $(LIBRARIES) -o $(kBuildFolder)/$(@:.cpp=$(MEXEXT))

######### TARGETS #########

.SECONDARY: $(OBJ_FILES) $(CUDA_OBJ_FILES)

all: make_dirs cnntrain classify genweights

cnntrain: cnntrain_mex.cpp
classify: classify_mex.cpp
genweights: genweights_mex.cpp

make_dirs:
	mkdir -p $(kObjFolder)
	mkdir -p $(kCudaObjFolder)
	mkdir -p $(kBuildFolder)


