PROJECT := sanalyzer
CONFIGS := Makefile.config

include $(CONFIGS)

OBJ_DIR := obj
SRC_DIR := src
INC_DIR := include
LIB_DIR := lib
PREFIX := $(INSTALL_DIR)

LIB := $(LIB_DIR)/lib$(PROJECT).so

CXX ?= g++

CXX_FLAGS ?=
INCLUDES ?=
LDFLAGS ?=
LINK_LIBS ?=

INCLUDES += -I$(INC_DIR)
INCLUDES += -I$(SANITIZER_TOOL_DIR)/gpu_src/include
INCLUDES += -I$(NV_NVBIT_DIR)/include

INCLUDES += -I$(CPP_TRACE_DIR)/include
LDFLAGS += -L$(CPP_TRACE_DIR)/lib -Wl,-rpath=$(CPP_TRACE_DIR)/lib
LINK_LIBS += -lcpp_trace

INCLUDES += -I$(PY_FRAME_DIR)/include
LDFLAGS += -L$(PY_FRAME_DIR)/lib -Wl,-rpath=$(PY_FRAME_DIR)/lib
LINK_LIBS += -lpy_frame


CXX_FLAGS += -std=c++17

ifeq ($(DEBUG), 1)
	CXX_FLAGS += -g -O0
else
	CXX_FLAGS += -O3
endif

SRCS := $(notdir $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*/*.cpp))
OBJS := $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SRCS)))

all: dirs libs
dirs: $(OBJ_DIR) $(LIB_DIR)
libs: $(LIB)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(LIB): $(OBJS)
	$(CXX) $(LDFLAGS) -fPIC -shared -o $@ $^ $(LINK_LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -fPIC -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/*/%.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -fPIC -c $< -o $@

.PHONY: clean
clean:
	-rm -rf $(OBJ_DIR) $(LIB_DIR) $(PREFIX)


.PHONY: install
install: all
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	cp -r $(LIB) $(PREFIX)/lib
	cp -r $(INC_DIR)/$(PROJECT).h $(PREFIX)/include
