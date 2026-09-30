# =============================================================================
#  schwarz2lvl - two-level algebraic Schwarz preconditioner (MPI + Eigen)
#
#  make                 build bin/schwarz2lvl and bin/matgen
#  make tests           build the tests in bin/
#  make check           build and run the tests with 1, 2, 4 processes
#  make doc             Doxygen documentation in doc/html
#  make clean
#
#  Optional features (1 = on):   USE_MUMPS=1 USE_UMFPACK=1 USE_PARDISO=0
#  Library locations can be overridden on the command line or in a file
#  make.inc (see scripts/make.inc.g100 for CINECA Galileo100).
# =============================================================================
-include make.inc

ifeq ($(origin CXX),default)
CXX = mpicxx
endif
CXXFLAGS    ?= -std=c++17 -O3 -march=native -DNDEBUG
WARNINGS    ?= -Wall -Wextra -Wpedantic
USE_MUMPS   ?= 1
USE_UMFPACK ?= 1
USE_PARDISO ?= 0

EIGEN_INC   ?= /usr/include/eigen3
METIS_INC   ?=
METIS_LIB   ?= -lmetis
MUMPS_INC   ?=
MUMPS_LIB   ?= -ldmumps -lmumps_common -lpord -lscalapack-openmpi -llapack -lblas
SUITESPARSE_INC ?= /usr/include/suitesparse
SUITESPARSE_LIB ?= -lumfpack -lamd -lsuitesparseconfig
MKL_INC     ?= $(MKLROOT)/include
MKL_LIB     ?= -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# ----------------------------------------------------------------------------
MODULES  := core linalg partition preconditioner solver utils
INCLUDES := $(addprefix -Iinclude/,$(MODULES)) -Iexternal/spectra/include -isystem $(EIGEN_INC)
DEFINES  := -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
LIBS     := $(METIS_LIB)
ifneq ($(METIS_INC),)
INCLUDES += -I$(METIS_INC)
endif
ifeq ($(USE_MUMPS),1)
DEFINES  += -DUSE_MUMPS
LIBS     += $(MUMPS_LIB)
ifneq ($(MUMPS_INC),)
INCLUDES += -I$(MUMPS_INC)
endif
endif
ifeq ($(USE_UMFPACK),1)
DEFINES  += -DUSE_UMFPACK
INCLUDES += -I$(SUITESPARSE_INC)
LIBS     += $(SUITESPARSE_LIB)
endif
ifeq ($(USE_PARDISO),1)
DEFINES  += -DUSE_PARDISO
INCLUDES += -I$(MKL_INC)
LIBS     += $(MKL_LIB)
endif

BUILD := build
BIN   := bin
LIB_SRC  := $(filter-out src/main.cpp src/matgen.cpp,$(wildcard $(addprefix src/,$(addsuffix /*.cpp,$(MODULES)))))
LIB_OBJ  := $(patsubst src/%.cpp,$(BUILD)/%.o,$(LIB_SRC))
TEST_SRC := $(wildcard test/*.cpp)
TESTS    := $(patsubst test/%.cpp,$(BIN)/%,$(TEST_SRC))

.PHONY: all tests check doc clean
all: $(BIN)/schwarz2lvl $(BIN)/matgen

$(BIN)/schwarz2lvl: $(BUILD)/main.o $(LIB_OBJ) | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

$(BIN)/matgen: $(BUILD)/matgen.o $(BUILD)/utils/matrix_generators.o $(BUILD)/partition/matrix_market_io.o \
               $(BUILD)/partition/csr_matrix.o | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN)/test_%: $(BUILD)/test/test_%.o $(LIB_OBJ) | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

$(BUILD)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(WARNINGS) $(DEFINES) $(INCLUDES) -MMD -MP -c $< -o $@

$(BUILD)/test/%.o: test/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(WARNINGS) $(DEFINES) $(INCLUDES) -Itest -MMD -MP -c $< -o $@

$(BIN):
	@mkdir -p $(BIN)

tests: $(TESTS)

check: tests
	./scripts/run_tests.sh

doc:
	doxygen Doxyfile

clean:
	rm -rf $(BUILD) $(BIN) doc/html doc/latex

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)