# =============================================================================
#  schwarz2lvl - two-level algebraic Schwarz preconditioner (MPI + Eigen)
#
#  make                 build bin/schwarz2lvl and bin/matgen
#  make tests           build the tests in bin/
#  make check           build and run every test with 1, 2 and 4 processes (CHECK_NP)
#  make doc             Doxygen documentation in docs/Doxygen/html
#  make clean
#
#  Required: an MPI C++17 compiler (CXX, default mpicxx) and METIS 5. Eigen 3.4.0, Spectra 1.0.1
#  and AMGCL are bundled in external/ (header-only).
#  Optional features (1 = on, 0 = off):
#    USE_MUMPS    distributed coarse solver (MUMPS)           default: on if dmumps_c.h is found
#    USE_UMFPACK  local LU backend (SuiteSparse UMFPACK)       default: on if umfpack.h is found
#    USE_PARDISO  local backend (Intel MKL PARDISO)           default: off
#    USE_CPARDISO distributed coarse solver (MKL cluster_sparse_solver; needs the MKL BLACS
#                 library of the MPI in use, MKL_BLACS_LIB)   default: off
#    USE_AMGCL    AMG V-cycles as approximate local solvers (--local-approx amg; bundled
#                 header-only library, nothing to link)      default: on
#  Every variable below can be overridden on the command line (make USE_MUMPS=0) or in a file
#  make.inc next to this Makefile (see README.md for CINECA Galileo100).
#  Compile flags are not tracked as dependencies: run `make clean` after changing a USE_* switch.
# =============================================================================
-include make.inc

ifeq ($(origin CXX),default)
CXX = mpicxx
endif
CXXFLAGS    ?= -std=c++17 -O3 -march=native -DNDEBUG
WARNINGS    ?= -Wall -Wextra -Wpedantic

EIGEN_INC   ?= external/eigen3
METIS_INC   ?=
METIS_LIB   ?= -lmetis
MUMPS_INC   ?=
MUMPS_LIB   ?= -ldmumps -lmumps_common -lpord -lscalapack-openmpi -llapack -lblas
SUITESPARSE_INC ?= /usr/include/suitesparse
SUITESPARSE_LIB ?= -lumfpack -lamd -lsuitesparseconfig
MKL_INC     ?= $(MKLROOT)/include
MKL_LIB     ?= -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
MKL_BLACS_LIB ?= -lmkl_blacs_intelmpi_lp64
AMGCL_INC   ?= external/amgcl

# 1 if the compiler finds header $(1) with the extra flags $(2), 0 otherwise. (A '#' written inside
# a function call keeps its backslash in GNU make 4.3: it comes from a variable instead.)
HASH := \#
have_header = $(shell printf '$(HASH)include <$(1)>\n' | $(CXX) $(2) -E -x c++ - >/dev/null 2>&1 && echo 1 || echo 0)

USE_MUMPS   ?= $(call have_header,dmumps_c.h,$(if $(MUMPS_INC),-I$(MUMPS_INC)))
USE_UMFPACK ?= $(call have_header,umfpack.h,-I$(SUITESPARSE_INC))
USE_PARDISO ?= 0
USE_CPARDISO ?= 0
USE_AMGCL   ?= 1

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
ifeq ($(USE_CPARDISO),1)
DEFINES  += -DUSE_CPARDISO
ifneq ($(USE_PARDISO),1)
INCLUDES += -I$(MKL_INC)
LIBS     += $(MKL_LIB)
endif
LIBS     += $(MKL_BLACS_LIB)
endif
ifeq ($(USE_AMGCL),1)
DEFINES  += -DUSE_AMGCL -DAMGCL_NO_BOOST
INCLUDES += -isystem $(AMGCL_INC)
endif

# Early, readable errors instead of a failing compilation (skipped by `make clean`).
ifneq ($(filter-out clean doc,$(or $(MAKECMDGOALS),all)),)
ifeq ($(shell command -v $(firstword $(CXX)) 2>/dev/null),)
$(error compiler '$(firstword $(CXX))' not found: install an MPI implementation (e.g. libopenmpi-dev) or set CXX)
endif
ifeq ($(call have_header,metis.h,$(if $(METIS_INC),-I$(METIS_INC))),0)
$(error metis.h not found: install METIS 5 (e.g. libmetis-dev) or set METIS_INC / METIS_LIB)
endif
$(info schwarz2lvl: CXX=$(CXX)  USE_MUMPS=$(USE_MUMPS) USE_UMFPACK=$(USE_UMFPACK) USE_PARDISO=$(USE_PARDISO) USE_CPARDISO=$(USE_CPARDISO) USE_AMGCL=$(USE_AMGCL)  EIGEN_INC=$(EIGEN_INC))
endif

BUILD := build
BIN   := bin
LIB_SRC  := $(filter-out src/main.cpp src/matgen.cpp,$(wildcard $(addprefix src/,$(addsuffix /*.cpp,$(MODULES)))))
LIB_OBJ  := $(patsubst src/%.cpp,$(BUILD)/%.o,$(LIB_SRC))
TEST_SRC := $(wildcard test/*.cpp)
TEST_OBJ := $(patsubst test/%.cpp,$(BUILD)/test/%.o,$(TEST_SRC))
TESTS    := $(patsubst test/%.cpp,$(BIN)/%,$(TEST_SRC))

# The test objects are built through a chain of pattern rules: without this, make deletes them as
# intermediate files after linking, and recompiles every test at the next `make tests` / `make check`.
.SECONDARY: $(TEST_OBJ)

# make check: every test with every process count of CHECK_NP. Open MPI refuses more processes
# than cores unless --oversubscribe is given: it is added automatically when mpirun is Open MPI.
MPIEXEC       ?= mpirun
MPIEXEC_FLAGS ?= $(shell $(MPIEXEC) --version 2>/dev/null | grep -q 'Open MPI' && echo --oversubscribe)
CHECK_NP      ?= 1 2 4

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

# One line per (test, process count); the output of every run is kept in bin/<test>.np<N>.log.
check: tests
	@fail=0; \
	for t in $(TESTS); do \
	  for np in $(CHECK_NP); do \
	    printf '  %-22s np=%-3s ' "$$(basename $$t)" "$$np"; \
	    if OMP_NUM_THREADS=1 $(MPIEXEC) $(MPIEXEC_FLAGS) -np $$np $$t > $$t.np$$np.log 2>&1; then echo ok; \
	    else echo "FAILED (see $$t.np$$np.log)"; fail=1; fi; \
	  done; \
	done; \
	if [ $$fail -eq 0 ]; then echo "make check: all tests passed"; else echo "make check: some tests FAILED"; exit 1; fi

doc:
	@mkdir -p docs/Doxygen
	doxygen Doxyfile

clean:
	rm -rf $(BUILD) $(BIN) docs/Doxygen

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
