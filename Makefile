# ====================================================================
# TWO-LEVEL SCHWARZ SOLVER - STRUCTURAL ALGEBRAIC BUILD
# ====================================================================

# Compilatore e flag
CXX      := mpicxx
CXXFLAGS := -std=c++17 -O3 -Wall -DIDXTYPEWIDTH=32 -DREALTYPEWIDTH=64

# Nome dell'eseguibile principale
TARGET   := schwarz_solver

# Percorsi base delle cartelle
INCDIR   := include
SRCDIR   := src
OBJDIR   := obj
EXTDIR   := external
TESTDIR  := test
BINDIR   := bin

# Flag di inclusione
INCLUDES := -I$(INCDIR) \
            -I$(INCDIR)/solver \
            -I$(INCDIR)/preconditioner \
            -I$(INCDIR)/utils \
            -I$(INCDIR)/core \
            -I$(INCDIR)/linalg \
            -I$(INCDIR)/partition \
            -I$(EXTDIR)/eigen3 \
            -I$(EXTDIR)/metis/include \
            -I$(EXTDIR)/GKlib

# --- SORGENTI APPLICAZIONE PRINCIPALE ---
SOURCES  := $(SRCDIR)/main.cpp \
            $(SRCDIR)/partition/matrix_market_io.cpp \
            $(SRCDIR)/partition/graph_partitioner.cpp \
            $(SRCDIR)/partition/subdomain_topology.cpp \
            $(SRCDIR)/partition/domain_decomposer.cpp \
			$(SRCDIR)/partition/matrix_distributor.cpp \
            $(SRCDIR)/preconditioner/restriction_operator.cpp \
            $(SRCDIR)/preconditioner/partition_of_unity.cpp \
            $(SRCDIR)/preconditioner/one_level_preconditioner.cpp \
            $(SRCDIR)/preconditioner/local_block_splitting.cpp \
            $(SRCDIR)/preconditioner/local_eigensolver.cpp \
            $(SRCDIR)/preconditioner/coarse_space.cpp \
            $(SRCDIR)/preconditioner/additive_two_level_preconditioner.cpp \
            $(SRCDIR)/preconditioner/deflated_two_level_preconditioner.cpp \
			$(SRCDIR)/preconditioner/distributed_one_level_preconditioner.cpp \
			$(SRCDIR)/preconditioner/distributed_deflated_two_level_preconditioner.cpp \
			$(SRCDIR)/preconditioner/distributed_additive_two_level_preconditioner.cpp \
			$(SRCDIR)/preconditioner/distributed_coarse_space.cpp \
            $(SRCDIR)/solver/krylov_gmres.cpp \
            $(SRCDIR)/solver/krylov_bicgstab.cpp \
			$(SRCDIR)/solver/distributed_gmres.cpp \
			$(SRCDIR)/solver/distributed_bicgstab.cpp \
            $(SRCDIR)/linalg/halo_exchange.cpp \
			$(SRCDIR)/core/local_index_map.cpp \
			$(SRCDIR)/linalg/local_matrix.cpp \
			$(SRCDIR)/linalg/distributed_spmv.cpp 

# Mappatura oggetti applicazione principale
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/src/%.o)

# Oggetti condivisi con i test (TUTTI tranne il main.o principale per evitare conflitti)
SHARED_OBJECTS := $(filter-out $(OBJDIR)/src/main.o, $(OBJECTS))

# --- SORGENTI E TARGET DEI TEST (DINAMICI) ---
# Trova in automatico tutti i file .cpp dentro la cartella test/
TEST_SOURCES := $(wildcard $(TESTDIR)/*.cpp)
# Genera la lista dei singoli eseguibili di test (es. bin/test_halo)
TEST_TARGETS := $(patsubst $(TESTDIR)/%.cpp, $(BINDIR)/%, $(TEST_SOURCES))

# Linker flags
LDFLAGS  := -lmetis

# --------------------------------------------------------------------
# REGOLI DI BUILD
# --------------------------------------------------------------------

# Default: compila solo l'applicazione principale
all: $(TARGET)

# Regola per compilare l'eseguibile principale
$(TARGET): $(OBJECTS)
	@echo "Linking final app executable: $@"
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

# COMPILA SOLO I TEST: Crea tutti gli eseguibili dentro la cartella bin/
test: $(SHARED_OBJECTS) $(TEST_TARGETS)
	@echo "All tests compiled successfully inside '$(BINDIR)/'!"

# Regola generica per linkare un singolo test indipendente
$(BINDIR)/%: $(OBJDIR)/test/%.o $(SHARED_OBJECTS)
	@mkdir -p $(BINDIR)
	@echo "Linking test executable: $@"
	$(CXX) $(CXXFLAGS) $< $(SHARED_OBJECTS) $(LDFLAGS) -o $@

# Regola per compilare i file sorgenti in src/
$(OBJDIR)/src/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling C++ Source: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Regola per compilare i file sorgenti in test/
$(OBJDIR)/test/%.o: $(TESTDIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling C++ Test Source: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# REGOLA SPERIMENTALE: Esegui al volo un test specifico scrivendo "make test_nomefile"
# Esempio: "make test_halo" compila le dipendenze e lancia subito bin/test_halo
test_%: $(BINDIR)/test_%
	@echo "Running test: $<"
	@./$<

# --------------------------------------------------------------------
# UTILITY
# --------------------------------------------------------------------

# Generazione documentazione Doxygen
doc:
	@if command -v doxygen > /dev/null; then \
		echo "Creating required documentation directories..."; \
		mkdir -p docs/Doxygen; \
		echo "Generating Doxygen documentation using Doxyfile..."; \
		doxygen Doxyfile; \
		echo "Documentation generated successfully."; \
	else \
		echo "Error: 'doxygen' command not found."; \
		echo "To install it on Ubuntu/WSL, run: sudo apt install doxygen graphviz"; \
	fi

# Pulizia profonda
clean:
	@echo "Cleaning compiled objects, bins and executables..."
	rm -rf $(OBJDIR) $(BINDIR) $(TARGET)
	@echo "Cleaning all generated documentation directories..."
	rm -rf docs html latex

.PHONY: all clean doc test