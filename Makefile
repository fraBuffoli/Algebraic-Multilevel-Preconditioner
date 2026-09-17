# ====================================================================
# TWO-LEVEL SCHWARZ SOLVER - ROBUST LINKING VERSION
# ====================================================================

# Compilatore e flag (Manteniamo le macro per la coerenza dei tipi)
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic -DIDXTYPEWIDTH=32 -DREALTYPEWIDTH=64

# Nome dell'eseguibile finale
TARGET   := schwarz_solver

# Percorsi delle cartelle del tuo codice
INCDIR   := include
SRCDIR   := src
OBJDIR   := obj
EXTDIR   := external

# Flag di inclusione per il tuo codice e per Eigen3
INCLUDES := -I$(INCDIR) -I$(EXTDIR)/eigen3

# Linker flags: diciamo al sistema di agganciare la libreria METIS precompilata
LDFLAGS  := -lmetis

# Elenco dei file sorgenti e dei rispettivi oggetti
SOURCES  := $(SRCDIR)/main.cpp \
            $(SRCDIR)/matrix_market_io.cpp \
            $(SRCDIR)/graph_partitioner.cpp \
            $(SRCDIR)/restriction_operator.cpp \
            $(SRCDIR)/subdomain_topology.cpp \
            $(SRCDIR)/partition_of_unity.cpp \
            $(SRCDIR)/one_level_preconditioner.cpp \
            $(SRCDIR)/local_eigensolver.cpp \
            $(SRCDIR)/local_block_splitting.cpp

OBJECTS  := $(OBJDIR)/$(SRCDIR)/main.o \
            $(OBJDIR)/$(SRCDIR)/matrix_market_io.o \
            $(OBJDIR)/$(SRCDIR)/graph_partitioner.o \
            $(OBJDIR)/$(SRCDIR)/restriction_operator.o \
            $(OBJDIR)/$(SRCDIR)/subdomain_topology.o \
            $(OBJDIR)/$(SRCDIR)/partition_of_unity.o \
            $(OBJDIR)/$(SRCDIR)/one_level_preconditioner.o \
            $(OBJDIR)/$(SRCDIR)/local_block_splitting.o \
            $(OBJDIR)/$(SRCDIR)/local_eigensolver.o

HEADERS  := $(INCDIR)/sparse_matrix.hpp \
            $(INCDIR)/matrix_market_io.hpp \
            $(INCDIR)/graph_partitioner.hpp \
            $(INCDIR)/config.hpp \
            $(INCDIR)/restriction_operator.hpp \
            $(INCDIR)/subdomain_topology.hpp \
            $(INCDIR)/partition_of_unity.hpp \
            $(INCDIR)/preconditioner.hpp \
            $(INCDIR)/one_level_preconditioner.hpp \
            $(INCDIR)/local_block_splitting.hpp \
            $(INCDIR)/local_eigensolver.hpp \
            $(INCDIR)/timer.hpp

# Regola principale
all: $(TARGET)

# Regola di Link finale (Usa LDFLAGS per agganciare metis senza compilare .c)
$(TARGET): $(OBJECTS)
	@echo "Linking executable with METIS: $@"
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

# Regola per compilare i tuoi file C++ (src/)
$(OBJDIR)/$(SRCDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling C++ Source: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Regola di pulizia
clean:
	@echo "Cleaning compiled objects and executables..."
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
