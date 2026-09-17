# ====================================================================
# TWO-LEVEL SCHWARZ SOLVER - STRUCTURAL ALGEBRAIC BUILD
# ====================================================================

# Compilatore e flag (Macro per le larghezze dei tipi di METIS incluse)
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic -DIDXTYPEWIDTH=32 -DREALTYPEWIDTH=64

# Nome dell'eseguibile finale
TARGET   := schwarz_solver

# Percorsi base delle cartelle
INCDIR   := include
SRCDIR   := src
OBJDIR   := obj
EXTDIR   := external

# Flag di inclusione: indichiamo a g++ dove cercare gli header in automatico
INCLUDES := -I$(INCDIR) \
            -I$(INCDIR)/solver \
            -I$(INCDIR)/preconditioner \
            -I$(INCDIR)/utils \
            -I$(EXTDIR)/eigen3 \
            -I$(EXTDIR)/metis/include \
            -I$(EXTDIR)/GKlib

# Elenco blindato dei file sorgenti con i percorsi corretti delle sottocartelle
SOURCES  := $(SRCDIR)/main.cpp \
            $(SRCDIR)/utils/matrix_market_io.cpp \
            $(SRCDIR)/utils/graph_partitioner.cpp \
            $(SRCDIR)/preconditioner/restriction_operator.cpp \
            $(SRCDIR)/preconditioner/subdomain_topology.cpp \
            $(SRCDIR)/preconditioner/partition_of_unity.cpp \
            $(SRCDIR)/preconditioner/one_level_preconditioner.cpp \
            $(SRCDIR)/preconditioner/local_block_splitting.cpp \
            $(SRCDIR)/preconditioner/local_eigensolver.cpp \
            $(SRCDIR)/preconditioner/coarse_space.cpp \
            $(SRCDIR)/preconditioner/additive_two_level_preconditioner.cpp \
            $(SRCDIR)/preconditioner/deflated_two_level_preconditioner.cpp 

# Generazione automatica dei file oggetto speculari dentro la cartella obj/
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# Linker flags: aggancia la libreria METIS precompilata a livello di sistema
LDFLAGS  := -lmetis

# Regola principale di build
all: $(TARGET)

# Regola di Link finale (Unisce gli oggetti C++ e la libreria esterna)
$(TARGET): $(OBJECTS)
	@echo "Linking final executable with METIS: $@"
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

# Regola universale per compilare QUALSIASI file .cpp mantenendo l'albero delle cartelle
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling C++ Source: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Regola di pulizia profonda degli oggetti e dell'eseguibile
clean:
	@echo "Cleaning compiled objects and executables..."
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean