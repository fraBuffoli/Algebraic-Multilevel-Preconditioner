# ====================================================================
# TWO-LEVEL SCHWARZ SOLVER
# ====================================================================

# Compiler and flags
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic

# foders for source, object, and include files
INCDIR   := include
SRCDIR   := src
OBJDIR   := obj
EXTDIR   := external

INCLUDES := -I$(INCDIR) -I$(EXTDIR)/eigen3

TARGET   := schwarz_solver

# finds all files .cpp in src/
SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS  := $(SOURCES:%.cpp=$(OBJDIR)/%.o)

# header files (.hpp)
DEPENDS  := $(OBJECTS:%.o=%.d)

# Default target
all: $(TARGET)

# Rfinal link the executable from the object files
$(TARGET): $(OBJECTS)
	@echo "Linking executable: $@"
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@

-include $(DEPENDS)

# compile each .cpp file into a .o file in the obj/ directory, preserving the directory structure
$(OBJDIR)/$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) -MMD -MP $(INCLUDES) -c $< -o $@

# clean target to remove compiled objects and executables
clean:
	@echo "Cleaning compiled objects and executables..."
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
