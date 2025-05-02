# Makefile for Task 4 - Double Precision CUDA Version

NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11
NVCCFLAGS = -O2

TARGET = task4
OBJ = main.o heat_utils.o heat_kernel.o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJ)

main.o: main.cpp heat_utils.h
	$(CXX) $(CXXFLAGS) -c main.cpp

heat_utils.o: heat_utils.cpp heat_utils.h
	$(CXX) $(CXXFLAGS) -c heat_utils.cpp

heat_kernel.o: heat_kernel.cu heat_utils.h
	$(NVCC) $(NVCCFLAGS) -c heat_kernel.cu

clean:
	rm -f *.o $(TARGET)
