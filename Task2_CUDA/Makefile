NVCC = nvcc
CXX = g++
CXXFLAGS = -O2
TARGET = task2

all: $(TARGET)

$(TARGET): main.o heat_utils.o heat_kernel.o
	$(NVCC) -o $(TARGET) main.o heat_utils.o heat_kernel.o

main.o: main.cpp heat_utils.h
	$(CXX) $(CXXFLAGS) -c main.cpp

heat_utils.o: heat_utils.cpp heat_utils.h
	$(CXX) $(CXXFLAGS) -c heat_utils.cpp

heat_kernel.o: heat_kernel.cu
	$(NVCC) -c heat_kernel.cu -o heat_kernel.o

clean:
	rm -f *.o $(TARGET)
