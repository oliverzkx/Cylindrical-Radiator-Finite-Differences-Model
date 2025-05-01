NVCC = nvcc
CXX = g++
TARGET = task2

all: $(TARGET)

$(TARGET): main.cpp heat_kernel.cu
	$(NVCC) -O2 -o $(TARGET) main.cpp heat_kernel.cu

clean:
	rm -f $(TARGET)
