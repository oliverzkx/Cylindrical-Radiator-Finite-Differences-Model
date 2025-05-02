# Makefile for Task 3 CUDA

NVCC = nvcc
CFLAGS = -O2 -std=c++11

TARGET = task3
SRCS = main.cpp heat_utils.cpp heat_kernel.cu
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(NVCC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
