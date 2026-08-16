NVCC = nvcc
ARCH = -arch=sm_89
FLAGS = -O3 --use_fast_math -lcublas
SRC = src/main.cu

sgemm: $(SRC)
	$(NVCC) $(ARCH) $(FLAGS) $(SRC) -o sgemm

clean:
	rm -f sgemm *.o


	