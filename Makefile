NVCC = nvcc
CCBIN = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
FLAGS = -ccbin $(CCBIN) -arch=sm_89 -O3

test: test.cu
	$(NVCC) $(FLAGS) -o test.exe test.cu

clean:
	del test.exe