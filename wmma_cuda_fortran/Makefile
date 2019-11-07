PLATFORM=linux86-64
YEAR=2019
main: main.o wmma_mod.o
	pgfortran -o $@ $^ -Mcuda=cc70
main.o: wmma_mod.o main.CUF

.SUFFIXES: .CUF .o
.CUF.o:
	pgfortran -fast -o $@ -c -Mcuda=cc70 -I $(PGI)/$(PLATFORM)/$(YEAR)/examples/CUDA-Fortran/TensorCores/Utils $<
.PHONY: clean
clean:
	-rm -f *.o *.mod core main
