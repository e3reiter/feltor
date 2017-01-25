ifeq ($(strip $(shell hostname)),e3rgpu)	# uniquely identify system
CC=g++																		# the host c++ compiler
MPICC = mpic++
MPICFLAGS+= -DMPICH_IGNORE_CXX_SEEK
OPT=-O3                                   # the optimization flag for the host
OMPFLAG=-fopenmp                          # openmp flag for CC and MPICC
INCLUDE = -I$(HOME)/include               # cusp and thrust libraries
LIBS=-lnetcdf -lcurl -lhdf5 -lhdf5_hl     # netcdf library for file output
NVCCARCH=-arch=sm_61                      # nvcc gpu compute capability
GLFLAGS = -lglfw -lGL
JSONLIB = -ljsoncpp
endif
