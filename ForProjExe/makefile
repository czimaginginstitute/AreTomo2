PRJHOME = $(HOME)/CuProjs/TiltAxis-03-21-2016
CUDAHOME = /usr/local/cuda
CUDAINC = $(CUDAHOME)/include
CUDALIB = $(CUDAHOME)/lib64
PRJINC = $(HOME)/Projs/Include
PRJLIB = $(HOME)/Projs/Lib
CUPRJINC = $(HOME)/CuProjs/Include
CUPRJLIB = $(HOME)/CuProjs/Lib
#-----------------------------
CUSRCS = ../Recon/GBackProj.cu \
	../Recon/GForProj.cu \
	../Recon/GRWeighting.cu \
	../Util/GBinImage2D.cu
CUCPPS = $(patsubst %.cu, %.cpp, $(CUSRCS))
#------------------------------------------
SRCS = ../Util/CSimpleFuncs.cpp \
	../Util/CParseArgs.cpp \
	../Util/CSaveTempMrc.cpp \
	../MrcUtil/CTomoStack.cpp \
	../MrcUtil/CLoadStack.cpp \
	../MrcUtil/CSaveStack.cpp \
	../Recon/CCalcProjs.cpp \
	./CInput.cpp \
	./CProcessThread.cpp \
	./CMain.cpp \
	./CForProjExe.cpp \
	$(CUCPPS)
OBJS = $(patsubst %.cpp, %.o, $(SRCS))
#-------------------------------------
CC = g++
CFLAG = -c -g -pthread -m64
NVCC = nvcc
CUFLAG = -O3 -arch=sm_20  -Xptxas -dlcm=ca 
#-----------------------------------------
cuda: $(CUCPPS)

compile: $(OBJS)

exe: $(OBJS)
	@g++ -g -pthread -m64 $(OBJS) \
	$(PRJLIB)/libmrcfile.a \
	$(PRJLIB)/libutil.a \
	$(CUPRJLIB)/libcuutil.a \
	$(CUPRJLIB)/libcuutilfft.a \
	-L$(CUDALIB) -L/usr/lib64 \
	-lcufft -lcudart -lcuda -lc -lm -lpthread \
	-o TomoForProj
	@echo TomoForProj has been generated.

%.cpp: %.cu
	@$(NVCC) -cuda $(CUFLAG) -I$(PRJINC) -I$(CUPRJINC) $< -o $@
	@echo $< has been compiled.

%.o: %.cpp
	@$(CC) $(CFLAG) -I$(PRJINC) -I$(CUPRJINC) -I$(CUDAINC) \
		$< -o $@
	@echo $< has been compiled.

clean:
	@rm -f $(OBJS) $(CUCPPS) *.h~ makefile~ DfCorrOffline
	
