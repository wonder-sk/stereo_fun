TEMPLATE = app	
CONFIG += qt release warn_on

HEADERS = \
	src/cuda_util_cpu.h \
	src/cuda_util.h \
	src/disparity.h \
	src/disparity_kernels.h \
	src/disparity2_kernels.h \
	src/util_kernels.h \
	src/census_kernels.cu

SOURCES = \
	src/cuda_util_cpu.cpp \
	src/main.cpp \
	src/disparity.cpp

CUDA_SOURCES = \
	src/cuda_util.cu \
	src/disparity_kernels.cu \
	src/disparity2_kernels.cu \
	src/util_kernels.cu \
	src/census_kernels.cu

QMAKE_CXXFLAGS_RELEASE = -march=native -O3 -pipe -fomit-frame-pointer
CUDA_ARCH= sm_11

#DESTDIR = bin
MOC_DIR = obj
OBJECTS_DIR = obj
RCC_DIR = obj
UI_DIR = obj


#
# CUDA
#
CUDA_DIR = /opt/cuda
INCLUDEPATH += $$CUDA_DIR/include \
    $$CUDA_DIR/sdk/C/common/inc
QMAKE_LIBDIR += $$CUDA_DIR/lib64 \
    $$CUDA_DIR/sdk/C/lib/
LIBS += -lcudart
cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
cuda.commands = $$CUDA_DIR/bin/nvcc -O3 -arch $$CUDA_ARCH -c --compiler-bindir /usr/bin/gcc-4.3.4 -Xcompiler -fPIC -Xcompiler $$join(QMAKE_CXXFLAGS,",") \
    $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependcy_type = TYPE_C
cuda.depend_command = nvcc -M -arch $$CUDA_ARCH -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPAT,'" -I "','-I "','"') ${QMAKE_FILE_NAME} \
	| sed "s,^.*: ,," | sed "s,^ *,,"  | tr -d '\\\n'
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_COMPILERS += cuda
OTHER_FILES += src/gpupipeline/output_kernels.cu \
    src/gpupipeline/multibandblend_kernels.cu \
    src/gpupipeline/input_kernels.cu \
    src/gpupipeline/imagepyramid_kernels.cu \
    src/gpupipeline/autofocus_kernels.cu

QMAKE_LFLAGS += -Wl,-R,\'$$CUDA_DIR/lib64\'