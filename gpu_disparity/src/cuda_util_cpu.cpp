#include "cuda_util_cpu.h"

#include <stdio.h>
#include <stdlib.h>

bool CudaUtil::checkError(const char *msg)
{
    cudaError_t err = cudaGetLastError();

    if(cudaSuccess != err) {
        fprintf(stderr, "** CUDA error %s: %s.\n", msg, cudaGetErrorString(err));
        return true;
    }
    return false;
}

QImage CudaUtil::imageFromGPU(void* gpuBuffer, int width, int height)
{
	const int size= sizeof(int) * width * height;
	int* data= (int*)malloc(size);

	cudaMemcpy(data, gpuBuffer, size, cudaMemcpyDeviceToHost);
	//checkCudaError("imageFromBufferGPU");

	QImage ret= imageFromCPU(data, width, height);
	free(data);

	return ret;
}

QImage CudaUtil::imageFromGPU(void* gpuBuffer, QSize size)
{
	return imageFromGPU(gpuBuffer, size.width(), size.height());
}

QImage CudaUtil::imageFromCPU(int* cpuBuffer, int width, int height)
{
	// Es necesario el const para que el copy haga un deep copy (ver doc de QImage)
	return QImage((const uchar*)cpuBuffer, width, height, QImage::Format_RGB32).copy();
}


void* CudaUtil::uploadImage(QImage image)
{
	void* ret;
	int size= image.width() * image.height() * sizeof(int);

	cudaMalloc(&ret, size);
	cudaMemcpy(ret, image.bits(), size, cudaMemcpyHostToDevice);
	checkError("CudaUtil::uploadImage");

	return ret;
}

void CudaUtil::setBuffer(void* gpuBuffer, QImage image)
{
	int size= image.width() * image.height() * sizeof(int);
	cudaMemcpy(gpuBuffer, image.bits(), size, cudaMemcpyHostToDevice);
}