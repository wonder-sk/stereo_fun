#ifndef CUDA_UTIL_CPU_H
#define CUDA_UTIL_CPU_H

#include <QtGui>
#include <cuda_runtime_api.h>

class CudaUtil
{
public:
	static bool checkError(const char *msg);

	// Baja la imagen a un buffer local y genera una QImage
	static QImage imageFromGPU(void* gpuBuffer, QSize size);
	static QImage imageFromGPU(void* gpuBuffer, int width, int height);

	static int divCeil(int a, int b) { return (a/b) + ((a % b)?1:0); }

	static void* uploadImage(QImage image);
	static void setBuffer(void* gpuBuffer, QImage image);

private:
	// Genera una QImage a partir de los datos en buffer en RAM, no lo elimina
	static QImage imageFromCPU(int* cpuBuffer, int width, int height);
};

#endif // CUDA_UTIL_CPU_H