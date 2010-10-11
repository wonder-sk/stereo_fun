#include "disparity.h"
#include "cuda_util_cpu.h"
#include "disparity_kernels.h"

Disparity::Disparity(QImage leftImage, QImage rightImage)
{
	lastTime= -1;
	
	this->leftImage= leftImage;
	this->rightImage= rightImage;	

	left= CudaUtil::uploadImage(leftImage);
	right= CudaUtil::uploadImage(rightImage);

	size= leftImage.size();

	// 32 BPP disparity map, initialized to black
	const int disparityBytes= size.width() * size.height() * sizeof(uint32_t);
	cudaMalloc(&disparity, disparityBytes);
	cudaMemset(disparity, 0, disparityBytes);
	
	CudaUtil::checkError("Disparity::Disparity: Uploading images.");
}

Disparity::~Disparity()
{
	cudaFree(left);
	cudaFree(right);
	cudaFree(disparity);
	
	CudaUtil::checkError("Disparity::~Disparity: Freeing buffers.");
}

void Disparity::calculate()
{
	lastTime= gpu_disparity_naive((int*)left, (int*)right, (int*)disparity, size.width(), size.height());
	CudaUtil::checkError("Disparity::calculate");
}

QImage Disparity::getDisparity()
{
	return CudaUtil::imageFromGPU(disparity, size);
}
