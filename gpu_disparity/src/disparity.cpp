#include "disparity.h"
#include "cuda_util_cpu.h"

#include "disparity_kernels.h"
#include "disparity2_kernels.h"
#include "util_kernels.h"
#include "census_kernels.h"

Disparity::Disparity(QImage leftImage, QImage rightImage)
{
	lastTime= -1;
	
	this->leftImage= leftImage.convertToFormat(QImage::Format_RGB32);
	this->rightImage= rightImage.convertToFormat(QImage::Format_RGB32);	

	// Upload images in 32bpp format
	leftColor= CudaUtil::uploadImage(leftImage);
	rightColor= CudaUtil::uploadImage(rightImage);

	size= leftImage.size();
	bytes= size.width() * size.height();
	colorBytes= bytes * 4;

	// Alloc and fill grayscale left image
	cudaMalloc(&left, bytes);
	gpu_rgb_to_gray((int*)leftColor, (char*)left, size.width() * size.height());
	
	// Alloc and fill grayscale right image
	cudaMalloc(&right, bytes);
	gpu_rgb_to_gray((int*)rightColor, (char*)right, size.width() * size.height());
	
	// Alloc a 32bpp disparity map, initialize to 0
	cudaMalloc(&disparity, colorBytes);
	cudaMemset(disparity, 0, colorBytes);
	cudaMalloc(&disparityColor, colorBytes);

	// Alloc census-transformed buffers
	cudaMalloc(&leftCensus5, bytes * 4);
	cudaMalloc(&rightCensus5, bytes * 4);
	
	cudaMalloc(&leftCensus7, bytes * 8);
	cudaMalloc(&rightCensus7, bytes * 8);
	
	CudaUtil::checkError("Disparity::Disparity");
}

Disparity::~Disparity()
{
	cudaFree(left);
	cudaFree(right);
	cudaFree(disparity);
	
	cudaFree(leftColor);
	cudaFree(rightColor);
	cudaFree(disparityColor);
	
	cudaFree(leftCensus5);
	cudaFree(rightCensus5);
	
	cudaFree(leftCensus7);	
	cudaFree(rightCensus7);
	
	CudaUtil::checkError("Disparity::~Disparity: Freeing buffers.");
}

void Disparity::transform()
{
	float leftTime;
	float rightTime;
	
	leftTime = gpu_census_transform_5x5((char*)left , (int*)leftCensus5 , size.width(), size.height());
	rightTime= gpu_census_transform_5x5((char*)right, (int*)rightCensus5, size.width(), size.height());
	qDebug() << "Census 5x5 transform time:" << leftTime << "ms. left +" << rightTime << "ms. right";
	
 	leftTime = gpu_census_transform_7x7((char*)left , (int*)leftCensus7 , size.width(), size.height());
 	rightTime= gpu_census_transform_7x7((char*)right, (int*)rightCensus7, size.width(), size.height());
  	qDebug() << "Census 7x7 transform time:" << leftTime << "ms. left +" << rightTime << "ms. right";
}

void Disparity::calculate()
{
//	lastTime= gpu_disparity_naive((char*)left, (char*)right, (int*)disparity, size.width(), size.height());
//  lastTime= gpu_disparity_constant((char*)left, (char*)right, (int*)disparity, size.width(), size.height());
 	lastTime= gpu_disparity_census_5x5((int*)leftCensus5, (int*)rightCensus5, (int*)disparity, size.width(), size.height());
// 	lastTime= gpu_disparity_census_7x7((int*)leftCensus7, (int*)rightCensus7, (int*)disparity, size.width(), size.height());
	
	CudaUtil::checkError("Disparity::calculate");
}

QImage Disparity::getDisparity()
{
	gpu_gray32_to_colorscale((int*)disparity, (int*)disparityColor, size.width() * size.height(), 0, 55);	
	QImage ret= CudaUtil::imageFromGPU(disparityColor, size);
	CudaUtil::checkError("Disparity::getDisparity");
	
	return ret;
}
