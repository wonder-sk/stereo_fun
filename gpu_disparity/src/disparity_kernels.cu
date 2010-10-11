#include "disparity_kernels.h"
#include <cuda_runtime_api.h>
#include "cuda_util.h"
#include "cuda_util.cu"
#include "disparity_util.cu"

// Textures used for sampling the input images
texture<uchar4, 2, cudaReadModeNormalizedFloat> tex1;
texture<uchar4, 2, cudaReadModeNormalizedFloat> tex2;
#define fetch1f(x, y) tex2D(tex1, (x)+0.5f, ((y)+0.5f))
#define fetch1(x, y)  float4_to_uchar4(fetch1f(x, y))
#define fetch2f(x, y) tex2D(tex2, (x)+0.5f, ((y)+0.5f))
#define fetch2(x, y)  float4_to_uchar4(fetch2f(x, y))


/**
**	Naive implementation: Use only fetches from textures
**/

__device__ inline
float sadCost_naive(/* tex1, tex2, */ int x0, int y0, int distance, int r)
{
	float cost= 0;
	
	for(int x= x0-r; x <= x0+r; x++)
		for(int y= y0-r; y <= y0+r; y++)
			//cost += sad(fetch1f(x, y), fetch2f(x-distance, y));
		
			cost += fabsf(getY(fetch1f(x, y)) - getY(fetch2f(x-distance, y)));

	return cost;
}

__device__ inline
float nccCost_naive(/* tex1, tex2, */ int x0, int y0, int distance, int r)
{
	float lrSum= 0;
	float llSum= 0;
	float rrSum= 0;

/*	float lAvr= 0;
	float rAvr= 0;
	for(int x= x0-r; x <= x0+r; x++) {
		for(int y= y0-r; y <= y0+r; y++) {
			lAvr += getY(fetch1f(x, y));
			rAvr += getY(fetch2f(x-distance, y));
		}
	}
	lAvr /= (r+r+1) * (r+r+1);
	rAvr /= (r+r+1) * (r+r+1);*/
	
	for(int x= x0-r; x <= x0+r; x++) {
		for(int y= y0-r; y <= y0+r; y++) {
			float l= getY(fetch1f(x, y));
			float r= getY(fetch2f(x-distance, y));

			lrSum += l * r;
			llSum += l * l;
			rrSum += r * r;
		}
	}

	return lrSum / sqrtf(llSum * rrSum);
}

__global__
void disparity_naive(int* output, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= width or y >= height)
		return;

	const int maxDistance= 64;
	const int patchRadius= 2;
	
	int minDist= 0;
	float minCost= 99999.0f;
	
	for(int distance= 0; distance<=maxDistance; distance++) {
		float cost= sadCost_naive(x, y, distance, patchRadius);

		if(cost < minCost) {
			minCost= cost;
			minDist= distance;
		}
	}
	
    output[x + y * width]= rgb2int(minDist, minDist, minDist);
}

extern "C"
float gpu_disparity_naive(
	int* left, int* right, int* disparity, int width, int height)
{
	cudaFuncSetCacheConfig(disparity_naive, cudaFuncCachePreferL1);

	const int block_size= 16;
    dim3 grid(DIV_CEIL(width, block_size), DIV_CEIL(height, block_size));
    dim3 threads(block_size, block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Bind rgb input as a texture
    cudaChannelFormatDesc channelDesc= cudaCreateChannelDesc<uchar4>();
    cudaBindTexture2D(0, tex1, left, channelDesc, width, height, width * 4);
	cudaBindTexture2D(0, tex2, right, channelDesc, width, height, width * 4);

	disparity_naive<<<grid, threads>>>(disparity, width, height);

	cudaUnbindTexture(tex1);
	cudaUnbindTexture(tex2);

    cudaEventRecord(stop, 0);
    float time;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}