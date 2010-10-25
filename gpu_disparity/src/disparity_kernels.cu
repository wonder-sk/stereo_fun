#include "disparity_kernels.h"
#include <cuda_runtime_api.h>
#include "cuda_util.h"
#include "cuda_util.cu"

// Textures used for sampling the input images
texture<uchar, 2> tex1;
texture<uchar, 2> tex2;

#define fetch1(x, y) tex2D(tex1, (x)+0.5f, (y)+0.5f)
#define fetch2(x, y) tex2D(tex2, (x)+0.5f, (y)+0.5f)

#define fetch1f(x, y) (fetch1(x, y) / 255.0f)
#define fetch2f(x, y) (fetch2(x, y) / 255.0f)


/**
**	Naive implementation: Use only fetches from textures
**/

__device__ inline
float mnccCost(int x0, int y0, int distance, int r)
{
	float ll= 0;
	float rr= 0;
	float lr= 0;

	for(int x= x0-r; x <= x0+r; x++) {
		for(int y= y0-r; y <= y0+r; y++) {
			float l= fetch1f(x, y);
			float r= fetch2f(x-distance, y);

			ll += l * l;
			rr += r * r;
			lr += l * r;
		}
	}

	// cost = 1 / MoravecNCC
	return (ll + rr) / lr;
}

__device__ inline
float sadCost(int x0, int y0, int distance, int r)
{
	float cost= 0;
	
	for(int x= x0-r; x <= x0+r; x++)
		for(int y= y0-r; y <= y0+r; y++)
			cost += fabsf(fetch1f(x, y) - fetch2f(x-distance, y));			

	return cost;
}

__global__
void disparity_naive(int* output, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= width or y >= height)
		return;

	const int maxDistance= 64;
	const int patchRadius= 3;
	
	int minDist;
	float minCost;
	
	for(int distance= 0; distance <= maxDistance; distance++) {
		float cost= sadCost(x, y, distance, patchRadius);

		if(!distance || cost < minCost) {
			minCost= cost;
			minDist= distance;
		}
	}
 	
 	output[x + y * width]= minDist;
}

extern "C"
float gpu_disparity_naive(
	char* left, char* right, int* disparity, int width, int height)
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
    cudaChannelFormatDesc channelDesc= cudaCreateChannelDesc<uchar>();
    cudaBindTexture2D(0, tex1, left, channelDesc, width, height, width);
	cudaBindTexture2D(0, tex2, right, channelDesc, width, height, width);

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