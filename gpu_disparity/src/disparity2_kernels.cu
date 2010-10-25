#include "disparity_kernels.h"
#include <cuda_runtime_api.h>
#include "cuda_util.h"
#include "cuda_util.cu"

__constant__ uchar leftData[16384];

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
float mnccCost(int x0, int y0, int distance, int r, int width)
{
	float ll= 0;
	float rr= 0;
	float lr= 0;

	for(int dx= -r; dx <= r; dx++) {
		for(int dy= -r; dy <= r; dy++) {

			const int lx= clamp(x0 + dx, 0, width-1);
			const int ly= dy + r;
			float l= leftData[lx + ly * width] / 255.0f;
			
			float r= fetch2f(x0 + dx - distance, y0 + dy);

			ll += l * l;
			rr += r * r;
			lr += l * r;
		}
	}

	// cost = 1 / MoravecNCC
	return (ll + rr) / lr;
}

__device__ inline
float sadCost(int x0, int y0, int distance, int r, int width)
{
	float cost= 0;
	
	for(int dx= -r; dx <= r; dx++) {
		for(int dy= -r; dy <= r; dy++) {

			const int lx= clamp(x0 + dx, 0, width-1);
			const int ly= dy + r;
			const float leftValue= leftData[lx + ly * width] / 255.0f;
			
			cost += fabsf(leftValue - fetch2f(x0 + dx - distance, y0 + dy));
			//cost += fabsf(fetch1f(x0 + dx, y0 + dy) - fetch2f(x0 + dx - distance, y0 + dy));
		}
	}

	return cost;
}

__global__
void disparity_constant(int* output, int width, int height, int y, int blockRadius)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x >= width) return;

	const int maxDistance= 64;
	
	int minDist;
	float minCost;
	
	for(int distance= 0; distance <= maxDistance; distance++) {
		float cost= sadCost(x, y, distance, blockRadius, width);

		if(!distance || cost < minCost) {
			minCost= cost;
			minDist= distance;
		}
	}
 	
 	output[x + y * width]= minDist;
}

extern "C"
float gpu_disparity_constant(
	char* left, char* right, int* disparity, int width, int height)
{
	cudaFuncSetCacheConfig(disparity_constant, cudaFuncCachePreferL1);

	const int block_size= 256;
    dim3 grid(DIV_CEIL(width, block_size));
    dim3 threads(block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Bind rgb input as a texture
    cudaChannelFormatDesc channelDesc= cudaCreateChannelDesc<uchar>();
	cudaBindTexture2D(0, tex1, left, channelDesc, width, height, width);
    cudaBindTexture2D(0, tex2, right, channelDesc, width, height, width);
	
	const int blockRadius= 2;
	const int blockDiam= 2*blockRadius + 1;
	
	for(int y=blockRadius; y < height-blockDiam; y++) {

		cudaMemcpyToSymbol(leftData, left + width * (y-blockRadius), width * blockDiam, 0, cudaMemcpyDeviceToDevice);
		
		disparity_constant<<<grid, threads>>>(disparity, width, height, y, blockRadius);
	}
	
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