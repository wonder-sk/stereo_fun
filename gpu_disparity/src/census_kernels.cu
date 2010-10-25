#include "census_kernels.h"
#include "cuda_util.h"

texture<uchar, 2> tex1;
#define fetch1(x, y) tex2D(tex1, (x)+0.5f, (y)+0.5f)

texture<uint, 2> texLeft5;
#define fetchLeft5(x, y) tex2D(texLeft5, (x)+0.5f, (y)+0.5f)
texture<uint, 2> texRight5;
#define fetchRight5(x, y) tex2D(texRight5, (x)+0.5f, (y)+0.5f)

texture<uint2, 2> texLeft7;
#define fetchLeft7(x, y) tex2D(texLeft7, (x)+0.5f, (y)+0.5f)
texture<uint2, 2> texRight7;
#define fetchRight7(x, y) tex2D(texRight7, (x)+0.5f, (y)+0.5f)


///
/// CENSUS TRANSFORM 5x5
///

__global__
void census_transform_5x5(int* output, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= width or y >= height)
		return;

	int res= 0;
	uchar center= fetch1(x, y);
	
	for(int dy=-2; dy<=2; dy++)
		for(int dx=-2; dx<=2; dx++)
			res= (res << 1) | (fetch1(x+dx, y+dy) < center);

 	output[x + y * width]= res;
}

extern "C"
float gpu_census_transform_5x5(char* input, int* output, int width, int height)
{
	cudaFuncSetCacheConfig(census_transform_5x5, cudaFuncCachePreferL1);

	const int block_size= 16;
    dim3 grid(DIV_CEIL(width, block_size), DIV_CEIL(height, block_size));
    dim3 threads(block_size, block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Bind rgb input as a texture
    cudaChannelFormatDesc channelDesc= cudaCreateChannelDesc<uchar>();
    cudaBindTexture2D(0, tex1, input, channelDesc, width, height, width);

	census_transform_5x5<<<grid, threads>>>(output, width, height);

	cudaUnbindTexture(tex1);

    cudaEventRecord(stop, 0);
    float time;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

///
/// CENSUS TRANSFORM 7x7
///

__global__
void census_transform_7x7(int* output, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= width or y >= height)
		return;

	int res1= 0;
	int res2= 0;
	uchar center= fetch1(x, y);

	for(int dy=-3; dy<0; dy++)
		for(int dx=-3; dx<=3; dx++)
			res1= (res1 << 1) | (fetch1(x+dx, y+dy) < center);

	for(int dy=0; dy<=3; dy++)
		for(int dx=-3; dx<=3; dx++)
			res2= (res2 << 1) | (fetch1(x+dx, y+dy) < center);
		
	const int baseIndex= 2 * (x + y * width);
 	output[baseIndex + 0]= res1;
	output[baseIndex + 1]= res2;
}

extern "C"
float gpu_census_transform_7x7(char* input, int* output, int width, int height)
{
	cudaFuncSetCacheConfig(census_transform_5x5, cudaFuncCachePreferL1);

	const int block_size= 16;
    dim3 grid(DIV_CEIL(width, block_size), DIV_CEIL(height, block_size));
    dim3 threads(block_size, block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Bind rgb input as a texture
    cudaChannelFormatDesc channelDesc= cudaCreateChannelDesc<uchar>();
    cudaBindTexture2D(0, tex1, input, channelDesc, width, height, width);

	census_transform_7x7<<<grid, threads>>>(output, width, height);

	cudaUnbindTexture(tex1);

    cudaEventRecord(stop, 0);
    float time;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

///
/// CENSUS DISPARITY 5x5
///

__device__ inline
int census_cost_5x5(int* left, int* right, int x0, int y0, int distance, int r, int width, int height)
{
	int cost= 0;

	for(int x= max(x0-r, 0); x <= min(x0+r, width-1); x++) {
		for(int y= max(y0-r, 0); y <= min(y0+r, height-1); y++) {
		
			uint le= left[x + y * width];
			uint re= right[x - distance + y * width];

			cost += __popc(le ^ re);
		}
	}

	return cost;
}

__global__
void disparity_census_5x5(int* left, int* right, int* output, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= width or y >= height)
		return;

	const int maxDistance= 64;
	const int patchRadius= 3;

	int minDist;
	int minCost;

	for(int distance= 0; distance <= maxDistance; distance++) {
		int cost= census_cost_5x5(left, right, x, y, distance, patchRadius, width, height);

		if(!distance || cost < minCost) {
			minCost= cost;
			minDist= distance;
		}
	}

 	output[x + y * width]= minDist;
}


extern "C"
float gpu_disparity_census_5x5(int* left, int* right, int* disparity, int width, int height)
{
	cudaFuncSetCacheConfig(disparity_census_5x5, cudaFuncCachePreferL1);

	const int block_size= 16;
    dim3 grid(DIV_CEIL(width, block_size), DIV_CEIL(height, block_size));
    dim3 threads(block_size, block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	disparity_census_5x5<<<grid, threads>>>(left, right, disparity, width, height);

    cudaEventRecord(stop, 0);
    float time;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

///
/// CENSUS DISPARITY 7x7
///

__device__ inline
int census_cost_7x7(int* left, int* right, int x0, int y0, int distance, int r, int width, int height)
{
	int cost= 0;

	uint le, re;

	for(int x= max(x0-r, 0); x <= min(x0+r, width-1); x++) {
		for(int y= max(y0-r, 0); y <= min(y0+r, height-1); y++) {

			int baseLeft = 2 * (x + y * width);
			int baseRight= 2 * (x - distance + y * width);
			
			le= left[baseLeft];
			re= right[baseRight];
			cost += __popc(le ^ re);

			le= left[baseLeft + 1];
			re= right[baseRight + 1];
			cost += __popc(le ^ re);			
		}
	}

	return cost;
}

__global__
void disparity_census_7x7(int* left, int* right, int* output, int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= width or y >= height)
		return;

	const int maxDistance= 64;
	const int patchRadius= 4;

	int minDist;
	int minCost;

	for(int distance= 0; distance <= maxDistance; distance++) {
		int cost= census_cost_7x7(left, right, x, y, distance, patchRadius, width, height);

		if(!distance || cost < minCost) {
			minCost= cost;
			minDist= distance;
		}
	}

 	output[x + y * width]= minDist;
}


extern "C"
float gpu_disparity_census_7x7(int* left, int* right, int* disparity, int width, int height)
{
	cudaFuncSetCacheConfig(disparity_census_7x7, cudaFuncCachePreferL1);

	const int block_size= 16;
    dim3 grid(DIV_CEIL(width, block_size), DIV_CEIL(height, block_size));
    dim3 threads(block_size, block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	disparity_census_7x7<<<grid, threads>>>(left, right, disparity, width, height);

    cudaEventRecord(stop, 0);
    float time;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}