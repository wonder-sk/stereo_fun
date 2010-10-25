#include "util_kernels.h"
#include "cuda_util.h"

/**
 * RGB TO GRAY
 */

__global__
void rgb_to_gray(int* input, char* output, int pixels)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= pixels) return;
	
	int color= input[i];
	output[i]= (red(color)*11 + green(color)*16 + blue(color)*5) / 32;
}

extern "C"
void gpu_rgb_to_gray(int* input, char* output, int pixels)
{
	dim3 blockDim(256);
	dim3 gridDim(DIV_CEIL(pixels, blockDim.x));

	rgb_to_gray<<<gridDim, blockDim>>>(input, output, pixels);
}

/**
 * GRAY TO RGB
 */

__global__
void gray_to_rgb(char* input, int* output, int pixels)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= pixels) return;

	char gray= input[i];
	output[i]= rgb2int(gray, gray, gray);
}

extern "C"
void gpu_gray_to_rgb(char* input, int* output, int pixels)
{
	dim3 blockDim(256);
	dim3 gridDim(DIV_CEIL(pixels, blockDim.x));

	gray_to_rgb<<<gridDim, blockDim>>>(input, output, pixels);
}

/**
 * GRAY32 TO COLORSCALE
 */

inline __device__
int getColor(float v, float vmin, float vmax)
{
   // http://local.wasp.uwa.edu.au/~pbourke/texture_colour/colourramp/
   float r = 1.0, g = 1.0, b = 1.0;
   float dv;

   v= clamp(v, vmin, vmax);
   dv = vmax - vmin;

   if (v < (vmin + 0.25f * dv)) {
      r = 0;
      g = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5f * dv)) {
      r = 0;
      b = 1 + 4 * (vmin + 0.25f * dv - v) / dv;
   } else if (v < (vmin + 0.75f * dv)) {
      r = 4 * (v - vmin - 0.5f * dv) / dv;
      b = 0;
   } else {
      g = 1 + 4 * (vmin + 0.75f * dv - v) / dv;
      b = 0;
   }

	const uchar ri= clamp((int)(r * 255.0f), 0, 255);
	const uchar gi= clamp((int)(g * 255.0f), 0, 255);
	const uchar bi= clamp((int)(b * 255.0f), 0, 255);

	return rgb2int(ri, gi, bi);
}

__global__
void gray32_to_colorscale(int* input, int* output, int pixels, int vmin, int vmax)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= pixels) return;

	output[i]= getColor(input[i], vmin, vmax);
}

extern "C"
void gpu_gray32_to_colorscale(int* input, int* output, int pixels, int vmin, int vmax)
{
	dim3 blockDim(256);
	dim3 gridDim(DIV_CEIL(pixels, blockDim.x));

	gray32_to_colorscale<<<gridDim, blockDim>>>(input, output, pixels, vmin, vmax);
}
