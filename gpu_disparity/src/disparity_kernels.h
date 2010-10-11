#ifndef DISPARITY_KERNELS_H
#define DISPARITY_KERNELS_H

extern "C"
float gpu_disparity_naive(
	int* left, int* right, int* disparity, int width, int height);
	
#endif