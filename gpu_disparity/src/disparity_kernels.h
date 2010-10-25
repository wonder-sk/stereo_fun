#ifndef DISPARITY_KERNELS_H
#define DISPARITY_KERNELS_H

extern "C"
float gpu_disparity_naive(
	char* left, char* right, int* disparity, int width, int height);

#endif