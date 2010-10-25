#ifndef DISPARITY2_KERNELS_H
#define DISPARITY2_KERNELS_H

extern "C"
float gpu_disparity_constant(
	char* left, char* right, int* disparity, int width, int height);
	
#endif