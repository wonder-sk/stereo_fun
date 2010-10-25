#ifndef CENSUS_KERNELS
#define CENSUS_KERNELS

extern "C"
float gpu_census_transform_5x5(char* input, int* output, int width, int height);

extern "C"
float gpu_census_transform_7x7(char* input, int* output, int width, int height);

extern "C"
float gpu_disparity_census_5x5(int* left, int* right, int* disparity, int width, int height);

extern "C"
float gpu_disparity_census_7x7(int* left, int* right, int* disparity, int width, int height);


#endif