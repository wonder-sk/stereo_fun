#ifndef UTIL_KERNELS
#define UTIL_KERNELS

extern "C"
void gpu_rgb_to_gray(int* input, char* output, int pixels);

extern "C"
void gpu_gray_to_rgb(char* input, int* output, int pixels);

extern "C"
void gpu_gray32_to_colorscale(int* input, int* output, int pixels, int vmin, int vmax);

#endif