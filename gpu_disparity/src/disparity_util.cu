//
#ifndef DISPARITY_UTIL_CU
#define DISPARITY_UTIL_CU

#include "cuda_util.h"

__device__ inline
uint sad(uchar4 c1, uchar4 c2)
{
	return abs(c1.x-c2.x) + abs(c1.y-c2.y) + abs(c1.z-c2.z);
}

__device__ inline
uint ssd(uchar4 c1, uchar4 c2)
{
	short dx= c1.x - c2.x;
	short dy= c1.y - c2.y;
	short dz= c1.z - c2.z;

	return dx*dx + dy*dy + dz*dz;
}

__device__ inline
uint sad(int c1, int c2)
{
	return abs(red(c1)-red(c2)) + abs(green(c1)-green(c2)) + abs(blue(c1)-blue(c2));
}

__device__ inline
uint ssd(int c1, int c2)
{
	short dx= red(c1) - red(c2);
	short dy= green(c1) - green(c2);
	short dz= blue(c1) - blue(c2);

	return dx*dx + dy*dy + dz*dz;
}

__device__ inline
float sad(float4 c1, float4 c2)
{
	return fabsf(c1.x-c2.x) + fabsf(c1.y-c2.y) + fabsf(c1.z-c2.z);
}

__device__ inline
float ssd(float4 c1, float4 c2)
{
	float dx= c1.x - c2.x;
	float dy= c1.y - c2.y;
	float dz= c1.z - c2.z;

	return dx*dx + dy*dy + dz*dz;
}

#endif