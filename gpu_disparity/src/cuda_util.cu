#include "cuda_util.h"

#ifndef CUDA_UTIL_CU
#define CUDA_UTIL_CU

inline
__device__ void int2floats(uint color, float &r, float &g, float &b)
{
	r= red(color) / 255.0f;
	g= green(color) / 255.0f;
	b= blue(color) / 255.0f;
}

inline
__device__ uint floats2int(float r, float g, float b)
{
	const uchar rOut= r * 255.0f;
	const uchar gOut= g * 255.0f;
	const uchar bOut= b * 255.0f;

	return rgb2int(rOut, gOut, bOut);
}

inline
__device__ uchar4 float4_to_uchar4(float4 col)
{
	uchar4 r;
	r.x= (unsigned char)(col.x * 255.0f);
	r.y= (unsigned char)(col.y * 255.0f);
	r.z= (unsigned char)(col.z * 255.0f);
	return r;
}

inline
__device__ short4 float4_to_short4(float4 col)
{
	short4 r;
	r.x= (short)(col.x * 255.0f);
	r.y= (short)(col.y * 255.0f);
	r.z= (short)(col.z * 255.0f);
	return r;
}

inline
__device__ void rgb2hsv(float r, float g, float b, float& h, float& s, float& v)
{
	float cMax= max(r, max(g, b));
	float cMin= min(r, min(g, b));
	float cDelta= cMax-cMin;

	v= cMax;

	if(cDelta == 0) {
		h= 0;
		s= 0;
	} else {
		s= cDelta/cMax;

		const float deltaR= ((cMax-r)/6.0f + cDelta/2.0f) / cDelta;
		const float deltaG= ((cMax-g)/6.0f + cDelta/2.0f) / cDelta;
		const float deltaB= ((cMax-b)/6.0f + cDelta/2.0f) / cDelta;

		if     (r == cMax) h= deltaB - deltaG;
		else if(g == cMax) h= (1.0f/3.0f) + deltaR - deltaB;
		else if(b == cMax) h= (2.0f/3.0f) + deltaG - deltaR;

		if(h < 0) h+=1.0f;
		if(h > 1) h-=1.0f;
	}
}

inline
__device__ void hsv2rgb(float h, float s, float v, float &r, float &g, float &b)
{
	if(s == 0) {
		r= v;
		g= v;
		b= v;
	} else {
		const float varH= (h==1.0)? 0 : h*6;
		const int   varI= int(varH);
		const float var1= v * (1-s);
		const float var2= v * (1-s * (varH - varI));
		const float var3= v * (1-s * (1 - (varH - varI)));

		if      ( varI == 0 ) { r = v     ; g = var3 ; b = var1; }
		else if ( varI == 1 ) { r = var2 ; g = v     ; b = var1; }
		else if ( varI == 2 ) { r = var1 ; g = v     ; b = var3; }
		else if ( varI == 3 ) { r = var1 ; g = var2 ; b = v;     }
		else if ( varI == 4 ) { r = var3 ; g = var1 ; b = v;     }
		else                  { r = v     ; g = var1 ; b = var2; }
	}

	r= clampTop(r, 1.0f);
	g= clampTop(g, 1.0f);
	b= clampTop(b, 1.0f);
}

inline __device__ float getY(float r, float g, float b) { return    0.299f*r +   0.587f*g +   0.114f*b; }
inline __device__ float getU(float r, float g, float b) { return -0.14713f*r - 0.28886f*g +   0.436f*b; }
inline __device__ float getV(float r, float g, float b) { return    0.615f*r - 0.51499f*g - 0.10001f*b; }

inline __device__ float getY(float4 color) { return    0.299f*color.z +   0.587f*color.y +   0.114f*color.x; }
inline __device__ float getU(float4 color) { return -0.14713f*color.z - 0.28886f*color.y +   0.436f*color.x; }
inline __device__ float getV(float4 color) { return    0.615f*color.z - 0.51499f*color.y - 0.10001f*color.x; }

inline __device__ float getY(uint color) { return getY(red(color), green(color), blue(color))/255.0f; }
inline __device__ float getU(uint color) { return getU(red(color), green(color), blue(color))/255.0f; }
inline __device__ float getV(uint color) { return getV(red(color), green(color), blue(color))/255.0f; }


inline
__device__ void rgb2yuv(float r, float g, float b, float& y, float& u, float& v)
{
	y= getY(r, g, b);
	u= getU(r, g, b);
	v= getV(r, g, b);
}

inline
__device__ void yuv2rgb(float y, float u, float v, float& r, float& g, float& b)
{
	r= y + 1.13983f * v;
	g= y - 0.39465f * u - 0.5806f * v;
	b= y + 2.03211f * u;

	r= clampTop(r, 1.0f);
	g= clampTop(g, 1.0f);
	b= clampTop(b, 1.0f);
}

inline
__device__ uint setY(uint color, float newY)
{
	float r, g, b;

	const float y= newY;
	const float u= getU(color);
	const float v= getV(color);

	yuv2rgb(y,u,v, r,g,b);

	return floats2int(r, g, b);
}

/*
inline
__device__ float cosineInterp(float y1, float y2, float k)
{
	float k2= (1-cos(k*PI))/2;   
	return lerp(y1, y2, k2)
}
*/

#endif