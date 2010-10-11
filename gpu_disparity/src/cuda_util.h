#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#ifndef uchar
#define uchar	unsigned char
#endif
#ifndef uint
#define uint 	unsigned int
#endif

#define rgb2int(r, g, b) (0xff000000 | ((r) << 16) | ((g) << 8) | (b))
#define gray(r, g, b) (((r)*11 + (g)*16 + (b)*5) / 32)
#define blue(c)	 ((c) & 0xff)
#define green(c) (((c)>>8) & 0xff)
#define red(c)  (((c)>>16) & 0xff)

#define rgb2long(r, g, b) (0xffff000000000000L | ((long)(r) << 32) | ((long)(g) << 16) | ((long)(b)))
#define lblue(c)	 (((long)(c)) & 0xffff)
#define lgreen(c) (((long)(c)>>16) & 0xffff)
#define lred(c)  (((long)(c)>>32) & 0xffff)

#define clamp(x, a, b) min(max(x, a), b)
#define clampTop(x, b) min(x, b)
#define clampDown(x, a) max(x, a)

#ifndef DIV_CEIL
#define DIV_CEIL(a, b) ((a)/(b) + (((a) % (b))?1:0))
#endif

#define lerp(x1, x2, k) ((1.0-k)*(x1) + (k)*(x2))

#define pow2(x) ((x)*(x))

#endif // CUDA_UTIL_H