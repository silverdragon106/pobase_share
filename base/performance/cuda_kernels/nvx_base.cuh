#pragma once
#include "nvx_base.h"
#include "nvx_types.h"

//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers, only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)

inline i32 _cuDivUp(i32 a, i32 b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
inline i32 _cuDivDown(i32 a, i32 b) 
{
	return a / b;
}

//Align a to nearest higher multiple of b
inline i32 _cuAlignUp(i32 a, i32 b) 
{
	return (a % b != 0) ? (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
inline i32 _cuAlignDown(i32 a, i32 b)
{
	return a - a % b;
}

template <typename T, typename U>
__device__ static inline void _cuTrans2x2(T* tr, U x, U y, U& tx, U& ty)
{
	tx = tr[0] * x + tr[1] * y;
	ty = tr[2] * x + tr[3] * y;
}

template <typename T, typename U>
__device__ static inline void _cuTrans2x3(T* tr, U& x, U& y)
{
	U px = x;
	U py = y;
	x = tr[0] * px + tr[1] * py + tr[2];
	y = tr[3] * px + tr[4] * py + tr[5];
}

template <typename T, typename U>
__device__ static inline bool _cvIsPointInRect(const cuRect<T>& tr, U& x, U& y)
{
	return (x >= tr.x1 && x < tr.x2 && y >= tr.y1 && y < tr.y2);
}

template <typename T>
__device__ __host__ static inline bool _cuIsPositive(T val)
{
	return (val > 0 && val < CUDA_PO_MAXINT);
}

__device__ static inline int _cuAbs(int val)
{
	if (val >= 0)
		return val;
	return -val;
}

template <typename T>
__device__ __host__ static inline T _cuSgn(T val)
{
	return (T)((val) >= 0 ? (1) : (-1));
}

template <typename T>
__device__ __host__ static inline T _cuGetAngle(cuVector2d<T> pos2d)
{
	T angle = atan2(pos2d.y, pos2d.x);
	if (angle < 0)
	{
		angle += CUDA_PO_PI2;
	}
	return angle;
}

template <typename T>
__device__ __host__ static inline T _cuDistPtInLine(T cax, T cay, cuVector2d<T> normalize_ba)
{
	T len;
	cuVector2d<T> ca(cax, cay);
	cuVector2d<T> normalize_ca = ca.normalize(len);
	return normalize_ca.dotProduct(normalize_ba)*len;
}

template <typename T>
__device__ __host__ static inline T _cuDistPt2Line(T cx, T cy, T ax, T ay)
{
	return std::abs(cx*ay - cy * ax);
}

template <typename T>
__device__ __host__ static inline T _cuDistPt2Line(cuVector2d<T> ca, cuVector2d<T> normalize_ba)
{
	return std::abs(ca.x*normalize_ba.y - ca.y*normalize_ba.x);
}

template <typename T>
__device__ __host__ static inline T _cuDistPt2Line(T cax, T cay, cuVector2d<T> normalize_ba)
{
	return std::abs(cax*normalize_ba.y - cay * normalize_ba.x);
}

template <typename T>
__device__ __host__ static inline T _cuDistPt2Line(cuVector2d<T> pt, cuVector2d<T> line_pt1, cuVector2d<T> line_pt2)
{
	cuVector2d<T> normalized_ba = line_pt2 - line_pt1;
	cuVector2d<T> ca = line_pt1 - pt;

	normalized_ba.normalize();
	return std::abs(ca.crossProduct(normalized_ba));
}

template <typename T>
__device__ __host__ static inline f64 _cuRatio(T x1, T x2)
{
	f64 mx1 = min(x1, x2);
	f64 mx2 = max(x1, x2);
	return (mx2 > CUDA_PO_EPSILON) ? (mx1 / mx2) : 0;
}

template <typename T>
__device__ __host__ static inline void _cuSwap(T& x1, T& x2)
{
	T xtmp;
	xtmp = x1; x1 = x2; x2 = xtmp;
}

template <typename T, typename U>
__device__ static inline void cuTrans2x2(T* tr, U x, U y, U& tx, U& ty)
{
	tx = tr[0] * x + tr[1] * y;
	ty = tr[2] * x + tr[3] * y;
}

template <typename T, typename U>
__device__  static inline void cuTrans2x3(T* tr, U& x, U& y)
{
	U px = x;
	U py = y;
	x = tr[0] * px + tr[1] * py + tr[2];
	y = tr[3] * px + tr[4] * py + tr[5];
}

template <typename T, typename U>
__device__  static inline bool cuIsPointInRect(const cuRect<T>& tr, U& x, U& y)
{
	return (x >= tr.x1 && x < tr.x2 && y >= tr.y1 && y < tr.y2);
}

template <typename T>
__device__ __host__ static inline bool cuIsPositive(T val)
{
	return (val > 0 && val < CUDA_PO_MAXINT);
}

static inline bool _cuBitCheck(i32 val, i32 chk)
{
	return (val & chk) == chk;
}
