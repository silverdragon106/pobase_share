#pragma once

#include "define.h"

#pragma pack(push, 4)

#define PO_NORM_L1				0
#define PO_NORM_L2				1

#define PO_DECOMP_LU			0
#define PO_DECOMP_SVD			1

#define PO_TYPE_NONE			0
#define PO_TYPE_U8				1
#define PO_TYPE_U16				2
#define PO_TYPE_U32				3
#define PO_TYPE_U64				4
#define PO_TYPE_S8				5
#define PO_TYPE_S16				6
#define PO_TYPE_S32				7
#define PO_TYPE_S64				8
#define PO_TYPE_F32				9
#define PO_TYPE_F64				10

#define PO_MAT_ROW_F64(a,b)		((f64*)a.data + (b)*a.w)

#define SF(y, x)				((f32*)(src_data + y*src_step))[x]
#define SD(y, x)				((f64*)(src_data + y*src_step))[x]
#define DF(y, x)				((f32*)(dst_data + y*dst_step))[x]
#define DD(y, x)				((f64*)(dst_data + y*dst_step))[x]

#define DET2(m)					((f64)m(0,0)*m(1,1) - (f64)m(0,1)*m(1,0))
#define DET3(m)					(m(0, 0)*((f64)m(1, 1)*m(2, 2) - (f64)m(1, 2)*m(2, 1)) - \
								m(0, 1)*((f64)m(1, 0)*m(2, 2) - (f64)m(1, 2)*m(2, 0)) + \
								m(0, 2)*((f64)m(1, 0)*m(2, 1) - (f64)m(1, 1)*m(2, 0)))
struct PoMat
{
	i32				w;
	i32				h;
	i32				elem;
	i32				wstep;
	i32				size;
	u8*				data;

public:
	PoMat();
	PoMat(i32 h, i32 w, i32 elem);
};

class CPOMatrix
{
public:
	CPOMatrix();
	~CPOMatrix();

	static void					initMatrix(PoMat& mat, i32 h, i32 w, i32 elem);
	static void					freeMatrix(PoMat& mat);
	static void					copyMatrix(const PoMat& src_mat, PoMat& dst_mat);
	static bool					checkMatrix(const PoMat& mat, i32 h, i32 w, i32 elem_size);

	static bool					isValidMatrix(const PoMat& mat);
	static bool					zeroMatrix(PoMat& mat);
	static bool					setIdentity(PoMat& mat);

	static bool					transpose(const PoMat& src_mat, PoMat& dst_mat);
	static bool					multiply(const PoMat& a_mat, const PoMat& b_mat, PoMat& dst_mat);
	static bool					multiplyDiag(PoMat& mat, f64 rate);
	static bool					inverse(const PoMat& src_mat, PoMat& dst_mat, i32 method);

	template<typename T>
	static f64					norm(T* p1, T* p2, i32 count, i32 method);
};

#pragma pack(pop)

template<typename T>
f64 CPOMatrix::norm(T* p1, T* p2, i32 count, i32 method)
{
	if (!p1 || !p2 || count <= 0)
	{
		return PO_MAXINT;
	}

	T tmp, value = 0;
	switch (method)
	{
		case PO_NORM_L1:
		{
			for (i32 i = 0; i < count; i++)
			{
				value += std::abs(p1[i] - p2[i]);
			}
			return value / count;
		}
		case PO_NORM_L2:
		{
			for (i32 i = 0; i < count; i++)
			{
				tmp = p1[i] - p2[i];
				value += tmp*tmp;
			}
			return value / count;
		}
	}
	return PO_MAXINT;
}