#include "matrix.h"
#include "base.h"

//////////////////////////////////////////////////////////////////////////
inline i32 datasize_(i32 elem_type)
{
	switch (elem_type)
	{
		case PO_TYPE_U8:	return sizeof(u8);
		case PO_TYPE_U16:	return sizeof(u16);
		case PO_TYPE_U32:	return sizeof(u32);
		case PO_TYPE_U64:	return sizeof(u64);
		case PO_TYPE_S8:	return sizeof(i8);
		case PO_TYPE_S16:	return sizeof(i16);
		case PO_TYPE_S32:	return sizeof(i32);
		case PO_TYPE_S64:	return sizeof(i64);
		case PO_TYPE_F32:	return sizeof(f32);
		case PO_TYPE_F64:	return sizeof(f64);
	}
	return 0;
}

template<typename T>
void identity_(const u8* src, i32 h, i32 w)
{
	T* s0 = (T*)src;
	i32 i, count = po::_min(w, h);
	for (i = 0; i < count; i++)
	{
		s0[w*i + i] = 1;
	}
}

template<typename T>
void transpose_(const u8* src, i32 h, i32 w, u8* dst)
{
	i32 i, j, index;
	T *s0 = (T*)src;
	T *d0 = (T*)dst, *dtmp;

	for (i = 0; i < w; i++)
	{
		dtmp = d0 + i*h;
		index = i;
		for (j = 0; j < h; j++)
		{
			dtmp[j] = s0[index];
			index += w;
		}
	}
}

template<typename T>
void multiply_(const u8* a_src, i32 ah, i32 aw, const u8* b_src, i32 bh, i32 bw, u8* dst)
{
	if (aw != bh)
	{
		return;
	}

	T val;
	i32 i, j, k, index;
	T *a0 = (T*)a_src, *a_tmp;
	T *b0 = (T*)b_src;
	T *d0 = (T*)dst, *d_tmp;

	for (i = 0; i < ah; i++)
	{
		a_tmp = a0 + i*aw;
		d_tmp = d0 + i*bw;
		for (j = 0; j < bw; j++)
		{
			val = 0;
			index = j;
			for (k = 0; k < aw; k++)
			{
				val += a_tmp[k] * b0[index];
				index += bw;
			}
			d_tmp[j] = val;
		}
	}
}

template<typename T>
void multiply_diag_(const u8* src, i32 h, i32 w, f64 rate)
{
	T* s0 = (T*)src;
	i32 i, index, count = po::_min(w, h);
	for (i = 0; i < count; i++)
	{
		index = i*w + i;
		s0[index] = (T)(rate*s0[index]);
	}
}

template<typename T> 
i32 lu_impl(T* A, i32 astep, i32 m, T* b, i32 bstep, i32 n, T eps)
{
	i32 i, j, k, p = 1;
	for (i = 0; i < m; i++)
	{
		k = i;
		for (j = i + 1; j < m; j++)
		{
			if (std::abs(A[j*astep + i]) > std::abs(A[k*astep + i]))
			{
				k = j;
			}
		}

		if (std::abs(A[k*astep + i]) < eps)
		{
			return 0;
		}

		if (k != i)
		{
			for (j = i; j < m; j++)
			{
				std::swap(A[i*astep + j], A[k*astep + j]);
			}
			if (b)
			{
				for (j = 0; j < n; j++)
				{
					std::swap(b[i*bstep + j], b[k*bstep + j]);
				}
			}
			p = -p;
		}

		T d = -1 / A[i*astep + i];
		for (j = i + 1; j < m; j++)
		{
			T alpha = A[j*astep + i] * d;
			for (k = i + 1; k < m; k++)
			{
				A[j*astep + k] += alpha*A[i*astep + k];
			}

			if (b)
			{
				for (k = 0; k < n; k++)
				{
					b[j*bstep + k] += alpha*b[i*bstep + k];
				}
			}
		}
	}

	if (b)
	{
		for (i = m - 1; i >= 0; i--)
		{
			for (j = 0; j < n; j++)
			{
				T s = b[i*bstep + j];
				for (k = i + 1; k < m; k++)
				{
					s -= A[i*astep + k] * b[k*bstep + j];
				}
				b[i*bstep + j] = s / A[i*astep + i];
			}
		}
	}

	return p;
}

//////////////////////////////////////////////////////////////////////////
PoMat::PoMat()
{
	memset(this, 0, sizeof(PoMat));
}

PoMat::PoMat(i32 h, i32 w, i32 elem)
{
	this->w = w;
	this->h = h;
	this->elem = elem;
	this->wstep = w*datasize_(elem);
	this->size = h*this->wstep;
	this->data = new u8[this->size];
	memset(this->data, 0, this->size);
}

CPOMatrix::CPOMatrix()
{
}

CPOMatrix::~CPOMatrix()
{
}

void CPOMatrix::initMatrix(PoMat& mat, i32 h, i32 w, i32 elem)
{
	i32 elem_size = datasize_(elem);
	if (checkMatrix(mat, h, w, elem_size))
	{
		mat.w = w;
		mat.h = h;
		mat.elem = elem;
		mat.wstep = w*elem_size;
		memset(mat.data, 0, mat.size);
		return;
	}

	freeMatrix(mat);

	mat.w = w;
	mat.h = h;
	mat.elem = elem;
	mat.wstep = w*elem_size;
	mat.size = h*mat.wstep;
	mat.data = new u8[mat.size];
	memset(mat.data, 0, mat.size);
}

void CPOMatrix::freeMatrix(PoMat& mat)
{
	mat.w = 0;
	mat.h = 0;
	mat.elem = 0;
	mat.wstep = 0;
	mat.size = 0;
	POSAFE_DELETE_ARRAY(mat.data);
}

void CPOMatrix::copyMatrix(const PoMat& src_mat, PoMat& dst_mat)
{
	if (!isValidMatrix(src_mat))
	{
		return;
	}

	initMatrix(dst_mat, src_mat.h, src_mat.w, src_mat.elem);
	CPOBase::memCopy(dst_mat.data, src_mat.data, dst_mat.size);
}

bool CPOMatrix::checkMatrix(const PoMat& mat, i32 h, i32 w, i32 elem_size)
{
	if (!isValidMatrix(mat) || mat.size < w*h*elem_size)
	{
		return false;
	}
	return true;
}

bool CPOMatrix::isValidMatrix(const PoMat& mat)
{
	if (!mat.data || mat.size <= 0 || mat.w <= 0 || mat.h <= 0)
	{
		return false;
	}
	return true;
}

bool CPOMatrix::zeroMatrix(PoMat& mat)
{
	if (!isValidMatrix(mat))
	{
		return false;
	}
	memset(mat.data, 0, mat.size);
	return true;
}

bool CPOMatrix::setIdentity(PoMat& mat)
{
	if (!zeroMatrix(mat))
	{
		return false;
	}
	switch (mat.elem)
	{
		case PO_TYPE_U8:
		{
			identity_<u8>(mat.data, mat.h, mat.w);
			break;
		}
		case PO_TYPE_U16:
		{
			identity_<u16>(mat.data, mat.h, mat.w);
			break;
		}
		case PO_TYPE_U32:
		{
			identity_<u32>(mat.data, mat.h, mat.w);
			break;
		}
		case PO_TYPE_U64:
		{
			identity_<u64>(mat.data, mat.h, mat.w);
			break;
		}
		case PO_TYPE_S8:
		{
			identity_<i8>(mat.data, mat.h, mat.w);
			break;
		}
		case PO_TYPE_S16:
		{
			identity_<i16>(mat.data, mat.h, mat.w);
			break;
		}
		case PO_TYPE_S32:
		{
			identity_<i32>(mat.data, mat.h, mat.w);
			break;
		}
		case PO_TYPE_S64:
		{
			identity_<i64>(mat.data, mat.h, mat.w);
			break;
		}
		case PO_TYPE_F32:
		{
			identity_<f32>(mat.data, mat.h, mat.w);
			break;
		}
		case PO_TYPE_F64:
		{
			identity_<f64>(mat.data, mat.h, mat.w);
			break;
		}
		default:
		{
			return false;
		}
	}
	return true;
}

bool CPOMatrix::transpose(const PoMat& src_mat, PoMat& dst_mat)
{
	initMatrix(dst_mat, src_mat.w, src_mat.h, src_mat.elem);
	switch (dst_mat.elem)
	{
		case PO_TYPE_U8:
		case PO_TYPE_S8:
		{
			transpose_<i8>(src_mat.data, src_mat.h, src_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_U16:
		case PO_TYPE_S16:
		{
			transpose_<i16>(src_mat.data, src_mat.h, src_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_U32:
		case PO_TYPE_S32:
		case PO_TYPE_F32:
		{
			transpose_<i32>(src_mat.data, src_mat.h, src_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_U64:
		case PO_TYPE_S64:
		case PO_TYPE_F64:
		{
			transpose_<i64>(src_mat.data, src_mat.h, src_mat.w, dst_mat.data);
			break;
		}
		default:
		{
			return false;
		}
	}
	return true;
}

bool CPOMatrix::multiply(const PoMat& a_mat, const PoMat& b_mat, PoMat& dst_mat)
{
	if (a_mat.w != b_mat.h || a_mat.elem != b_mat.elem)
	{
		return false;
	}

	initMatrix(dst_mat, a_mat.h, b_mat.w, a_mat.elem);
	switch (dst_mat.elem)
	{
		case PO_TYPE_U8:
		{
			multiply_<u8>(a_mat.data, a_mat.h, a_mat.w, b_mat.data, b_mat.h, b_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_U16:
		{
			multiply_<u16>(a_mat.data, a_mat.h, a_mat.w, b_mat.data, b_mat.h, b_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_U32:
		{
			multiply_<u32>(a_mat.data, a_mat.h, a_mat.w, b_mat.data, b_mat.h, b_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_U64:
		{
			multiply_<u64>(a_mat.data, a_mat.h, a_mat.w, b_mat.data, b_mat.h, b_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_S8:
		{
			multiply_<i8>(a_mat.data, a_mat.h, a_mat.w, b_mat.data, b_mat.h, b_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_S16:
		{
			multiply_<i16>(a_mat.data, a_mat.h, a_mat.w, b_mat.data, b_mat.h, b_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_S32:
		{
			multiply_<i32>(a_mat.data, a_mat.h, a_mat.w, b_mat.data, b_mat.h, b_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_S64:
		{
			multiply_<i64>(a_mat.data, a_mat.h, a_mat.w, b_mat.data, b_mat.h, b_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_F32:
		{
			multiply_<f32>(a_mat.data, a_mat.h, a_mat.w, b_mat.data, b_mat.h, b_mat.w, dst_mat.data);
			break;
		}
		case PO_TYPE_F64:
		{
			multiply_<f64>(a_mat.data, a_mat.h, a_mat.w, b_mat.data, b_mat.h, b_mat.w, dst_mat.data);
			break;
		}
		default:
		{
			return false;
		}
	}
	return true;
}

bool CPOMatrix::multiplyDiag(PoMat& mat, f64 rate)
{
	if (mat.w != mat.h || !isValidMatrix(mat))
	{
		return false;
	}

	switch (mat.elem)
	{
		case PO_TYPE_U8:
		{
			multiply_diag_<u8>(mat.data, mat.h, mat.w, rate);
			break;
		}
		case PO_TYPE_U16:
		{
			multiply_diag_<u16>(mat.data, mat.h, mat.w, rate);
			break;
		}
		case PO_TYPE_U32:
		{
			multiply_diag_<u32>(mat.data, mat.h, mat.w, rate);
			break;
		}
		case PO_TYPE_U64:
		{
			multiply_diag_<u64>(mat.data, mat.h, mat.w, rate);
			break;
		}
		case PO_TYPE_S8:
		{
			multiply_diag_<i8>(mat.data, mat.h, mat.w, rate);
			break;
		}
		case PO_TYPE_S16:
		{
			multiply_diag_<i16>(mat.data, mat.h, mat.w, rate);
			break;
		}
		case PO_TYPE_S32:
		{
			multiply_diag_<i32>(mat.data, mat.h, mat.w, rate);
			break;
		}
		case PO_TYPE_S64:
		{
			multiply_diag_<i64>(mat.data, mat.h, mat.w, rate);
			break;
		}
		case PO_TYPE_F32:
		{
			multiply_diag_<f32>(mat.data, mat.h, mat.w, rate);
			break;
		}
		case PO_TYPE_F64:
		{
			multiply_diag_<f64>(mat.data, mat.h, mat.w, rate);
			break;
		}
		default:
		{
			return false;
		}
	}
	return true;
}

bool CPOMatrix::inverse(const PoMat& src_mat, PoMat& dst_mat, i32 method)
{
	if (src_mat.w != src_mat.h || (src_mat.elem != PO_TYPE_F32 && src_mat.elem != PO_TYPE_F64))
	{
		return false;
	}
	if (method == PO_DECOMP_SVD)
	{
		return false;
	}

	initMatrix(dst_mat, src_mat.h, src_mat.w, src_mat.elem);
	u8* src_data = src_mat.data;
	u8* dst_data = dst_mat.data;
	i32 src_step = src_mat.wstep;
	i32 dst_step = dst_mat.wstep;

	switch (src_mat.w)
	{
		case 2:
		{
			if (src_mat.elem == PO_TYPE_F32)
			{
				f64 d = DET2(SF);
				if (d == 0)
				{
					return false;
				}

				f64 t0, t1;
				t0 = SF(0, 0)*d;
				t1 = SF(1, 1)*d;
				DF(1, 1) = (f32)t0;
				DF(0, 0) = (f32)t1;
				t0 = -SF(0, 1)*d;
				t1 = -SF(1, 0)*d;
				DF(0, 1) = (f32)t0;
				DF(1, 0) = (f32)t1;
			}
			else if (src_mat.elem == PO_TYPE_F64)
			{
				f64 d = DET2(SD);
				if (d == 0)
				{
					return false;
				}

				f64 t0, t1;
				d = 1 / d;
				t0 = SD(0, 0)*d;
				t1 = SD(1, 1)*d;
				DD(1, 1) = t0;
				DD(0, 0) = t1;
				t0 = -SD(0, 1)*d;
				t1 = -SD(1, 0)*d;
				DD(0, 1) = t0;
				DD(1, 0) = t1;
			}
			break;
		}
		case 3:
		{
			if (src_mat.elem == PO_TYPE_F32)
			{
				f64 d = DET3(SF);
				if (d == 0)
				{
					return false;
				}

				f64 t[9];
				d = 1 / d;

				t[0] = (((f64)SF(1, 1) * SF(2, 2) - (f64)SF(1, 2) * SF(2, 1)) * d);
				t[1] = (((f64)SF(0, 2) * SF(2, 1) - (f64)SF(0, 1) * SF(2, 2)) * d);
				t[2] = (((f64)SF(0, 1) * SF(1, 2) - (f64)SF(0, 2) * SF(1, 1)) * d);

				t[3] = (((f64)SF(1, 2) * SF(2, 0) - (f64)SF(1, 0) * SF(2, 2)) * d);
				t[4] = (((f64)SF(0, 0) * SF(2, 2) - (f64)SF(0, 2) * SF(2, 0)) * d);
				t[5] = (((f64)SF(0, 2) * SF(1, 0) - (f64)SF(0, 0) * SF(1, 2)) * d);

				t[6] = (((f64)SF(1, 0) * SF(2, 1) - (f64)SF(1, 1) * SF(2, 0)) * d);
				t[7] = (((f64)SF(0, 1) * SF(2, 0) - (f64)SF(0, 0) * SF(2, 1)) * d);
				t[8] = (((f64)SF(0, 0) * SF(1, 1) - (f64)SF(0, 1) * SF(1, 0)) * d);

				DF(0, 0) = (f32)t[0]; DF(0, 1) = (f32)t[1]; DF(0, 2) = (f32)t[2];
				DF(1, 0) = (f32)t[3]; DF(1, 1) = (f32)t[4]; DF(1, 2) = (f32)t[5];
				DF(2, 0) = (f32)t[6]; DF(2, 1) = (f32)t[7]; DF(2, 2) = (f32)t[8];
			}
			else if (src_mat.elem == PO_TYPE_F64)
			{
				f64 d = DET3(SD);
				if (d == 0)
				{
					return false;
				}
					
				f64 t[9];
				d = 1 / d;

				t[0] = (SD(1, 1) * SD(2, 2) - SD(1, 2) * SD(2, 1)) * d;
				t[1] = (SD(0, 2) * SD(2, 1) - SD(0, 1) * SD(2, 2)) * d;
				t[2] = (SD(0, 1) * SD(1, 2) - SD(0, 2) * SD(1, 1)) * d;

				t[3] = (SD(1, 2) * SD(2, 0) - SD(1, 0) * SD(2, 2)) * d;
				t[4] = (SD(0, 0) * SD(2, 2) - SD(0, 2) * SD(2, 0)) * d;
				t[5] = (SD(0, 2) * SD(1, 0) - SD(0, 0) * SD(1, 2)) * d;

				t[6] = (SD(1, 0) * SD(2, 1) - SD(1, 1) * SD(2, 0)) * d;
				t[7] = (SD(0, 1) * SD(2, 0) - SD(0, 0) * SD(2, 1)) * d;
				t[8] = (SD(0, 0) * SD(1, 1) - SD(0, 1) * SD(1, 0)) * d;

				DD(0, 0) = t[0]; DD(0, 1) = t[1]; DD(0, 2) = t[2];
				DD(1, 0) = t[3]; DD(1, 1) = t[4]; DD(1, 2) = t[5];
				DD(2, 0) = t[6]; DD(2, 1) = t[7]; DD(2, 2) = t[8];
			}
			break;
		}
		default:
		{
			PoMat tmp;
			i32 result = 0, w = src_mat.w;
			copyMatrix(src_mat, tmp);
			setIdentity(dst_mat);

			//calc invert matrix
			if (src_mat.elem == PO_TYPE_F32)
			{
				result = lu_impl<f32>((f32*)tmp.data, w, w, (f32*)dst_data, w, w, FLT_EPSILON * 10);
			}
			else if (src_mat.elem == PO_TYPE_F64)
			{
				result = lu_impl<f64>((f64*)tmp.data, w, w, (f64*)dst_data, w, w, DBL_EPSILON * 100);
			}

			//free buffer
			freeMatrix(tmp);
			if (result == 0)
			{
				setIdentity(dst_mat);
				return false;
			}
			break;
		}
	}
	return true;
}
