#pragma once
#include "struct.h"

class CImageProc;

template <typename T>
void CImageProc::makeIntegralImage(i32* int_img_ptr, T* img_ptr, i32 w, i32 h)
{
	if (int_img_ptr == NULL || img_ptr == NULL || w*h <= 0)
	{
		return;
	}

	T* tmp_img_ptr = img_ptr;
	i32* tmp_int_img_ptr = int_img_ptr;

	i32 x, y, s = 0;
	for (x = 0; x < w; x++)
	{
		s += *tmp_img_ptr;
		*tmp_int_img_ptr = s;
		tmp_img_ptr++;
		tmp_int_img_ptr++;
	}
	for (y = 1; y < h; y++)
	{
		s = 0;
		for (x = 0; x < w; x++)
		{
			s += *tmp_img_ptr;
			*tmp_int_img_ptr = *(tmp_int_img_ptr - w) + s;
			tmp_img_ptr++;
			tmp_int_img_ptr++;
		}
	}
}

template <typename T>
void CImageProc::makeIntegralImageSq(i32* int_img_ptr, T* img_ptr, i32 w, i32 h)
{
	if (int_img_ptr == NULL || img_ptr == NULL || w*h <= 0)
	{
		return;
	}

	T* tmp_img_ptr = img_ptr;
	i32* tmp_int_img_ptr = int_img_ptr;

	i32 x, y, tmp, s = 0;
	for (x = 0; x < w; x++, tmp_img_ptr++, tmp_int_img_ptr++)
	{
		tmp = *tmp_img_ptr;
		tmp = tmp*tmp;
		s += tmp;
		*tmp_int_img_ptr = s;
	}
	for (y = 1; y < h; y++)
	{
		s = 0;
		for (x = 0; x < w; x++, tmp_img_ptr++, tmp_int_img_ptr++)
		{
			tmp = *tmp_img_ptr;
			tmp = tmp*tmp;
			s += tmp;
			*tmp_int_img_ptr = *(tmp_int_img_ptr - w) + s;
		}
	}
}

template <typename T>
void CImageProc::makeRobertGradImage(T* edge_img_ptr, u8* img_ptr, i32 w, i32 h)
{
	if (edge_img_ptr == NULL || img_ptr == NULL || w <= 0 || h <= 0)
	{
		return;
	}

	u16 a, b;
	i32 x, y, g1, g2;
	u8* tmp_img_ptr = img_ptr;
	T* tmp_edge_img_ptr = edge_img_ptr;
	memset(tmp_edge_img_ptr, 0, sizeof(T)*w);

	for (y = 1; y < h; y++)
	{
		tmp_img_ptr = img_ptr + y*w + 1;
		tmp_edge_img_ptr = edge_img_ptr + y*w;
		tmp_edge_img_ptr[0] = 0;
		tmp_edge_img_ptr++;

		for (x = 1; x < w; x++)
		{
			a = *(u16*)(tmp_img_ptr - 1);
			b = *(u16*)(tmp_img_ptr - w - 1);
			g1 = std::abs((i32)(a >> 8) - (b & 0xFF));
			g2 = std::abs((i32)(a & 0xFF) - (b >> 8));
			if (g1 < g2)
			{
				*tmp_edge_img_ptr = g2 + (g1 >> 2);
			}
			else
			{
				*tmp_edge_img_ptr = g1 + (g2 >> 2);
			}
			tmp_edge_img_ptr++; tmp_img_ptr++;
		}
	}
}

template <typename T>
void CImageProc::makeSubPixel(T* grad_img_ptr, i32 w, i32 h, Pixelf* pixel_ptr, i32 count)
{
	i32 px, py, index;
	i32 w1 = w - 1;
	i32 h1 = h - 1;
	i32 a, b, c, d, e, k1, k2;

	for (i32 i = 0; i < count; i++, pixel_ptr++)
	{
		px = (i32)(pixel_ptr->x);
		py = (i32)(pixel_ptr->y);
		

		if (px > 0 && py > 0 && px < w1 && py < h1)
		{
			pixel_ptr->x = px + 0.5f;
			pixel_ptr->y = py + 0.5f;

			index = py*w + px;
			a = grad_img_ptr[index - 1];
			b = grad_img_ptr[index + 1];
			c = grad_img_ptr[index - w];
			d = grad_img_ptr[index + w];
			e = grad_img_ptr[index];

			k1 = a + b + e;
			k2 = c + d + e;
			if (k1)
			{
				pixel_ptr->x += (f32)(b - a) /k1;
			}
			if (k2)
			{
				pixel_ptr->y += (f32)(d - c) /k2;
			}
		}
		else
		{
			pixel_ptr->x = px;
			pixel_ptr->y = py;
		}
	}
}

template <typename T>
void CImageProc::makePaddingBinary(u8* pad_img_ptr, T* img_ptr, i32 w, i32 h, i32 padding_size)
{
	if (img_ptr == NULL || pad_img_ptr == NULL || w <= 0 || h <= 0)
	{
		return;
	}

	i32 x, y;
	i32 w2 = w + 2 * padding_size;
	i32 h2 = h + 2 * padding_size;
	memset(pad_img_ptr, 0, w2*h2);

	//fill padding and change with 0, 1
	u8* tmp_pad_img_ptr;
	for (y = 0; y < h; y++)
	{
		tmp_pad_img_ptr = pad_img_ptr + (y + padding_size)*w2 + padding_size;
		for (x = 0; x < w; x++)
		{
			if (*img_ptr)
			{
				*tmp_pad_img_ptr = 1;
			}
			img_ptr++; tmp_pad_img_ptr++;
		}
	}
}

template <typename T>
void CImageProc::makePaddingImage(u8* pad_img_ptr, T* img_ptr, i32 w, i32 h, u8 bg_pixel, i32 padding_size)
{
	if (img_ptr == NULL || pad_img_ptr == NULL || w <= 0 || h <= 0)
	{
		return;
	}

	i32 w1 = w + 2 * padding_size;
	i32 h1 = h + 2 * padding_size;
	i32 wsize = w*sizeof(T);
	memset(pad_img_ptr, bg_pixel, w1*h1);

	//fill padding only with gray image
	for (i32 y = 0; y < h; y++)
	{
		memcpy(pad_img_ptr + (y + padding_size)*w1 + padding_size, img_ptr + y*w, wsize);
	}
}

template <typename T, typename U>
bool CImageProc::fillContourTrace(T* img_ptr, i32 w, i32 h, u8* pad_img_ptr, i32 x, i32 y, i32 mode, i32 nk, i32 val, Contour<U>* contour_ptr)
{
	i32 w1 = w + 2;
	i32 index = (y + 1)*w1 + x + 1;

	i32 i, ni, dir[8], step;
	i32 next_index, st_index = index;

	if (mode == 0)
	{
		//counter-clockwise
		dir[0] = 1; dir[1] = 1 - w1; dir[2] =-w1; dir[3] = -1 - w1;
		dir[4] =-1; dir[5] = w1 - 1; dir[6] = w1; dir[7] = w1 + 1;
	}
	else
	{
		//clockwise
		dir[0] = 1; dir[1] = w1 + 1; dir[2] = w1; dir[3] = w1 - 1;
		dir[4] =-1; dir[5] =-1 - w1; dir[6] =-w1; dir[7] = 1 - w1;
	}
	
	i32 cstp = 3;
	i32 ccount = 0;
	bool is_closed = false;
	Pixel<U>* pixel_ptr = NULL;
	if (contour_ptr)
	{
		pixel_ptr = contour_ptr->m_pixel_ptr;
	}

	pad_img_ptr[index]++;
	img_ptr[y*w + x] = val;
	if (contour_ptr)
	{
		pixel_ptr[ccount++] = Pixel<U>(x, y);
	}

	while (true)
	{
		for (i = 1; i <= 8; i++)
		{
			ni = (i + nk) % 8;
			next_index = index + dir[ni];
			step = pad_img_ptr[next_index];
			if (step == 0)
			{
				continue;
			}
			if (step >= cstp)
			{
				is_closed = true;
				break;
			}

			cstp = (next_index == st_index) ? 2 : 3;//start point has to new line
			index = next_index;

			if (pad_img_ptr[index] == 1)
			{
				x = (index % w1) - 1;
				y = (index / w1) - 1;
			}
			
			pad_img_ptr[index]++;
			img_ptr[y*w + x] = val;
			nk = (ni + 4) % 8;

			//store contour pixels
			if (contour_ptr)
			{
				pixel_ptr[ccount++] = Pixel<U>(x, y);
			}
			break;
		}
		if (is_closed || i == 9)
		{
			break;
		}
	}

	if (contour_ptr)
	{
		contour_ptr->m_pixel_count = ccount;
		contour_ptr->m_is_closed = is_closed;
	}
	return is_closed;
}

template <typename T>
void CImageProc::gradL2Norm(T* gx_ptr, T* gy_ptr, T* grad_ptr, i32 w, i32 h, bool usd_l2_norm)
{
	if (gx_ptr == NULL || gy_ptr == NULL || grad_ptr == NULL)
	{
		return;
	}

	i32 i, wh = w*h;
	if (usd_l2_norm)
	{
		//calc normal gradient by L2Norm
		i32 abs_gx, abs_gy;
		for (i = 0; i < wh; i++)
		{
			abs_gx = std::abs(*gx_ptr);
			abs_gy = std::abs(*gy_ptr);
			*grad_ptr = (short)std::sqrt((abs_gx*abs_gx) + (abs_gy*abs_gy));
			
			gx_ptr++;
			gy_ptr++;
			grad_ptr++;
		}
	}
	else
	{
		//calc approx gradient by abs
		for (i = 0; i < wh; i++, grad_ptr++, gx_ptr++, gy_ptr++)
		{
			*grad_ptr = std::abs(*gx_ptr) + std::abs(*gy_ptr);
		}
	}
}
