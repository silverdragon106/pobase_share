#pragma once
#include "contour_trace.h"

template <class T, class U>
bool CContourTrace::traceContour(T* pad_img_ptr, i32 w, i32 x, i32 y, i32 fore_pixel, i32 back_pixel,
								i32 mode, Contour<U>* contour_ptr)
{
	if (!contour_ptr)
	{
		return false;
	}

	i32 dir[8];
	i32 index = y*w + x;
	i32 i = 0, ni = 0, nk = 0, pixel_count = 0;
	i32 nind, npixel, stind = index;
	i32 pixel_index = 0;
	i32 pixel_malloc_count = contour_ptr->m_malloc_pixel_count;
	Pixel<U>* pixel_ptr = contour_ptr->m_pixel_ptr;
	bool is_closed = false;
	
	if (!pixel_ptr)
	{
		return false;
	}
	
	if (mode == kSearchModeFore)
	{
		//clockwise
		nk = 6;
		dir[0] = 1; dir[1] = w + 1; dir[2] = w; dir[3] = w - 1;
		dir[4] =-1; dir[5] =-1 - w; dir[6] =-w; dir[7] = 1 - w;
	}
	else if (mode == kSearchModeBack)
	{
		//counter-clockwise
		nk = 3;
		dir[0] = 1; dir[1] = 1 - w; dir[2] =-w; dir[3] = -1 - w;
		dir[4] =-1; dir[5] = w - 1; dir[6] = w; dir[7] = w + 1;
	}
	else
	{
		return false;
	}

	/* 시작픽셀을 보관한다. */
	if (pad_img_ptr[index] == fore_pixel)
	{
		pad_img_ptr[index] = back_pixel;
		pixel_ptr[pixel_count++] = Pixel<U>(x - 1, y - 1);
	}
	
	/* 륜곽선을 따라 지정한 방향으로 순환한다.*/
	while (true)
	{
		for (i = 1; i < 8; i++)
		{
			ni = (i + nk) % 8;
			nind = index + dir[ni];
			npixel = pad_img_ptr[nind];
			if (nind == stind)
			{
				/* 순환도중에 시작픽셀을 만나면 고리형륜곽으로 판정한다.*/
				if (pixel_count > 2)
				{
					is_closed = true;
					break;
				}
			}
			if (npixel == back_pixel)
			{
				i = 8;
				break;
			}
			if (npixel == fore_pixel && i > 1 && i < 7)
			{
				break;
			}
		}
		if (i >= 8 || is_closed)
		{
			break;
		}

		x = (nind % w) - 1;
		y = (nind / w) - 1;
		nk = (ni + 4) % 8;
		index = nind;
		
		//store contour pixels
		pixel_count++;
		pad_img_ptr[index] = back_pixel;
		if (mode == kSearchModeFore)
		{
			pixel_index++;
		}
		else
		{
			pixel_index = (pixel_index + pixel_malloc_count - 1) % pixel_malloc_count;
		}

		pixel_ptr[pixel_index] = Pixel<U>(x, y);
	}

	if (mode == kSearchModeFore)
	{
		contour_ptr->m_end_pixel_index = pixel_index;
	}
	else
	{
		contour_ptr->m_start_pixel_index = pixel_index;
	}

	contour_ptr->m_is_closed = is_closed;
	contour_ptr->m_pixel_count += pixel_count;
	return true;
}


