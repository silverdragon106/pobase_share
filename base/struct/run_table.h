#pragma once

#include "struct.h"

#pragma pack(push, 4)

class CImgRunTable
{
public:
	i32					m_width;
	i32					m_height;
	i32					m_pixels;
	i32					m_run_count;
	i32*				m_pxy_ptr;
	u16*				m_run2_ptr;

	u16					_st_pos;
	u16					_count_pixel;

public:
	CImgRunTable();
	~CImgRunTable();

	i32*				getPxyTable();
	u16*				getRunTable();
	i32					getArraySize();
	i32					getMaxArraySize();

	void				initBuffer(i32 nw, i32 nh);
	void				checkBuffer(i32 nw, i32 nh);
	void				freeBuffer();

	i32					memSize();
	i32					memRead(u8*& buffer_ptr, i32& buffer_size);
	i32					memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool				fileRead(FILE* fp);
	bool				fileWrite(FILE* fp);
	void				setValue(CImgRunTable* run_table_ptr);

	bool				makeRunTable(u8* img_ptr, i32 nw, i32 nh, bool vx_method = true);
	bool				makeRunTable(u8* img_ptr, u8* mask_img_ptr, i32 nw, i32 nh, bool vx_method = true);
	bool				makeRunTable(u8* img_ptr, i32 nw, i32 nh, i32 val, bool vx_method = true);
	bool				makeInvRunTable(u8* img_ptr, i32 nw, i32 nh, bool vx_method = true);
	bool				makeImageFromRun(u8* img_ptr, i32 nw, i32 nh, i32 val, bool vx_method = true);
	u8*					makeImageFromRun(i32& nw, i32& nh, i32 val, bool vx_method = true);

	CImgRunTable&		operator=(const CImgRunTable& robj);

	// inline functions
	inline void buildStart(i32 w, i32 h) 
	{
		_st_pos = 0;
		_count_pixel = 0;
		m_run_count = 0;
		m_pixels = 0;
		checkBuffer(w, h);
	}
	inline void buildStop()
	{
	}
	inline void updateNewPixel(i32 pos)
	{
		m_pixels++;
		_count_pixel++;
		if (_count_pixel == 1)
		{
			_st_pos = pos;
		}
	}
	inline void updateFreePixel()
	{
		if (_count_pixel > 0)
		{
			m_run2_ptr[m_run_count] = _st_pos;
			m_run2_ptr[m_run_count + 1] = _count_pixel;
			m_run_count += 2;
			_count_pixel = 0;
		}
	}
	inline void	updateNewLine(i32 pos)
	{
		updateFreePixel();
		m_pxy_ptr[pos + 1] = m_run_count;
	}

	template <typename T>
	bool makeRunTable(Recti range, i32 run_count, T* run_row_ptr, T* run_col_begin_ptr, T* run_col_end_ptr)
	{
		i32 nw = range.getWidth();
		i32 nh = range.getHeight();
		i32 x_pos = range.x1;
		i32 y_pos = range.y1;

		if (nw <= 0 || nh <= 0)
		{
			return false;
		}
		if (!run_row_ptr || !run_col_begin_ptr || !run_col_end_ptr)
		{
			return false;
		}

		buildStart(nw, nh);

		i32 i, j, tmp_y_pos, ns, ne, nc;
		i32 next_ypos, index = 0;
		i32 prev_ypos = -1, pixel_count = 0;

		//build run table
		for (i = 0; i < run_count; i++)
		{
			tmp_y_pos = run_row_ptr[i] - y_pos;

			if (tmp_y_pos != prev_ypos)
			{
				next_ypos = tmp_y_pos;
				for (j = prev_ypos + 1; j <= next_ypos; j++)
				{
					m_pxy_ptr[j] = index;
				}
				prev_ypos = next_ypos;
			}
			if (tmp_y_pos < 0 || tmp_y_pos >= nh)
			{
				break;
			}

			ns = run_col_begin_ptr[i] - x_pos;
			ne = run_col_end_ptr[i] - x_pos;
			ns = po::_max(ns, 0);
			ne = po::_min(ne, nw - 1);
			nc = ne - ns + 1;
			if (nc > 0)
			{
				m_run2_ptr[index++] = ns;
				m_run2_ptr[index++] = nc;
				pixel_count += nc;
			}
		}

		//build pxy table
		for (i = prev_ypos + 1; i <= nh; i++)
		{
			m_pxy_ptr[i] = index;
		}

		m_width = nw;
		m_height = nh;
		m_pixels = pixel_count;
		m_run_count = index;

		buildStop();
		return true;
	}
};

#pragma pack(pop)