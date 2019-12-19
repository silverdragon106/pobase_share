#pragma once
#include "define.h"

class CHistogram;

template <typename T, typename U>
void CHistogram::meanFilter(T* data_ptr, i32 data_num, i32 ksize, U* smooth_data_ptr)
{
	if (!data_ptr || !smooth_data_ptr
		|| data_num <= 0 || data_num <= 2*ksize || ksize<=0)
	{
		return;
	}

	i32 i;
	f32 s = data_ptr[0];
	for (i = 1; i <= ksize; i++)
	{
		s += data_ptr[data_num - i];
		s += data_ptr[i];
	}

	smooth_data_ptr[0] = s;
	for (i = 1; i < data_num; i++)
	{
		smooth_data_ptr[i] = smooth_data_ptr[i - 1];
		smooth_data_ptr[i] -= data_ptr[(i + data_num - ksize - 1) % data_num];
		smooth_data_ptr[i] += data_ptr[(i + ksize) % data_num];
	}

	for (i = 0; i < data_num; i++)
	{
		smooth_data_ptr[i] /= (2 * ksize + 1);
	}
}

template <typename T>
i32 CHistogram::findPeakPoints(T* short_data_ptr, T* long_data_ptr, i32 data_num,
							i32 limit_count, f32 min_rate, i32vector& peak_index_vec)
{
	peak_index_vec.clear();

	if (!short_data_ptr || !long_data_ptr || data_num <= 0 || limit_count <= 0)
	{
		return 0;
	}

	i32 index = -1;
	i32 last_index = -1;
	i32 peak_index = -1;
	T cur_value = -1;
	T peak_value = -PO_MAXINT;

	i32 loop_count = 0;
	i32 over_mode = 0; //0: unknown, 1: lower, 2: higher
	f32vector peak_value_vec;

	while (true)
	{
		loop_count++;
		index = (index + 1) % data_num;
		if (index == last_index || loop_count >= 2 * data_num)
		{
			break;
		}

		cur_value = short_data_ptr[index];
		if (cur_value <= long_data_ptr[index]) //lower
		{
			if (over_mode == 2)
			{
				peak_index_vec.push_back(peak_index);
				peak_value_vec.push_back(peak_value);
			}
			over_mode = 1;
			peak_index = -1;
			peak_value = -PO_MAXINT;
		}
		else //higher
		{
			if (over_mode == 1)
			{
				//set last index
				if (last_index < 0)
				{
					last_index = index;
				}
				over_mode = 2;
			}
			if (over_mode == 2)
			{
				if (peak_value < cur_value)
				{
					peak_value = cur_value; peak_index = index;
				}
			}
		}
	}

	/* 피크자료들을 정렬하고 개수를 제한한다. */
	f32 tmp;
	i32 i, j, itmp;
	i32 peak_count = (i32)peak_index_vec.size();
	limit_count = po::_min(limit_count, peak_count);
	
	if (peak_count <= 0)
	{
		return 0;
	}

	for (i = 0; i < limit_count;i++)
	{
		for (j = i + 1; j < peak_count; j++)
		{
			if (peak_value_vec[i] < peak_value_vec[j])
			{
				tmp = peak_value_vec[i];
				peak_value_vec[i] = peak_value_vec[j];
				peak_value_vec[j] = tmp;

				itmp = peak_index_vec[i];
				peak_index_vec[i] = peak_index_vec[j];
				peak_index_vec[j] = itmp;
			}
		}
	}

	f32 th_min_similar = peak_value_vec[0] * min_rate;
	for (i = 0; i < limit_count; i++)
	{
		if (peak_value_vec[i] < th_min_similar)
		{
			limit_count = i;
			break;
		}
	}
	return limit_count;
}
