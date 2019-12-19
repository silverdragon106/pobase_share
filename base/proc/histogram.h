#pragma once

#include "struct.h"
#include "memory_pool.h"
#include <opencv2/opencv.hpp>

class CHistogram : public CPOMemPool
{
public:
	CHistogram();
	virtual ~CHistogram();

	template <typename T, typename U>
	static void				meanFilter(T* data_ptr, i32 data_count, i32 ksize, U* smooth_data_ptr);

	template <typename T>
	static i32				findPeakPoints(T* short_data_ptr, T* long_data_ptr,
									i32 data_count, i32 limit_count, f32 min_rate, i32vector& peak_index_vec);
};

#include "histogram-inl.h"