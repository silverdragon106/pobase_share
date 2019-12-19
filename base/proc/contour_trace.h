#pragma once

#include "struct.h"
#include "memory_pool.h"
#include <opencv2/opencv.hpp>

enum ContourSearchMode
{
	kSearhModeNone = 0,
	kSearchModeFore,
	kSearchModeBack
};

class CContourTrace: public CPOMemPool
{
public:
	CContourTrace();
	virtual ~CContourTrace();

	i32				getContourPixel(u8* edge_img_ptr, i32 w, i32 h, i32 fore_pxiel, i32 pixel_num, 
								i32 min_contour_pixel, ContouruVector& contour_vec);
	
	template <class T, class U>
	bool			traceContour(T* pad_img_ptr, i32 w, i32 x, i32 y, i32 fore_pixel, i32 back_pixel,
								i32 mode, Contour<U>* contour_ptr);
};

#include "contour_trace-inl.h"