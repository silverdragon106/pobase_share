#pragma once

#include "struct.h"
#include <opencv2/opencv.hpp>

#define PO_THRESH_BINARY		0x01
#define PO_THRESH_BINARY_INV	0x02
#define PO_THRESH_FITTING		0x04
#define PO_THRESH_OTSU			0x08

class CThreshold
{
public:
	CThreshold();
	virtual ~CThreshold();

	static void				threshold(u8* src_img_ptr, i32 w, i32 h, i32 mode = PO_THRESH_BINARY,
									u8* dst_img_ptr = NULL, u8* mask_img_ptr = NULL, i32 th_value = 0x7F);

	static u8				calcThreshold(i32* hist_ptr, i32 hist_min, i32 hist_max);

	static u8				getThresholdLowBand(u8* img_ptr, u8* mask_img_ptr, i32 w, i32 h, f32 th_rate = 0.75f);
	static u8				getThresholdWithOtsu(u8* img_ptr, i32 w, i32 h);
	static u8				getThresholdWithOtsuEx(u8* img_ptr, i32 w, i32 h);
	static u8				getThresholdWithOtsu(u8* img_ptr, u8* mask_img_ptr, i32 w, i32 h, bool* need_invert_ptr);
	static u8				getThresholdWithOtsuEx(u8* img_ptr, u8* mask_img_ptr, i32 w, i32 h, bool* need_invert_ptr);

	static u8*				makeBackground(u8* src_img_ptr, i32 w, i32 h);
};
