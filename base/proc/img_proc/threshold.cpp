#include "threshold.h"
#include "base.h"
#include "proc/image_proc.h"

//#define POR_TESTMODE

CThreshold::CThreshold()
{
}

CThreshold::~CThreshold()
{
}

u8 CThreshold::getThresholdLowBand(u8* img_ptr, u8* mask_img_ptr, i32 w, i32 h, f32 th_rate)
{
	if (!img_ptr)
	{
		return 0;
	}

	/* 히스토그람을 추출한다. */
	i32 hist[256];
	i32 i, sum = 0, mean_sum = 0, max_sum = 0;
	memset(hist, 0, sizeof(i32) * 256);
	CImageProc::makeHistogram(img_ptr, mask_img_ptr, w, h, hist, NULL);

	for (i = 0; i < 256; i++)
	{
		max_sum += hist[i];
	}
	max_sum *= th_rate;

	for (i = 0; i < 256; i++)
	{
		sum += hist[i];
		mean_sum += i * hist[i];
		if (sum > max_sum)
		{
			break;
		}
	}
	if (sum <= 0)
	{
		return 0;
	}

	i32 hist_mean = mean_sum / sum;
	return calcThreshold(hist, hist_mean, i);
}

u8 CThreshold::calcThreshold(i32* hist_ptr, i32 hist_min, i32 hist_max)
{
	u8 optimal_th_st = (hist_min + hist_max) / 2;
	u8 optimal_th_ed = optimal_th_st;

	if (!hist_ptr)
	{
		return optimal_th_st;
	}

	i32 i, s;
	f32 m0, d0, ww, m, beta, p0;
	f32 delta;

	//total mean value of image
	s = 0; m0 = 0;
	for (i = hist_min; i <= hist_max; i++)
	{
		s += hist_ptr[i];
		m0 += i * hist_ptr[i];
	}
	m0 /= s; //toal mean gray level value 

	//optimal threshold value determination
	ww = m = beta = d0 = 0;
	for (i = hist_min; i < hist_max - 1; i++)
	{
		p0 = (f32)hist_ptr[i] / s;
		ww = ww + p0;
		m = m + i * p0;
		if (ww >= 0.999999)
		{
			break;//2000.12.7
		}

		//To avoid difference between DEBUG and RELEASE
		d0 = (m0 * ww - m) * (m0 * ww - m) / (ww * (1 - ww));
		delta = beta - d0;
		if (std::abs(delta) < PO_EPSILON)
		{
			optimal_th_ed = i + 1;
		}
		else if (beta < d0)
		{
			optimal_th_st = i + 1;
			optimal_th_ed = optimal_th_st;
			beta = d0;
		}
	}
	return (optimal_th_st + optimal_th_ed) / 2;
}

u8 CThreshold::getThresholdWithOtsu(u8* img_ptr, i32 w, i32 h)
{
	return getThresholdWithOtsu(img_ptr, NULL, w, h, NULL);
}

u8 CThreshold::getThresholdWithOtsu(u8* img_ptr, u8* mask_img_ptr, i32 w, i32 h, bool* need_invert_ptr)
{
	if (!img_ptr)
	{
		return 0;
	}

	i32 hist[256];
	i32 border_hist[256];
	memset(hist, 0, sizeof(i32) * 256);
	memset(border_hist, 0, sizeof(i32) * 256);

	/* 히스토그람을 추출한다. */
	CImageProc::makeHistogram(img_ptr, mask_img_ptr, w, h, hist, border_hist);

	/* 2진화턱값결정을 위한 최대최소값을 계산한다.*/
	i32 i;
	i32 acc_hist = 0, hist_min = 0, hist_max = 0;

	for (i = 0; i < 256; i++)
	{
		acc_hist += hist[i];
		if (acc_hist > h)
		{
			hist_min = i; break;  /* 히스토그람최소값 */
		}
	}

	acc_hist = 0;
	for (i = 255; i > 0; i--)
	{
		acc_hist += hist[i];
		if (acc_hist > h)
		{
			hist_max = i; break;  /* 히스토그람최대값 */
		}
	}

	if ((hist_max - hist_min) < 2)
	{
		return (hist_min + hist_max) / 2;
	}

	/* 무리내분산과 무리사이분산을 리용한 턱값결정을 진행한다. */
	u8 opt_th = calcThreshold(hist, hist_min, hist_max);

	/* 에지히스토그람을 리용하여 바탕색으로 결정한다. */
	if (need_invert_ptr)
	{
		*need_invert_ptr = CImageProc::isInvertedBackground(opt_th, hist, border_hist);
	}
	return opt_th;
}

u8 CThreshold::getThresholdWithOtsuEx(u8* img_ptr, i32 w, i32 h)
{
	return getThresholdWithOtsuEx(img_ptr, NULL, w, h, NULL);
}

u8 CThreshold::getThresholdWithOtsuEx(u8* img_ptr, u8* mask_img_ptr, i32 w, i32 h, bool* need_invert_ptr)
{
	if (!img_ptr)
	{
		return 0;
	}

	i32 hist[256];
	i32 border_hist[256];
	memset(hist, 0, sizeof(i32) * 256);
	memset(border_hist, 0, sizeof(i32) * 256);

	/* 히스토그람을 추출한다. */
	CImageProc::makeHistogram(img_ptr, mask_img_ptr, w, h, hist, border_hist);

	/* 2진화턱값결정을 위한 최대최소값을 계산한다.*/
	i32 i;
	i32 acc_hist = 0, hist_min = 0, hist_max = 0;

	for (i = 0; i < 256; i++)
	{
		acc_hist += hist[i];
		if (acc_hist > h)
		{
			hist_min = i; break;  /* 히스토그람최소값 */
		}
	}

	acc_hist = 0;
	for (i = 255; i > 0; i--)
	{
		acc_hist += hist[i];
		if (acc_hist > h)
		{
			hist_max = i; break;  /* 히스토그람최대값 */
		}
	}

	if ((hist_max - hist_min) <= 0)
	{
		return (hist_min + hist_max) / 2;
	}

	/* 무리내분산과 무리사이분산을 리용한 턱값결정을 진행한다. */
	u8 opt_th0 = calcThreshold(hist, hist_min, hist_max);
	u8 opt_th1 = calcThreshold(hist, hist_min, opt_th0);
	u8 opt_th2 = calcThreshold(hist, opt_th0, hist_max);
	u8 opt_th = calcThreshold(hist, opt_th1, opt_th2);

	/* 에지히스토그람을 리용하여 바탕색으로 결정한다. */
	if (need_invert_ptr)
	{
		*need_invert_ptr = CImageProc::isInvertedBackground(opt_th, hist, border_hist);
	}
	return opt_th;
}

void CThreshold::threshold(u8* src_img_ptr, i32 w, i32 h, i32 mode, 
						u8* dst_img_ptr, u8* mask_img_ptr, i32 th_value)
{
	if (!src_img_ptr || w*h <= 0)
	{
		return;
	}

 	i32 i, size = w*h;
	u8 min_value = 0, max_value = 0xFF;
 	dst_img_ptr = (dst_img_ptr != NULL) ? dst_img_ptr : src_img_ptr;
	
	//fitting with polynormal background
	if (CPOBase::bitCheck(mode, PO_THRESH_FITTING))
	{
		u8* bkg_img_ptr = CThreshold::makeBackground(src_img_ptr, w, h);
		if (!bkg_img_ptr)
		{
			return;
		}

		u8* src_scan_ptr = src_img_ptr;
		u8* bkg_scan_ptr = bkg_img_ptr;
		if (CPOBase::bitCheck(mode, PO_THRESH_BINARY_INV))
		{
			for (i = 0; i < size; i++)
			{
				*src_scan_ptr = po::_max((i32)*bkg_scan_ptr - *src_scan_ptr, 0);
				src_scan_ptr++;  bkg_scan_ptr++;
			}
		}
		else
		{
			for (i = 0; i < size; i++)
			{
				*src_scan_ptr = po::_max((i32)*src_scan_ptr - *bkg_scan_ptr, 0);
				src_scan_ptr++;  bkg_scan_ptr++;
			}
		}
		CPOBase::bitAdd(mode, PO_THRESH_OTSU);
		CPOBase::bitRemove(mode, PO_THRESH_BINARY_INV);
		POSAFE_DELETE_ARRAY(bkg_img_ptr);

#if defined(POR_TESTMODE)
		CImageProc::saveImgOpenCV(PO_LOG_PATH"update_image.bmp", src_img_ptr, w, h);
#endif
	}

	if (CPOBase::bitCheck(mode, PO_THRESH_OTSU))
	{
		th_value = CThreshold::getThresholdWithOtsuEx(src_img_ptr, mask_img_ptr, w, h, NULL);
	}
	if (CPOBase::bitCheck(mode, PO_THRESH_BINARY_INV))
	{
		CPOBase::swap(min_value, max_value);
	}

	//threshold
	if (mask_img_ptr)
	{
		//threshold with mask image
		for (i = 0; i < size; i++, src_img_ptr++, dst_img_ptr++)
		{
			if (mask_img_ptr[i] > 0)
			{
				*dst_img_ptr = (*src_img_ptr >= th_value) ? max_value : min_value;
			}
		}
	}
	else
	{
		//threshold without mask image
		for (i = 0; i < size; i++, src_img_ptr++, dst_img_ptr++)
		{
			*dst_img_ptr = (*src_img_ptr >= th_value) ? max_value : min_value;
		}
	}
}

u8* CThreshold::makeBackground(u8* src_img_ptr, i32 w, i32 h)
{
	//please refer document....
	//[Shading Surface Estimation using Piecewise Polynomials for Binarizing
	//Unevenly Illuminated Document Images].pdf
	if (!src_img_ptr || w*h <= 0)
	{
		return NULL;
	}

	//resize
	i32 tw = 480, th = 320;
	u8* dst_img_ptr = CImageProc::makeThumbImage(src_img_ptr, w, h, 1, tw, th);

	//make matrix
	i32 x, y, i, j, v;
	f64 a = 2 * sqrt(2) / tw;
	f64 b = 2 * sqrt(2) / th;
	f64 c = -sqrt(2);
	f64 fx, fx2, fx3, fy, fy2, fy3, fv;
	u8* dst_scan_ptr = dst_img_ptr;

	f64 ca[100], cb[10], cc[10], cx[10];
	memset(ca, 0, sizeof(f64) * 100);
	memset(cb, 0, sizeof(f64) * 10);
	for (y = 0; y < th; y++)
	{
		for (x = 0; x < tw; x++)
		{
			fx = a * x + c; fx2 = fx * fx; fx3 = fx2 * fx;
			fy = b * y + c; fy2 = fy * fy; fy3 = fy2 * fy;
			v = *dst_scan_ptr; dst_scan_ptr++;

			cc[0] = 1; cc[1] = fx; cc[2] = fy;
			cc[3] = fx * fy; cc[4] = fx2; cc[5] = fy2;
			cc[6] = fx2 * fy; cc[7] = fx * fy2; cc[8] = fx3; cc[9] = fy3;

			for (i = 0; i < 10; i++)
			{
				j = i * 10;
				ca[j + 0] += cc[i] * cc[0]; ca[j + 1] += cc[i] * cc[1]; ca[j + 2] += cc[i] * cc[2];
				ca[j + 3] += cc[i] * cc[3]; ca[j + 4] += cc[i] * cc[4]; ca[j + 5] += cc[i] * cc[5];
				ca[j + 6] += cc[i] * cc[6]; ca[j + 7] += cc[i] * cc[7]; ca[j + 8] += cc[i] * cc[8];
				ca[j + 9] += cc[i] * cc[9];
				cb[i] += v * cc[i];
			}
		}
	}

	//calc coffecient matrix
	cv::Mat cv_A(10, 10, CV_64FC1, ca);
	cv::Mat cv_B(10, 1, CV_64FC1, cb);
	cv::Mat cv_X(10, 1, CV_64FC1, cx);
	cv::solve(cv_A, cv_B, cv_X, cv::DECOMP_SVD);

	//build background
	dst_scan_ptr = dst_img_ptr;
	for (y = 0; y < th; y++)
	{
		for (x = 0; x < tw; x++)
		{
			fx = a * x + c; fx2 = fx * fx; fx3 = fx2 * fx;
			fy = b * y + c; fy2 = fy * fy; fy3 = fy2 * fy;
			fv = cx[0] + cx[1] * fx + cx[2] * fy + cx[3] * fx * fy + 
				cx[4]*fx2 + cx[5] * fy2 + cx[6] * fx2*fy + cx[7] * fx*fy2 +
				cx[8] * fx3 + cx[9] * fy3;
			*dst_scan_ptr = po::_max(po::_min(fv, 0xFF), 0);
			dst_scan_ptr++;
		}
	}

	//restore size
	u8* new_img_ptr = po_new u8[w*h];
	cv::Mat cv_new_img(h, w, CV_8UC1, new_img_ptr);
	cv::Mat cv_bkg_img(th, tw, CV_8UC1, dst_img_ptr);
	cv::resize(cv_bkg_img, cv_new_img, cv::Size(w, h));

#if defined(POR_TESTMODE)
	CImageProc::saveImgOpenCV(PO_LOG_PATH"background.bmp", new_img_ptr, w, h);
#endif

	//free buffer
	POSAFE_DELETE_ARRAY(dst_img_ptr);
	return new_img_ptr;
}