#include "ssim_processor.h"
#include "proc/image_proc.h"
#include "logger/logger.h"
#include "math/squareroots.h"

//#define POR_TESTMODE

//////////////////////////////////////////////////////////////////////////
SSIMMap::SSIMMap()
{
	mean_lum_ptr = NULL;
	mean_struct_sq_ptr = NULL;
	edge_img_ptr = NULL;
	ssim_map_ptr = NULL;

	is_external = false;
}

SSIMMap::~SSIMMap()
{
	freeBuffer();
}

void SSIMMap::initBuffer(i32 w, i32 h, bool is_external)
{
	freeBuffer();
	if (w <= 0 || h <= 0)
	{
		return;
	}

	if (!is_external)
	{
		i32 wh = w*h;
		mean_lum_ptr = new u8[wh];
		mean_struct_sq_ptr = new u16[wh];
		edge_img_ptr = new u16[wh];
		ssim_map_ptr = new f32[wh];
		memset(mean_lum_ptr, 0, sizeof(u8)*wh);
		memset(mean_struct_sq_ptr, 0, sizeof(u16)*wh);
		memset(edge_img_ptr, 0, sizeof(u16)*wh);
		memset(ssim_map_ptr, 0, sizeof(f32)*wh);
	}
	this->is_external = is_external;
}

void SSIMMap::freeBuffer()
{
	if (is_external)
	{
		mean_lum_ptr = NULL;
		mean_struct_sq_ptr = NULL;
		edge_img_ptr = NULL;
		ssim_map_ptr = NULL;
		is_external = false;
	}
	else
	{
		POSAFE_DELETE_ARRAY(mean_lum_ptr);
		POSAFE_DELETE_ARRAY(mean_struct_sq_ptr);
		POSAFE_DELETE_ARRAY(edge_img_ptr);
		POSAFE_DELETE_ARRAY(ssim_map_ptr);
	}
}

//////////////////////////////////////////////////////////////////////////
CSSIMProcessor::CSSIMProcessor()
{
	initInstance();
}

CSSIMProcessor::~CSSIMProcessor()
{
	exitInstance();
}

void CSSIMProcessor::initInstance()
{
}

void CSSIMProcessor::exitInstance()
{
}

void CSSIMProcessor::makeSSIMMap(SSIMMap& ssim_map, u8* img_ptr, i32 w, i32 h)
{
	if (w <= 0 || h <= 0 || !img_ptr)
	{
		return;
	}

	i32 wh = w*h, ns = 5;
	ssim_map.initBuffer(w, h);
	
	u16* edge_img_ptr = ssim_map.edge_img_ptr;
	u16* mean_struct_sq_ptr = ssim_map.mean_struct_sq_ptr;
	u8* mean_lum_ptr = ssim_map.mean_lum_ptr;

	//use temp buffer of mem pool
	i32* int_img_ptr = new i32[wh];
	i32* int_edge_ptr = new i32[wh];

	//make integral image for mean image
	CImageProc::makeIntegralImage(int_img_ptr, img_ptr, w, h);

	//make edge2 integral image for sq2 image
	CImageProc::makeRobertGradImage(edge_img_ptr, img_ptr, w, h);
	CImageProc::makeIntegralImageSq(int_edge_ptr, edge_img_ptr, w, h);

	i32 x1, y1, x2, y2;
	i32 p1, p2, p3, p4;
	i32 x, y, index, patch_size;
	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
		{
			index = y*w + x;
			x1 = po::_max(x - ns, 0);
			y1 = po::_max(y - ns, 0);
			x2 = po::_min(x + ns, w - 1);
			y2 = po::_min(y + ns, h - 1);
			patch_size = (y2 - y1)*(x2 - x1);

			p1 = y2*w + x2;
			p2 = y2*w + x1;
			p3 = y1*w + x2;
			p4 = y1*w + x1;

			mean_lum_ptr[index] = (int_img_ptr[p1] + int_img_ptr[p4] - int_img_ptr[p2] - int_img_ptr[p3]) / patch_size;
			mean_struct_sq_ptr[index] = (int_edge_ptr[p1] + int_edge_ptr[p4] - int_edge_ptr[p2] - int_edge_ptr[p3]) / patch_size;
		}
	}

	POSAFE_DELETE_ARRAY(int_img_ptr);
	POSAFE_DELETE_ARRAY(int_edge_ptr);
}

f32 CSSIMProcessor::checkStructSimilar(SSIMMap& ssim_map1, SSIMMap& ssim_map2, i32 w, i32 h)
{
	if (w <= 0 || h <= 0)
	{
		return 0;
	}

	i32 ns = 5;
	f32 mssim = 0;
	i32 mssim_num = 0;
	
	u8* mean_lum_ptr1 = ssim_map1.mean_lum_ptr;
	u8* mean_lum_ptr2 = ssim_map2.mean_lum_ptr;
	u16* struct_sq_ptr1 = ssim_map1.mean_struct_sq_ptr;
	u16* struct_sq_ptr2 = ssim_map2.mean_struct_sq_ptr;
	u16* edge_img_ptr1 = ssim_map1.edge_img_ptr;
	u16* edge_img_ptr2 = ssim_map2.edge_img_ptr;

#ifdef POR_TESTMODE
	cv::Mat cv_mean_img1(h, w, CV_8UC1, mean_lum_ptr1);
	cv::Mat cv_mean_img2(h, w, CV_8UC1, mean_lum_ptr2);
	cv::Mat cv_sq_img1(h, w, CV_16UC1, struct_sq_ptr1);
	cv::Mat cv_sq_img2(h, w, CV_16UC1, struct_sq_ptr2);
	cv::imwrite("d:\\mu_a.bmp", cv_mean_img1);
	cv::imwrite("d:\\mu_b.bmp", cv_mean_img2);
	cv::imwrite("d:\\sq_a.bmp", cv_sq_img1);
	cv::imwrite("d:\\sq_b.bmp", cv_sq_img2);
#endif

	f32* ssim_map_ptr = ssim_map2.ssim_map_ptr;
	f32* tmp_ssim_map_ptr = ssim_map_ptr;

	//make cross gradient structual term
	i32 i, wh = w*h;
	i32 *edge_img_ptr12 = new i32[wh]; 
	i32 *edge_int_ptr12 = new i32[wh];
	i32* tmp_edge_img_ptr12 = edge_img_ptr12;

	for (i = 0; i < wh; i++)
	{
		*tmp_edge_img_ptr12 = (*edge_img_ptr1)*(*edge_img_ptr2);
		tmp_edge_img_ptr12++;
		edge_img_ptr1++;
		edge_img_ptr2++;
	}
	CImageProc::makeIntegralImage(edge_int_ptr12, edge_img_ptr12, w, h);

	//calc MSSIM value
	i32 x, y, index, pnum;
	i32 x1, y1, x2, y2;
	i32 p1, p2, p3, p4;
	f32 lum, st, mgxy, similar;

#ifdef POR_TESTMODE
	u8* mean_img_ptr = new u8[w*h];
	u8* st_img_ptr = new u8[w*h];
	memset(mean_img_ptr, 0, w*h);
	memset(st_img_ptr, 0, w*h);
#endif

	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
		{
			index = y*w + x;
			x1 = po::_max(x - ns, 0);
			y1 = po::_max(y - ns, 0);
			x2 = po::_min(x + ns, w - 1);
			y2 = po::_min(y + ns, h - 1);
			pnum = (y2 - y1)*(x2 - x1);

			p1 = y2*w + x2;
			p2 = y2*w + x1;
			p3 = y1*w + x2;
			p4 = y1*w + x1;

			lum = 1.0f - std::abs((f32)mean_lum_ptr1[index] - mean_lum_ptr2[index]) / 320;
			mgxy = (f32)(edge_int_ptr12[p1] + edge_int_ptr12[p4] - edge_int_ptr12[p2] - edge_int_ptr12[p3]) / pnum;
			st = (2 * mgxy + 58.25f) / ((f32)struct_sq_ptr1[index] + struct_sq_ptr2[index] + 58.25f);

#ifdef POR_TESTMODE
			mean_img_ptr[y*w + x] = (u8)(255 * po_min(1.0f, lum));
			st_img_ptr[y*w + x] = (u8)(255 * po_min(1.0f, st));
#endif
			similar = lum*st;
			mssim += similar;
			mssim_num++;
			tmp_ssim_map_ptr[index] = similar;
		}
	}

#ifdef POR_TESTMODE
	cv::Mat edge1(h, w, CV_16UC1, ssim_map1.edge_img_ptr);
	cv::Mat edge2(h, w, CV_16UC1, ssim_map2.edge_img_ptr);
	cv::imwrite("d:\\edge1.bmp", edge1);
	cv::imwrite("d:\\edge2.bmp", edge2);
	CImageProc::saveImgOpenCV("d:\\mu.bmp", mean_img_ptr, w, h);
	CImageProc::saveImgOpenCV("d:\\st.bmp", st_img_ptr, w, h);
	POSAFE_DELETE_ARRAY(mean_img_ptr);
	POSAFE_DELETE_ARRAY(st_img_ptr);
#endif

	POSAFE_DELETE_ARRAY(edge_img_ptr12);
	POSAFE_DELETE_ARRAY(edge_int_ptr12);

	if (mssim_num <= 0)
	{
		return 1.0f;
	}
	return mssim / mssim_num;
}
