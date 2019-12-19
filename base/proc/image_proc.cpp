#include "image_proc.h"
#include "base.h"

#if defined(POR_WITH_OVX)
#include "performance/openvx_pool/ovx_graph_pool.h"
#elif defined(POR_WITH_OCLCV)
#include <opencv2/core/ocl.hpp>
#endif

static i32 g_bit_mask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
i32 g_thumb_width = PO_THUMB_WIDTH;
i32 g_thumb_height = PO_THUMB_HEIGHT;

CImageProc::CImageProc()
{
}

CImageProc::~CImageProc()
{
}

f32 CImageProc::getImageFocus(u8* img_ptr, i32 w, i32 h, Recti rt, i32 lapth)
{
	i32 nw = rt.getWidth();
	i32 nh = rt.getHeight();
	if (!img_ptr || nw <= 0 || nh <= 0)
	{
		return -1;
	}

	//analysis of focus measure operators in shape-from-focus https://www.researchgate.net/publication/234073157
	i32 i, j, ns = 2;
	i32 size = nw*nh;
	f32 mean_th3 = 0;
	f32 mean_th5 = 0;
	f32 lap3 = 0, lap5 = 0;
	f32* new_img_ptr3;
	f32* new_img_ptr5;

	//check memorypool size and buffer
	CPOMemPool::initBuffer(8 * size);
	CPOMemPool::getZeroBuffer(new_img_ptr3, size);
	CPOMemPool::getZeroBuffer(new_img_ptr5, size);
	if (!new_img_ptr3 || !new_img_ptr5)
	{
		return -1;
	}
	
	//apply laplacian filter 
	cv::Rect cv_range(rt.x1, rt.y1, nw, nh);
	cv::Mat cv_image(h, w, CV_8UC1, img_ptr);
	cv::Mat cv_lap3_image(nh, nw, CV_32F, new_img_ptr3);
	cv::Mat cv_lap5_image(nh, nw, CV_32F, new_img_ptr5);
	cv::Mat cv_croped_image;
	cv::GaussianBlur(cv_image(cv_range), cv_croped_image, cv::Size(3, 3), 0);
	cv::Laplacian(cv_croped_image, cv_lap3_image, CV_32F, 3);
	cv::Laplacian(cv_croped_image, cv_lap5_image, CV_32F, 5);

	//calc mean gradient of image
	i32 index;
	i32 ww = nw - ns;
	i32 hh = nh - ns;
	if (nw < ns * 2 || nh < ns * 2)
	{
		CPOMemPool::releaseBuffer();
		return 0;
	}

	f32 tmp;
	f32* tmp_img_ptr;
	for (i = ns; i < hh; i++)
	{
		for (j = ns; j < ww; j++)
		{
			index = i*nw + j;
			tmp_img_ptr = new_img_ptr3 + index;
			tmp = std::abs(*tmp_img_ptr); *tmp_img_ptr = tmp; mean_th3 += tmp;

			tmp_img_ptr = new_img_ptr5 + index;
			tmp = std::abs(*tmp_img_ptr); *tmp_img_ptr = tmp; mean_th5 += tmp;
		}
	}
	size = (nw - ns * 2)*(nh - ns * 2);
	mean_th3 = po::_max(mean_th3 / size, (f32)lapth);
	mean_th5 = po::_max(mean_th5 / size, (f32)lapth*3.75f);

	//calc focus with gradient-map
	for (i = ns; i < hh; i++)
	{
		for (j = ns; j < ww; j++)
		{
			index = i*nw + j;
			if (new_img_ptr3[index] > mean_th3)
			{
				lap3++;
			}
			if (new_img_ptr5[index] > mean_th5)
			{
				lap5++;
			}
		}
	}
	CPOMemPool::releaseBuffer();

	if (lap5 <= 0)
	{
		return 0;
	}
	return po::_min(1.0f, lap3/lap5) * 100;
}

f32 CImageProc::getImageFocus(u8* img_ptr, i32 w, i32 h, i32 lapth)
{
	return getImageFocus(img_ptr, w, h, Recti(0, 0, w, h), lapth);
}

void CImageProc::invertImage(u8* img_ptr, i32 w, i32 h, i32 channel)
{
	if (!img_ptr || w*h*channel <= 0)
	{
		return;
	}
	cv::Mat cv_img(h, w, CV_8UC(channel), img_ptr);
	cv::bitwise_not(cv_img, cv_img);
}

void CImageProc::convertImgYFlip8(u8* img_ptr, i32 w, i32 h, u8* new_img_ptr)
{
	if (!img_ptr || !new_img_ptr || w*h <= 0)
	{
		return;
	}

	u8* tmp_img_ptr1 = img_ptr;
	u8* tmp_img_ptr2 = new_img_ptr + (h - 1)*w;

	for (i32 i = 0; i < h; i++)
	{
		memcpy(tmp_img_ptr2, tmp_img_ptr1, w);
		tmp_img_ptr1 += w;
		tmp_img_ptr2 -= w;
	}
}
#ifdef POR_DEVICE
u8* CImageProc::convertcvMatToImg(cv::Mat& mat, i32& nw, i32& nh)
{
	nw = mat.cols;
	nh = mat.rows;
	if (nw <= 0 || nh <= 0)
	{
		return NULL;
	}

	i32 i, stp = (i32)mat.step;
	u8* data_ptr = mat.data;
	u8* img_ptr = po_new u8[nw*nh];
	for (i = 0; i < nh; i++)
	{
		CPOBase::memCopy(img_ptr + i*nw, data_ptr + i*stp, nw);
	}
	return img_ptr;
}
#endif

void CImageProc::applyTransform(u8* dst_img_ptr, u8* src_img_ptr, i32& w, i32& h,
							i32 rotation, bool flip_x, bool flip_y, bool is_invert, i32 channel)
{
	if (!src_img_ptr || !dst_img_ptr || w <= 0 || h <= 0)
	{
		w = 0; h = 0;
		return;
	}
	if (src_img_ptr == dst_img_ptr && //원천화상과 목적화상이 같은경우 화상변환조작이 없으면 탈퇴
		rotation == kPORotation0 && !flip_x && !flip_y && !is_invert)
	{
		return;
	}

	//apply invert
	cv::Mat cv_src(h, w, CV_8UC(channel), src_img_ptr);
	if (is_invert)
	{
		cv_src = 255 - cv_src;
	}
	
	//apply flip_x and flip_y
	bool is_flip_y = false;
	if (flip_x)
	{
		is_flip_y = !is_flip_y;
		rotation = (rotation + 2) % kPORotationCount;
	}
	if (flip_y)
	{
		is_flip_y = !is_flip_y;
	}

	if (is_flip_y)
	{
		cv::flip(cv_src, cv_src, 0); //flipcode == 0 : vertical flip
	}

	i32 nw, nh;
	u8* tmp_img_ptr = NULL;
	if (src_img_ptr == dst_img_ptr)
	{
		tmp_img_ptr = po_new u8[w*h*channel];
		memcpy(tmp_img_ptr, src_img_ptr, w*h*channel);
		cv_src = cv::Mat(h, w, CV_8UC(channel), tmp_img_ptr);
	}
	switch (rotation)
	{
		case kPORotation90:
		{
			nw = h; nh = w;
			cv::Mat cv_dst(nh, nw, CV_8UC(channel), dst_img_ptr);

#if defined(POR_SUPPORT_TIOPENCV)
			cv::transpose(cv_src, cv_src);
			cv::flip(cv_src, cv_dst, 0); //flipcode == 0 : vertical flip
#else
			cv::rotate(cv_src, cv_dst, cv::ROTATE_90_CLOCKWISE);
#endif
			break;
		}
		case kPORotation180:
		{
			nw = w; nh = h;
			cv::Mat cv_dst(nh, nw, CV_8UC(channel), dst_img_ptr);

#if defined(POR_SUPPORT_TIOPENCV)
			cv::flip(cv_src, cv_dst, -1); //flipcode < 0 : vertical, horizontal flip
#else
			cv::rotate(cv_src, cv_dst, cv::ROTATE_180);
#endif
			break;
		}
		case kPORotation270:
		{
			nw = h; nh = w;
			cv::Mat cv_dst(nh, nw, CV_8UC(channel), dst_img_ptr);

#if defined(POR_SUPPORT_TIOPENCV)
			cv::transpose(cv_src, cv_src);
			cv::flip(cv_src, cv_dst, 1); //flipcode > 0 : horizontal flip
#else
			cv::rotate(cv_src, cv_dst, cv::ROTATE_90_COUNTERCLOCKWISE);
#endif
			break;
		}
		default:
		{
			nw = w; nh = h;
			if (dst_img_ptr != src_img_ptr)
			{
				memcpy(dst_img_ptr, src_img_ptr, w*h*channel); //dst = src
			}
			break;
		}
	}
	w = nw; h = nh;
	POSAFE_DELETE_ARRAY(tmp_img_ptr);
}

void CImageProc::applyTransform(ImageData& dst_img, ImageData& src_img,
							i32 rotation, bool flip_x, bool flip_y, bool is_invert)
{
	i32 w = src_img.w;
	i32 h = src_img.h;
	i32 channel = src_img.channel;
	dst_img.initBuffer(w, h, channel);

	applyTransform(dst_img.img_ptr, src_img.img_ptr, w, h, rotation, flip_x, flip_y, is_invert, channel);
	dst_img.update(w, h, channel);
}

void CImageProc::applyTransform(ImageData& img_data, i32 rotation, bool flip_x, bool flip_y, bool is_invert)
{
	i32 w = img_data.w;
	i32 h = img_data.h;
	i32 channel = img_data.channel;
	u8* img_ptr = img_data.img_ptr;
	
	applyTransform(img_ptr, img_ptr, w, h, rotation, flip_x, flip_y, is_invert, channel);
	img_data.update(w, h, channel);
}

u8* CImageProc::makeZeroImage(Recti rt)
{
	if (rt.isEmpty())
	{
		return NULL;
	}

	i32 w = rt.getWidth();
	i32 h = rt.getHeight();
	u8* img_ptr = po_new u8[w*h];
	memset(img_ptr, 0, w*h);
	return img_ptr;
}

u8* CImageProc::copyImage(u8* img_ptr, i32 w, i32 h)
{
	if (!img_ptr || w * h <= 0)
	{
		return NULL;
	}

	u8* new_img_ptr = po_new u8[w*h];
	memcpy(new_img_ptr, img_ptr, w*h);
	return new_img_ptr;
}

u8* CImageProc::copyImage(u8* img_ptr, Recti range)
{
	i32 w = range.getWidth();
	i32 h = range.getHeight();
	if (!img_ptr || w * h <= 0)
	{
		return NULL;
	}

	u8* new_img_ptr = po_new u8[w*h];
	memcpy(new_img_ptr, img_ptr, w*h);
	return new_img_ptr;
}

u8* CImageProc::cropImage(const Img img, vector2di crop_size)
{
	if (!img.isValid())
	{
		return NULL;
	}

	i32 nw = img.w;
	i32 w = po::_min(img.w, crop_size.x);
	i32 h = po::_min(img.h, crop_size.y);
	i32 channel = img.channel;

	if (w*h*channel <= 0)
	{
		return NULL;
	}

	u8* new_img_ptr = po_new u8[w*h*channel];
	u8* dst_img_ptr = new_img_ptr;
	u8* src_img_ptr = img.img_ptr;
	memset(new_img_ptr, 0, w*h*channel);

	i32 i, w_stride = w*channel;
	i32 nw_stride = nw*channel;
	for (i = 0; i < h; i++)
	{
		CPOBase::memCopy(dst_img_ptr, src_img_ptr, w_stride);
		dst_img_ptr += w_stride;
		src_img_ptr += nw_stride;
	}
	return new_img_ptr;
}

u8* CImageProc::cropImage(const Img img , Recti crop_rt)
{
	if (!img.isValid())
	{
		return NULL;
	}

	i32 x1 = crop_rt.x1;
	i32 y1 = crop_rt.y1;
	i32 w = crop_rt.getWidth();
	i32 h = crop_rt.getHeight();
	i32 nw = img.w;
	i32 channel = img.channel;

	Recti inter_rt = Recti(0, 0, img.w, img.h).intersectRect(crop_rt);
	i32 w2 = inter_rt.getWidth();
	i32 h2 = inter_rt.getHeight();
	if (w2*h2 <= 0)
	{
		return NULL;
	}

	u8* new_img_ptr = po_new u8[w*h*channel];
	u8* dst_img_ptr = new_img_ptr;
	u8* src_img_ptr = img.img_ptr + (y1*nw + x1)*channel;
	memset(new_img_ptr, 0, w*h*channel);

	i32 w_stride = w*channel;
	i32 w2_stride = w2*channel;
	i32 nw_stride = nw*channel;
	for (i32 i = 0; i < h2; i++)
	{
		CPOBase::memCopy(dst_img_ptr, src_img_ptr, w2_stride);
		dst_img_ptr += w_stride;
		src_img_ptr += nw_stride;
	}
	return new_img_ptr;
}

bool CImageProc::cropImage(u8* dst_img_ptr, Img img, Recti crop_rt)
{
	if (!img.isValid() || !dst_img_ptr)
	{
		return false;
	}

	i32 x1 = crop_rt.x1;
	i32 y1 = crop_rt.y1;
	i32 dst_w = crop_rt.getWidth();
	i32 dst_h = crop_rt.getHeight();
	i32 src_w = img.w;
	i32 channel = img.channel;

	Recti inter_rt = Recti(0, 0, img.w, img.h).intersectRect(crop_rt);
	i32 w0 = inter_rt.getWidth();
	i32 h0 = inter_rt.getHeight();
	if (w0*h0 <= 0)
	{
		return false;
	}

	u8* dst_scan_ptr = dst_img_ptr;
	u8* src_scan_ptr = img.img_ptr + (y1*src_w + x1)*channel;

	i32 dst_stride = dst_w * channel;
	i32 src_stride = src_w * channel;
	i32 stride = w0 * channel;
	memset(dst_img_ptr, 0, dst_stride*dst_h);

	for (i32 i = 0; i < h0; i++)
	{
		CPOBase::memCopy(dst_scan_ptr, src_scan_ptr, stride);
		dst_scan_ptr += dst_stride;
		src_scan_ptr += src_stride;
	}
	return true;
}

u8* CImageProc::cropImage(const ImgPart img_part, Recti crop_rt)
{
	if (!img_part.isValid())
	{
		return NULL;
	}

	i32 w1 = crop_rt.getWidth();
	i32 h1 = crop_rt.getHeight();

	Recti rt0 = img_part.range;
	i32 w0 = rt0.getWidth();
	i32 h0 = rt0.getHeight();
	i32 channel = img_part.channel;

	Recti inter_rt = rt0.intersectRect(crop_rt);
	i32 x = inter_rt.x1 - rt0.x1;
	i32 y = inter_rt.y1 - rt0.y1;
	i32 w2 = inter_rt.getWidth();
	i32 h2 = inter_rt.getHeight();
	if (w2*h2 <= 0)
	{
		return NULL;
	}

	u8* new_img_ptr = po_new u8[w1*h1*channel];
	u8* src_img_ptr = img_part.img_ptr + (y*w0 + x)*channel;
	u8* dst_img_ptr = new_img_ptr;
	memset(new_img_ptr, 0, w1*h1*channel);

	i32 w0_stride = w0*channel;
	i32 w1_stride = w1*channel;
	i32 w2_stride = w2*channel;
	for (i32 i = 0; i < h2; i++)
	{
		CPOBase::memCopy(dst_img_ptr, src_img_ptr, w2_stride);
		src_img_ptr += w0_stride;
		dst_img_ptr += w1_stride;
	}
	return new_img_ptr;
}

bool CImageProc::cropImage(const ImgPart img_part, Recti crop_rt, Img& dst_img)
{
	if (!img_part.isValid())
	{
		return false;
	}

	i32 w1 = crop_rt.getWidth();
	i32 h1 = crop_rt.getHeight();

	Recti rt0 = img_part.range;
	i32 w0 = rt0.getWidth();
	i32 h0 = rt0.getHeight();
	i32 channel = img_part.channel;

	Recti inter_rt = rt0.intersectRect(crop_rt);
	i32 x = inter_rt.x1 - rt0.x1;
	i32 y = inter_rt.y1 - rt0.y1;
	i32 w2 = inter_rt.getWidth();
	i32 h2 = inter_rt.getHeight();
	if (w2*h2 <= 0)
	{
		return false;
	}

	dst_img.initBuffer(w1, h1, channel);
	u8* src_img_ptr = img_part.img_ptr + (y*w0 + x)*channel;
	u8* dst_img_ptr = dst_img.img_ptr;
	memset(dst_img_ptr, 0, dst_img.getImgSize());

	i32 i, w2_stride = w2*channel;
	i32 w0_stride = w0*channel;
	i32 w1_stride = w1*channel;
	for (i = 0; i < h2; i++)
	{
		CPOBase::memCopy(dst_img_ptr, src_img_ptr, w2_stride);
		src_img_ptr += w0_stride;
		dst_img_ptr += w1_stride;
	}
	return true;
}

bool CImageProc::cropImage(ImageData& img_data, vector2di crop_size)
{
	if (img_data.w <= crop_size.x && img_data.h <= crop_size.y)
	{
		return true;
	}

	cv::Rect roi(0, 0, crop_size.x, crop_size.y);
	cv::Mat cv_img(img_data.h, img_data.w, CV_8UC(img_data.channel), img_data.img_ptr);
	cv::Mat cv_crop_img = cv_img(roi).clone();

	img_data.w = cv_crop_img.cols;
	img_data.h = cv_crop_img.rows;
	CPOBase::memCopy(img_data.img_ptr, cv_crop_img.data, img_data.getImageSize());
	return true;
}

u8* CImageProc::loadImgOpenCV(const char* filename, i32& w, i32& h)
{
	w = 0; h = 0;
	if (!filename)
	{
		return NULL;
	}

	cv::Mat cv_img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	w = cv_img.cols;
	h = cv_img.rows;
	if (w*h <= 0)
	{
		return NULL;
	}

	u8* img_ptr = po_new u8[w*h];
	CPOBase::memCopy(img_ptr, (u8*)cv_img.data, w*h);
	return img_ptr;
}

#ifdef POR_DEVICE
cv::Mat CImageProc::loadImgOpenCV(const potstring& filename, i32 mode)
{
	FILE* fp = tfopen(filename.c_str(), _T("rb"));
	if (!fp)
	{
		return cv::Mat::zeros(1, 1, CV_8U);
	}

	long size = CPOBase::fileSize(fp);
	char* buffer_ptr = po_new char[size];
	CPOBase::fileRead(buffer_ptr, size, fp);

	cv::_InputArray arr(buffer_ptr, size);
	cv::Mat cv_img = cv::imdecode(arr,mode);
	POSAFE_DELETE_ARRAY(buffer_ptr);

	fclose(fp);
	return cv_img;
}
#endif

bool CImageProc::saveImgOpenCV(const char* filename, u8* img_ptr, i32 w, i32 h, i32 channel)
{
	if (!img_ptr || w*h*channel <= 0 || !filename)
	{
		return false;
	}

	cv::Mat cv_img(h, w, CV_8UC(channel), img_ptr);
	return cv::imwrite(filename, cv_img);
}

bool CImageProc::saveImgOpenCV(const char* filename, ImageData* img_data_ptr)
{
	if (!img_data_ptr || !img_data_ptr->isValid())
	{
		return false;
	}
	return saveImgOpenCV(filename, img_data_ptr->img_ptr, img_data_ptr->w, img_data_ptr->h, img_data_ptr->channel);
}

bool CImageProc::saveImgOpenCV(const char* filename, const ImageData& img_data)
{
	if (!img_data.isValid())
	{
		return false;
	}
	return saveImgOpenCV(filename, img_data.img_ptr, img_data.w, img_data.h, img_data.channel);
}	

bool CImageProc::saveImgOpenCV(const char* filename, const ImgPart img_part)
{
	Recti rt = img_part.getRange();
	i32 w = rt.getWidth();
	i32 h = rt.getHeight();
	return saveImgOpenCV(filename, img_part.img_ptr, w, h, img_part.channel);
}

bool CImageProc::saveImgOpenCV(const wchar_t* filename, u8* img_ptr, i32 w, i32 h, i32 channel)
{
#if defined(POR_SUPPORT_UNICODE)
	if (!img_ptr || w*h*channel <= 0 || !filename)
	{
		return false;
	}

	u8vector encode_img;
	encode_img.reserve(w*h);
	cv::Mat cv_img(h, w, CV_8UC(channel), img_ptr);
	if (!cv::imencode(".bmp", cv_img, encode_img))
	{
		return false;
	}

	FILE* fp = _wfopen(filename, L"wb");
	if (!fp)
	{
		return false;
	}
	CPOBase::fileWrite(encode_img.data(), (i32)encode_img.size(), fp);
	fclose(fp);
#endif
	return true;
}

bool CImageProc::saveBinImgOpenCV(const char* filename, u8* img_ptr, i32 w, i32 h)
{
	if (!img_ptr || w*h <= 0 || !filename)
	{
		return false;
	}

	i32 i, wh = w*h;
	u8* new_img_ptr = po_new u8[wh];
	memcpy(new_img_ptr, img_ptr, wh);
	for (i = 0; i < wh; i++)
	{
		if (new_img_ptr[i])
		{
			new_img_ptr[i] = 0xFF;
		}
	}

	cv::Mat cv_img(h, w, CV_8UC1, new_img_ptr);
	bool is_success = cv::imwrite(filename, cv_img);
	POSAFE_DELETE_ARRAY(new_img_ptr);
	return is_success;
}

bool CImageProc::saveBinXImgOpenCV(const char* filename, u8* img_ptr, i32 w, i32 h, i32 val)
{
	if (!img_ptr || w * h <= 0 || !filename)
	{
		return false;
	}

	i32 i, wh = w * h;
	u8* new_img_ptr = po_new u8[wh];
	for (i = 0; i < wh; i++)
	{
		new_img_ptr[i] = po::_min(0xFF, img_ptr[i] * val);
	}

	cv::Mat cv_img(h, w, CV_8UC1, new_img_ptr);
	bool is_success = cv::imwrite(filename, cv_img);
	POSAFE_DELETE_ARRAY(new_img_ptr);
	return is_success;
}

bool CImageProc::saveImgOpenCV(const potstring& filename, const Img img)
{
	return saveImgOpenCV(filename.data(), img.img_ptr, img.w, img.h, img.channel);
}

bool CImageProc::saveImgOpenCV(const potstring& filename, const ImgPart img_part)
{
	Recti rt = img_part.getRange();
	i32 w = rt.getWidth();
	i32 h = rt.getHeight();
	return saveImgOpenCV(filename.data(), img_part.img_ptr, w, h, img_part.channel);
}

bool CImageProc::saveImgOpenCV(const potstring& filename, const ImgExpr img_expr)
{
	return saveImgOpenCV(filename.c_str(), img_expr.img_ptr, img_expr.w, img_expr.h, img_expr.channel);
}

bool CImageProc::saveImgOpenCV(const potstring& filename, const ImageData& img_data)
{
	return saveImgOpenCV(filename.c_str(), img_data.img_ptr, img_data.w, img_data.h, img_data.channel);
}

bool CImageProc::saveImgOpenCV(const potstring& filename, ImageData* img_data_ptr)
{
	if (!img_data_ptr)
	{
		return false;
	}
	return saveImgOpenCV(filename.c_str(), img_data_ptr->img_ptr, img_data_ptr->w, img_data_ptr->h, img_data_ptr->channel);
}

cv::Mat CImageProc::decodeImgOpenCV(const u8vector& encoded_vec, i32 mode)
{
	cv::Mat cv_img = cv::imdecode(encoded_vec, mode);
	return cv_img;
}

void CImageProc::fillContourForBlob(u8* img_ptr, i32 w, i32 h, u8 val)
{
	if (!img_ptr)
	{
		return;
	}

	i32 w1 = w + 2;
	i32 h1 = h + 2;
	u8* padding_img_ptr = po_new u8[w1*h1];
	memset(padding_img_ptr, 0, w1*h1);

	i32 x, y, ind;
	i32 hpos, hpos1;
	for (y = 0; y < h; y++)
	{
		hpos = y*w;
		hpos1 = (y + 1)*w1 + 1;
		for (x = 0; x < w; x++)
		{
			if (img_ptr[hpos + x] == 0)
			{
				continue;
			}

			ind = hpos1 + x;
			padding_img_ptr[ind + 1]++; 
			padding_img_ptr[ind - 1]++;
			padding_img_ptr[ind - w1]++; 
			padding_img_ptr[ind + w1]++;
		}
	}

	for (y = 0; y < h; y++)
	{
		hpos = y*w;
		hpos1 = (y + 1)*w1 + 1;
		for (x = 0; x < w; x++)
		{
			ind = hpos + x;
			if (img_ptr[ind] == 0 || padding_img_ptr[hpos1 + x] == 4)
			{
				continue;
			}
			img_ptr[ind] = val;
		}
	}

	POSAFE_DELETE_ARRAY(padding_img_ptr);
}

void CImageProc::fillContourForBlob2(u8* img_ptr, i32 w, i32 h, u8 fg, u8 bg)
{
	if (!img_ptr)
	{
		return;
	}

	i32 w2 = w + 2;
	i32 h2 = h + 2;
	i32 x, y, hpos, prev, pixel;
	u8* padding_img_ptr = po_new u8[w2*h2];
	CImageProc::makePaddingBinary(padding_img_ptr, img_ptr, w, h);
	
	for (y = 0; y < h; y++)
	{
		prev = 0;
		hpos = y*w;
		for (x = 0; x < w; x++)
		{
			pixel = img_ptr[hpos + x];
			if (prev == 0 && pixel == fg)
			{
				fillContourTrace(img_ptr, w, h, padding_img_ptr, x, y, 0, 4, bg, (Contouru*)NULL);
				pixel = bg;
			}
			else if (prev == fg && pixel == 0)
			{
				fillContourTrace(img_ptr, w, h, padding_img_ptr, x - 1, y, 0, 0, bg, (Contouru*)NULL);
			}
			prev = pixel;
		}
	}
	POSAFE_DELETE_ARRAY(padding_img_ptr);
}

void CImageProc::makeHistogram(u8* img_ptr, u8* mask_img_ptr, i32 w, i32 h, i32* hist_ptr, i32* border_hist_ptr)
{
	if (!img_ptr || !hist_ptr)
	{
		return;
	}

	i32 i, wh = w*h;
	u8 mask_pixel, img_pixel;
	u8* tmp_img_ptr = img_ptr;
	u8* tmp_mask_ptr = mask_img_ptr;

	if (mask_img_ptr)
	{
		/* 마스크화상이 있는 경우*/
		if (border_hist_ptr)
		{
			/* 마스크화상에 에지픽셀히스토그람이 있는 경우 */
			for (i = 0; i < wh; i++)
			{
				img_pixel = *tmp_img_ptr;
				mask_pixel = *tmp_mask_ptr;

				if (mask_pixel)
				{
					hist_ptr[img_pixel]++;
					if (mask_pixel == kPOEdgePixel)
					{
						border_hist_ptr[img_pixel]++;
					}
				}
				tmp_img_ptr++; tmp_mask_ptr++;
			}
		}
		else
		{
			/* 마스크화상에 에지픽셀히스토그람이 없는 경우 */
			for (i = 0; i < wh; i++)
			{
				img_pixel = *tmp_img_ptr;
				mask_pixel = *tmp_mask_ptr;

				if (mask_pixel)
				{
					hist_ptr[img_pixel]++;
				}
				tmp_img_ptr++; tmp_mask_ptr++;
			}
		}
	}
	else
	{
		/* 마스크화상이 없는 경우 */
		if (border_hist_ptr)
		{
			/* 마스크화상이 없고 에지픽셀히스토그람이 있는 경우*/
			for (i = 0; i < wh; i++)
			{
				img_pixel = *tmp_img_ptr;
				if (img_pixel == kPOEdgePixel)
				{
					border_hist_ptr[img_pixel]++;
				}
				hist_ptr[img_pixel]++;
				tmp_img_ptr++;
			}
		}
		else
		{
			/* 마스크화상과 에지픽셀히스토그람이 다 없는 경우 */
			for (i = 0; i < wh; i++)
			{
				img_pixel = *tmp_img_ptr;
				hist_ptr[img_pixel]++;
				tmp_img_ptr++;
			}
		}
	}
}

bool CImageProc::isInvertedBackground(i32 th, i32* hist_ptr, i32* border_hist_ptr)
{
	if (!border_hist_ptr || !hist_ptr)
	{
		return false;
	}

	//update accmulated weight
	i32 acc_border_black = 0, acc_border_white = 0;
	i32 acc_content_black = 0, acc_content_white = 0;
	for (i32 i = 0; i < 256; i++)
	{
		if (i <= th)
		{
			acc_border_black += border_hist_ptr[i];
			acc_content_black += hist_ptr[i];
			continue;
		}
		acc_border_white += border_hist_ptr[i];
		acc_content_white += hist_ptr[i];
	}

	//determine invert background with border histogram
	if (acc_border_black > 2 * acc_border_white)
	{
		return false;
	}
	else if (acc_border_white > 2 * acc_border_black)
	{
		return true;
	}

	//determine background level with content histogram
	return acc_content_white > acc_content_black;
}

void CImageProc::fillOutsideContourForBlob(u16* img_ptr, Recti& range, i32 count)
{
	if (!img_ptr)
	{
		return;
	}

	i32 w = range.getWidth();
	i32 h = range.getHeight();
	i32 w1 = w + 2;
	i32 h1 = h + 2;

	bool* is_used_ptr = po_new bool[count];
	u8* padding_img_ptr = po_new u8[w1*h1];
	memset(is_used_ptr, 0, count);
	CImageProc::makePaddingBinary(padding_img_ptr, img_ptr, w, h);

	u16* tmp_img_ptr = img_ptr;
	i32 x, y, prev, pixel, edge;

	for (y = 0; y < h; y++)
	{
		prev = 0;
		for (x = 0; x < w; x++)
		{
			pixel = *tmp_img_ptr;
			tmp_img_ptr++;

			if (prev == 0 && pixel > 0 && pixel < kPOEdgeInner)
			{
				if (is_used_ptr[pixel])
				{
					edge = kPOEdgeInner;
				}
				else
				{
					edge = kPOEdgeOutter;
					is_used_ptr[pixel] = true;
				}
				fillContourTrace(img_ptr, w, h, padding_img_ptr, x, y, 0, 4, edge, (Contouru*)NULL);
				pixel = edge;
			}
			else if (prev > 0 && prev < kPOEdgeInner && pixel == 0)
			{
				if (is_used_ptr[prev])
				{
					edge = kPOEdgeInner;
				}
				else
				{
					edge = kPOEdgeOutter;
					is_used_ptr[prev] = true;
				}
				fillContourTrace(img_ptr, w, h, padding_img_ptr, x - 1, y, 0, 0, edge, (Contouru*)NULL);
			}
			prev = pixel;
		}
	}

	tmp_img_ptr = img_ptr;
	i32 i, wh = w*h;
	for (i = 0; i < wh; i++, tmp_img_ptr++)
	{
		*tmp_img_ptr = (*tmp_img_ptr == kPOEdgeOutter) ? 1: 0;
	}

	POSAFE_DELETE_ARRAY(is_used_ptr);
	POSAFE_DELETE_ARRAY(padding_img_ptr);
}

void CImageProc::registerThumbSize(i32 w, i32 h)
{
	if (w <= 0 || h <= 0)
	{
		return;
	}
	g_thumb_width = w;
	g_thumb_height = h;
}

u8* CImageProc::makeThumbImage(u8* img_ptr, i32 w, i32 h, i32 channel, i32& tw, i32& th)
{
	if (!img_ptr || w*h*channel <= 0 || tw*th <= 0)
	{
		return NULL;
	}

	f32 s_normal = po::_max((f32)w / tw, (f32)h / th);
	f32 s_rotation = po::_max((f32)w / th, (f32)h / tw);
	f32 s0 = po::_min(s_normal, s_rotation);
	tw = CPOBase::int_cast(w / s0);
	th = CPOBase::int_cast(h / s0);

	cv::Mat thumb_img;
	cv::Mat cv_img(h, w, CV_8UC(channel), img_ptr);
	cv::resize(cv_img, thumb_img, cv::Size(tw, th));

	tw = thumb_img.cols;
	th = thumb_img.rows;
	u8* thumb_img_ptr = po_new u8[tw*th*channel];
	memcpy(thumb_img_ptr, thumb_img.data, tw*th*channel);
	return thumb_img_ptr;
}

ImgExpr CImageProc::makeThumbImage(ImageData* img_data_ptr)
{
	i32 nw = g_thumb_width;
	i32 nh = g_thumb_height;
	i32 w = img_data_ptr->w;
	i32 h = img_data_ptr->h;
	i32 channel = img_data_ptr->channel;
	u8* img_ptr = makeThumbImage(img_data_ptr->img_ptr, w, h, channel, nw, nh);
	return ImgExpr(img_ptr, nw, nh, channel);
}

void CImageProc::makeThumbImage(ImageData& src_img, i32 nw, i32 nh, ImageData& dst_img)
{
	if (!src_img.isValid() || nw <= 0 || nh <= 0)
	{
		return;
	}

	f32 s_normal = po::_max((f32)src_img.w / nw, (f32)src_img.h / nh);
	f32 s_rotation = po::_max((f32)src_img.w / nh, (f32)src_img.h / nw);
	f32 s0 = po::_min(s_normal, s_rotation);
	i32 w1 = CPOBase::int_cast(src_img.w / s0);
	i32 h1 = CPOBase::int_cast(src_img.h / s0);

	cv::Mat cv_thumb_img;
	cv::Mat cv_img = cv::Mat(src_img.h, src_img.w, CV_8UC(src_img.channel), src_img.img_ptr);
	cv::resize(cv_img, cv_thumb_img, cv::Size(w1, h1));
	dst_img.copyImage(cv_thumb_img.data, w1, h1, src_img.channel);
}

void CImageProc::makeThumbImage(ImageData& src_img, ImageData& dst_img)
{
	makeThumbImage(src_img, g_thumb_width, g_thumb_height, dst_img);
}

bool CImageProc::convertColor(ImageData& src_img, ImageData& dst_img, i32 cvt_mode)
{
	if (!src_img.isValid() || !dst_img.img_ptr)
	{
		return false;
	}

	i32 w = src_img.w;
	i32 h = src_img.h;
	i32 src_channel = src_img.channel;
	bool is_processed = false;

	switch (src_channel)
	{
		case kPOGrayChannels:
		{
			switch (cvt_mode)
			{
				case kPOColorCvt2Gray:
				case kPOColorCvt2Red:
				case kPOColorCvt2Green:
				case kPOColorCvt2Blue:
				case kPOColorCvt2Intensity:
				{
					if (dst_img.img_ptr != src_img.img_ptr)
					{
						CPOBase::memCopy(dst_img.img_ptr, src_img.img_ptr, w*h);
					}
					dst_img.update(w, h, 1);
					break;
				}
				case kPOColorCvt2RGB:
				{
					is_processed = vxConvertColor(src_img, dst_img, cvt_mode);
					if (!is_processed)
					{
						cv::Mat cv_src_img(h, w, CV_8UC1, src_img.img_ptr);
						cv::Mat cv_dst_img(h, w, CV_8UC3, dst_img.img_ptr);
						cv::cvtColor(cv_src_img, cv_dst_img, CV_GRAY2RGB);
					}
					dst_img.update(w, h, 3);
					break;
				}
				default:
				{
					return false;
				}
			}
			break;
		}
		case kPORGBChannels:
		{
			switch (cvt_mode)
			{
				case kPOColorCvt2Gray:
				{
					is_processed = vxConvertColor(src_img, dst_img, cvt_mode);
					if (!is_processed)
					{
						cv::Mat cv_src_img(h, w, CV_8UC3, src_img.img_ptr);
						cv::Mat cv_dst_img(h, w, CV_8UC1, dst_img.img_ptr);
						cv::cvtColor(cv_src_img, cv_dst_img, CV_RGB2GRAY);
					}
					dst_img.update(w, h, 1);
					break;
				}
				case kPOColorCvt2RGB:
				{
					if (dst_img.img_ptr != src_img.img_ptr)
					{
						CPOBase::memCopy(dst_img.img_ptr, src_img.img_ptr, w*h);
					}
					dst_img.update(w, h, 3);
					break;
				}
				case kPOColorCvt2YUV:
				{
					is_processed = vxConvertColor(src_img, dst_img, cvt_mode);
					if (!is_processed)
					{
						cv::Mat cv_src_img(h, w, CV_8UC3, src_img.img_ptr);
						cv::Mat cv_dst_img(h, w, CV_8UC3, dst_img.img_ptr);
						cv::cvtColor(cv_src_img, cv_dst_img, CV_RGB2YUV);
					}
					dst_img.update(w, h, 3);
					break;
				}
				case kPOColorCvt2HSV:
				{
					is_processed = vxConvertColor(src_img, dst_img, cvt_mode);
					if (!is_processed)
					{
						cv::Mat cv_src_img(h, w, CV_8UC3, src_img.img_ptr);
						cv::Mat cv_dst_img(h, w, CV_8UC3, dst_img.img_ptr);
						cv::cvtColor(cv_src_img, cv_dst_img, CV_RGB2HSV);
						dst_img.update(w, h, 3);
					}
					break;
				}
				case kPOColorCvt2Red:
				{
					is_processed = vxConvertColor(src_img, dst_img, cvt_mode);
					if (!is_processed)
					{
						cv::Mat cv_channel[3];
						cv::Mat cv_src_img(h, w, CV_8UC3, src_img.img_ptr);
						cv::split(cv_src_img, cv_channel);
						CPOBase::memCopy(dst_img.img_ptr, cv_channel[2].data, w*h);
					}
					dst_img.update(w, h, 1);
					break;
				}
				case kPOColorCvt2Green:
				{
					is_processed = vxConvertColor(src_img, dst_img, cvt_mode);
					if (!is_processed)
					{
						cv::Mat cv_channel[3];
						cv::Mat cv_src_img(h, w, CV_8UC3, src_img.img_ptr);
						cv::split(cv_src_img, cv_channel);
						CPOBase::memCopy(dst_img.img_ptr, cv_channel[1].data, w*h);
					}
					dst_img.update(w, h, 1);
					break;
				}
				case kPOColorCvt2Blue:
				{
					is_processed = vxConvertColor(src_img, dst_img, cvt_mode);
					if (!is_processed)
					{
						cv::Mat cv_channel[3];
						cv::Mat cv_src_img(h, w, CV_8UC3, src_img.img_ptr);
						cv::split(cv_src_img, cv_channel);
						CPOBase::memCopy(dst_img.img_ptr, cv_channel[0].data, w*h);
					}
					dst_img.update(w, h, 1);
					break;
				}
				case kPOColorCvt2Hue:
				{
					is_processed = vxConvertColor(src_img, dst_img, cvt_mode);
					if (!is_processed)
					{
						cv::Mat cv_src_img(h, w, CV_8UC3, src_img.img_ptr);
						cv::Mat cv_tmp_img(h, w, CV_8UC3);
						cv::Mat cv_channel[3];
						cv::cvtColor(cv_src_img, cv_tmp_img, CV_RGB2HSV);
						cv::split(cv_tmp_img, cv_channel);
						CPOBase::memCopy(dst_img.img_ptr, cv_channel[0].data, w*h);
					}
					dst_img.update(w, h, 1);
					break;
				}
				case kPOColorCvt2Saturation:
				{
					is_processed = vxConvertColor(src_img, dst_img, cvt_mode);
					if (!is_processed)
					{
						cv::Mat cv_src_img(h, w, CV_8UC3, src_img.img_ptr);
						cv::Mat cv_tmp_img(h, w, CV_8UC3);
						cv::Mat cv_channel[3];
						cv::cvtColor(cv_src_img, cv_tmp_img, CV_RGB2HSV);
						cv::split(cv_tmp_img, cv_channel);
						CPOBase::memCopy(dst_img.img_ptr, cv_channel[1].data, w*h);
					}
					dst_img.update(w, h, 1);
					break;
				}
				case kPOColorCvt2Intensity:
				{
					is_processed = vxConvertColor(src_img, dst_img, cvt_mode);
					if (!is_processed)
					{
						cv::Mat cv_src_img(h, w, CV_8UC3, src_img.img_ptr);
						cv::Mat cv_dst_img(h, w, CV_8UC1, dst_img.img_ptr);
						cv::Mat cv_channel[3];
						cv::split(cv_src_img, cv_channel);
						cv_dst_img = (cv_channel[0] + cv_channel[1] + cv_channel[2]) / 3;
					}
					dst_img.update(w, h, 1);
					break;
				}
				default:
				{
					return false;
				}
			}
			break;
		}
		default: 
		{
			return false;
		}
	}

	//copy image information
	dst_img.copyImageInfo(&src_img);
	return true;
}

bool CImageProc::vxConvertColor(ImageData& src_img, ImageData& dst_img, i32 cvt_mode)
{
	if (!src_img.isValid() || !dst_img.img_ptr)
	{
		return false;
	}

	bool is_processed = false;
	switch (cvt_mode)
	{
		case kPOColorCvt2Gray:
		case kPOColorCvt2RGB:
		case kPOColorCvt2YUV:
		{
#if defined(POR_WITH_OVX)
			if (g_vx_gpool_ptr)
			{
				CGImgProcCvtColor* graph_ptr = (CGImgProcCvtColor*)g_vx_gpool_ptr->fetchGraph(
							kGImgProcCvtColor, &src_img, &dst_img, cvt_mode);
				if (graph_ptr)
				{
					is_processed = graph_ptr->process();
					g_vx_gpool_ptr->releaseGraph(graph_ptr);
				}
			}
#endif
			break;
		}
		case kPOColorCvt2Red:
		case kPOColorCvt2Green:
		case kPOColorCvt2Blue:
		{
#if defined(POR_WITH_OVX)
			if (g_vx_gpool_ptr)
			{
				CGImgProcCvtSplit* graph_ptr = (CGImgProcCvtSplit*)g_vx_gpool_ptr->fetchGraph(
							kGImgProcCvtSplit, &src_img, &dst_img, cvt_mode);
				if (graph_ptr)
				{
					is_processed = graph_ptr->process();
					g_vx_gpool_ptr->releaseGraph(graph_ptr);
				}
			}
#endif
			break;
		}
		case kPOColorCvt2Hue:
		case kPOColorCvt2Saturation:
		{
#if defined(POR_WITH_OVX)
			if (g_vx_gpool_ptr)
			{
				CGImgProcCvtHSVSplit* graph_ptr = (CGImgProcCvtHSVSplit*)g_vx_gpool_ptr->fetchGraph(
							kGImgProcCvtHSVSplit, &src_img, &dst_img, cvt_mode);
				if (graph_ptr)
				{
					is_processed = graph_ptr->process();
					g_vx_gpool_ptr->releaseGraph(graph_ptr);
				}
			}
#endif
			break;
		}
		case kPOColorCvt2Intensity:
		{
#if defined(POR_WITH_OVX)
			if (g_vx_gpool_ptr)
			{
				CGImgProcCvtIntensity* graph_ptr = (CGImgProcCvtIntensity*)g_vx_gpool_ptr->fetchGraph(
							kGImgProcCvtIntensity, &src_img, &dst_img, cvt_mode);
				if (graph_ptr)
				{
					is_processed = graph_ptr->process();
					g_vx_gpool_ptr->releaseGraph(graph_ptr);
				}
			}
#endif
			break;
		}

	}
	return is_processed;
}

bool CImageProc::convertColor(ImageData* src_img_ptr, ImageData* dst_img_ptr, i32 cvt_mode)
{
	if (!src_img_ptr || !dst_img_ptr)
	{
		return false;
	}
	return convertColor(src_img_ptr[0], dst_img_ptr[0], cvt_mode);
}

bool CImageProc::convertColor(Img& src_img, Img& dst_img, i32 cvt_mode)
{
	ImageData src_img_data(src_img);
	ImageData dst_img_data(dst_img);
	if (!convertColor(src_img_data, dst_img_data, cvt_mode))
	{
		return false;
	}

	dst_img.update(dst_img_data.w, dst_img_data.h, dst_img_data.channel);
	return true;
}

void CImageProc::fillImageBorder(u8* img_ptr, i32 w, i32 h, u8 val)
{
	if (!img_ptr)
	{
		return;
	}

	memset(img_ptr, val, w);
	memset(img_ptr + (h-1)*w, val, w);

	for (i32 i = 1; i < h; i++)
	{ 
		memset(img_ptr + i*w - 1, val, 2);
	}
}

u8* CImageProc::extendImage(u8* img_ptr, i32 w, i32 h, i32 r)
{
	i32 nw = w + 2 * r;
	i32 nh = h + 2 * r;
	if (!img_ptr || w*h <= 0)
	{
		return NULL;
	}

	u8* new_img_ptr = po_new u8[nw*nh];
	memset(new_img_ptr, 0, nw*nh);
	extendImage(new_img_ptr, img_ptr, w, h, r);
	return new_img_ptr;
}

void CImageProc::extendImage(u8* new_img_ptr, u8* img_ptr, i32 w, i32 h, i32 r)
{
	if (!new_img_ptr || !img_ptr || w <= 0 || h <= 0)
	{
		return;
	}

	i32 nw = w + 2 * r;
	i32 nh = h + 2 * r;
	for (i32 i = 0; i < h; i++)
	{
		memcpy(new_img_ptr + (i + r)*nw + r, img_ptr + i*w, w);
	}
}

u8* CImageProc::affineWarpImageOpenCV(u8* img_ptr, i32 w, i32 h, f32* tr)
{
	if (!tr)
	{
		return NULL;
	}

	u8* new_img_ptr = po_new u8[w*h];
	affineWarpImageOpenCV(new_img_ptr, Recti(0, 0, w, h), Img(img_ptr, w, h), tr);
	return new_img_ptr;
}

void CImageProc::affineWarpImageOpenCV(u8* img_ptr, Recti rt, Img img, f32* tr)
{
	//img_ptr, rt: destnation
	//img: source
	//tr: destnation -> source
	if (img_ptr == NULL || img.img_ptr == NULL)
	{
		return;
	}

	i32 w = img.w;
	i32 h = img.h;
	u8* tmp_img_ptr = img.img_ptr;

	i32 x1 = rt.x1;
	i32 y1 = rt.y1;
	i32 nw = rt.getWidth();
	i32 nh = rt.getHeight();

	f32 mtr[9];
	f32 ttr[9];
	CPOBase::init3x3(ttr, 0, 1.0f, x1, y1);
	cv::Mat cv_mat_tr1(3, 3, CV_32FC1, tr);
	cv::Mat cv_mat_tr2(3, 3, CV_32FC1, ttr);
	cv::Mat cv_mat_tr3(3, 3, CV_32FC1, mtr);
    cv_mat_tr3 = cv_mat_tr1*cv_mat_tr2;

    cv::Mat cv_img_after(nh, nw, CV_8UC1, img_ptr);
    cv::Mat cv_img_before(h, w, CV_8UC1, tmp_img_ptr);
    cv::warpPerspective(cv_img_before, cv_img_after, cv_mat_tr3, cv::Size(nw, nh), CV_INTER_AREA | CV_WARP_INVERSE_MAP);
}

u8* CImageProc::affineWarpImage(u8* img_ptr, i32 w, i32 h, f32* tr)
{
	if (!tr)
	{
		return NULL;
	}

	u8* new_img_ptr = po_new u8[w*h];
	affineWarpImage(new_img_ptr, Recti(0, 0, w, h), Img(img_ptr, w, h), tr);
	return new_img_ptr;
}

void CImageProc::affineWarpImage(u8* img_ptr, Recti rt, Img img, f32* tr)
{
	if (img_ptr == NULL || img.img_ptr == NULL)
	{
		return;
	}

	i32 w = img.w;
	i32 h = img.h;
	u8* img_ptr1 = img.img_ptr;

	i32 x1 = rt.x1;
	i32 y1 = rt.y1;
	i32 nw = rt.getWidth();
	i32 nh = rt.getHeight();
	memset(img_ptr, 0, nw*nh);

	f32 px, py;
	i32 index, tmp, stmp;
	i32 x, y, nx, ny, kx, ky, kk, ikx, iky;
	for (y = 0; y < nh; y++)
	{
		for (x = 0; x < nw; x++)
		{
			px = (f32)x + x1;
			py = (f32)y + y1;
			CPOBase::trans3x3(tr, px, py);
			if (px < 0 || px >= w || py < 0 || py >= h)
			{
				img_ptr++; continue;
			}

			//linear-interpolation image warpping...
			nx = (i32)px; kx = (i32)((nx + 1 - px) * 100); ikx = 100 - kx;
			ny = (i32)py; ky = (i32)((ny + 1 - py) * 100); iky = 100 - ky;

			index = ny*w + nx;
			kk = kx*ky; tmp = kk* img_ptr1[index]; stmp = kk;
			if (nx < w - 1)
			{
				kk = ikx*ky; tmp += kk*img_ptr1[index + 1]; stmp += kk;
				if (ny < h - 1)
				{
					kk = kx*iky; tmp += kk*img_ptr1[index + w]; stmp += kk;
					kk = ikx*iky; tmp += kk*img_ptr1[index + w + 1]; stmp += kk;
				}
			}
			else if (ny < h - 1)
			{
				kk = kx*iky; tmp += kk*img_ptr1[index + w]; stmp += kk;
			}

			if (stmp > 0)
			{
				*img_ptr = (u8)(tmp / stmp);
			}
			img_ptr++;
		}
	}
}

void CImageProc::gaussianBlur(u8* img_ptr, i32 w, i32 h, i32 nw, f32 sigma)
{
	if (!img_ptr || w <= 0 || h <= 0)
	{
		return;
	}

	u8* new_img_ptr = po_new u8[w*h];
	CPOBase::memCopy(new_img_ptr, img_ptr, w*h);

	cv::Mat cv_img(h, w, CV_8UC1, new_img_ptr);
	cv::Mat cv_img_blur(h, w, CV_8UC1, img_ptr);
	cv::GaussianBlur(cv_img, cv_img_blur, cv::Size(nw, nw), 0);
	POSAFE_DELETE_ARRAY(new_img_ptr);
}

i32 CImageProc::getZeroPixels(u8* img_ptr, i32 w, i32 h)
{
	if (!img_ptr || w <= 0 || h <= 0)
	{
		return 0;
	}

	cv::Mat cv_img(h, w, CV_8UC1, img_ptr);
	return w*h - cv::countNonZero(cv_img);
}

i32 CImageProc::getNonZeroPixels(u8* img_ptr, i32 w, i32 h)
{
	if (!img_ptr || w <= 0 || h <= 0)
	{
		return 0;
	}

	cv::Mat cv_img(h, w, CV_8UC1, img_ptr);
	return cv::countNonZero(cv_img);
}

void CImageProc::erodeImage(u8* img_ptr, i32 w, i32 h, i32 cov)
{
	if (!img_ptr || w*h <= 0)
	{
		return;
	}

	i32 erode_size = 2 * cov + 1;
	bool is_processed = false;

#if defined(POR_WITH_OVX)
	if (g_vx_gpool_ptr)
	{
		CGImgProcErode* graph_ptr = (CGImgProcErode*)g_vx_gpool_ptr->fetchGraph(
				kGImgProcErode, erode_size, img_ptr, w, h, img_ptr);
		if (graph_ptr)
		{
			is_processed = graph_ptr->process();
			g_vx_gpool_ptr->releaseGraph(graph_ptr);
		}
	}
#elif defined(POR_WITH_OCLCV)
	cv::ocl::setUseOpenCL(true);

	cv::Mat cv_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erode_size, erode_size));
	cv::Mat cv_src_img(h, w, CV_8UC1, img_ptr);
	i32 sub_w = cv_src_img.cols;
	i32 sub_h = cv_src_img.rows;

	cv::UMat u_tmp_img;
	cv::UMat u_src_img = cv::UMat::zeros(CPOBase::round(sub_h, 256), CPOBase::round(sub_w, 256), CV_8UC1);
	cv_src_img.copyTo(u_src_img(cv::Rect(0,0, sub_w, sub_h)));
	cv::erode(u_src_img, u_tmp_img, cv_element);
	u_tmp_img(cv::Rect(0,0, sub_w, sub_h)).copyTo(cv_src_img);

	cv::ocl::setUseOpenCL(false);
#endif
	if (!is_processed)
	{
		cv::Mat cv_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erode_size, erode_size));
		cv::Mat cv_src_img(h, w, CV_8UC1, img_ptr);
		cv::Mat cv_erode_img;
		cv::erode(cv_src_img, cv_erode_img, cv_element);
		CPOBase::memCopy(img_ptr, (u8*)cv_erode_img.data, w*h);
	}
}

void CImageProc::dilateImage(u8* img_ptr, i32 w, i32 h, i32 cov)
{
    if (!img_ptr || w*h <= 0)
    {
        return;
    }

	i32 dilate_size = 2 * cov + 1;
	bool is_processed = false;

#if defined(POR_WITH_OVX)
	if (g_vx_gpool_ptr)
	{
		CGImgProcDilate* graph_ptr = (CGImgProcDilate*)g_vx_gpool_ptr->fetchGraph(
					kGImgProcDilate, dilate_size, img_ptr, w, h, img_ptr);
		if (graph_ptr)
		{
			is_processed = graph_ptr->process();
			g_vx_gpool_ptr->releaseGraph(graph_ptr);
		}
	}
#elif defined(POR_WITH_OCLCV)
	cv::ocl::setUseOpenCL(true);

	cv::Mat cv_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilate_size, dilate_size));
	cv::Mat cv_src_img(h, w, CV_8UC1, img_ptr);
	i32 sub_w = cv_src_img.cols;
	i32 sub_h = cv_src_img.rows;

	cv::UMat u_tmp_img;
	cv::UMat u_src_img = cv::UMat::zeros(CPOBase::round(sub_h, 256), CPOBase::round(sub_w, 256), CV_8UC1);
	cv_src_img.copyTo(u_src_img(cv::Rect(0,0, sub_w, sub_h)));
	cv::dilate(u_src_img, u_tmp_img, cv_element);
	u_tmp_img(cv::Rect(0,0, sub_w, sub_h)).copyTo(cv_src_img);

	cv::ocl::setUseOpenCL(false);
#endif
	if (!is_processed)
	{
		cv::Mat cv_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilate_size, dilate_size));
		cv::Mat cv_src_img(h, w, CV_8UC1, img_ptr);
		cv::Mat cv_dilate_img;
		cv::dilate(cv_src_img, cv_dilate_img, cv_element);
		CPOBase::memCopy(img_ptr, (u8*)cv_dilate_img.data, w*h);
	}
}

void CImageProc::addImage(u8* dst_img_ptr, u8* img_ptr, i32 w, i32 h)
{
	if (!img_ptr || !dst_img_ptr || w*h <= 0)
	{
		return;
	}
	
	cv::Mat cv_dst_img(h, w, CV_8UC1, dst_img_ptr);
	cv::Mat cv_src_img(h, w, CV_8UC1, img_ptr);

	cv::Mat cv_tmp;
	cv_tmp = cv_dst_img + cv_src_img;
	cv_tmp.convertTo(cv_dst_img, CV_8UC1);
}

void CImageProc::addInvImage(u8* dst_img_ptr, u8* img_ptr, i32 w, i32 h)
{
	if (!img_ptr || !dst_img_ptr || w*h <= 0)
	{
		return;
	}

	cv::Mat cv_dst_img(h, w, CV_8UC1, dst_img_ptr);
	cv::Mat cv_src_img(h, w, CV_8UC1, img_ptr);

	cv::Mat cv_src_tmp, cv_tmp;
	cv::bitwise_not(cv_src_img, cv_src_tmp);
	cv_tmp = cv_dst_img + cv_src_tmp;
	cv_tmp.convertTo(cv_dst_img, CV_8UC1);
}

void CImageProc::getGradient(u8* img_ptr, i32 index, i32 w, f32& nx, f32& ny)
{
	if (!img_ptr)
	{
		return;
	}

	i32 nindex = index - w;
	i32 sindex = index + w;
	i32 nn = img_ptr[nindex];
	i32 ss = img_ptr[sindex];
	i32 ww = img_ptr[index + 1];
	i32 ee = img_ptr[index - 1];
	i32 nw = img_ptr[nindex + 1];
	i32 ne = img_ptr[nindex - 1];
	i32 sw = img_ptr[sindex + 1];
	i32 se = img_ptr[sindex - 1];

	nx = (f32)(nw + 2 * ww + sw - ne - 2 * ee - se);
	ny = (f32)(sw + 2 * ss + se - nw - 2 * nn - ne);
	if (nx == 0 && ny == 0)
	{
		return;
	}

	f32 len = std::sqrt(nx*nx + ny*ny);
	nx = nx / len;
	ny = ny / len;
}

void CImageProc::absDiffImage(u8* img_ptr1, u8* img_ptr2, u8* mask_img_ptr, i32 w, i32 h, u8* dst_img_ptr)
{
	if (!img_ptr1 || !img_ptr2 || !dst_img_ptr)
	{
		return;
	}

	i32 i, wh = w*h;
	if (mask_img_ptr) //with mask image
	{
		for (i = 0; i < wh; i++)
		{
			if (*mask_img_ptr != 0)
			{
				*dst_img_ptr = std::abs(*img_ptr1 - *img_ptr2);
			}

			img_ptr1++;
			img_ptr2++;
			dst_img_ptr++;
			mask_img_ptr++;
		}
	}
	else //with nonmask
	{
		for (i = 0; i < wh; i++)
		{
			*dst_img_ptr = std::abs(*img_ptr1 - *img_ptr2);

			img_ptr1++;
			img_ptr2++;
			dst_img_ptr++;
		}
	}
}

void CImageProc::maxImage(u8* img_ptr1, u8* img_ptr2, u8* mask_img_ptr, i32 w, i32 h, u8* dst_img_ptr)
{
	if (!img_ptr1 || !img_ptr2 || !dst_img_ptr)
	{
		return;
	}

	i32 i, wh = w*h;
	if (mask_img_ptr) //with mask image
	{
		for (i = 0; i < wh; i++)
		{
			if (*mask_img_ptr != 0)
			{
				*dst_img_ptr = po::_max(*img_ptr1, *img_ptr2);
			}

			img_ptr1++;
			img_ptr2++;
			dst_img_ptr++;
			mask_img_ptr++;
		}
	}
	else //with nonmask
	{
		for (i = 0; i < wh; i++)
		{
			*dst_img_ptr = po::_max(*img_ptr1, *img_ptr2);

			img_ptr1++;
			img_ptr2++;
			dst_img_ptr++;
		}
	}
}

void CImageProc::suppressionImage(u8* img_ptr, i32 w, i32 h, i32 sup)
{
	i32 i, wh = w*h;
	if (!img_ptr || wh <= 0 || sup <= 0)
	{
		return;
	}

	for (i = 0; i < wh; i++)
	{
		if (*img_ptr <= sup)
		{
			*img_ptr = 0;
		}
		else
		{
			*img_ptr -= sup;
		}
		img_ptr++;
	}
}

void CImageProc::maskIgnoreImage(u8* img_ptr, i32 w, i32 h, u8* igr_img_ptr)
{
	i32 i, wh = w*h;
	if (!img_ptr || !igr_img_ptr || wh <= 0)
	{
		return;
	}

	for (i = 0; i < wh; i++)
	{
		if (*igr_img_ptr > 0)
		{
			*img_ptr = 0;
		}
		img_ptr++; igr_img_ptr++;
	}
}

bool CImageProc::imgPackGray2Bin(u8* src_buffer_ptr, i32 w, i32 h, i32 src_step, u8* new_src_buffer_ptr, i32 nw)
{
	if (!src_buffer_ptr || !new_src_buffer_ptr || w*h*nw <= 0 || src_step != 1)
	{
		return false;
	}

	i32 i, j, tmp, bit_index = 0;
	memset(new_src_buffer_ptr, 0, nw*h);

	for (i=0; i<h; i++)
	{
		tmp = 0;
		bit_index = 0;
		u8* tmp_buffer_ptr = src_buffer_ptr + i*w;
		u8* dst_buffer_ptr = new_src_buffer_ptr + i*nw;

		for (j=0; j<w; j++)
		{
			if (*tmp_buffer_ptr)
			{
				tmp |= g_bit_mask[bit_index];
			}

			if (++bit_index == 8)
			{
				bit_index = 0;
				*dst_buffer_ptr = tmp;
				dst_buffer_ptr++;
				tmp = 0;
			}
			tmp_buffer_ptr++;
		}
		if (bit_index != 0)
		{
			*dst_buffer_ptr = tmp;
			dst_buffer_ptr++;
		}
	}
	return true;
}

bool CImageProc::imgPackBin2Gray(u8* src_buffer_ptr, i32 nw, i32 h, u8* dst_buffer_ptr, i32 w, i32 dst_step, i32 offset_x, i32 offset_y)
{
	if (!src_buffer_ptr || !dst_buffer_ptr || w*h*nw <= 0 || dst_step != 1)
	{
		return false;
	}

	i32 w1 = w-2*offset_x;
	i32 h1 = h-2*offset_y;
	i32 i, j, tmp, bit_index = 0;
	memset(dst_buffer_ptr, 0, w*h);

    for (i=0; i<h1; i++)
	{
		u8* tmp_buffer_ptr = src_buffer_ptr + i*nw;
		u8* tmp_dst_buffer_ptr = dst_buffer_ptr+ (i+offset_y)*w + offset_x;
		tmp = tmp_buffer_ptr[0];
        bit_index = 8;

		for (j=0; j<w1; j++)
		{
			if (tmp & 0x01)
			{
				*tmp_dst_buffer_ptr = 0xFF;
			}
			tmp = tmp >> 1;
			if (--bit_index == 0)
			{
				tmp_buffer_ptr++;
				tmp = *tmp_buffer_ptr;
				bit_index = 8;
			}
			tmp_dst_buffer_ptr++;
		}
	}
	return true;
}

#if defined(POR_WITH_OVX)
#include "performance/openvx_app/openvx_app.h"

void CImageProc::saveImgOpenVx(const char* filename, vx_image vx_img)
{
	vx_status status = VX_SUCCESS;

	void *data_vx_img = NULL;
	vx_map_id map_id_vx_img = 0;
	vx_rectangle_t rect_vx_img;
	vx_imagepatch_addressing_t addr_vx_img;

	VX_CHKRET(vxGetValidRegionImage(vx_img, &rect_vx_img));
	VX_CHKRET(vxMapImagePatch(vx_img, &rect_vx_img, 0, &map_id_vx_img, &addr_vx_img, &data_vx_img, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

	if (addr_vx_img.stride_x == 1)
	{
		i32 w = rect_vx_img.end_x - rect_vx_img.start_x;
		i32 h = rect_vx_img.end_y - rect_vx_img.start_y;
		cv::Mat img(h, w, CV_8UC1, (u8*)data_vx_img, addr_vx_img.stride_y);
		cv::imwrite(filename, img);
	}
	else if (addr_vx_img.stride_x == 2)
	{
		i32 w = rect_vx_img.end_x - rect_vx_img.start_x;
		i32 h = rect_vx_img.end_y - rect_vx_img.start_y;
		cv::Mat img(h, w, CV_16SC1, (u8*)data_vx_img, addr_vx_img.stride_y);
		cv::imwrite(filename, img);
	}

	if (map_id_vx_img > 0)
	{
		VX_CHKRET(vxUnmapImagePatch(vx_img, map_id_vx_img));
	}
}
#endif

void CImageProc::makePaddingImage(const ImgPart& dst_img_part, const ImgPart& src_img_part)
{
	if (!dst_img_part.isValid() || !src_img_part.isValid() || dst_img_part.channel != src_img_part.channel)
	{
		return;
	}

	makePaddingImage(dst_img_part.img_ptr, src_img_part.img_ptr,
		dst_img_part.getWidth(), dst_img_part.getHeight(), src_img_part.getWidth(), src_img_part.getHeight(),
		src_img_part.channel);
}

void CImageProc::makePaddingImage(u8* dst_img_ptr, u8* src_img_ptr, i32 dw, i32 dh, i32 sw, i32 sh, i32 channel)
{
	if (!dst_img_ptr || !src_img_ptr || dw * dh <= 0 || sw * sh <= 0)
	{
		return;
	}

	i32 w = po::_min(sw, dw);
	i32 h = po::_min(sh, dh);
	i32 dst_stride = dw * channel;
	i32 src_stride = sw * channel;
	i32 stride = w * channel;

	for (i32 i = 0; i < h; i++)
	{
		memcpy(dst_img_ptr + dst_stride * i, src_img_ptr + src_stride * i, stride);
	}
}
