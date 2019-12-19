#pragma once

#include "struct.h"
#include "memory_pool.h"
#include "img_proc/threshold.h"
# include <opencv2/opencv.hpp>

#if defined(POR_WITH_OVX)
#if !defined(POR_IMVS2_ON_AM5728)
	#include "VX/vx.h"
#else
	#include "TI/tivx.h"
#endif
#endif

#define PO_THUMB_WIDTH		128
#define PO_THUMB_HEIGHT		96

class CImageProc : public CThreshold, public CPOMemPool
{
public:
	CImageProc();
	virtual ~CImageProc();

	//non-static function 
	f32						getImageFocus(u8* img_ptr, i32 w, i32 h, Recti rt, i32 lapth);
	f32						getImageFocus(u8* img_ptr, i32 w, i32 h, i32 lapth);

	//new, crop, convert
	static u8*				makeZeroImage(Recti rt);
	static u8*				copyImage(u8* img_ptr, i32 w, i32 h);
	static u8*				copyImage(u8* img_ptr, Recti range);
	static u8*				cropImage(const Img img, vector2di crop_size);
	static u8*				cropImage(const Img img, Recti crop_rt);
	static u8*				cropImage(const ImgPart img_part, Recti crop_rt);
	static bool				cropImage(const ImgPart img_part, Recti crop_rt, Img& dst_img);
	static bool				cropImage(u8* dst_img_ptr, Img img, Recti crop_rt);
	static bool				cropImage(ImageData& img_data, vector2di crop_size);
	static void				invertImage(u8* img_ptr, i32 w, i32 h, i32 channel = 1);
	static u8*				extendImage(u8* img_ptr, i32 w, i32 h, i32 r);
	static void				extendImage(u8* dst_img_ptr, u8* img_ptr, i32 w, i32 h, i32 r);
	static void				convertImgYFlip8(u8* img_ptr, i32 w, i32 h, u8* dst_img_ptr);
	static u8*				convertcvMatToImg(cv::Mat& mat, i32& nw, i32& nh);
	static void				applyTransform(u8* dst_img_ptr, u8* src_img_ptr, i32& w, i32& h,
									i32 rotation, bool flip_x, bool flip_y, bool is_invert, i32 channel = 1);
	static void				applyTransform(ImageData& dst_img, ImageData& src_img,
									i32 rotation, bool flip_x, bool flip_y, bool is_invert);
	static void				applyTransform(ImageData& img_data,
									i32 rotation, bool flip_x, bool flip_y, bool is_invert);

	//warp image
	static u8*				affineWarpImage(u8* img_ptr, i32 w, i32 h, f32* tr);
	static u8*				affineWarpImageOpenCV(u8* img_ptr, i32 w, i32 h, f32* tr);
	static void				affineWarpImage(u8* img_ptr, Recti rt, Img image, f32* tr);
	static void				affineWarpImageOpenCV(u8* img_ptr, Recti rt, Img image, f32* tr);

	//integral image
    template <typename T>
	static void				makeIntegralImage(i32* int_img_ptr, T* img_ptr, i32 w, i32 h);
    template <typename T>
	static void				makeIntegralImageSq(i32* int_img_ptr, T* img_ptr, i32 w, i32 h);

	//processing
	static void				gaussianBlur(u8* img_ptr, i32 w, i32 h, i32 nw, f32 sigma = 0);
	static void				addImage(u8* dst_img_ptr, u8* img_ptr, i32 w, i32 h);
	static void				addInvImage(u8* dst_img_ptr, u8* img_ptr, i32 w, i32 h);
	static void				dilateImage(u8* img_ptr, i32 w, i32 h, i32 cov);
	static void				erodeImage(u8* img_ptr, i32 w, i32 h, i32 cov);
	static void				absDiffImage(u8* img_ptr1, u8* img_ptr2, u8* mask_img_ptr, i32 w, i32 h, u8* dst_img_ptr);
	static void				maxImage(u8* img_ptr1, u8* img_ptr2, u8* mask_img_ptr, i32 w, i32 h, u8* dst_img_ptr);
	static void				suppressionImage(u8* img_ptr, i32 w, i32 h, i32 sup);
	static void				maskIgnoreImage(u8* img_ptr, i32 w, i32 h, u8* ignore_img_ptr);
	
	/* 히스토그람 */
	static void				makeHistogram(u8* img_ptr, u8* mask_img_ptr, i32 w, i32 h, i32* hist_ptr, i32* border_hist_ptr);
	static bool				isInvertedBackground(i32 th, i32* hist_ptr, i32* border_hist_ptr);

	//thumbnail
	static void				registerThumbSize(i32 w, i32 h);
	static u8*				makeThumbImage(u8* img_ptr, i32 w, i32 h, i32 channel, i32& tw, i32& th);
	static ImgExpr			makeThumbImage(ImageData* img_data_ptr);
	static void				makeThumbImage(ImageData& src_img, ImageData& dst_img);
	static void				makeThumbImage(ImageData& src_img, i32 nw, i32 nh, ImageData& dst_img);
	
	//color
	static bool				convertColor(ImageData& src_img, ImageData& dst_img, i32 cvt_mode);
	static bool				convertColor(ImageData* src_img_ptr, ImageData* dst_img_ptr, i32 cvt_mode);
	static bool				convertColor(Img& src_img, Img& dst_img, i32 cvt_mode);
	static bool				vxConvertColor(ImageData& src_img, ImageData& dst_img, i32 cvt_mode);

	//edge
	static void				getGradient(u8* img_ptr, i32 index, i32 w, f32& nx, f32& ny);
    template <typename T>
	static void				gradL2Norm(T* gx, T* gy, T* grad, i32 w, i32 h, bool bL2Norm);
    template <typename T>
	static void				makeRobertGradImage(T* edge_img_ptr, u8* img_ptr, i32 w, i32 h);
    template <typename T>
	static void				makeSubPixel(T* pGradImg, i32 w, i32 h, Pixelf* ppixel, i32 count);
	
	//border, contour
	static void				fillImageBorder(u8* img_ptr, i32 w, i32 h, u8 val);
	static void				fillContourForBlob(u8* img_ptr, i32 w, i32 h, u8 val);
	static void				fillContourForBlob2(u8* img_ptr, i32 w, i32 h, u8 fg, u8 eg);
	static void				fillOutsideContourForBlob(u16* img_ptr, Recti& range, i32 count);

	static void				makePaddingImage(const ImgPart& dst_img_part, const ImgPart& src_img_part);
	static void				makePaddingImage(u8* dst_img_ptr, u8* src_img_ptr, i32 dw, i32 dh, i32 sw, i32 sh, i32 channel = 1);

    template <typename T>
	static void				makePaddingBinary(u8* pad_img_ptr, T* img_ptr, i32 w, i32 h, i32 padding_size = 1);
    template <typename T>
	static void				makePaddingImage(u8* pad_img_ptr, T* img_ptr, i32 w, i32 h, u8 bg_pixel, i32 padding_size = 1);
    template <typename T, typename U>
	static bool				fillContourTrace(T* img_ptr, i32 w, i32 h, u8* pad_img_ptr,
									i32 x, i32 y, i32 mode, i32 ni, i32 val, Contour<U>* contour_ptr);

	//save, load
	static u8*				loadImgOpenCV(const char* filename, i32& w, i32& h);
	static cv::Mat			loadImgOpenCV(const potstring& filename, i32 mode);

	static bool				saveImgOpenCV(const char* filename, u8* img_ptr, i32 w, i32 h, i32 channel = kPOGrayChannels);
	static bool				saveImgOpenCV(const char* filename, ImageData* img_data_ptr);
	static bool				saveImgOpenCV(const char* filename, const ImageData& img_data);
	static bool				saveImgOpenCV(const char* filename, const ImgPart img_part);
	static bool				saveImgOpenCV(const wchar_t* filename, u8* img_ptr, i32 w, i32 h, i32 channel = kPOGrayChannels);
	static bool				saveImgOpenCV(const potstring& filename, const Img img);
	static bool				saveImgOpenCV(const potstring& filename, const ImgPart img_part);
	static bool				saveImgOpenCV(const potstring& filename, const ImgExpr img_expr);
	static bool				saveImgOpenCV(const potstring& filename, const ImageData& img_data);
	static bool				saveImgOpenCV(const potstring& filename, ImageData* img_data_ptr);
	static bool				saveBinImgOpenCV(const char* filename, u8* img_ptr, i32 w, i32 h);
	static bool				saveBinXImgOpenCV(const char* filename, u8* img_ptr, i32 w, i32 h, i32 val);

	static cv::Mat			decodeImgOpenCV(const u8vector& encoded_vec, i32 mode);

	//count
	static i32				getZeroPixels(u8* img_ptr, i32 w, i32 h);
	static i32				getNonZeroPixels(u8* img_ptr, i32 w, i32 h);

	static bool				imgPackGray2Bin(u8* src_buffer_ptr, i32 w, i32 h, i32 src_step, u8* new_src_buffer_ptr, i32 nw);
	static bool				imgPackBin2Gray(u8* src_buffer_ptr, i32 nw, i32 h, u8* dst_buffer_ptr, i32 w, i32 dst_step, i32 offset_x, i32 offset_y);

#if defined(POR_WITH_OVX)
	static void				saveImgOpenVx(const char* filename, vx_image vx_img);
#endif
};

#include "image_proc-inl.h"
