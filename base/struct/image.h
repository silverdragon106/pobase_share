#pragma once

#include "define.h"
#include "struct/rect.h"
#include "struct/vector2d.h"

#pragma pack(push, 4)

enum ImgPartFlags
{
	kImgPartNone	= 0x00,
	kImgPartSanpped	= 0x01
};

struct ImgExpr
{
	i32					w;
	i32					h;
	i32					channel;
	u8*					img_ptr;

public:
	ImgExpr();
	ImgExpr(u8* img_ptr, i32 nw, i32 nh, i32 nchannel = 1);

	bool				isValid() const;
};

struct ImgPart
{
	Recti				range;
	i32					channel;
	u8*					img_ptr;
	i32					flag;

public:
	ImgPart();
	ImgPart(u8* img_ptr, i32 nw, i32 nh, i32 nchannel = kPOGrayChannels, i32 flag = 0);
	ImgPart(u8* img_ptr, Recti range, i32 nchannel = kPOGrayChannels, i32 flag = 0);

	void				init();

	const bool			isValid() const;
	const Recti			getRange() const;
	const Recti*		getRangePtr() const;
	const i32			getWidth() const { return range.getWidth(); }; 
	const i32			getHeight() const { return range.getHeight(); };

	inline bool			isGrayImage() const { return (channel == kPOGrayChannels); };
	inline bool			isColorImage() const { return (channel != kPOGrayChannels); };
};

struct Img
{
	i32					w;
	i32					h;
	i32					channel;
	u8*					img_ptr;
	bool				is_external_alloc; //external memory used

public:
	Img();
	Img(i32 nw, i32 nh, i32 nchannel = 1);
	Img(u8* img_ptr, i32 nw, i32 nh, i32 nchannel = 1);
	~Img();

	Img&				setValue(Img* other_ptr);

	bool				initBuffer(i32 nw, i32 nh, i32 nchannel = kPOGrayChannels);
	bool				setImage(const ImgPart img_part, bool is_external = true);
	bool				setImage(const ImgExpr img_expr, bool is_external = true);
	bool				setImage(u8* img_ptr, i32 nw, i32 nh, i32 nchannel, bool is_external = true);
	bool				copyImage(u8* img_ptr, i32 nw, i32 nh, i32 nchannel = kPOGrayChannels);
	bool				copyImage(Img* image_ptr);
	bool				copyImage(Img& image);
	void				freeBuffer();

	void				update(i32 w, i32 h, i32 channel);

	i32					memSize();
	i32					memRead(u8*& buffer_ptr, i32& buffer_size);
	i32					memWrite(u8*& buffer_ptr, i32& buffer_size);
	bool				fileRead(FILE* fp);
	bool				fileWrite(FILE* fp);

	bool				isValid() const;
	bool				isLessBuffer(i32 nw, i32 nh, i32 nchannel);
	bool				isSameSize(Img& other);

	inline bool			isGrayImage() { return (channel == kPOGrayChannels); };
	inline bool			isColorImage() { return (channel != kPOGrayChannels); };

	inline i32			getImgSize() { return w*h*channel; };
	inline ImgExpr		toImgExpr() { return ImgExpr(img_ptr, w, h, channel); };
};

struct ImageData
{
	i32					w;
	i32					h;
	i32					channel;
	u8*					img_ptr;
	i64					time_stamp;
	bool				is_snaped;
	bool				is_calibed;
	f32					pixel_per_mm;
	i32					reserved;

	i32					buffer_size;
	bool				is_external_alloc;

public:
	ImageData();
	ImageData(const Img& img);
	ImageData(const ImageData& img_data);
	ImageData(u8* img_ptr, i32 w, i32 h, i32 channel = kPOGrayChannels);
	ImageData(u8* img_ptr, i32 w, i32 h, i32 channel, i64 tm_stamp, bool is_calib, f32 pixel2mm);
	~ImageData();

	i32					memSize();
	i32					memRead(u8*& buffer_ptr, i32& buffer_size);
	i32					memWrite(u8*& buffer_ptr, i32& buffer_size);

	i32					memImgSize();
	i32					memImgWrite(u8*& buffer_ptr, i32& buffer_size);

	void				freeBuffer();
	void				releaseImage();
	void				initBuffer(ImageData* img_data_ptr);
	void				initBuffer(const ImageData& img_data);
	void				initBuffer(i32 nw, i32 nh, i32 nchannel = kPOGrayChannels);
	void				initInternalBuffer(i32 nw, i32 nh, i32 nchannel = kPOGrayChannels);
	void				update(i32 w, i32 h, i32 channel);
	void				setSnapImage(bool is_snap);
	void				setTimeStamp(i64 tmstamp);
	void				setImageData(u8* img_ptr, i32 w, i32 h, i32 channel = kPOGrayChannels);
	void				copyImage(ImageData* img_data_ptr);
	void				copyImage(const ImageData& img_data);
	void				copyImage(const ImageData& img_data, i32 pw, i32 ph);
	void				copyImage(u8* img_ptr, i32 w, i32 h, i32 channel = kPOGrayChannels);
	void				cropImage(i32 max_width, i32 max_height);
	void				copyImageInfo(const ImageData* img_data_ptr);
	void				copyToImage(ImageData* img_data_ptr, i32 out_channel = kPOAnyChannels);
	void				copyToImage(ImageData& img_data, i32 out_channel = kPOAnyChannels);
	void				copyToImage(u8* img_ptr, i32& w, i32& h, i32& channel, i32 out_channel = kPOAnyChannels);

	bool				isValid() const;
	bool				isSnapImage() const;
	bool				isLessBuffer(ImageData* img_data_ptr);
	bool				isSameBuffer(ImageData* img_data_ptr);
	bool				isSameFormat(ImageData* img_data_ptr);
	bool				isLessBuffer(const ImageData& img_data);
	bool				isSameBuffer(const ImageData& img_data);
	bool				isSameFormat(const ImageData& img_data);
	bool				isSameSize(const ImageData& img_data);

	Img					toImg() const;
	ImgPart				toImgPart() const;
	ImgExpr				toImgExpr() const;

	inline bool			isGrayImage() { return (channel == kPOGrayChannels); };
	inline bool			isColorImage() { return (channel != kPOGrayChannels); };

	inline u8*			getImageBuffer() const { return img_ptr; };
	inline i32			getImageSize() const { return w*h*channel; };
	inline i32			getImageStride() const { return w*channel; };

	void				convertToGray();
	void				reduceImage(ImageData& img_data, i32 nw, i32 nh);
};
#pragma pack(pop)