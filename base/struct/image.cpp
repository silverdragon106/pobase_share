#include "image.h"
#include "base.h"

#if defined(POR_DEVICE)
#include "proc/image_proc.h"
#endif

//////////////////////////////////////////////////////////////////////////
ImgExpr::ImgExpr()
{
	CPOBase::memZero(this);
}

ImgExpr::ImgExpr(u8* img_ptr, i32 nw, i32 nh, i32 nchannel)
{
	this->w = nw;
	this->h = nh;
	this->channel = nchannel;
	this->img_ptr = img_ptr;
}

bool ImgExpr::isValid() const
{
	return (img_ptr != NULL && w*h*channel > 0);
}

//////////////////////////////////////////////////////////////////////////
ImgPart::ImgPart()
{
	init();
}

ImgPart::ImgPart(u8* img_ptr, i32 nw, i32 nh, i32 nchannel, i32 nflag)
{
	this->range = Recti(0, 0, nw, nh);
	this->channel = nchannel;
	this->img_ptr = img_ptr;
	this->flag = flag;
}

ImgPart::ImgPart(u8* img_ptr, Recti range, i32 nchannel, i32 nflag)
{
	this->range = range;
	this->channel = nchannel;
	this->img_ptr = img_ptr;
	this->flag = flag;
}

void ImgPart::init()
{
	memset(this, 0, sizeof(ImgPart));
}

const bool ImgPart::isValid() const
{
	i32 w = range.getWidth();
	i32 h = range.getHeight();
	return (img_ptr && w*h*channel > 0);
}

const Recti ImgPart::getRange() const
{
	return range;
}

const Recti* ImgPart::getRangePtr() const
{
	return &range;
}

//////////////////////////////////////////////////////////////////////////
Img::Img()
{
	CPOBase::memZero(this);
}

Img::Img(i32 nw, i32 nh, i32 nchannel)
{
	if (nw > 0 && nh > 0 && nchannel > 0)
	{
		w = nw;
		h = nh;
		channel = nchannel;
		is_external_alloc = false;
		img_ptr = po_new u8[w*h*channel];
		memset(img_ptr, 0, w*h*channel);
	}
	else
	{
		CPOBase::memZero(this);
	}
}

Img::Img(u8* img_ptr, i32 nw, i32 nh, i32 nchannel)
{
	if (img_ptr != NULL && nw > 0 && nh > 0 && nchannel > 0)
	{
		w = nw;
		h = nh;
		channel = nchannel;
		this->img_ptr = img_ptr;
		is_external_alloc = true;
	}
	else
	{
		CPOBase::memZero(this);
	}
}

Img::~Img()
{
	freeBuffer();
}

bool Img::initBuffer(i32 nw, i32 nh, i32 nchannel)
{
	//check image size
	if (nw <= 0 || nh <= 0 || nchannel <= 0)
	{
		return false;
	}

	//reallocate when less
	if (isLessBuffer(nw, nh, nchannel))
	{
		freeBuffer();
		img_ptr = po_new u8[nw*nh*nchannel];
	}
	w = nw; h = nh; channel = nchannel;
	return true;
}

bool Img::setImage(u8* img_ptr, i32 nw, i32 nh, i32 nchannel,  bool is_external)
{
	//check source image valid
	if (img_ptr == NULL || nw <= 0 || nh <= 0 || nchannel <= 0)
	{
		return false;
	}
	freeBuffer();

	w = nw;
	h = nh;
	channel = nchannel;
	this->img_ptr = img_ptr;
	is_external_alloc = is_external;
	return true;
}

bool Img::setImage(const ImgPart img_part, bool is_external)
{
	//check source image valid
	if (!img_part.isValid())
	{
		return false;
	}
	freeBuffer();

	Recti range = img_part.getRange();
	w = range.getWidth();
	h = range.getHeight();
	channel = img_part.channel;
	this->img_ptr = img_part.img_ptr;
	is_external_alloc = is_external;
	return true;
}

bool Img::setImage(const ImgExpr img_expr, bool is_external)
{
	//check source image valid
	if (!img_expr.isValid())
	{
		return false;
	}
	freeBuffer();

	w = img_expr.w;
	h = img_expr.h;
	channel = img_expr.channel;
	this->img_ptr = img_expr.img_ptr;
	is_external_alloc = is_external;
	return true;
}

bool Img::copyImage(u8* img_ptr, i32 nw, i32 nh, i32 nchannel)
{
	//check source image valid
	if (img_ptr == NULL || nw <= 0 || nh <= 0 || nchannel <= 0)
	{
		return false;
	}

	if (isLessBuffer(nw, nh, nchannel))
	{
		freeBuffer();
		this->img_ptr = po_new u8[nw*nh*nchannel];
	}

	w = nw;
	h = nh;
	channel = nchannel;
	is_external_alloc = false;
	CPOBase::memCopy(this->img_ptr, img_ptr, nw*nh*nchannel);
	return true;
}

bool Img::copyImage(Img* img_ptr)
{
	return copyImage(img_ptr->img_ptr, img_ptr->w, img_ptr->h, img_ptr->channel);
}

bool Img::copyImage(Img& image)
{
	return copyImage(image.img_ptr, image.w, image.h, image.channel);
}

void Img::freeBuffer()
{
	if (!is_external_alloc)
	{
		POSAFE_DELETE_ARRAY(img_ptr);
	}

	w = 0;
	h = 0;
	channel = 0;
	img_ptr = NULL;
	is_external_alloc = false;
}

void Img::update(i32 w, i32 h, i32 channel)
{
	this->w = w;
	this->h = h;
	this->channel = channel;
}

i32 Img::memSize()
{
	return CPOBase::getImageSize(w, h, channel);
}

i32 Img::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	freeBuffer();
	CPOBase::memReadImage(img_ptr, w, h, channel, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 Img::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWriteImage(img_ptr, w, h, channel, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool Img::fileRead(FILE* fp)
{
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	i32 nw, nh, nchannel;
	CPOBase::fileRead(nw, fp);
	CPOBase::fileRead(nh, fp);
	CPOBase::fileRead(nchannel, fp);
	if (initBuffer(nw, nh, nchannel))
	{
		CPOBase::fileRead(img_ptr, nw*nh*nchannel, fp);
	}
	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool Img::fileWrite(FILE* fp)
{
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);
	CPOBase::fileWriteImage(img_ptr, w, h, channel, fp);
	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

Img& Img::setValue(Img* other_ptr)
{
	copyImage(other_ptr);
	return *this;
}

bool Img::isSameSize(Img& other)
{
	return (w == other.w && h == other.h && channel == other.channel);
}

bool Img::isValid() const
{
	return (img_ptr != NULL && w*h*channel > 0); 
}

bool Img::isLessBuffer(i32 nw, i32 nh, i32 nchannel)
{
	return (is_external_alloc || w*h*channel < nw*nh*nchannel); 
}

//////////////////////////////////////////////////////////////////////////
ImageData::ImageData()
{
	memset(this, 0, sizeof(ImageData));
}

ImageData::ImageData(const Img& img)
{
	memset(this, 0, sizeof(ImageData));
	w = img.w;
	h = img.h;
	channel = img.channel;
	img_ptr = img.img_ptr;
	is_external_alloc = true;
}

ImageData::ImageData(const ImageData& img_data)
{
	memset(this, 0, sizeof(ImageData));
	copyImage(img_data);
}

ImageData::ImageData(u8* img_ptr, i32 w, i32 h, i32 channel)
{
	memset(this, 0, sizeof(ImageData));
	this->img_ptr = img_ptr;
	this->w = w;
	this->h = h;
	this->channel = channel;
	this->is_external_alloc = true;
}

ImageData::ImageData(u8* img_ptr, i32 w, i32 h, i32 channel, i64 tmstamp, bool is_calib, f32 pixel2mm)
{
	memset(this, 0, sizeof(ImageData));
	this->img_ptr = img_ptr;
	this->w = w;
	this->h = h;
	this->channel = channel;
	this->is_external_alloc = true;

	this->is_calibed = is_calib;
	this->pixel_per_mm = pixel2mm;
	this->time_stamp = tmstamp;
}

ImageData::~ImageData()
{
	freeBuffer();
}

void ImageData::initBuffer(ImageData* img_data_ptr)
{
	if (!img_data_ptr)
	{
		return;
	}
	initBuffer(img_data_ptr->w, img_data_ptr->h, img_data_ptr->channel);
}

void ImageData::initBuffer(const ImageData& img_data)
{
	initBuffer(img_data.w, img_data.h, img_data.channel);
}

void ImageData::initBuffer(i32 nw, i32 nh, i32 nchannel)
{
	if (is_external_alloc || buffer_size < nw*nh*nchannel)
	{
		freeBuffer();

		buffer_size = nw*nh*nchannel;
		img_ptr = po_new u8[buffer_size];
		memset(img_ptr, 0, buffer_size);
	}

	w = nw;
	h = nh;
	channel = nchannel;
	is_snaped = false;
	is_calibed = false;
	is_external_alloc = false;
}

void ImageData::initInternalBuffer(i32 nw, i32 nh, i32 nchannel)
{
	if (is_external_alloc)
	{
		w = nw; h = nh; channel = nchannel;
		return;
	}
	initBuffer(nw, nh, nchannel);
}

void ImageData::setImageData(u8* img_ptr, i32 w, i32 h, i32 channel)
{
	freeBuffer();

	this->w = w;
	this->h = h;
	this->channel = channel;
	this->img_ptr = img_ptr;

	is_snaped = false;
	is_calibed = false;
	is_external_alloc = true;
	pixel_per_mm = 0;
}

void ImageData::freeBuffer()
{
	if (!is_external_alloc)
	{
		POSAFE_DELETE_ARRAY(img_ptr);
	}
	
	img_ptr = NULL;
	buffer_size = 0;
	releaseImage();
}

void ImageData::releaseImage()
{
	w = 0;
	h = 0;
	channel = 0;

	time_stamp = 0;
	is_snaped = false;
	is_calibed = false;
	pixel_per_mm = 0;
	is_external_alloc = false;
}

void ImageData::update(i32 dw, i32 dh, i32 dchannel)
{
	this->w = dw;
	this->h = dh;
	this->channel = dchannel;
}

void ImageData::setSnapImage(bool is_snap)
{
	this->is_snaped = is_snap;
}

void ImageData::setTimeStamp(i64 tm_stamp)
{
	this->time_stamp = tm_stamp;
}

bool ImageData::isValid() const
{
	if (w*h*channel <= 0 || img_ptr == NULL)
	{
		return false;
	}
	return true;
}

bool ImageData::isSnapImage() const
{
	return is_snaped;
}

Img ImageData::toImg() const
{
	return Img(img_ptr, w, h, channel); 
}

ImgPart ImageData::toImgPart() const
{ 
	return ImgPart(img_ptr, w, h, channel, (is_snaped ? kImgPartSanpped : kImgPartNone));
}

ImgExpr ImageData::toImgExpr() const
{
	return ImgExpr(img_ptr, w, h, channel); 
}

i32 ImageData::memSize()
{
	i32 len = 0;
	len += sizeof(w);
	len += sizeof(h);
	len += sizeof(channel);
	if (!img_ptr || w*h*channel <= 0)
	{
		return len;
	}

	len += sizeof(time_stamp);
	len += sizeof(is_snaped);
	len += sizeof(is_calibed);
	len += sizeof(pixel_per_mm);
	len += w*h*channel;
	return len;
}

i32 ImageData::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;
	CPOBase::memRead(w, buffer_ptr, buffer_size);
	CPOBase::memRead(h, buffer_ptr, buffer_size);
	CPOBase::memRead(channel, buffer_ptr, buffer_size);
	if (w <= 0 || h <= 0 || channel <= 0)
	{
		return buffer_ptr - buffer_pos;
	}

	CPOBase::memRead(time_stamp, buffer_ptr, buffer_size);
	CPOBase::memRead(is_snaped, buffer_ptr, buffer_size);
	CPOBase::memRead(is_calibed, buffer_ptr, buffer_size);
	CPOBase::memRead(pixel_per_mm, buffer_ptr, buffer_size);

	initBuffer(w, h, channel);
	CPOBase::memRead(img_ptr, w*h*channel, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 ImageData::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;
	if (!isValid())
	{
		i32 nw = 0, nh = 0, nchannel = 0;
		CPOBase::memWrite(nw, buffer_ptr, buffer_size);
		CPOBase::memWrite(nh, buffer_ptr, buffer_size);
		CPOBase::memWrite(nchannel, buffer_ptr, buffer_size);
		return buffer_ptr - buffer_pos;
	}

	CPOBase::memWrite(w, buffer_ptr, buffer_size);
	CPOBase::memWrite(h, buffer_ptr, buffer_size);
	CPOBase::memWrite(channel, buffer_ptr, buffer_size);
	CPOBase::memWrite(time_stamp, buffer_ptr, buffer_size);
	CPOBase::memWrite(is_snaped, buffer_ptr, buffer_size);
	CPOBase::memWrite(is_calibed, buffer_ptr, buffer_size);
	CPOBase::memWrite(pixel_per_mm, buffer_ptr, buffer_size);
	CPOBase::memWrite(img_ptr, w*h*channel, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 ImageData::memImgSize()
{
	i32 len = 0;
	len += sizeof(w);
	len += sizeof(h);
	len += sizeof(channel);
	if (!img_ptr || w*h*channel <= 0)
	{
		return len;
	}

	len += w*h*channel;
	return len;
}

i32 ImageData::memImgWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;
	if (!isValid())
	{
		i32 nw = 0, nh = 0, nchannel = 0;
		CPOBase::memWrite(nw, buffer_ptr, buffer_size);
		CPOBase::memWrite(nh, buffer_ptr, buffer_size);
		CPOBase::memWrite(nchannel, buffer_ptr, buffer_size);
		return buffer_ptr - buffer_pos;
	}

	CPOBase::memWrite(w, buffer_ptr, buffer_size);
	CPOBase::memWrite(h, buffer_ptr, buffer_size);
	CPOBase::memWrite(channel, buffer_ptr, buffer_size);
	CPOBase::memWrite(img_ptr, w*h*channel, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool ImageData::isLessBuffer(ImageData* img_data_ptr)
{
	if (!img_data_ptr)
	{
		return false;
	}
	return isLessBuffer(img_data_ptr[0]);
}

bool ImageData::isLessBuffer(const ImageData& img_data)
{
	i32 nw = img_data.w;
	i32 nh = img_data.h;
	i32 nchannel = img_data.channel;
	return (nw*nh*nchannel > buffer_size);
}

bool ImageData::isSameBuffer(ImageData* img_data_ptr)
{
	if (!img_data_ptr)
	{
		return false;
	}
	return isSameBuffer(img_data_ptr[0]);
}

bool ImageData::isSameBuffer(const ImageData& img_data)
{
	i32 nw = img_data.w;
	i32 nh = img_data.h;
	i32 nchannel = img_data.channel;
	return ((nw*nh*nchannel == buffer_size) && nw*nh*nchannel > 0);
}

bool ImageData::isSameFormat(ImageData* img_data_ptr)
{
	if (!img_data_ptr)
	{
		return false;
	}
	return isSameFormat(img_data_ptr[0]);
}

bool ImageData::isSameFormat(const ImageData& img_data)
{
	i32 nw = img_data.w;
	i32 nh = img_data.h;
	i32 nchannel = img_data.channel;
	return (w == nw && h == nh && channel == nchannel);
}

bool ImageData::isSameSize(const ImageData& img_data)
{
	i32 nw = img_data.w;
	i32 nh = img_data.h;
	return (w == nw && h == nh);
}

void ImageData::copyImage(u8* src_img_ptr, i32 nw, i32 nh, i32 nchannel)
{
	if (!src_img_ptr || nw*nh*nchannel <= 0)
	{
		return;
	}

	if (buffer_size < nw*nh*nchannel)
	{
		freeBuffer();
		initBuffer(nw, nh, nchannel);
	}

	w = nw;
	h = nh;
	channel = nchannel;
	CPOBase::memCopy(img_ptr, src_img_ptr, w*h*channel);
}

void ImageData::copyImage(ImageData* img_data_ptr)
{
	if (!img_data_ptr || !img_data_ptr->isValid())
	{
		return;
	}

	initBuffer(img_data_ptr);

	w = img_data_ptr->w;
	h = img_data_ptr->h;
	channel = img_data_ptr->channel;
	copyImageInfo(img_data_ptr);
	CPOBase::memCopy(img_ptr, img_data_ptr->img_ptr, w*h*channel);
}

void ImageData::copyImage(const ImageData& img_data)
{
	if (!img_data.isValid() || &img_data == this)
	{
		return;
	}

	initBuffer(img_data);

	w = img_data.w;
	h = img_data.h;
	channel = img_data.channel;
	copyImageInfo(&img_data);
	CPOBase::memCopy(img_ptr, img_data.img_ptr, w*h*channel);
}

void ImageData::copyImage(const ImageData& img_data, i32 pw, i32 ph)
{
#if defined(POR_DEVICE)
	if (!img_data.isValid() || pw * ph <= 0)
	{
		return;
	}
	if (img_data.w == pw && img_data.h == ph)
	{
		copyImage(img_data);
	}
	else
	{
		initBuffer(pw, ph, img_data.channel);
		copyImageInfo(&img_data);
		CImageProc::makePaddingImage(img_ptr, img_data.img_ptr,
								pw, ph, img_data.w, img_data.h, img_data.channel);
	}
#endif
}

void ImageData::copyImageInfo(const ImageData* img_data_ptr)
{
	if (!img_data_ptr)
	{
		return;
	}
	is_snaped = img_data_ptr->is_snaped;
	is_calibed = img_data_ptr->is_calibed;
	pixel_per_mm = img_data_ptr->pixel_per_mm;
	time_stamp = img_data_ptr->time_stamp;
	reserved = img_data_ptr->reserved;
}

void ImageData::copyToImage(ImageData* img_data_ptr, i32 out_channel)
{
	if (!isValid() || !img_data_ptr || img_data_ptr == this)
	{
		return;
	}

	out_channel = out_channel > 0 ? out_channel : channel;
	img_data_ptr->initInternalBuffer(w, h, out_channel);
	if (out_channel == channel)
	{
		CPOBase::memCopy(img_data_ptr->img_ptr, img_ptr, w*h*channel);
		return;
	}

#if defined(POR_DEVICE)
	switch (out_channel)
	{
		case kPOGrayChannels: //Convert to Grayscale
		{
			CImageProc::convertColor(this, img_data_ptr, kPOColorCvt2Gray);
			break;
		}
		case kPORGBChannels: //Convert to RGB
		{
			CImageProc::convertColor(this, img_data_ptr, kPOColorCvt2RGB);
			break;
		}
	}
#endif
}

void ImageData::copyToImage(ImageData& img_data, i32 out_channel)
{
	copyToImage(&img_data, out_channel);
}

void ImageData::copyToImage(u8* dst_img_ptr, i32& dw, i32& dh, i32& dchannel, i32 out_channel)
{
	if (!dst_img_ptr || !isValid())
	{
		return;
	}

	ImageData dst_img_data(dst_img_ptr, w, h, dchannel);
	copyToImage(dst_img_data, out_channel);

	dw = dst_img_data.w;
	dh = dst_img_data.h;
	dchannel = dst_img_data.channel;
}

void ImageData::convertToGray()
{
#if defined(POR_DEVICE)
	CImageProc::convertColor(*this, *this, kPOColorCvt2Gray);
#endif
}

void ImageData::reduceImage(ImageData& img_data, i32 nw, i32 nh)
{
#if defined(POR_DEVICE)
	CImageProc::makeThumbImage(img_data, nw, nh, *this);
#endif
}

void ImageData::cropImage(i32 max_width, i32 max_height)
{
#if defined(POR_DEVICE)
	CImageProc::cropImage(*this, vector2di(max_width, max_height));
#endif
}