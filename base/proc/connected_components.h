#pragma once
#include "define.h"
#include "struct.h"

#ifdef POR_SUPPORT_TIOPENCV
#include "opencv2/opencv.hpp"

enum MyConnectedComponentsTypes
{
    MY_CC_STAT_LEFT = 0, //!< The leftmost (x) coordinate which is the inclusive start of the bounding
    //!< box in the horizontal direction.
    MY_CC_STAT_TOP = 1, //!< The topmost (y) coordinate which is the inclusive start of the bounding
    //!< box in the vertical direction.
    MY_CC_STAT_WIDTH = 2, //!< The horizontal size of the bounding box
    MY_CC_STAT_HEIGHT = 3, //!< The vertical size of the bounding box
    MY_CC_STAT_AREA = 4, //!< The total area (in pixels) of the connected component
    MY_CC_STAT_MAX = 5
};

i32 myConnectedComponents(cv::InputArray _img, cv::OutputArray _labels,
                        i32 connectivity = 8, i32 ltype = CV_32S);

i32 myConnectedComponentsWithStats(cv::InputArray image, cv::OutputArray labels,
                        cv::OutputArray stats, cv::OutputArray centroids,
                        i32 connectivity = 8, i32 ltype = CV_32S);
#endif

struct CCRotatedRect
{
	vector2df				center;
	vector2df				size;
	f32						angle; //rad
};

struct ConnComp
{
public:
	u32						area;
	u16						left;
	u16						top;
	u16						width;
	u16						height;
	vector2df				center;

	//contour information
	u16						contour_count;
	b8vector				is_closed_vec;
	u16vector				segment_vec;
	ptfvector				pixel_vec;
	
	//shape features
	f32						orient_width;
	f32						orient_height;
	f32						orient_angle;
	f32						rectangularity;
	f32						circularity;
	f32						compactness;
	i32						border_pixels;

	//tempoary used
	u16						index;
	bool					is_valid;

public:
	ConnComp();
	~ConnComp();

	void					updateBlobFeatures();
};

class CConnectedComponents
{
public:
	CConnectedComponents();
	virtual ~CConnectedComponents();

	void					checkBuffer(i32 w, i32 h);
	void					freeBuffer();

	ConnComp*				getConnectedComponents(u8* img_ptr, u16* idx_img_ptr, i32 w, i32 h, i32& count);
	void					getConnectedComponentEdge(u8* img_ptr, u16* idx_img_ptr, i32 w, i32 h, ConnComp* cc_ptr, i32 count, i32 flag = kPOPixelOperNone);

	void					addEdge2ConnectedComponent(ConnComp* cc_ptr);
	void					updateSubPixelEdge(u8* img_ptr, i32 w, i32 h, ConnComp* cc_ptr, i32 count);
	void					updateBlobFeatures(ConnComp* cc_ptr, i32 count);

private:
	void					testConnectedComponentEdge(ConnComp* cc_ptr, i32 count, i32 w, i32 h);

public:
	i32						m_tmp_size;
	i32						m_tmp_pixel_count;
	Pixelf*					m_tmp_pixel_ptr;
	bool					m_tmp_edge_closed;
};
