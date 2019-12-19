#include "connected_components.h"
#include "image_proc.h"
#include "base.h"

//#define POR_TESTMODE

//////////////////////////////////////////////////////////////////////////
ConnComp::ConnComp()
{
	area = 0;
	contour_count = 0;
	left = 0; top = 0; width = 0; height = 0;
	center = vector2df(0, 0);

	segment_vec.clear();
	pixel_vec.clear();

	index = 0;
	is_valid = false;

	orient_width = 0;
	orient_height = 0;
	orient_angle = 0;
	rectangularity = 0;
	circularity = 0;
	compactness = 0;
	border_pixels = 0;
}

ConnComp::~ConnComp()
{
	segment_vec.clear();
	pixel_vec.clear();
}

void ConnComp::updateBlobFeatures()
{
	if (!is_valid || !CPOBase::isPositive(contour_count))
	{
		//return, if connect component is invalid
		return;
	}

	std::vector<cv::Point2f> pt_vec;
	std::vector<cv::Point2f> approx_pt_vec;
	std::vector<cv::Point2f> hull;
	i32 i, count = (i32)segment_vec[1];
	pt_vec.resize(count);
	for (i = 0; i < count; i++)
	{
		Pixelf& tmp = pixel_vec[i];
		pt_vec[i] = cv::Point2f(tmp.x, tmp.y);
	}

	cv::approxPolyDP(pt_vec, approx_pt_vec, 2.0f, true);
	cv::convexHull(approx_pt_vec, hull);
	f32 perimeter = cv::arcLength(approx_pt_vec, true);
	f32 hull_perimeter = cv::arcLength(hull, true);
	cv::RotatedRect rot_rect = cv::minAreaRect(approx_pt_vec);
	
	orient_width = rot_rect.size.width;
	orient_height = rot_rect.size.height;
	orient_angle = CPOBase::degToRad(rot_rect.angle);
	rectangularity = po::_min(1.0f, (f32)area / (orient_width*orient_height));
	circularity = po::_min(1.0f, (f32)(4 * PO_PI*area) / (hull_perimeter*hull_perimeter));
	compactness = po::_min(1.0f, (f32)(4 * PO_PI*area) / (perimeter*perimeter));
}

//////////////////////////////////////////////////////////////////////////
CConnectedComponents::CConnectedComponents()
{
	m_tmp_size = 0;
	m_tmp_pixel_count = 0;
	m_tmp_pixel_ptr = NULL;
	m_tmp_edge_closed = false;
}

CConnectedComponents::~CConnectedComponents()
{
	freeBuffer();
}

void CConnectedComponents::checkBuffer(i32 w, i32 h)
{
	i32 size = w*h / 2;
	if (m_tmp_size < size)
	{
		POSAFE_DELETE_ARRAY(m_tmp_pixel_ptr);
		m_tmp_pixel_ptr = po_new Pixelf[size];
		m_tmp_size = size;
	}
	m_tmp_pixel_count = 0;
}

void CConnectedComponents::freeBuffer()
{
	m_tmp_size = 0;
	POSAFE_DELETE_ARRAY(m_tmp_pixel_ptr);
}

ConnComp* CConnectedComponents::getConnectedComponents(u8* img_ptr, u16* idx_img_ptr, i32 w, i32 h, i32& count)
{
	cv::Mat cv_img(h, w, CV_8UC1, img_ptr);
	cv::Mat cv_label_img(h, w, CV_16UC1, idx_img_ptr);
	cv::Mat blob_stat_vec, center_vec;

#ifdef POR_SUPPORT_TIOPENCV
    count = myConnectedComponentsWithStats(cv_img, cv_label_img, blob_stat_vec, center_vec, 8, CV_16U);
#else
    count = cv::connectedComponentsWithStats(cv_img, cv_label_img, blob_stat_vec, center_vec, 8, CV_16U);
#endif

	if (count <= 1)
	{
		return NULL;
	}

	count = count - 1; //index:0 - background
	ConnComp* tmp_cc_ptr;
	ConnComp* cc_ptr = po_new ConnComp[count];

	i32 i, ni;
	for (i = 0; i < count; i++)
	{
		ni = i + 1;
		tmp_cc_ptr = cc_ptr + i;

#ifdef POR_SUPPORT_TIOPENCV
        tmp_cc_ptr->area = blob_stat_vec.at<i32>(ni, MY_CC_STAT_AREA);
        tmp_cc_ptr->left = blob_stat_vec.at<i32>(ni, MY_CC_STAT_LEFT);
        tmp_cc_ptr->top = blob_stat_vec.at<i32>(ni, MY_CC_STAT_TOP);
        tmp_cc_ptr->width = blob_stat_vec.at<i32>(ni, MY_CC_STAT_WIDTH);
        tmp_cc_ptr->height = blob_stat_vec.at<i32>(ni, MY_CC_STAT_HEIGHT);
        tmp_cc_ptr->center = vector2df(center_vec.at<f64>(ni, 0), center_vec.at<f64>(ni, 1));
#else
		tmp_cc_ptr->area = blob_stat_vec.at<i32>(ni, cv::CC_STAT_AREA);
		tmp_cc_ptr->left = blob_stat_vec.at<i32>(ni, cv::CC_STAT_LEFT);
		tmp_cc_ptr->top = blob_stat_vec.at<i32>(ni, cv::CC_STAT_TOP);
		tmp_cc_ptr->width = blob_stat_vec.at<i32>(ni, cv::CC_STAT_WIDTH);
		tmp_cc_ptr->height = blob_stat_vec.at<i32>(ni, cv::CC_STAT_HEIGHT);
        tmp_cc_ptr->center = vector2df(center_vec.at<f64>(ni, 0), center_vec.at<f64>(ni, 1));
#endif

		tmp_cc_ptr->index = ni;
		tmp_cc_ptr->is_valid = true;
	}
	return cc_ptr;
}

void CConnectedComponents::getConnectedComponentEdge(u8* img_ptr, u16* idx_img_ptr, i32 w, i32 h, ConnComp* cc_ptr, i32 count, i32 flag)
{
	i32 x, y, prev, pixel, hpos;
	i32 ext_size = (w + 2)*(h + 2);
	
	//extend image for padding
	u8* pad_img_ptr = po_new u8[ext_size];
	CImageProc::makePaddingBinary(pad_img_ptr, idx_img_ptr, w, h);
		
	//contour trace
	for (y = 0; y < h; y++)
	{
		prev = 0;
		hpos = y*w;
		for (x = 0; x < w; x++)
		{
			pixel = idx_img_ptr[hpos + x];

			if (prev == kPOBackPixel && pixel < kPOValidPixel && pixel > kPOBackPixel)
			{
				Contourf contour(m_tmp_pixel_ptr);
				CImageProc::fillContourTrace(idx_img_ptr, w, h, pad_img_ptr, x, y, 0, 4, kPOEdgeOutter, &contour);

				m_tmp_pixel_count = contour.getContourPixelNum();
				m_tmp_edge_closed = contour.isClosedContour();

				if (CPOBase::bitCheck(flag, kPOPixelOperOutterEdge))
				{
					if (!CPOBase::bitCheck(flag, kPOPixelOperClosedEdge) || (m_tmp_edge_closed && m_tmp_pixel_count > 4))
					{
						addEdge2ConnectedComponent(cc_ptr + pixel - 1);
					}
				}
				pixel = kPOEdgeOutter;
			}
			else if (prev > kPOBackPixel && prev < kPOValidPixel && pixel == kPOBackPixel)
			{
				Contourf contour(m_tmp_pixel_ptr);
				CImageProc::fillContourTrace(idx_img_ptr, w, h, pad_img_ptr, x - 1, y, 0, 0, kPOEdgeInner, &contour);

				m_tmp_pixel_count = contour.getContourPixelNum();
				m_tmp_edge_closed = contour.isClosedContour();
				if (CPOBase::bitCheck(flag, kPOPixelOperInnerEdge))
				{
					if (!CPOBase::bitCheck(flag, kPOPixelOperClosedEdge) || (m_tmp_edge_closed && m_tmp_pixel_count > 4))
					{
						addEdge2ConnectedComponent(cc_ptr + prev - 1);
					}
				}
			}
			prev = pixel;
		}
	}

	if (CPOBase::bitCheck(flag, kPOPixelOperSubPixel))
	{
		updateSubPixelEdge(img_ptr, w, h, cc_ptr, count);
	}

	testConnectedComponentEdge(cc_ptr, count, w, h);
	POSAFE_DELETE_ARRAY(pad_img_ptr);
}

void CConnectedComponents::updateSubPixelEdge(u8* img_ptr, i32 w, i32 h, ConnComp* cc_ptr, i32 count)
{
	//calc gradent and normal vector
	cv::Mat gradx, grady;
	cv::Mat abs_gradx, abs_grady, grad;
	cv::Mat img_blur(h, w, CV_8UC1, img_ptr);

	cv::Sobel(img_blur, gradx, CV_16S, 1, 0);
	cv::Sobel(img_blur, grady, CV_16S, 0, 1);
	abs_gradx = cv::abs(gradx);
	abs_grady = cv::abs(grady);
	cv::addWeighted(abs_gradx, 0.5, abs_grady, 0.5, 0, grad);
	u16* grad_img_ptr = (u16*)grad.data;

	//make sub-pixeling
	Pixelf* pixel_ptr;
	i32 i, contour_pixels;
	for (i = 0; i < count; i++)
	{
		contour_pixels = (i32)cc_ptr[i].pixel_vec.size();
		pixel_ptr = cc_ptr[i].pixel_vec.data();
		CImageProc::makeSubPixel(grad_img_ptr, w, h, pixel_ptr, contour_pixels);
	}
}

void CConnectedComponents::addEdge2ConnectedComponent(ConnComp* cc_ptr)
{
	if (!cc_ptr->is_valid)
	{
		//return, if connect component is invalid
		return;
	}

	b8vector& is_closed = cc_ptr->is_closed_vec;
	u16vector& segment_vec = cc_ptr->segment_vec;
	ptfvector& pixel_vec = cc_ptr->pixel_vec;

	if (m_tmp_pixel_count > 0)
	{
		i32 st_pos, ed_pos, size;
		size = (i32)segment_vec.size();
		if (size > 0)
		{
			st_pos = segment_vec[size - 1];
		}
		else
		{
			st_pos = 0;
		}
		
		ed_pos = st_pos + m_tmp_pixel_count;
		cc_ptr->contour_count++;

		is_closed.push_back(m_tmp_edge_closed);
		segment_vec.push_back(st_pos);
		segment_vec.push_back(ed_pos);
		pixel_vec.resize(ed_pos);

		Pixelf* pixel_ptr = pixel_vec.data() + st_pos;
		for (i32 i = 0; i < m_tmp_pixel_count; i++)
		{
			pixel_ptr[i] = m_tmp_pixel_ptr[i];
		}
	}
}

void CConnectedComponents::testConnectedComponentEdge(ConnComp* cc_ptr, i32 count, i32 w, i32 h)
{
#if defined(POR_TESTMODE)
	cv::Mat drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::RNG rng(12345);
	cv::Vec3b color;
	ConnComp* tmp_cc_ptr;
	Pixelf* pixel_ptr;

	i32 i, j, k, contour_count;
	i32 stpos, edpos;
	for (i = 0; i < count; i++)
	{
		tmp_cc_ptr = cc_ptr + i;
		if (!tmp_cc_ptr->is_valid)
		{
			continue;
		}

		contour_count = tmp_cc_ptr->contour_count;
		pixel_ptr = tmp_cc_ptr->pixel_vec.data();

		for (j = 0; j < contour_count; j++)
		{
			stpos = tmp_cc_ptr->segment_vec[j * 2];
			edpos = tmp_cc_ptr->segment_vec[j * 2 + 1];
			color = cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			for (k = stpos; k < edpos; k++)
			{
				drawing.at<cv::Vec3b>(cv::Point((i32)(pixel_ptr[k].x+0.5f), (i32)(pixel_ptr[k].y+0.5f))) = color;
			}
		}
	}
	
	cv::imwrite(PO_DEBUG_PATH"conn_comp_test_edge.bmp", drawing);
#endif
}

void CConnectedComponents::updateBlobFeatures(ConnComp* cc_ptr, i32 count)
{
	if (!cc_ptr || count <= 0)
	{
		return;
	}
	for (i32 i = 0; i < count; i++)
	{
		cc_ptr[i].updateBlobFeatures();
	}
}
