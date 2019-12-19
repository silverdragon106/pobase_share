#include "find_circlegrid.h"
#include "base.h"
#include "proc/image_proc.h"
#include "proc/fit_shape.h"
#include "struct/camera_calib.h"

//#define POR_TESTMODE

//////////////////////////////////////////////////////////////////////////
CFindCircleGrid::CFindCircleGrid()
{
}

CFindCircleGrid::~CFindCircleGrid()
{
}

bool CFindCircleGrid::findCorners(const ImageData* img_data_ptr, CalibBoard* calib_board_ptr)
{
	if (!img_data_ptr || !calib_board_ptr)
	{
		return false;
	}
	return findCornersWithAnchor(img_data_ptr, NULL, calib_board_ptr);
}

bool CFindCircleGrid::findCornersWithAnchor(const ImageData* img_data_ptr,
										CameraCalib* calib_param_ptr, CalibBoard* calib_board_ptr)
{
#if defined(POR_TESTMODE)
	CImageProc::saveImgOpenCV(PO_LOG_PATH"calib.bmp", img_data_ptr->img_ptr, img_data_ptr->w, img_data_ptr->h);
#endif

	f32 lowth = 0.65f;	// 1/(1.25^2)
	f32 highth = 1.55f;	// 1.25^2
	f32 anchrate = 1.0f;
	bool is_anchor = false;
	bool is_anchor_found = false;

	if (!img_data_ptr || !calib_board_ptr)
	{
		printlog_lv1("Input image and param is invalid in find circle corner.");
		return false;
	}

	CameraCalib cam_calib_param;
	if (calib_param_ptr)
	{
		cam_calib_param = calib_param_ptr->getValue();
		if (cam_calib_param.m_calib_board_type == kCalibAnchorCircleGrid)
		{
			f32 rate = cam_calib_param.u.anchorgrid_calib.anchor_diameter_ratio;
			anchrate = rate*rate - 1.0f / (rate*rate);
			is_anchor = true;
		}
	}

	i32 w = img_data_ptr->w;
	i32 h = img_data_ptr->h;
	u8* img_ptr = img_data_ptr->img_ptr;
	u8* new_img_ptr = po_new u8[w*h];
	u16* idx_img_ptr = po_new u16[w*h];

	//gaussian blur for statified process
	cv::Mat cv_img(h, w, CV_8UC1, img_ptr);
	cv::Mat cv_img_blur(h, w, CV_8UC1, new_img_ptr);
	cv::GaussianBlur(cv_img, cv_img_blur, cv::Size(3, 3), 0);

	//get all connected components from image
	i32 i, count = 0, cc_count = 0;
	ConnComp* cc_ptr = NULL;

	//threshold image and extract all connected-components...
	CImageProc::threshold(new_img_ptr, w, h, PO_THRESH_FITTING | PO_THRESH_BINARY_INV);
	cc_ptr = m_conn_comp.getConnectedComponents(new_img_ptr, idx_img_ptr, w, h, count);
	if (!cc_ptr || count <= 1)
	{
		POSAFE_DELETE_ARRAY(new_img_ptr);
		POSAFE_DELETE_ARRAY(idx_img_ptr);
		printlog_lv1(QString("ConnectComponents count[%1] invalid.").arg(count));
		return false;
	}
	
	//get edge image from connected components
	m_conn_comp.checkBuffer(w, h);
	m_conn_comp.getConnectedComponentEdge(new_img_ptr, idx_img_ptr, w, h, cc_ptr, count,
					kPOPixelOperSubPixel | kPOPixelOperClosedEdge |
					(cam_calib_param.m_calib_board_type == kCalibAnchorCircleGrid ? KPOPixelOperAllEdge : kPOPixelOperOutterEdge));

#if defined(POR_TESTMODE)
	cv::Vec3b color(0, 0, 255);
	cv::Mat draw_label(h, w, CV_16U, idx_img_ptr);
	cv::Mat drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::imwrite(PO_LOG_PATH"calibgrid_label.bmp", draw_label);
#endif

	//find peak connect component area with bounding area
	f32vector cc_square;
	f32vector ccratio;
	f32vector cchist;
	cc_square.resize(count);
	ccratio.resize(count);
	cchist.resize(count);

	i32 x1, y1, x2, y2;
	i32 thinterval = 7, hcount = 0;
	for (i = 0; i < count; i++)
	{
		x1 = cc_ptr[i].left;
		y1 = cc_ptr[i].top;
		x2 = cc_ptr[i].left + cc_ptr[i].width;
		y2 = cc_ptr[i].top + cc_ptr[i].height;
		if (x1 < thinterval || y1 < thinterval || x2 > w - thinterval || y2 > h - thinterval)
		{
			cc_ptr[i].is_valid = false;
			continue;
		}

		cc_square[i] = cc_ptr[i].area;
		ccratio[i] = CPOBase::ratio(cc_ptr[i].width, cc_ptr[i].height);
		if (ccratio[i] < 0.75f || cc_square[i] < 10.0f)
		{
			cc_ptr[i].is_valid = false;
			continue;
		}
		cchist[hcount++] = cc_square[i];
	}

	f32 ccth = CPOBase::getPeakValueFromHist(cchist.data(), hcount, 25, 0.8, 1.2);

	hcount = 0;
	for (i = 0; i < count; i++)
	{
		cc_count = cc_ptr[i].contour_count;
		if (cc_count == 1)
		{
			if (cc_square[i] < ccth* lowth || cc_square[i] > ccth* highth)
			{
				cc_ptr[i].is_valid = false;
				continue;
			}
		}
		else if (cc_count == 2)
		{
			if (!calib_param_ptr || cc_square[i] < ccth*anchrate* lowth || cc_square[i] > ccth*anchrate* highth)
			{
				cc_ptr[i].is_valid = false;
				continue;
			}
		}
		else
		{
			cc_ptr[i].is_valid = false;
			continue;
		}
		hcount++;
	}

	//check corner candidate points count
	if (hcount < 5)
	{
		POSAFE_DELETE_ARRAY(cc_ptr);
		POSAFE_DELETE_ARRAY(new_img_ptr);
		POSAFE_DELETE_ARRAY(idx_img_ptr);
		printlog_lv1(QString("The count of corner point is less[%1].").arg(hcount));
		return false;
	}

	//fit all point with ellipse
	i32 j, founds;
	f32 cc_ratio = 1.0f;
	f32 cc_radius_th = std::sqrt(ccth) / 2;
	f32 r[2];
	vector2df pt[2];
	FittedEllipse fit_ellipse;

	CornerVector& corner_vec = calib_board_ptr->corner_vec;
	corner_vec.clear();

	if (cam_calib_param.m_calib_board_type == kCalibAnchorCircleGrid)
	{
		cc_ratio = cam_calib_param.u.anchorgrid_calib.anchor_diameter_ratio;
		cc_ratio = 1.0f / (cc_ratio*cc_ratio);
	}

	for (i = 0; i < count; i++)
	{
		if (!cc_ptr[i].is_valid)
		{
			continue;
		}

		founds = 0;
		cc_ptr[i].is_valid = false;
		cc_count = cc_ptr[i].contour_count;
		b8vector& is_closed_vec = cc_ptr[i].is_closed_vec;
		u16vector& segment_vec = cc_ptr[i].segment_vec;
		ptfvector& contour_pixel_vec = cc_ptr[i].pixel_vec;
		Pixelf* contour_pixel_ptr;

		for (j = 0; j < cc_count; j++)
		{
			contour_pixel_ptr = (Pixelf*)contour_pixel_vec.data();
			if (!is_closed_vec[j] ||
				CFitShapef::fitEllipse(contour_pixel_ptr, segment_vec[j*2], segment_vec[j*2 + 1], fit_ellipse) > 1.0f)
			{
				continue;
			}
			if (CPOBase::ratio(fit_ellipse.r1, fit_ellipse.r2) < 0.4f)
			{
				continue;
			}

			r[founds] = (fit_ellipse.r1 + fit_ellipse.r2) / 2;
			pt[founds] = fit_ellipse.center;
			if (++founds >= 2)
			{
				break;
			}

#if defined(POR_TESTMODE)
			cv::ellipse(drawing, cv::Point((i32)fit_ellipse.center.x, (i32)fit_ellipse.center.y),
				cv::Size(po::_max(1, (i32)fit_ellipse.r1), po::_max(1, (i32)fit_ellipse.r2)),
				CPOBase::radToDeg(fit_ellipse.an), 0, 360, color);
#endif
		}
		if (founds != cc_count)
		{
			continue;
		}

		if (founds == 1 && r[0] >= cc_radius_th*lowth && r[0] <= cc_radius_th*highth)
		{
			//grid point
			corner_vec.push_back(CornerPoint(pt[0]));
		}
		else if (founds == 2 && calib_param_ptr)
		{
			//anchor point
			f32 rtmp = (r[0] + r[1]) / 2;
			f32 ratio = CPOBase::ratio(r[0], r[1]);
			if (rtmp >= cc_radius_th*lowth && rtmp <= cc_radius_th*highth)
			{
				if (ratio > 0.7f*cc_ratio && ratio < 1.25f*cc_ratio)
				{
					if (is_anchor_found && is_anchor)
					{
						POSAFE_DELETE_ARRAY(cc_ptr);
						POSAFE_DELETE_ARRAY(new_img_ptr);
						POSAFE_DELETE_ARRAY(idx_img_ptr);

						return false;
					}
					corner_vec.push_back(CornerPoint(pt[0]));
					calib_board_ptr->anchor_point = CornerPoint(pt[0]);
					is_anchor_found = true;
				}
			}
		}
	}

#if defined(POR_TESTMODE)
	cv::imwrite(PO_LOG_PATH"CalibGrid_FitEllipse.bmp", drawing);
#endif

	//free buffer
	POSAFE_DELETE_ARRAY(cc_ptr);
	POSAFE_DELETE_ARRAY(new_img_ptr);
	POSAFE_DELETE_ARRAY(idx_img_ptr);
	
	if (calib_board_ptr->corner_vec.size() < 5 || (is_anchor && !is_anchor_found))
	{
		calib_board_ptr->corner_vec.clear();
		printlog_lv1("Corner count is small or can't found anchor point.");
		return false;
	}
	return true;
}