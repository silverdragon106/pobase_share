#include "find_chessboard.h"
#include "struct/camera_calib.h"
#include "base.h"
#include <opencv2/opencv.hpp>

//#define POR_TESTMODE

CFindChessborad::CFindChessborad()
{
}

CFindChessborad::~CFindChessborad()
{
}

bool CFindChessborad::findCorners(const ImageData* img_data_ptr, CalibBoard* calib_board_ptr)
{
	if (img_data_ptr->w == 0 || img_data_ptr->h == 0 || img_data_ptr->img_ptr == 0)
	{
		return false;
	}

	i32 w = img_data_ptr->w;
	i32 h = img_data_ptr->h;
	f32 scale = w / 1024.0f;
	if (scale < 1) scale = 1;

	// Compose Gray Image
	cv::Mat src(h, w, CV_8UC1);
	memcpy(src.data, img_data_ptr->img_ptr, w*h);
	cv::Mat gray = src.clone();

	// Equalize Lighting - Remove Highlights
	f64 vmin;
	cv::Mat blurred;
	cv::blur(gray, blurred, cv::Size(201, 201));
	blurred = cv::Scalar::all(255) - blurred;
	cv::minMaxLoc(blurred, &vmin, 0);
	blurred -= vmin;
	gray += blurred;

	// Binary & Extract Rectangles
	i32 ksize = (i32)(3 * scale) * 2 + 1;
	cv::threshold(gray, gray, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::medianBlur(gray, gray, ksize - 2);
	cv::dilate(gray, gray, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize)));
	cv::erode(gray, gray, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize - 2, ksize - 2)));

	// Invert for contour detection - assumed that black rectangle with white background
	gray = cv::Scalar::all(255) - gray;

#if defined(POR_TESTMODE)
	cv::Mat g;
	cv::resize(gray, g, cv::Size(gray.cols / 2, gray.rows / 2));
	cv::imshow("Gray", gray);
#endif

	// Find Contours of Rectangles
	std::vector<cv::Vec4i> hierachy;
	std::vector< std::vector<cv::Point> > blobs;
	cv::findContours(gray.clone(), blobs, hierachy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

	// No Rectangle, Return it.
	if (blobs.size() == 0)
	{
		return false;
	}

	std::vector<f32> ratios, squares;
	ratios.resize(blobs.size());
	squares.resize(blobs.size());

	static const i32 kMinApproxDistance = 11 * scale;
	for (i32 i = 0; i < blobs.size(); i++)
	{
		cv::approxPolyDP(blobs[i], blobs[i], kMinApproxDistance, true);

		cv::Rect box = cv::boundingRect(blobs[i]);
		ratios[i] = box.height / (f32)box.width;
		squares[i] = sqrt(box.width * box.height);
	}

	f32 ratio = CPOBase::getPeakValueFromHist(ratios.data(), (i32)ratios.size(), 0.1, 0.8, 1.2);
	//float square = CPOBase::GetPeakValueFromHist(squares.data(), (i32)squares.size(), 4, 0.8, 1.2);

#if defined(POR_TESTMODE)
	cv::Mat clrSrc;
	cv::cvtColor(src, clrSrc, CV_GRAY2RGB);
#endif

	// Draw Mask
	f32 tr, ts;
	static const i32 margin = 11 * scale;
	const i32 kMinCornerDistance = 21 * scale;
	calib_board_ptr->corner_vec.clear();
	for (i32 i = 0; i < blobs.size(); i++)
	{
		vector2dd center;
		cv::Rect box = cv::boundingRect(blobs[i]);
		center.x = box.x + box.width / 2;
		center.y = box.y + box.height / 2;

		tr = box.height / (f32)box.width;
		ts = sqrt(box.width * box.height);

		if (hierachy[i][3] != -1)
		{
			continue;
		}
		if (tr < ratio * 0.7 || tr > ratio * 1.3)
		{
			continue;
		}

		//if (ts < square * 0.7 || ts > square * 1.4)
		//	continue;

		if (blobs[i].size() != 4)
		{
			continue;
		}

		// For Chessboard Rectangle
		for (i32 j = 0; j < blobs[i].size(); j++)
		{
			vector2dd pt = vector2dd(blobs[i][j].x, blobs[i][j].y);

			// Find Close Points
			bool is_found = false;
			for (i32 k = 0; k < calib_board_ptr->corner_vec.size(); k++)
			{
				if (abs(calib_board_ptr->corner_vec[k].point.x - pt.x) < kMinCornerDistance
					&& abs(calib_board_ptr->corner_vec[k].point.y - pt.y) < kMinCornerDistance)
				{
					is_found = true;
					calib_board_ptr->corner_vec[k].point = (calib_board_ptr->corner_vec[k].point + pt) / 2;
					break;
				}
			}

			if (is_found)
			{
				continue;
			}

			// Update Corner Point - erode7, dilate5
			vector2dd dir = (pt - center);
			dir.normalize();

			pt = pt + dir * 3;

			if (pt.x < kMinApproxDistance || pt.x > w - kMinApproxDistance)
			{
				continue;
			}
			if (pt.y < kMinApproxDistance || pt.y > h - kMinApproxDistance)
			{
				continue;
			}

			calib_board_ptr->corner_vec.push_back(CornerPoint(pt));
#if defined(POR_TESTMODE)
			cv::circle(clrSrc, cv::Point(pt.x + 0.5, pt.y + 0.5), 3, cv::Scalar(0, 255, 0), -1);
#endif
		}
	}

#if defined(POR_TESTMODE)
	cv::resize(clrSrc, g, cv::Size(gray.cols / 2, gray.rows / 2));
	cv::imshow("Corners", g);
#endif

	if (calib_board_ptr->corner_vec.size() < 5)
	{
		return false;
	}
	return true;
}