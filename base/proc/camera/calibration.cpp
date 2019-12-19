#include "calibration.h"
#include "base.h"
#include "proc/image_proc.h"

#if defined(POR_DEVICE)
	#include <opencv2/opencv.hpp>

  #if defined(POR_WITH_OVX)
	#include "performance/openvx_pool/ovx_graph_pool.h"

	extern OvxContext g_vx_context;
	extern OvxResourcePool g_vx_resource_pool;
  #endif
#endif

CCamCalibration::CCamCalibration()
{
	m_calib_param_ptr = NULL;

#if defined(POR_WITH_OVX)
	m_vx_remap = NULL;
#endif
}

CCamCalibration::~CCamCalibration()
{
	exitInstance();
}

bool CCamCalibration::initInstance()
{
	m_calib_param_ptr = NULL;
	return true;
}

void CCamCalibration::exitInstance()
{
	m_calib_param_ptr = NULL;
	freeUndistortMap();
}

void CCamCalibration::setCalibParam(CameraCalib* calib_param_ptr, i32 max_w, i32 max_h)
{
	POMutexLocker l(m_undist_mutex);
	m_calib_param_ptr = calib_param_ptr;
	updateUndistortMap(max_w, max_h);
}

CameraCalib CCamCalibration::getCalibParam()
{
	if (!m_calib_param_ptr)
	{
		return CameraCalib();
	}
	return m_calib_param_ptr->getValue();
}

bool CCamCalibration::findCorners(const ImageData* img_data_ptr,
							CameraCalib* calib_param_ptr, CalibBoard* calib_board_ptr)
{
	if (!img_data_ptr || !calib_param_ptr || !calib_board_ptr || !img_data_ptr->isValid())
	{
		return false;
	}

	switch (calib_param_ptr->m_calib_board_type)
	{
		case kCalibChessBoard:
		{
			if (!m_chessboard_finder.findCorners(img_data_ptr, calib_board_ptr))
			{
				printlog_lvs2("Find corner is failed in chessboard.", LOG_SCOPE_CAM);
				return false;
			}
			break;
		}
		case kCalibCircleGrid:
		{
			if (!m_circlegrid_finder.findCorners(img_data_ptr, calib_board_ptr))
			{
				printlog_lvs2("Find corner is failed in circle grid.", LOG_SCOPE_CAM);
				return false;
			}
			break;
		}
		case kCalibAnchorCircleGrid:
		{
			if (!m_circlegrid_finder.findCornersWithAnchor(img_data_ptr, calib_param_ptr, calib_board_ptr))
			{
				printlog_lvs2("Find corner is failed in anchor circle grid.", LOG_SCOPE_CAM);
				return false;
			}
			break;
		}
		default:
		{
			return false;
		}
	}
	return true;
}

bool CCamCalibration::buildIndex(const ImageData* img_data_ptr, CalibBoard* calib_board_ptr)
{
	enum
	{
		kLinkUp = 0,
		kLinkLeft,
		kLinkRight,
		kLinkDown,
	};

	struct FCTree
	{
		i32		cnum;		// Count of Closest points
		i32		closest[8];	// Indices of 4 closest points
		f32		dist[8];	// 4 Closest Distances
		i32		link[8];	// Indices of Linked Points
		f32		nd[4];		// What directions of neighbor it has.
		i32		ni[4];		// What directions of neighbor it has.

		bool	is_ok;
		
		i32		ix, iy;
		bool	build;

		FCTree()
		{
			memset(this, 0, sizeof(FCTree));
			ix = iy = -65536;
			nd[0] = nd[1] = nd[2] = nd[3] = -1;
			ni[0] = ni[1] = ni[2] = ni[3] = -1;
			for (i32 i = 0; i < 8; i++)
			{
				closest[i] = -1;
				link[i] = -1;
			}
		}
	};

	CornerVector& corners = calib_board_ptr->corner_vec;
	std::vector<FCTree>	fctree;
	std::vector<vector2dd> points(corners.size());

	i32 i, j, count;
	i32 cnum = 4;	// nearest count
	i32 dnum = 0;	// distance count

	count = (i32)corners.size();
	for (i = 0; i < count; i++)
	{
		points[i] = corners[i].point;
	}

	//////////////////////////////////////////////////////////////////////////
	// Get Averaging distance
	i32 corner_count = (i32)corners.size();
	f32* dist_buff = po_new f32[corner_count * cnum];
	memset(dist_buff, 0, sizeof(f32) * corner_count * cnum);

	i32 indices[9];
	f32 dists[9];
	fctree.resize(corner_count);
	for (i = 0; i < corner_count; i++)
	{
		CPOBase::findNearest(points, points[i], cnum + 1, indices, dists);

		fctree[i].cnum = cnum;
		fctree[i].is_ok = true;
		for (j = 0; j < cnum; j++)
		{
			fctree[i].closest[j] = indices[j + 1];
			fctree[i].dist[j] = sqrt(dists[j + 1]);
			dist_buff[dnum++] = fctree[i].dist[j];
		}
	}
	f32 avg_dist = CPOBase::getPeakValueFromHist(dist_buff, dnum, 2, 0.8, 1.2);

	//////////////////////////////////////////////////////////////////////////
	// Get Averaging Angle
	f32 axis = 0;
	i32 anum = 0;
	for (i = 0; i < corner_count; i++)
	{
		vector2dd pt = points[i];
		for (j = 0; j < fctree[i].cnum; j++)
		{
			// If not enough length, it must be eliminated.
			if (fctree[i].dist[j] < avg_dist*0.8)
			{

			}
			else if (fctree[i].dist[j] > avg_dist*1.2)
			{
				fctree[i].closest[j] = -1;
			}
			else
			{
				//Calculate Angle
				vector2dd ngb = points[fctree[i].closest[j]];
				vector2dd dir = ngb - pt;
				if (abs(dir.x) < abs(dir.y))	// if greater than 45 degree, then get normal vector
				{
					f32 tmp = dir.x;
					dir.x = dir.y;
					dir.y = -tmp;
				}

				if (dir.x < 0)
				{
					dir *= -1;
				}

				f32 an = atan2(dir.x, dir.y);
				axis += an;
				anum++;
			}
		}
	}
	axis /= anum;

	//////////////////////////////////////////////////////////////////////////
	// Finding and Marking Neighbor
	i32 cx = img_data_ptr->w / 2, cy = img_data_ptr->h / 2;
	f32 fx, fy;
	i32 fseed = -1;
	i32 comp[4] = { 1, 2, 4, 8 };
	for (i = 0; i < corner_count; i++)
	{
		i32 nd = 0;
		vector2dd pt = points[i];
		for (j = 0; j < fctree[i].cnum; j++)
		{
			if (fctree[i].closest[j] == -1)
			{
				continue;
			}

			// Get Direction
			vector2dd dir = points[fctree[i].closest[j]] - pt;
			fx = dir.x;	fy = dir.y;

			// Get X-Axis Angle
			if (dir.x < 0)	dir *= -1;
			f32 an = atan2(dir.x, dir.y);

			// if Out of degree(11.25:PI/16, 88.75:PI*7/16), it is a wrong neighbor.
			f32 dan = abs(an - axis);
			f32 ep = 1 / 8.0;
			if (dan > PO_PI_HALF*ep && dan < PO_PI_HALF * (1 - ep))
			{
				fctree[i].closest[j] = -1;
				continue;
			}
			else if (dan > PO_PI_HALF*(1 + ep))
			{
				fctree[i].closest[j] = -1;
				continue;
			}

			// If X Axis
			i32 td = -1;
			if (dan <= PO_PI_HALF * ep)
			{
				// neighbor must be left or right
				if (fx < 0)	td = kLinkLeft;
				else		td = kLinkRight;
			}
			// If Y Axis
			else if (dan >= PO_PI_HALF * (1 - ep))
			{
				dan = abs(PO_PI_HALF - dan);

				// neighbor must be up or down
				if (fy < 0)	td = kLinkUp;
				else		td = kLinkDown;
			}
			else
			{
				// Error.
				fctree[i].is_ok = false;
				break;
			}

			// Empty
			if (fctree[i].nd[td] == -1)
			{
				fctree[i].link[j] = td;
				fctree[i].nd[td] = dan;
				fctree[i].ni[td] = j;
			}
			// If already set.
			else
			{
				// if old one is useless
				if (fctree[i].ni[td] > dan)
				{
					i32 old = fctree[i].ni[td];
					fctree[i].link[old] = -1;

					fctree[i].link[j] = td;
					fctree[i].nd[td] = dan;
					fctree[i].ni[td] = j;
				}
				else // if new one is useless
				{
					fctree[i].link[j] = -1;
					continue;
				}
			}

			nd |= comp[td];
		}

		// Check Valid Neighbor (Only Left and Right(Up and Down) is an error.)
		if ((nd == (comp[kLinkLeft] | comp[kLinkRight])) ||	// Only Left and Right
			(nd == (comp[kLinkUp] | comp[kLinkDown])) ||	// Only Up and Down
			((nd & (nd - 1)) == 0) ||						// Only One direction
			(nd == 0))										// Not Linked
		{
			fctree[i].is_ok = false;
			continue;
		}

		if (fseed == -1)
		{
			if (abs(points[i].x - cx) < cx / 2 && abs(points[i].y - cy) < cy / 2)
			{
				fseed = i;
			}
		}
	}

	if (fseed == -1)
	{
		POSAFE_DELETE_ARRAY(dist_buff);
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	// Indexing
	std::vector<i32> seeds;
	seeds.reserve(corner_count);
	seeds.push_back(fseed);
	fctree[fseed].ix = 0;
	fctree[fseed].iy = 0;
	fctree[fseed].build = true;

	// For calculating coverage
	i32 iminX = 65536, iminY = 65536;

	for (u32 ii = 0; ii < seeds.size(); ii++)
	{
		i = seeds[ii];
		if (!fctree[i].is_ok || !fctree[i].build)
		{
			continue;
		}

		for (j = 0; j < fctree[i].cnum; j++)
		{
			i32 ix, iy;
			i32 idx = fctree[i].closest[j];	// Get Index of neighbor

			switch (fctree[i].link[j])
			{
				case kLinkLeft:
				{
					ix = fctree[i].ix - 1;
					iy = fctree[i].iy;
					break;
				}
				case kLinkRight:
				{
					ix = fctree[i].ix + 1;
					iy = fctree[i].iy;
					break;
				}
				case kLinkUp:
				{
					ix = fctree[i].ix;
					iy = fctree[i].iy - 1;
					break;
				}
				case kLinkDown:
				{
					ix = fctree[i].ix;
					iy = fctree[i].iy + 1;
					break;
				}
				default:
				{
					continue;
				}
			}

			// Indexing Error Checking
			if (fctree[idx].ix == -65536 && fctree[idx].iy == -65536)
			{
				fctree[idx].ix = ix;
				fctree[idx].iy = iy;
			}
			else
			{
				if (fctree[idx].ix != ix || fctree[idx].iy != iy)
				{
					return false;
				}
			}

			if (!fctree[idx].build && fctree[idx].is_ok)
			{
				fctree[idx].build = true;
				seeds.push_back(idx);
			}

			if (iminX > ix)	iminX = ix;
			if (iminY > iy)	iminY = iy;

		}
	}

	// Re-indexing
	i32 pnum = 0;
	for (i = 0; i < corner_count; i++)
	{
		if (!fctree[i].is_ok || !fctree[i].build)
		{
			continue;
		}
		fctree[i].ix -= iminX;
		fctree[i].iy -= iminY;
		pnum++;
	}

	// Arrange
	for (i = 0; i < corner_count; i++)
	{
		if (!fctree[i].is_ok || !fctree[i].build)
		{
			continue;
		}
		for (j = i + 1; j < corner_count; j++)
		{
			if (!fctree[j].is_ok || !fctree[i].build)
			{
				continue;
			}
			if (fctree[i].iy > fctree[j].iy)
			{
				CPOBase::swap(fctree[i], fctree[j]);
				CPOBase::swap(points[i], points[j]);

			}
			else if (fctree[i].iy == fctree[j].iy)
			{
				if (fctree[i].ix > fctree[j].ix)
				{
					CPOBase::swap(fctree[i], fctree[j]);
					CPOBase::swap(points[i], points[j]);
				}
			}
		}
	}

	corners.clear();
	for (i = 0; i < corner_count; i++)
	{
		if (!fctree[i].is_ok || !fctree[i].build)
		{
			continue;
		}
		CornerPoint corner(points[i]);
		corner.index_x = fctree[i].ix;
		corner.index_y = fctree[i].iy;

		corners.push_back(corner);
	}

	//calc coverage 
	count = (i32)corners.size();
	cvPointVector2f cpts, hull;
	cpts.resize(count);
	for (i = 0; i < count; i++)
	{
		cpts[i] = cv::Point2f(corners[i].point.x, corners[i].point.y);
	}

	cv::convexHull(cpts, hull);

	i32 w = img_data_ptr->w;
	i32 h = img_data_ptr->h;
	calib_board_ptr->board_coverage = std::abs(cv::contourArea(hull))/(w*h);

	testCalibBoardConvexHull(hull, w, h);
	POSAFE_DELETE_ARRAY(dist_buff);
	return true;
}

bool CCamCalibration::detectBoard(const ImageData* img_data_ptr, CameraCalib* calib_param_ptr, CalibBoard* calib_board_ptr)
{
	if (!img_data_ptr || !calib_board_ptr || !calib_param_ptr)
	{
		return false;
	}

	calib_board_ptr->w = img_data_ptr->w;
	calib_board_ptr->h = img_data_ptr->h;
	calib_board_ptr->snap_type = kCalibNormalSnap;
	calib_board_ptr->board_coverage = 0;

	if (!findCorners(img_data_ptr, calib_param_ptr, calib_board_ptr)
		|| calib_board_ptr->corner_vec.size() < 4 * 4)
	{
		printlog_lvs2("Find corner is failed in detect board process.", LOG_SCOPE_CAM);
		return false;
	}

	if (!buildIndex(img_data_ptr, calib_board_ptr))
	{
		printlog_lvs2("Build index is failed in detect board process.", LOG_SCOPE_CAM);
		return false;
	}

	//check calib board coverage
	if (calib_board_ptr->board_coverage < 0.3f)
	{
		printlog_lvs2(QString("The current board coverage[%1] is small.")
						.arg(calib_board_ptr->board_coverage), LOG_SCOPE_CAM);
		return false;
	}

	printlog_lv1(QString("CalibResult: point count[%1], coverage[%2].")
					.arg(calib_board_ptr->getPointCount()).arg(calib_board_ptr->board_coverage));
	return true;
}

i32 CCamCalibration::calibCamera(const i32 w, const i32 h, CalibBoardVector& board_vec)
{
	if (!m_calib_param_ptr)
	{
		return kCalibReturnNonParam;
	}

	switch (m_calib_param_ptr->getCalibType())
	{
		case kCalibTypeScale:
		{
			f32 sx = m_calib_param_ptr->u.scale_calib.calib_scale;
			return calibCameraScale(sx, sx, w, h);
		}
		case kCalibTypeXYScale:
		{
			f32 sx = m_calib_param_ptr->u.xy_scale_calib.horizontal_scale;
			f32 sy = m_calib_param_ptr->u.xy_scale_calib.vertical_scale;
			return calibCameraScale(sx, sy, w, h);
		}
		case kCalibTypeEdgeToEdge:
		case kCalibTypeXYEdgeToEdge:
		case kCalibTypeCircle:
		case kCalibType9Points:
		{
			return kCalibReturnInvalidParam;
		}
		case kCalibTypeGrid:
		{
			return calibCameraGrid(w, h, board_vec);
		}
		default:
		{
			break;
		}
	}
	return kCalibReturnNonParam;
}

i32 CCamCalibration::calibCameraScale(f64 sx, f64 sy, i32 w, i32 h)
{
	if (sx < PO_EPSILON || sy < PO_EPSILON)
	{
		return kCalibReturnInvalidParam;
	}

	f64 hw = w / 2;
	f64 hh = h / 2;

	m_calib_param_ptr->lock();
	{
		//[mm]->[calibed pixel]
		CPOBase::init3x3(m_calib_param_ptr->m_pose_matrix, 0, 1.0 / sy, hw, hh);

		//[image pixel]->[mm]
		CPOBase::init3x3(m_calib_param_ptr->m_inv_board_matrix, 0, sx, sy, -hw*sx, -hh*sy);

		//[calibed pixel]->[mm]->[image pixel]
		cv::Mat matPose(3, 3, CV_64FC1, m_calib_param_ptr->m_pose_matrix);
		cv::Mat matIPlate(3, 3, CV_64FC1, m_calib_param_ptr->m_inv_board_matrix);
		cv::Mat matIPosePlate(3, 3, CV_64FC1, m_calib_param_ptr->m_inv_pose_board_matrix);
		matIPosePlate = matIPlate.inv() * matPose.inv();

		//update calib
		m_calib_param_ptr->m_calibed_width = w;
		m_calib_param_ptr->m_calibed_height = h;
		m_calib_param_ptr->m_pixel_per_mm = sy;
		m_calib_param_ptr->m_calibed_result |= kCalibResultPixel2MM;
		m_calib_param_ptr->m_calibed_result |= (m_calib_param_ptr->m_calib_type & kCalibResultType);
		CPOBase::getNowTime(m_calib_param_ptr->m_calibed_time);

		//calibrate axis direction
		calibManualAxis();
		calibPivotPosition(w, h);
	}
	m_calib_param_ptr->unlock();
	return kCalibReturnSuccess;
}

i32 CCamCalibration::calibCameraGrid(const i32 w, const i32 h, CalibBoardVector& board_vec)
{
	i32 i, j;
	i32 board_width = 0, board_height = 0;
	i32 point_count, corner_count = PO_MAXINT;
	i32 board_count = (i32)board_vec.size();

	if (board_count <= 0)
	{
		return kCalibReturnLessSnaps;
	}
	else if (board_count < m_calib_param_ptr->getMinCalibSnap())
	{
		m_calib_param_ptr->m_calibed_result |= kCalibResultSnapLess;
	}

	//check all chessboard data
	CornerPoint* corner_ptr;
	CornerPoint* corner_vec_data;
	CalibBoard* board_ptr = NULL;
	CalibBoard* board_vec_ptr = board_vec.data();
	f32 interval = m_calib_param_ptr->u.chessboard_calib.chessboard_interval;

	for (i = 0; i < board_count; i++)
	{
		board_ptr = board_vec_ptr + i;
		board_width = board_ptr->w;
		board_height = board_ptr->h;
		corner_vec_data = board_ptr->corner_vec.data();
		point_count = board_ptr->getPointCount();
		corner_count = po::_min(point_count, corner_count);

		//update mm coord of each corners
		for (j = 0; j < point_count; j++)
		{
			corner_ptr = corner_vec_data + j;
			corner_ptr->mm_x = corner_ptr->index_x * interval;
			corner_ptr->mm_y = corner_ptr->index_y * interval;
		}
	}
	if (corner_count < kCamCalibMinPointCount)
	{
		return kCalibReturnLessCorners;
	}

	std::vector<cvPointVector2f> img_point_vec;
	std::vector<cvPointVector3f> obj_point_vec;

	for (i = 0; i < board_count; i++)
	{
		board_ptr = board_vec.data() + i;
		corner_vec_data = board_ptr->corner_vec.data();
		corner_count = board_ptr->getPointCount();

		cvPointVector2f* img_point_ptr = CPOBase::pushBackNew(img_point_vec);
		cvPointVector3f* obj_point_ptr = CPOBase::pushBackNew(obj_point_vec);
		img_point_ptr->resize(corner_count);
		obj_point_ptr->resize(corner_count);

		for (j = 0; j < corner_count; j++)
		{
			corner_ptr = corner_vec_data + j;
			(*obj_point_ptr)[j] = cv::Point3f(corner_ptr->mm_x, corner_ptr->mm_y, 0);
			(*img_point_ptr)[j] = cv::Point2f(corner_ptr->point.x, corner_ptr->point.y);
		}
	}

	cv::Mat dist_coeffs, cam_matrix;
	std::vector<cv::Mat> rvecs, tvecs;
	cv::calibrateCamera(obj_point_vec, img_point_vec, cv::Size(board_width, board_height), cam_matrix, dist_coeffs, rvecs, tvecs, 0,
					cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON));

	m_calib_param_ptr->lock();
	{
		CPOBase::getNowTime(m_calib_param_ptr->m_calibed_time);
		CPOBase::memCopy(m_calib_param_ptr->m_distort_coeffs, (f64*)dist_coeffs.data, 5);
		CPOBase::memCopy(m_calib_param_ptr->m_cam_matrix, (f64*)cam_matrix.data, 9);

		f64 err = findExactlyPlatePose(board_vec, cam_matrix, dist_coeffs, rvecs, tvecs);
		if (err > m_calib_param_ptr->getCalibThError())
		{
			return kCalibReturnToleranceError;
		}

		m_calib_param_ptr->m_calibed_width = w;
		m_calib_param_ptr->m_calibed_height = h;
		m_calib_param_ptr->m_calibed_result |= kCalibResultPixel2MM;
		m_calib_param_ptr->m_calibed_result |= kCalibResultUndistort;
		m_calib_param_ptr->m_calibed_result |= (m_calib_param_ptr->m_calib_type & kCalibResultType);

		//calibrate axis and anchor position with anchor data
		calibAxis(board_vec);
		calibPivotPosition(board_width, board_height);

		m_undist_mutex.lock();
		updateUndistortMap(board_width, board_height);
		m_undist_mutex.unlock();
	}
	m_calib_param_ptr->unlock();
	return kCalibReturnSuccess;
}

f64 CCamCalibration::findExactlyPlatePose(CalibBoardVector& board_vec, cv::Mat camMatrix, cv::Mat distCoeffs, std::vector<cv::Mat> rvecs, std::vector<cv::Mat> tvecs)
{
	i32 board_count = (i32)board_vec.size();
	if (board_count <= 0)
	{
		return PO_MAXINT;
	}

	i32 i, j, pt_count;
	i32 best_index = -1;
	f64 mat[9];
	f64 px, py, dist;
	f64 avg_dist, sig_dist, max_dist, calc_dist;
	f64 best_avg = 0, best_sig = 0, best_max = 0, best_dist = PO_MAXINT;
	CalibBoard* board_ptr;
	CalibBoard* board_vec_ptr = board_vec.data();
	CornerPoint* corner_ptr;
	CornerPoint* corner_data_ptr;

	//select best board result has minimal error
	for (i = 0; i < board_count; i++)
	{
		board_ptr = board_vec_ptr + i;
		CornerVector& pt_vec = board_ptr->corner_vec;
		pt_count = (i32)pt_vec.size();
		if (pt_count <= 0)
		{
			continue;
		}

		avg_dist = 0;
		sig_dist = 0;
		max_dist = 0;
		corner_data_ptr = pt_vec.data();
		convertToMat(mat, rvecs[i], tvecs[i]);

		for (j = 0; j < pt_count; j++)
		{
			corner_ptr = corner_data_ptr + j;
			getDistortedCoord(corner_ptr->mm_x, corner_ptr->mm_y, px, py, mat);
			dist = CPOBase::distance(px, py, corner_ptr->point.x, corner_ptr->point.y);
			avg_dist += dist;
			sig_dist += dist*dist;
			max_dist = po::_max(dist, max_dist);
		}

		avg_dist /= pt_count;
		sig_dist = sqrt(sig_dist / pt_count - avg_dist*avg_dist);
		calc_dist = avg_dist*0.25 + sig_dist*0.4 + max_dist*0.35; //calibration fitting error function
		if (calc_dist < best_dist)
		{
			best_dist = calc_dist;
			best_index = i;
			best_avg = avg_dist;
			best_max = max_dist;
			best_sig = sig_dist;
		}
	}
	if (best_index < 0)
	{
		return PO_MAXINT;
	}

	//calc similar transform with opencv icvGetRTMatrix function (in lypyramid.cpp)
	board_ptr = board_vec_ptr + best_index;
	CornerVector& pts = board_ptr->corner_vec;
	convertToMat(mat, rvecs[best_index], tvecs[best_index]);

	pt_count = (i32)pts.size();
	corner_data_ptr = pts.data();

	f64 mx, my;
	f64 sa[16], sb[4], m[4];
	cv::Mat A(4, 4, CV_64FC1, sa);
	cv::Mat B(4, 1, CV_64FC1, sb);
	cv::Mat M(4, 1, CV_64FC1, m);

	memset(sa, 0, sizeof(f64) * 16);
	memset(sb, 0, sizeof(f64) * 4);

	for (i = 0; i < pt_count; i++)
	{
		corner_ptr = corner_data_ptr + i;
		mx = corner_ptr->mm_x;
		my = corner_ptr->mm_y;
		px = corner_ptr->point.x;
		py = corner_ptr->point.y;

		sa[0] += mx*mx + my*my;
		sa[1] += 0;
		sa[2] += mx;
		sa[3] += my;

		sa[4] += 0;
		sa[5] += mx*mx + my*my;
		sa[6] += -my;
		sa[7] += mx;

		sa[8] += mx;
		sa[9] += -my;
		sa[10] += 1;
		sa[11] += 0;

		sa[12] += my;
		sa[13] += mx;
		sa[14] += 0;
		sa[15] += 1;

		sb[0] += mx*px + my*py;
		sb[1] += mx*py - my*px;
		sb[2] += px;
		sb[3] += py;
	}
	cv::solve(A, B, M, cv::DECOMP_SVD);

	//it's pose matrix from mm plane to pixel plane
	f64* u = m_calib_param_ptr->m_pose_matrix;
	CPOBase::init3x3(u);
	u[0] = m[0]; u[1] =-m[1]; u[2] = m[2];
	u[3] = m[1]; u[4] = m[0]; u[5] = m[3];

	//calc result calib parameter matrix
	cv::Mat matPlate(3, 3, CV_64FC1, mat);
	cv::Mat matPose(3, 3, CV_64FC1, u);
	cv::Mat matCam(3, 3, CV_64FC1, m_calib_param_ptr->m_cam_matrix);

	cv::Mat matInvCam(3, 3, CV_64FC1, m_calib_param_ptr->m_inv_cam_matrix);
	cv::Mat matInvPlate(3, 3, CV_64FC1, m_calib_param_ptr->m_inv_board_matrix);
	cv::Mat matInvPosePlate(3, 3, CV_64FC1, m_calib_param_ptr->m_inv_pose_board_matrix);

	matInvCam = matCam.inv();
	matInvPlate = matPlate.inv();
	matInvPosePlate = matPlate*matPose.inv();

	//calc scale from mm space to correctly pixel space
	f64 s = sqrt(m[0] * m[0] + m[1] * m[1]);
	m_calib_param_ptr->m_pixel_per_mm = 1.0 / s;

	m_calib_param_ptr->m_calibed_mean_error = best_avg;
	m_calib_param_ptr->m_calibed_sigma_error = best_sig;
	m_calib_param_ptr->m_calibed_max_error = best_max;
	m_calib_param_ptr->m_calibed_coverage = board_ptr->board_coverage;

	testCalibPlatePose(board_vec, best_index);
	return best_max / s;
}

i32 CCamCalibration::calibAxis(CalibBoardVector& boards)
{
	if (m_calib_param_ptr->checkMode(kCalibModeAutoAxis))
	{
		if (calibAutoAxis(boards) == kCalibReturnSuccess)
		{
			return kCalibReturnSuccess;
		}
	}

	calibManualAxis();
	return kCalibReturnSuccess;
}

i32 CCamCalibration::calibAutoAxis(CalibBoardVector& boards)
{
	if (!m_calib_param_ptr)
	{
		return kCalibReturnInvalidParam;
	}

	//auto axis calibration
	i32 i, board_count = (i32)boards.size();
	if (board_count <= 0)
	{
		return kCalibReturnLessSnaps;
	}

	//calib external axis with move to anchor
	bool bxdir = false;
	bool bydir = false;
	bool need_orth_axis = m_calib_param_ptr->checkMode(kCalibModeDecartAxis);

	f64 costh = cosf(CPOBase::degToRad(7));
	f64 sinth = sinf(CPOBase::degToRad(7));
	vector2dd xsdir, ysdir;
	vector2dd xpdir, ypdir;
	std::vector<vector2dd> mm_point_vec;
	std::vector<vector2dd> anch_point_vec;

	for (i = 0; i < board_count; i++)
	{
		CalibBoard& board = boards[i];
		switch (board.snap_type)
		{
		case kCalibAxisXPosSnap:
		case kCalibAxisXNegSnap:
		case kCalibAxisYPosSnap:
		case kCalibAxisYNegSnap:
		{
			if (i <= 0)
			{
				return kCalibReturnDiscontinuslyAnchor;
			}

			CalibBoard& prev_board = boards[i - 1];
			if (!prev_board.isAnchorBoard())
			{
				return kCalibReturnDiscontinuslyAnchor;
			}

			//undistort base point and moving point to mm space
			f64 mbx, mby, mpx, mpy;
			vector2dd anchor1 = prev_board.anchor_point.point;
			vector2dd anchor2 = board.anchor_point.point;
			undistortPoint(anchor1.x, anchor1.y, mbx, mby, kUndistMM);
			undistortPoint(anchor2.x, anchor2.y, mpx, mpy, kUndistMM);

			//calc direction in mm space
			f64 len;
			vector2dd anchdd, mmdd = vector2dd(mpx - mbx, mpy - mby);
			vector2dd normalizedir = mmdd;
			normalizedir.normalize(len);

			switch (board.snap_type)
			{
				case kCalibAxisXNegSnap:
				case kCalibAxisYNegSnap:
				{
					normalizedir = -normalizedir;
					mmdd = -mmdd;
					break;
				}
				}

				//check axis on mm space in multi-axis case
				switch (board.snap_type)
				{
					case kCalibAxisXPosSnap:
					case kCalibAxisXNegSnap:
					{
						if (bxdir && xsdir.dotProduct(normalizedir) < costh)
						{
							return kCalibReturnMultiAnchorXError;
						}
						else
						{
							xsdir = (xsdir + normalizedir).normalize();
							xpdir = (anchor2 - anchor1).normalize();
							bxdir = true;
						}
						break;
					}
					case kCalibAxisYPosSnap:
					case kCalibAxisYNegSnap:
					{
						if (bydir && ysdir.dotProduct(normalizedir) < costh)
						{
							return kCalibReturnMultiAnchorYError;
						}
						else
						{
							ysdir = (ysdir + normalizedir).normalize();
							ypdir = (anchor2 - anchor1).normalize();
							bydir = true;
						}
						break;
					}
					default:
						break;
				}

				if (!need_orth_axis)
				{
					//new axis on anchor mm space
					switch (board.snap_type)
					{
						case kCalibAxisXPosSnap:
						case kCalibAxisXNegSnap:
						{
							anchdd = vector2dd(len, 0);
							break;
						}
						case kCalibAxisYPosSnap:
						case kCalibAxisYNegSnap:
						{
							anchdd = vector2dd(0, len);
							break;
						}
					}

					mm_point_vec.push_back(mmdd);
					anch_point_vec.push_back(anchdd);
				}
				break;
			}
		}
	}
	if (!bxdir || !bydir || std::abs(xsdir.dotProduct(ysdir)) > sinth)
	{
		return kCalibReturnXYAnchorError;
	}

	//calc only rotation with reflection matrix
	f64 m[4];
	if (need_orth_axis)
	{
		//used only decart axis coordnidate
		vector2dd nydir = vector2dd(-xsdir.y, xsdir.x);
		if (nydir.dotProduct(ysdir) > costh)
		{
			ysdir = (nydir + ysdir).normalize();
			xsdir = vector2dd(ysdir.y, -ysdir.x);
		}
		else
		{
			ysdir = (ysdir - nydir).normalize();
			xsdir = vector2dd(-ysdir.y, ysdir.x);
		}
		m[0] = xsdir.x;
		m[1] = xsdir.y;
		m[2] = ysdir.x;
		m[3] = ysdir.y;
	}
	else
	{
		board_count = (i32)mm_point_vec.size();
		f64 dx, dy;
		f64* ma = po_new f64[board_count * 8];
		f64* anb = po_new f64[board_count * 2];
		memset(ma, 0, sizeof(f64)*board_count * 8);

		for (i = 0; i < board_count; i++)
		{
			dx = mm_point_vec[i].x;
			dy = mm_point_vec[i].y;
			ma[i * 8] = dx;
			ma[i * 8 + 1] = dy;
			ma[i * 8 + 6] = dx;
			ma[i * 8 + 7] = dy;

			dx = anch_point_vec[i].x;
			dy = anch_point_vec[i].y;
			anb[i * 2] = dx;
			anb[i * 2 + 1] = dy;
		}

		cv::Mat A((i32)mm_point_vec.size() * 2, 4, CV_64FC1, ma);
		cv::Mat B((i32)anch_point_vec.size() * 2, 1, CV_64FC1, anb);
		cv::Mat M(4, 1, CV_64FC1, m);
		cv::solve(A, B, M, cv::DECOMP_SVD);
	}

	//calc transform matrix from calibed pixel plane to anchor mm plane
	f64 u[9];
	CPOBase::init3x3(u);
	cv::Mat matAnchor(3, 3, CV_64FC1, u);
	cv::Mat matPose(3, 3, CV_64FC1, m_calib_param_ptr->m_pose_matrix);
	cv::Mat matIPoseAnch(3, 3, CV_64FC1, m_calib_param_ptr->m_inv_pose_absolute_matrix);

	u[0] = m[0]; u[1] = m[1];
	u[3] = m[2]; u[4] = m[3];
	matIPoseAnch = matAnchor*matPose.inv();
	m_calib_param_ptr->m_calibed_result |= kCalibResultAutoAxis;

	return kCalibReturnSuccess;
}

i32 CCamCalibration::calibManualAxis()
{
	if (!m_calib_param_ptr)
	{
		return kCalibReturnInvalidParam;
	}

	f64 tmp;
	f64* u = m_calib_param_ptr->m_inv_pose_absolute_matrix;
	CPOBase::init3x3(u);

	//calib axis scale from pose_matrix_ matrix
	f32 pixel2mm = m_calib_param_ptr->m_pixel_per_mm;
	u[0] *= pixel2mm; u[1] *= pixel2mm;
	u[3] *= pixel2mm; u[4] *= pixel2mm;

	//calib axis direction
	if (m_calib_param_ptr->checkMode(kCalibModeAxisSwapXY))
	{
		//swap r1 and r2
		tmp = u[0]; u[0] = u[3]; u[3] = tmp;
		tmp = u[1]; u[1] = u[4]; u[4] = tmp;
		tmp = u[2]; u[2] = u[5]; u[5] = tmp;
	}
	if (m_calib_param_ptr->checkMode(kCalibModeAxisInvertX))
	{
		//invert r1
		u[0] = -u[0];
		u[1] = -u[1];
		u[2] = -u[2];
	}
	if (m_calib_param_ptr->checkMode(kCalibModeAxisInvertY))
	{
		//invert r1
		u[3] = -u[3];
		u[4] = -u[4];
		u[5] = -u[5];
	}

	m_calib_param_ptr->m_calibed_result |= kCalibResultManualAxis;
	return kCalibReturnSuccess;
}

i32 CCamCalibration::calibPivotPosition(const i32 w, const i32 h)
{
	if (!m_calib_param_ptr)
	{
		return kCalibReturnInvalidParam;
	}

	/* 교정된 화상의 중심점을 원점으로 한다. */
	f64 tx, ty;
	f64* ipose_abs_ptr = m_calib_param_ptr->m_inv_pose_absolute_matrix;
	CPOBase::trans2x3(ipose_abs_ptr, (f64)w/2, (f64)h/2, tx, ty);
	ipose_abs_ptr[2] -= tx;
	ipose_abs_ptr[5] -= ty;

	/* m_absolute_pose_matrix을 계산한다. */
	cv::Mat matIPoseAbs(3, 3, CV_64FC1, ipose_abs_ptr);
	cv::Mat matAbsPose(3, 3, CV_64FC1, m_calib_param_ptr->m_absolute_pose_matrix);
	matAbsPose = matIPoseAbs.inv();
	
	m_calib_param_ptr->m_calibed_result |= kCalibResultPivot;
	return kCalibReturnSuccess;
}

void CCamCalibration::getDistortedCoord(f64 mx, f64 my, f64& px, f64& py, f64* tr)
{
	f64 dx, dy, dk, r2, r4, r6;
	f64 a1, a2, a3;
	f64* d = m_calib_param_ptr->m_distort_coeffs;
	f64* c = m_calib_param_ptr->m_cam_matrix;
	CPOBase::perspective2d(tr, mx, my, dx, dy);

	r2 = dx*dx + dy*dy;
	r4 = r2*r2;
	r6 = r4*r2;
	a1 = 2 * dx*dy;
	a2 = r2 + 2 * dx*dx;
	a3 = r2 + 2 * dy*dy;
	dk = (1 + d[0] * r2 + d[1] * r4 + d[4] * r6);

	dx = dx*dk + d[2] * a1 + d[3] * a2;
	dy = dy*dk + d[2] * a3 + d[3] * a1;
	CPOBase::trans2x3(c, dx, dy, px, py);
}

void CCamCalibration::getDistortedCoord(f64 px, f64 py, f64& dx, f64& dy)
{
	f64 dk, r2, r4, r6;
	f64 a1, a2, a3;
	f64* d = m_calib_param_ptr->m_distort_coeffs;
	f64* c = m_calib_param_ptr->m_cam_matrix;
	f64* inv_pb = m_calib_param_ptr->m_inv_pose_board_matrix;

	CPOBase::perspective2d(inv_pb, px, py, dx, dy);

	r2 = dx*dx + dy*dy;
	r4 = r2*r2;
	r6 = r4*r2;
	a1 = 2 * dx*dy;
	a2 = r2 + 2 * dx*dx;
	a3 = r2 + 2 * dy*dy;
	dk = (1 + d[0] * r2 + d[1] * r4 + d[4] * r6);

	px = dx*dk + d[2] * a1 + d[3] * a2;
	py = dy*dk + d[2] * a3 + d[3] * a1;
	CPOBase::trans2x3(c, px, py, dx, dy);
}

void CCamCalibration::convertToMat(f64* m9, cv::Mat& rvec, cv::Mat& tvec)
{
	f64* r = (f64*)rvec.data;
	f64* t = (f64*)tvec.data;

	convRodriguesToMat(r, m9);
	m9[2] = t[0];
	m9[5] = t[1];
	m9[8] = t[2];
}

void CCamCalibration::convRodriguesToMat(f64* r3, f64* r9)
{
	cv::Mat vecRot(3, 1, CV_64FC1, r3);
	cv::Mat matRot(3, 3, CV_64FC1, r9);
	cv::Rodrigues(vecRot, matRot);
}

void CCamCalibration::convMatrixToRodrigues(f64* r9, f64* r3)
{
	cv::Mat cv_rod_mat(3, 1, CV_64FC1, r3);
	cv::Mat cv_rot_mat(3, 3, CV_64FC1, r9);
	cv::Rodrigues(cv_rot_mat, cv_rod_mat);
}

bool CCamCalibration::undistortImage(ImageData& img_data)
{
	return undistortImage(img_data.img_ptr, img_data.w, img_data.h, img_data.channel);
}

bool CCamCalibration::undistortImage(u8* img_ptr, i32 w, i32 h, i32 channel)
{
	if (!m_calib_param_ptr || !img_ptr || w*h*channel <= 0)
	{
		return false;
	}

	bool is_calibed = m_calib_param_ptr->canbeCalib();
	if (!is_calibed)
	{
		return false;
	}

	switch (m_calib_param_ptr->getCalibType())
	{
		case kCalibTypeScale:
		{
			break;
		}
		case kCalibTypeXYScale:
		{
			undistortImageWrap(img_ptr, img_ptr, w, h, channel);
			break;
		}
		case kCalibTypeGrid:
		{
			undistortImageGrid(img_ptr, img_ptr, w, h, channel);
			break;
		}
		default:
		{
			return false;
		}
	}
	return true;
}

bool CCamCalibration::undistortImage(ImageData& dst_img, const ImageData& src_img)
{
	dst_img.initBuffer(src_img);
	return undistortImage(dst_img.img_ptr, src_img.img_ptr, src_img.w, src_img.h, src_img.channel);
}

bool CCamCalibration::undistortImage(u8* dst_img_ptr, u8* src_img_ptr, i32 w, i32 h, i32 channel)
{
	if (!m_calib_param_ptr || !dst_img_ptr || !src_img_ptr || w*h*channel <= 0)
	{
		return false;
	}

	bool is_calibed = m_calib_param_ptr->canbeCalib();
	if (!is_calibed)
	{
		CPOBase::memCopy(dst_img_ptr, src_img_ptr, w*h*channel);
		return false;
	}

	memset(dst_img_ptr, 0, w*h*channel);
	switch (m_calib_param_ptr->getCalibType())
	{
		case kCalibTypeScale:
		{
			CPOBase::memCopy(dst_img_ptr, src_img_ptr, w*h*channel);
			break;
		}
		case kCalibTypeXYScale:
		{
			undistortImageWrap(dst_img_ptr, src_img_ptr, w, h, channel);
			break;
		}
		case kCalibTypeGrid:
		{
			undistortImageGrid(dst_img_ptr, src_img_ptr, w, h, channel);
			break;
		}
		default:
		{
			return false;
		}
	}
	return true;
}

void CCamCalibration::undistortImageWrap(u8* dst_img_ptr, u8* src_img_ptr, i32 w, i32 h, i32 channel)
{
	if (!dst_img_ptr || !src_img_ptr || w*h*channel <= 0)
	{
		return;
	}

	if (m_calib_param_ptr->isCalibed())
	{
		bool is_processed = false;

#if defined(POR_WITH_OVX)
		i32 format = OvxHelper::imageFormat(channel);
		vx_context context = g_vx_context.getVxContext();
		OvxResource* ovx_src_ptr = g_vx_resource_pool.fetchImage(context, w, h, format);
		OvxResource* ovx_dst_ptr = g_vx_resource_pool.fetchImage(context, w, h, format);
		OvxResource* ovx_mat_ptr = g_vx_resource_pool.fetchMatrix(context, 2, 3, VX_TYPE_FLOAT32);

		if (ovx_src_ptr && ovx_dst_ptr && ovx_mat_ptr)
		{
			f32 tr[6];
			m_calib_param_ptr->lock();
			{
				f64* inv_pb = m_calib_param_ptr->m_inv_pose_board_matrix;
				CPOBase::transpose(tr, inv_pb, 3, 2);
			}
			m_calib_param_ptr->unlock();

			vx_image vx_src = (vx_image)ovx_src_ptr->m_resource;
			vx_image vx_dst = (vx_image)ovx_dst_ptr->m_resource;
			vx_matrix vx_mat = (vx_matrix)ovx_mat_ptr->m_resource;
			OvxHelper::writeImage(vx_src, src_img_ptr, w, h, 8*channel);
			OvxHelper::writeMatrix(vx_mat, tr);

			i32 ret_code = vxuWarpAffine(context, vx_src, vx_mat, VX_INTERPOLATION_BILINEAR, vx_dst);
			if (ret_code == VX_SUCCESS)
			{
				is_processed = true;
				OvxHelper::readImage(dst_img_ptr, w, h, vx_dst);
			}
		}
		g_vx_resource_pool.releaseResource(ovx_src_ptr);
		g_vx_resource_pool.releaseResource(ovx_mat_ptr);
		g_vx_resource_pool.releaseResource(ovx_dst_ptr);
#endif

		if (!is_processed)
		{
			u8* tmp_img_ptr = NULL;
			if (src_img_ptr == dst_img_ptr)
			{
				tmp_img_ptr = po_new u8[w*h*channel];
				memcpy(tmp_img_ptr, src_img_ptr, w*h*channel);
				src_img_ptr = tmp_img_ptr;
			}
			cv::Mat cv_tr_mat;
			cv::Mat cv_src_img(h, w, CV_8UC(channel), src_img_ptr);
			cv::Mat cv_dst_img(h, w, CV_8UC(channel), dst_img_ptr);

			m_calib_param_ptr->lock();
			{
				f64* inv_pb = m_calib_param_ptr->m_inv_pose_board_matrix;
				cv_tr_mat = cv::Mat(2, 3, CV_64FC1, inv_pb).clone();
			}
			m_calib_param_ptr->unlock();

			//warp affine transform
			cv::warpAffine(cv_src_img, cv_dst_img, cv_tr_mat, cv::Size(w, h), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
			POSAFE_DELETE_ARRAY(tmp_img_ptr);
		}
		return;
	}
	else if (dst_img_ptr != src_img_ptr) //copy content
	{
		CPOBase::memCopy(dst_img_ptr, src_img_ptr, w*h*channel);
	}

/* : Unused Method
{
	f64 dx, dy;
	i32 x, y, k, tmp, stmp;
	i32 nx, ny, kx, ky, ikx, iky, kk, index;
	f64* inv_pb = m_calib_param_ptr->m_inv_pose_board_matrix;
	i32 offset[3] = { channel, w*channel, (w + 1)*channel };
	u8* tmp_img_ptr;
	
	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
		{
			//calib pixel -> image pixel
			CPOBase::trans2x3(inv_pb, (f64)x, (f64)y, dx, dy);
			if (dx >= w || dx < 0 || dy >= h || dy < 0)
			{
				dst_img_ptr += channel;
				continue;
			}

			//linear-interpolation image warpping...
			nx = (i32)dx; kx = (i32)po::_max(1, (1 + nx - dx) * 50); ikx = 50 - kx;
			ny = (i32)dy; ky = (i32)po::_max(1, (1 + ny - dy) * 50); iky = 50 - ky;

			index = (ny*w + nx)*channel;
			tmp_img_ptr = src_img_ptr + index;
			for (k = 0; k < channel; k++)
			{
				kk = kx*ky; tmp = kk*tmp_img_ptr[k]; stmp = kk;
				if (nx < w - 1)
				{
					kk = ikx*ky; tmp += kk*tmp_img_ptr[k + offset[0]]; stmp += kk;
					if (ny < h - 1)
					{
						kk = kx*iky; tmp += kk*tmp_img_ptr[k + offset[1]]; stmp += kk;
						kk = ikx*iky; tmp += kk*tmp_img_ptr[k + offset[2]]; stmp += kk;
					}
				}
				else if (ny < h - 1)
				{
					kk = kx*iky; tmp += kk*tmp_img_ptr[index + offset[1]]; stmp += kk;
				}

				if (stmp > 0)
				{
					dst_img_ptr[k] = (u8)(tmp / stmp);
				}
			}
			dst_img_ptr += channel;
		}
	}
}*/
}

void CCamCalibration::undistortImageGrid(u8* dst_img_ptr, u8* src_img_ptr, i32 w, i32 h, i32 channel)
{
	if (!dst_img_ptr || !src_img_ptr || w*h <= 0)
	{
		return;
	}

	POMutexLocker l(m_undist_mutex);
	if (m_calib_param_ptr->isCalibed())
	{
#if defined(POR_WITH_OVX)
		bool is_processed = false;
		vx_remap vx_dist_remap = (vx_remap)m_vx_remap;
		if (vx_dist_remap && OvxHelper::getWidth(vx_dist_remap) == w && OvxHelper::getHeight(vx_dist_remap) == h)
		{
			if (g_vx_gpool_ptr)
			{
				CGImgProcRemap* graph_ptr = (CGImgProcRemap*)g_vx_gpool_ptr->fetchGraph(
							kGImgProcRemap, src_img_ptr, dst_img_ptr, w, h, channel, vx_dist_remap);
				if (graph_ptr)
				{
					is_processed = graph_ptr->process();
					g_vx_gpool_ptr->releaseGraph(graph_ptr);
				}
			}
		}
		if (is_processed)
		{
			return;
		}
#else
		i32 map_width = m_cv_undist_map1.cols;
		i32 map_height = m_cv_undist_map1.rows;
		if (map_width == w && map_height == h)
		{
			u8* tmp_img_ptr = NULL;
			if (src_img_ptr == dst_img_ptr)
			{
				tmp_img_ptr = po_new u8[w*h*channel];
				memcpy(tmp_img_ptr, src_img_ptr, w*h*channel);
				src_img_ptr = tmp_img_ptr;
			}

			cv::Mat cv_src_img(h, w, CV_8UC(channel), src_img_ptr);
			cv::Mat cv_dst_img(h, w, CV_8UC(channel), dst_img_ptr);
			cv::remap(cv_src_img, cv_dst_img, m_cv_undist_map1, m_cv_undist_map2, cv::INTER_LINEAR);
			POSAFE_DELETE_ARRAY(tmp_img_ptr);
			return;
		}
#endif
	}
	else if (dst_img_ptr != src_img_ptr) //copy content
	{
		CPOBase::memCopy(dst_img_ptr, src_img_ptr, w*h*channel);
	}

/*: Unused Method
{
	f64 dx, dy, dk, r2, r4, r6;
	f64 a1, a2, a3;
	f64 x1, y1;
	f64* d = m_calib_param_ptr->m_distort_coeffs;
	f64* m = m_calib_param_ptr->m_cam_matrix;
	f64* inv_pb = m_calib_param_ptr->m_inv_pose_board_matrix;
	i32 offset[3] = { channel, w*channel, (w + 1)*channel };
	u8* tmp_img_ptr;

	i32 x, y, k, tmp, stmp;
	i32 nx, ny, kx, ky, ikx, iky, kk, index;
	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
		{
			//calib pixel -> perspetive pixel
			CPOBase::perspective2d(inv_pb, (f64)x, (f64)y, dx, dy);

			//perspective pixel -> camera pixel with distort coeffs
			r2 = dx*dx + dy*dy;
			r4 = r2*r2;
			r6 = r4*r2;
			a1 = 2 * dx*dy;
			a2 = r2 + 2 * dx*dx;
			a3 = r2 + 2 * dy*dy;

			dk = (1 + d[0] * r2 + d[1] * r4 + d[4] * r6);
			dx = dx*dk + d[2] * a1 + d[3] * a2;
			dy = dy*dk + d[2] * a3 + d[3] * a1;

			//camera pixel -> pixel
			CPOBase::trans2x3(m, dx, dy, x1, y1);
	
			if (x1 >= w || x1 < 0 || y1 >= h || y1 < 0)
			{
				dst_img_ptr += channel;
				continue;
			}

			//linear-interpolation image warpping...
			nx = (i32)x1; kx = (i32)po::_max(1, (1 + nx - x1) * 50); ikx = 50 - kx;
			ny = (i32)y1; ky = (i32)po::_max(1, (1 + ny - y1) * 50); iky = 50 - ky;

			index = ny*w + nx;
			tmp_img_ptr = src_img_ptr + index;

			for (k = 0; k < channel; k++)
			{
				kk = kx*ky; tmp = kk* tmp_img_ptr[k]; stmp = kk;
				if (nx < w - 1)
				{
					kk = ikx*ky; tmp += kk*tmp_img_ptr[k + offset[0]]; stmp += kk;
					if (ny < h - 1)
					{
						kk = kx*iky; tmp += kk*tmp_img_ptr[k + offset[1]]; stmp += kk;
						kk = ikx*iky; tmp += kk*tmp_img_ptr[k + offset[2]]; stmp += kk;
					}
				}
				else if (ny < h - 1)
				{
					kk = kx*iky; tmp += kk*tmp_img_ptr[k + offset[1]]; stmp += kk;
				}

				if (stmp > 0)
				{
					dst_img_ptr[k] = (u8)(tmp / stmp);
				}
			}
			dst_img_ptr += channel;
		}
	}
}*/
}

bool CCamCalibration::getPerspectivePixel(f64 px, f64 py, f64& dx, f64& dy)
{
	dx = px;
	dy = py;

	if (!m_calib_param_ptr->checkState(kCalibResultUndistort))
	{
		return false;
	}

	f64* c = m_calib_param_ptr->m_cam_matrix;
	f64* d = m_calib_param_ptr->m_distort_coeffs;
	f64* h = m_calib_param_ptr->m_inv_cam_matrix;

	i32 niter = 0, iter = 5;

	f64 dk, r2, r4, r6;
	f64 a1, a2, a3, x1, y1, x2, y2;
	f64 eps = 0.0005 / po::_max(c[0], c[4]);

	CPOBase::trans2x3(h, px, py, x1, y1);
	dx = x1; dy = y1;

	while (true)
	{
		r2 = dx*dx + dy*dy;
		r4 = r2*r2;
		r6 = r4*r2;
		a1 = 2 * dx*dy;
		a2 = r2 + 2 * dx*dx;
		a3 = r2 + 2 * dy*dy;
		dk = (1 + d[0] * r2 + d[1] * r4 + d[4] * r6);
		x2 = (dx*dk + d[2] * a1 + d[3] * a2) - x1;
		y2 = (dy*dk + d[2] * a3 + d[3] * a1) - y1;
		if (CPOBase::length(x2, y2) < eps || niter++ > iter)
		{
			break;
		}
		dx = dx - x2;
		dy = dy - y2;
	}
	return true;
}

bool CCamCalibration::undistortPoint(f64 px, f64 py, f64& cx, f64& cy, i32 mode)
{
	cx = px;
	cy = py;

	if (!m_calib_param_ptr || !m_calib_param_ptr->isCalibed())
	{
		return false;
	}

	f64 dx, dy;
	f64* q = m_calib_param_ptr->m_pose_matrix;
	f64* p = m_calib_param_ptr->m_inv_board_matrix;
	f64* a = m_calib_param_ptr->m_inv_pose_absolute_matrix;

	//image pixel -> perspective pixel
	switch (m_calib_param_ptr->m_calib_type)
	{
		case kCalibTypeGrid:
		{
			if (!getPerspectivePixel(px, py, dx, dy))
			{
				return false;
			}
			break;
		}
		default:
		{
			dx = px;
			dy = py;
			break;
		}
	}

	switch (mode)
	{
		case kUndistCalibPixel:
		{
			CPOBase::perspective2d(p, dx, dy, px, py);
			CPOBase::trans2x3(q, px, py, cx, cy);
			break;
		}
		case kUndistMM:
		{
			CPOBase::perspective2d(p, dx, dy, cx, cy);
			break;
		}
		case kUndistCalibMM:
		{
			CPOBase::perspective2d(p, dx, dy, cx, cy);
			CPOBase::trans2x3(q, cx, cy, px, py);
			CPOBase::trans2x3(a, px, py, dx, dy);
			cx = dx;
			cy = dy;
			break;
		}
		case kUndistPerspective:
		{
			cx = dx;
			cy = dy;
			break;
		}
	}
	return true;
}

void CCamCalibration::testCalibPlatePose(CalibBoardVector& boards, i32 index)
{
	if (!chk_logcond(LOG_LV4, LOG_SCOPE_CAM))
	{
		return;
	}

	i32 i, count;
	f64 mat[9];
	f64 px, py, mx, my, cx, cy;
	f64 dist, max_pixel_dist = 0, max_mm_dist = 0;

	cv::Mat matPlate(3, 3, CV_64FC1, mat);
	cv::Mat matIPlate(3, 3, CV_64FC1, m_calib_param_ptr->m_inv_board_matrix);
	matPlate = matIPlate.inv();

	CalibBoard& board = boards[index];
	CornerPoint* pt = board.corner_vec.data();
	count = board.getPointCount();

	cv::Vec3b color1(0, 0, 255);
	cv::Vec3b color2(0, 255, 0);
	cv::Mat drawing = cv::Mat::zeros(board.h, board.w, CV_8UC3);

	for (i = 0; i < count; i++)
	{
		px = pt[i].point.x;
		py = pt[i].point.y;
		mx = pt[i].mm_x;
		my = pt[i].mm_y;

		getDistortedCoord(mx, my, cx, cy, mat);
		dist = CPOBase::distance(px, py, cx, cy);
		max_pixel_dist = po::_max(max_pixel_dist, dist);

		drawing.at<cv::Vec3b>(cv::Point((i32)(px + 0.5f), (i32)(py + 0.5f))) = color1;
		drawing.at<cv::Vec3b>(cv::Point((i32)(cx + 0.5f), (i32)(cy + 0.5f))) = color2;

		undistortPoint(px, py, cx, cy, kUndistMM);
		dist = CPOBase::distance(mx, my, cx, cy);
		max_mm_dist = po::_max(max_mm_dist, dist);
	}

	cv::imwrite(PO_LOG_PATH"CalibPlatePose.bmp", drawing);
}

void CCamCalibration::testCalibBoardConvexHull(cvPointVector2f& hull, i32 w, i32 h)
{
	if (!chk_logcond(LOG_LV4, LOG_SCOPE_CAM))
	{
		return;
	}

	cv::RNG rng(12345);
	cv::Mat drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::Vec3b color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

	std::vector<cv::Point> draw_hull;
	draw_hull.resize(hull.size());
	for (i32 i = 0; i < hull.size(); i++)
	{
		draw_hull[i] = hull[i];
	}

    cv::fillConvexPoly(drawing, draw_hull.data(), (i32)hull.size(), (cv::Scalar)color);
	cv::imwrite(PO_LOG_PATH"Calib_Convexhull.bmp", drawing);
}

void CCamCalibration::freeUndistortMap()
{
#if defined(POR_WITH_OVX)
	if (m_vx_remap)
	{
		vx_remap map = (vx_remap)m_vx_remap;
		vxReleaseRemap(&map);
		m_vx_remap = NULL;
	}
#else
	m_cv_undist_map1.release();
	m_cv_undist_map2.release();
#endif
}

void CCamCalibration::updateUndistortMap(i32 max_w, i32 max_h)
{
	if (!m_calib_param_ptr)
	{
		return;
	}

	//clear undistort map
	freeUndistortMap();

	CameraCalib tmp_calib_param = m_calib_param_ptr->getValue();
	if (!tmp_calib_param.isCalibed() || tmp_calib_param.getCalibType() != kCalibTypeGrid)
	{
		return;
	}

#if (1)
	if (max_w != tmp_calib_param.m_calibed_width)
	{
		printlog_lv2(QString("CalibError calib width:%1, camera width:%2")
						.arg(tmp_calib_param.m_calibed_width).arg(max_w));
	}
	if (max_h != tmp_calib_param.m_calibed_height)
	{
		printlog_lv2(QString("CalibError calib height:%1, camera height:%2")
						.arg(tmp_calib_param.m_calibed_height).arg(max_h));
	}
	
#if defined(POR_WITH_OVX)
	vx_context context = g_vx_context.getVxContext();
	m_vx_remap = vxCreateRemap(context, max_w, max_h, max_w, max_h);

	i32 x, y;
	f64 dx, dy;
	for (y = 0; y < max_h; y++)
	{
		for (x = 0; x < max_w; x++)
		{
			getDistortedCoord((f64)x, (f64)y, dx, dy);
			vxSetRemapPoint((vx_remap)m_vx_remap, (u32)x, (u32)y, (f32)dx, (f32)dy);
		}
	}
#else
	f64* d = tmp_calib_param.m_distort_coeffs;
	f64* m = tmp_calib_param.m_cam_matrix;
	f64* inv_pb = tmp_calib_param.m_inv_pose_board_matrix;

	std::vector<f32> dist_coeffs;
	dist_coeffs.push_back(d[0]);
	dist_coeffs.push_back(d[1]);
	dist_coeffs.push_back(d[2]);
	dist_coeffs.push_back(d[3]);
	dist_coeffs.push_back(d[4]);

	cv::Mat cv_cam_mat(3, 3, CV_64FC1, m);
	cv::Mat cv_R_mat(3, 3, CV_64FC1, inv_pb);
	cv::initUndistortRectifyMap(cv_cam_mat, dist_coeffs, cv_R_mat.inv(),
				cv::Mat::eye(3, 3, CV_64FC1), cv::Size(max_w, max_h), CV_16SC2, 
				m_cv_undist_map1, m_cv_undist_map2);
#endif
#endif
}

bool CCamCalibration::canbeCalib(f32& pixel_per_mm)
{
	pixel_per_mm = 0;
	if (!m_calib_param_ptr)
	{
		return false;
	}

	bool can_use = false;
	anlock_guard_ptr(m_calib_param_ptr);
	{
		can_use = m_calib_param_ptr->canbeCalib();
		pixel_per_mm = m_calib_param_ptr->getPixelPerMM();
	}
	return can_use;
}
