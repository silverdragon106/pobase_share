#include "camera_calib.h"
#include "base.h"

//////////////////////////////////////////////////////////////////////////
CornerPoint::CornerPoint()
{
	index_x = -1; index_y = -1;
	mm_x = -1; mm_y = -1;
	point = vector2dd(-1, -1);
}

CornerPoint::CornerPoint(vector2df pt)
{
	index_x = -1; index_y = -1;
	mm_x = -1; mm_y = -1;
	point = vector2dd(pt.x, pt.y);
}

CornerPoint::CornerPoint(vector2dd pt)
{
	index_x = -1; index_y = -1;
	mm_x = -1; mm_y = -1;
	point = pt;
}

CalibBoard::CalibBoard()
{
	w = h = 0;
	board_coverage = 0;
	corner_vec.clear();
	snap_type = kCalibNormalSnap;
	anchor_point = vector2dd(-1, -1);
}

i32 CalibBoard::memSize()
{
	i32 len = 0;

	len += sizeof(w);
	len += sizeof(h);
	len += sizeof(snap_type);
	len += (i32)CPOBase::getVectorMemSize(corner_vec);
	len += sizeof(anchor_point);
	len += sizeof(board_coverage);
	return len;
}

i32 CalibBoard::getPointCount()
{
	return (i32)corner_vec.size();
}

bool CalibBoard::isAnchorBoard()
{
	return anchor_point.isValid();
}

i32 CalibBoard::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(w, buffer_ptr, buffer_size);
	CPOBase::memRead(h, buffer_ptr, buffer_size);
	CPOBase::memRead(snap_type, buffer_ptr, buffer_size);
	CPOBase::memReadVector(corner_vec, buffer_ptr, buffer_size);
	CPOBase::memRead(anchor_point, buffer_ptr, buffer_size);
	CPOBase::memRead(board_coverage, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CalibBoard::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(w, buffer_ptr, buffer_size);
	CPOBase::memWrite(h, buffer_ptr, buffer_size);
	CPOBase::memWrite(snap_type, buffer_ptr, buffer_size);
	CPOBase::memWriteVector(corner_vec, buffer_ptr, buffer_size);
	CPOBase::memWrite(anchor_point, buffer_ptr, buffer_size);
	CPOBase::memWrite(board_coverage, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

//////////////////////////////////////////////////////////////////////////
CameraCalib::CameraCalib()
{
	reset();
}

void CameraCalib::reset()
{
	m_calib_type = kCalibTypeNone;
	m_calib_board_type = kCalibChessBoard;
	m_calib_mode = kCalibModeNone;
	memset(&u, 0, sizeof(u));

	m_calib_offset_x = 0;
	m_calib_offset_y = 0;
	m_calib_offset_z = 0;

	m_use_calib = false;

	u.anchorgrid_calib.dot_interval = kCamCalibDotInterval;
	u.anchorgrid_calib.dot_diameter = kCamCalibDotSize;
	u.anchorgrid_calib.anchor_diameter_ratio = kCamCalibAnchorRate;

	m_calibed_result = kCalibResultNone;
	m_pixel_per_mm = 1.0f;

	m_calibed_width = 0;
	m_calibed_height = 0;
	m_calibed_coverage = 0;
	m_calibed_mean_error = 0;
	m_calibed_max_error = 0;
	m_calibed_sigma_error = 0;
	memset(&m_calibed_time, 0, sizeof(DateTime));

	memset(&m_distort_coeffs, 0, 5 * sizeof(f64));
	CPOBase::init3x3(m_cam_matrix);
	CPOBase::init3x3(m_pose_matrix);
	CPOBase::init3x3(m_inv_board_matrix);
	CPOBase::init3x3(m_inv_cam_matrix);
	CPOBase::init3x3(m_inv_pose_board_matrix);
	CPOBase::init3x3(m_inv_pose_absolute_matrix);
	CPOBase::init3x3(m_absolute_pose_matrix);
}

void CameraCalib::init()
{
	lock_guard();
	reset();
}

i32 CameraCalib::memBoardSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_calib_type);
	len += sizeof(m_calib_board_type);
	len += sizeof(u);
	return len;
}

i32 CameraCalib::memBoardRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_calib_type, buffer_ptr, buffer_size);
	if (m_calib_type == kCalibTypeGrid)
	{
		CPOBase::memRead(m_calib_board_type, buffer_ptr, buffer_size);
		CPOBase::memRead(u, buffer_ptr, buffer_size);
	}
	else
	{
		printlog_lv1("Can't set calib board data.");
	}
	return buffer_ptr - buffer_pos;
}

i32 CameraCalib::memBoardWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_calib_type, buffer_ptr, buffer_size);
	if (m_calib_type != kCalibTypeGrid)
	{
		printlog_lv1("Can't set calib board data.");
	}
	else
	{
		CPOBase::memWrite(m_calib_board_type, buffer_ptr, buffer_size);
		CPOBase::memWrite(u, buffer_ptr, buffer_size);
	}
	return buffer_ptr - buffer_pos;
}

i32 CameraCalib::memOffsetSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_calib_offset_x);
	len += sizeof(m_calib_offset_y);
	len += sizeof(m_calib_offset_z);
	return len;
}

i32 CameraCalib::memOffsetRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_calib_offset_x, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calib_offset_y, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calib_offset_z, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraCalib::memParamSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_calib_type);
	len += sizeof(m_calib_board_type);
	len += sizeof(m_calib_mode);
	len += sizeof(u);
	return len;
}

i32 CameraCalib::memParamRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_calib_type, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calib_board_type, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calib_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(u, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraCalib::memParamWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_calib_type, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calib_board_type, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calib_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(u, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraCalib::memSize()
{
	lock_guard();
	i32 len = 0;

	len += sizeof(m_calib_type);
	len += sizeof(m_calib_board_type);
	len += sizeof(m_calib_mode);
	len += sizeof(m_calib_offset_x);
	len += sizeof(m_calib_offset_y);
	len += sizeof(m_calib_offset_z);
	len += sizeof(u);

	len += sizeof(m_use_calib);
	len += sizeof(m_calibed_result);
	len += sizeof(m_pixel_per_mm);

	len += sizeof(m_calibed_width);
	len += sizeof(m_calibed_height);
	len += sizeof(m_calibed_coverage);
	len += sizeof(m_calibed_mean_error);
	len += sizeof(m_calibed_max_error);
	len += sizeof(m_calibed_sigma_error);
	len += sizeof(m_calibed_time);

	len += sizeof(m_distort_coeffs);
	len += sizeof(m_cam_matrix);
	len += sizeof(m_pose_matrix);
	len += sizeof(m_inv_board_matrix);
	len += sizeof(m_inv_cam_matrix);
	len += sizeof(m_inv_pose_board_matrix);
	len += sizeof(m_inv_pose_absolute_matrix);
	len += sizeof(m_absolute_pose_matrix);
	return len;
}

i32 CameraCalib::memRead(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memRead(m_calib_type, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calib_board_type, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calib_mode, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calib_offset_x, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calib_offset_y, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calib_offset_z, buffer_ptr, buffer_size);
	CPOBase::memRead(u, buffer_ptr, buffer_size);

	CPOBase::memRead(m_use_calib, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calibed_result, buffer_ptr, buffer_size);
	CPOBase::memRead(m_pixel_per_mm, buffer_ptr, buffer_size);

	CPOBase::memRead(m_calibed_width, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calibed_height, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calibed_coverage, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calibed_mean_error, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calibed_max_error, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calibed_sigma_error, buffer_ptr, buffer_size);
	CPOBase::memRead(m_calibed_time, buffer_ptr, buffer_size);

	CPOBase::memRead(m_distort_coeffs, 5, buffer_ptr, buffer_size);
	CPOBase::memRead(m_cam_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memRead(m_pose_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memRead(m_inv_board_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memRead(m_inv_cam_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memRead(m_inv_pose_board_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memRead(m_inv_pose_absolute_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memRead(m_absolute_pose_matrix, 9, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

i32 CameraCalib::memWrite(u8*& buffer_ptr, i32& buffer_size)
{
	lock_guard();
	u8* buffer_pos = buffer_ptr;

	CPOBase::memWrite(m_calib_type, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calib_board_type, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calib_mode, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calib_offset_x, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calib_offset_y, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calib_offset_z, buffer_ptr, buffer_size);
	CPOBase::memWrite(u, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_use_calib, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calibed_result, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_pixel_per_mm, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_calibed_width, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calibed_height, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calibed_coverage, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calibed_mean_error, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calibed_max_error, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calibed_sigma_error, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_calibed_time, buffer_ptr, buffer_size);

	CPOBase::memWrite(m_distort_coeffs, 5, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_cam_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_pose_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_inv_board_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_inv_cam_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_inv_pose_board_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_inv_pose_absolute_matrix, 9, buffer_ptr, buffer_size);
	CPOBase::memWrite(m_absolute_pose_matrix, 9, buffer_ptr, buffer_size);
	return buffer_ptr - buffer_pos;
}

bool CameraCalib::fileRead(FILE* fp)
{
	lock_guard();
	if (!CPOBase::fileSignRead(fp, PO_SIGN_CODE))
	{
		return false;
	}

	CPOBase::fileRead(m_calib_type, fp);
	CPOBase::fileRead(m_calib_board_type, fp);
	CPOBase::fileRead(m_calib_mode, fp);
	CPOBase::fileRead(m_calib_offset_x, fp);
	CPOBase::fileRead(m_calib_offset_y, fp);
	CPOBase::fileRead(m_calib_offset_z, fp);
	CPOBase::fileRead(u, fp);

	CPOBase::fileRead(m_use_calib, fp);
	CPOBase::fileRead(m_calibed_result, fp);
	CPOBase::fileRead(m_pixel_per_mm, fp);

	CPOBase::fileRead(m_calibed_width, fp);
	CPOBase::fileRead(m_calibed_height, fp);
	CPOBase::fileRead(m_calibed_coverage, fp);
	CPOBase::fileRead(m_calibed_mean_error, fp);
	CPOBase::fileRead(m_calibed_max_error, fp);
	CPOBase::fileRead(m_calibed_sigma_error, fp);
	CPOBase::fileRead(m_calibed_time, fp);

	CPOBase::fileRead(m_distort_coeffs, 5, fp);
	CPOBase::fileRead(m_cam_matrix, 9, fp);
	CPOBase::fileRead(m_pose_matrix, 9, fp);
	CPOBase::fileRead(m_inv_board_matrix, 9, fp);
	CPOBase::fileRead(m_inv_cam_matrix, 9, fp);
	CPOBase::fileRead(m_inv_pose_board_matrix, 9, fp);
	CPOBase::fileRead(m_inv_pose_absolute_matrix, 9, fp);
	CPOBase::fileRead(m_absolute_pose_matrix, 9, fp);

	return CPOBase::fileSignRead(fp, PO_SIGN_ENDCODE);
}

bool CameraCalib::fileWrite(FILE* fp)
{
	lock_guard();
	CPOBase::fileSignWrite(fp, PO_SIGN_CODE);

	CPOBase::fileWrite(m_calib_type, fp);
	CPOBase::fileWrite(m_calib_board_type, fp);
	CPOBase::fileWrite(m_calib_mode, fp);
	CPOBase::fileWrite(m_calib_offset_x, fp);
	CPOBase::fileWrite(m_calib_offset_y, fp);
	CPOBase::fileWrite(m_calib_offset_z, fp);
	CPOBase::fileWrite(u, fp);

	CPOBase::fileWrite(m_use_calib, fp);
	CPOBase::fileWrite(m_calibed_result, fp);
	CPOBase::fileWrite(m_pixel_per_mm, fp);

	CPOBase::fileWrite(m_calibed_width, fp);
	CPOBase::fileWrite(m_calibed_height, fp);
	CPOBase::fileWrite(m_calibed_coverage, fp);
	CPOBase::fileWrite(m_calibed_mean_error, fp);
	CPOBase::fileWrite(m_calibed_max_error, fp);
	CPOBase::fileWrite(m_calibed_sigma_error, fp);
	CPOBase::fileWrite(m_calibed_time, fp);

	CPOBase::fileWrite(m_distort_coeffs, 5, fp);
	CPOBase::fileWrite(m_cam_matrix, 9, fp);
	CPOBase::fileWrite(m_pose_matrix, 9, fp);
	CPOBase::fileWrite(m_inv_board_matrix, 9, fp);
	CPOBase::fileWrite(m_inv_cam_matrix, 9, fp);
	CPOBase::fileWrite(m_inv_pose_board_matrix, 9, fp);
	CPOBase::fileWrite(m_inv_pose_absolute_matrix, 9, fp);
	CPOBase::fileWrite(m_absolute_pose_matrix, 9, fp);

	CPOBase::fileSignWrite(fp, PO_SIGN_ENDCODE);
	return true;
}

f32 CameraCalib::getLengthScale()
{
	lock_guard();
	if (!canbeCalib())
	{
		return 1.0f;
	}
	return m_pixel_per_mm;
}

f32 CameraCalib::getAreaScale()
{
	lock_guard();
	if (!canbeCalib())
	{
		return 1.0f;
	}
	return m_pixel_per_mm*m_pixel_per_mm;
}

f32 CameraCalib::convertToMM(f32 val)
{
	lock_guard();
	if (!canbeCalib())
	{
		return val;
	}
	return val*m_pixel_per_mm;
}

vector2df CameraCalib::convertToMM(const vector2df& pt)
{
	lock_guard();
	if (!canbeCalib())
	{
		return pt;
	}
	return m_pixel_per_mm*pt;
}

f32 CameraCalib::convertToMM2(f32 area)
{
	lock_guard();
	if (!canbeCalib())
	{
		return area;
	}
	return area*m_pixel_per_mm*m_pixel_per_mm;
}

vector2df CameraCalib::convertToAbsMM(const vector2df& pt)
{
	lock_guard();
	if (!canbeCalib())
	{
		return pt;
	}

	vector2df tmp_point = pt;
	CPOBase::trans2x3(m_inv_pose_absolute_matrix, tmp_point);

	tmp_point.x += m_calib_offset_x;
	tmp_point.y += m_calib_offset_y;
	return tmp_point;
}

f32 CameraCalib::convertToPixel(f32 val)
{
	lock_guard();
	if (!canbeCalib())
	{
		return val;
	}
	return val / m_pixel_per_mm;
}

f32 CameraCalib::convertToPixel2(f32 area)
{
	lock_guard();
	if (!canbeCalib())
	{
		return area;
	}
	return area / (m_pixel_per_mm * m_pixel_per_mm);
}

vector2df CameraCalib::convertToPixel(const vector2df& pt)
{
	lock_guard();
	if (!canbeCalib())
	{
		return pt;
	}
	return pt / m_pixel_per_mm;
}

vector2df CameraCalib::convertToAbsPixel(const vector2df& pt)
{
	lock_guard();
	if (!canbeCalib())
	{
		return pt;
	}

	vector2df tmp_point = pt;
	tmp_point.x -= m_calib_offset_x;
	tmp_point.y -= m_calib_offset_y;
	CPOBase::trans2x3(m_absolute_pose_matrix, tmp_point);
	return tmp_point;
}

CameraCalib CameraCalib::getValue()
{
	lock_guard();
	return *this;
}

void CameraCalib::setValue(CameraCalib& other)
{
	lock_guard();
	anlock_guard(other);
	*this = other;
}

bool CameraCalib::setUseCalib(bool use_calib)
{
	lock_guard();
	if (!CPOBase::bitCheck(m_calibed_result, kCalibResultPixel2MM))
	{
		return false;
	}

	m_use_calib = use_calib;
	return true;
}

i32 CameraCalib::getMinCalibSnap()
{
	lock_guard();
	if (m_calib_type == kCalibTypeGrid)
	{
		switch (m_calib_board_type)
		{
			case kCalibChessBoard:
			{
				return u.chessboard_calib.board_count_min;
			}
			case kCalibCircleGrid:
			{
				return u.circlegrid_calib.board_count_min;
			}
			case kCalibAnchorCircleGrid:
			{
				return u.anchorgrid_calib.board_count_min;
			}
		}
	}
	return 0;
}

i32 CameraCalib::getCalibThError()
{
	lock_guard();
	if (m_calib_type == kCalibTypeGrid)
	{
		switch (m_calib_board_type)
		{
			case kCalibChessBoard:
			{
				return u.chessboard_calib.threshold_calib_error;
			}
			case kCalibCircleGrid:
			{
				return u.circlegrid_calib.threshold_calib_error;
			}
			case kCalibAnchorCircleGrid:
			{
				return u.anchorgrid_calib.threshold_calib_error;
			}
		}
	}
	return 0;
}

bool CameraCalib::checkMode(i32 mode)
{
	lock_guard();
	return CPOBase::bitCheck(m_calib_mode, mode);
}

bool CameraCalib::checkState(i32 mode)
{
	lock_guard();
	return CPOBase::bitCheck(m_calibed_result, mode);
}