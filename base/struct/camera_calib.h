#pragma once

#include "define.h"
#include "struct.h"

#pragma pack(push, 4)

enum CalibType
{
	kCalibTypeNone = 0,
	kCalibTypeScale,
	kCalibTypeXYScale,
	kCalibTypeEdgeToEdge,
	kCalibTypeXYEdgeToEdge,
	kCalibTypeCircle,
	kCalibType9Points,
	kCalibTypeGrid
};

enum CalibBoardType
{
	kCalibAnyBoard = 0,
	kCalibChessBoard,
	kCalibCircleGrid,
	kCalibAnchorCircleGrid
};

enum CalibSnapType
{
	kCalibNormalSnap = 0,
	kCalibAxisXPosSnap,
	kCalibAxisXNegSnap,
	kCalibAxisYPosSnap,
	kCalibAxisYNegSnap,
	kCalibAnchorSnap
};

enum CalibOperationType
{
	kCalibOperationOffset = 0,
	kCalibOperationXYSwap,
	kCalibOperationXInvert,
	kCalibOperationYInvert,

	kCalibOperationCount
};

enum CalibReturnType
{
	kCalibReturnSuccess = 0,		//교정성공
	kCalibReturnNonParam,			//파라메터값이 없다
	kCalibReturnInvalidParam,		//파라메터값이 부정확
	kCalibReturnLessSnaps,			//교정을 위한 Snap이 부족하다.
	kCalibReturnLessCorners,		//교정을 위한 특징점이 부족하다
	kCalibReturnToleranceError,		//교정결과 거리오차가 너무크다.
	kCalibReturnDiscontinuslyAnchor,//Sanp이 기준점자리표계교정을 위한 자료가 아니다
	kCalibReturnMultiAnchorXError,	//Sanp이 X축이동화상이 아니다.(기준점자리표계교정)
	kCalibReturnMultiAnchorYError,	//Sanp이 Y축이동화상이 아니다.(기준점자리표계교정)
	kCalibReturnXYAnchorError,		//Snap을 가지고 기준점자리표계교정을 할수없다.

	kCalibReturnCodeCounts
};

enum CalibModeType
{
	kCalibModeNone			= 0x0000,
	kCalibModeUndistort		= 0x0001,
	kCalibModeFixAspect		= 0x0002,
	kCalibModeAutoAxis		= 0x0004,
	kCalibModeDecartAxis	= 0x0008,	//in AutoAxis mode
	kCalibModeAxisSwapXY	= 0x0010,	//in ManualAxis mode
	kCalibModeAxisInvertX	= 0x0020,	//in ManualAxis mode
	kCalibModeAxisInvertY	= 0x0040,	//in ManualAxis mode
	kCalibModeDirectProcess = 0x1000
};

enum CalibResultType
{
	kCalibResultNone		= 0x0000,
	kCalibResultType		= 0x00FF,
	kCalibResultPixel2MM	= 0x0100,
	kCalibResultUndistort	= 0x0200,
	kCalibResultAutoAxis	= 0x0400,
	kCalibResultManualAxis	= 0x0800,
	kCalibResultPivot		= 0x1000,
	kCalibResultSnapLess	= 0x2000
};

enum CalibUnitType
{
	CalibUnit_MM,
	CalibUnit_CM,
	CalibUnit_UM,
};

const i32 kCamCalibMinCount = 4;
const i32 kCamCalibMinPointCount = 4;
const f32 kCamCalibDotInterval = 10.0f;
const f32 kCamCalibDotSize = 5.0f;
const f32 kCamCalibAnchorRate = 1.5f;
const f32 kCamCalibThErrorMM = 4.0f; //4mm is available error of calibration

//////////////////////////////////////////////////////////////////////////
struct CornerPoint
{
	i32						index_x, index_y;	// X,Y index
	f64						mm_x, mm_y;			// Object Position
	vector2dd				point;

	CornerPoint();
	CornerPoint(vector2df pt);
	CornerPoint(vector2dd pt);

	inline bool				isValid() { return point.x >= 0 && point.y >= 0; };
};
typedef std::vector<CornerPoint> CornerVector;

struct CalibBoard
{
	i32						w;
	i32						h;
	i32						snap_type;
	f32						board_coverage;
	CornerPoint				anchor_point;
	CornerVector			corner_vec;

public:
	CalibBoard();

	i32						getPointCount();
	bool					isAnchorBoard();

	i32						memSize();
	i32						memWrite(u8*& buffer_ptr, i32& buffer_size);
	i32						memRead(u8*& buffer_ptr, i32& buffer_size);
};
typedef std::vector<CalibBoard> CalibBoardVector;

//////////////////////////////////////////////////////////////////////////
struct CalibScale
{
	f32						calib_scale;
};

struct CalibXYScale
{
	f32						horizontal_scale;
	f32						vertical_scale;
};

struct CalibEdgeToEdge
{
	f32						pos1[2];
	f32						pos2[2];
	f32						distance;
};

struct CalibXYEdgeToEdge
{
	f32						axis_x_pos1[2];
	f32						axis_x_pos2[2];
	f32						axis_x_distance;

	f32						axis_y_pos1[2];
	f32						axis_y_pos2[2];
	f32						axis_y_distance;
};

struct CalibChessBoard
{
	i32						board_count_min;
	f32						threshold_calib_error;
	f32						chessboard_interval;	//mm
};

struct CalibCircleGrid
{
	i32						board_count_min;
	f32						threshold_calib_error;
	f32						dot_interval;	//mm
	f32						dot_diameter;	//mm
};

struct CalibCircleAnch
{
	i32						board_count_min;
	f32						threshold_calib_error;
	f32						dot_interval;	//mm
	f32						dot_diameter;	//mm
	f32						anchor_diameter_ratio; //1~
};

class CameraCalib : public CLockGuard
{
public:
	CameraCalib();

	void					init();
	void					reset();

	CameraCalib				getValue();
	void					setValue(CameraCalib& other);
	bool					setUseCalib(bool use_calib);

	bool					checkMode(i32 mode);
	bool					checkState(i32 mode);
	i32						getMinCalibSnap();
	i32						getCalibThError();

	i32						memSize();
	i32						memParamSize();
	i32						memBoardSize();
	i32						memOffsetSize();

	i32						memRead(u8*& buffer_ptr, i32& buffer_size);
	i32						memParamRead(u8*& buffer_ptr, i32& buffer_size);
	i32						memBoardRead(u8*& buffer_ptr, i32& buffer_size);
	i32						memOffsetRead(u8*& buffer_ptr, i32& buffer_size);

	i32						memWrite(u8*& buffer_ptr, i32& buffer_size);
	i32						memParamWrite(u8*& buffer_ptr, i32& buffer_size);
	i32						memBoardWrite(u8*& buffer_ptr, i32& buffer_size);

	bool					fileRead(FILE* fp);
	bool					fileWrite(FILE* fp);

	f32						convertToMM(f32 val);
	f32						convertToMM2(f32 area);
	vector2df				convertToMM(const vector2df& pt);
	vector2df				convertToAbsMM(const vector2df& pt);

	f32						convertToPixel(f32 val);
	f32						convertToPixel2(f32 area);
	vector2df				convertToPixel(const vector2df& pt);
	vector2df				convertToAbsPixel(const vector2df& pt);

	f32						getLengthScale();
	f32						getAreaScale();

	inline i32				getCalibType()	{ lock_guard(); return m_calib_type; };
	inline i32				getCalibMode()	{ lock_guard(); return m_calib_mode; };
	inline i32				getCalibBoardType() { lock_guard(); return m_calib_board_type; };

	inline bool				isUsedCalib()	{ lock_guard(); return m_use_calib; }
	inline bool				canbeCalib()	{ lock_guard(); return isCalibed() && isUsedCalib(); };

	inline i32				getCalibedType()	 { lock_guard(); return (m_calibed_result & kCalibResultType); };
	inline bool				isCalibed()			 { lock_guard(); return (m_calibed_result & kCalibResultPixel2MM); };
	inline bool				isAxisCalibed()		 { lock_guard(); return (m_calibed_result & (kCalibResultAutoAxis | kCalibResultManualAxis)); };
	inline bool				isUndistortCalibed() { lock_guard(); return (m_calibed_result & kCalibResultUndistort); };

	inline f32				getPixelPerMM()		 { lock_guard(); return m_pixel_per_mm; };
	inline f32				getCalibOffsetX()	 { lock_guard(); return m_calib_offset_x; };
	inline f32				getCalibOffsetY()	 { lock_guard(); return m_calib_offset_y; };
	inline f32				getCalibOffsetZ()	 { lock_guard(); return m_calib_offset_z; };
	inline f64*				getInvPoseAbsolute() { lock_guard(); return m_inv_pose_absolute_matrix; };
	inline f64*				getAbsolutePose()	 { lock_guard(); return m_absolute_pose_matrix; };

public:
	CalibType				m_calib_type;
	CalibBoardType			m_calib_board_type;
	CalibModeType			m_calib_mode;
	f32						m_calib_offset_x;
	f32						m_calib_offset_y;
	f32						m_calib_offset_z;

	union
	{
		CalibScale			scale_calib;
		CalibXYScale		xy_scale_calib;
		CalibEdgeToEdge		edge_to_edge_calib;
		CalibXYEdgeToEdge	xy_edge_to_edge_calib;
		CalibChessBoard		chessboard_calib;
		CalibCircleGrid		circlegrid_calib;
		CalibCircleAnch		anchorgrid_calib;
	} u;

	//calibration result
	bool					m_use_calib;
	i32						m_calibed_result;
	f32						m_pixel_per_mm;

	i32						m_calibed_width;
	i32						m_calibed_height;
	f32						m_calibed_coverage;
	f32						m_calibed_mean_error;
	f32						m_calibed_max_error;
	f32						m_calibed_sigma_error;
	DateTime				m_calibed_time;

	/*
	1.[calibed pixel]	    <-	|PoseMatrix|	 <-	 [mm]
	2.[mm]					 ->	|PlateMatix|	  -> [perspectived pixel]
	3.[perspectived pixel]	 ->	|DistortCoffes|	  -> [camera internal pixel]
	4.[camera internalpixel] ->	|CameraMatix|	  -> [image pixel]
	----------------------------------------------------------------------------------
	5.[calibed pixel]		 ->	|IPoseAbsMatrix|  -> [abs mm]
	*/

	f64						m_distort_coeffs[5];			// lens coeffs: k1, k2, p1, p2, k3
	f64						m_cam_matrix[9];				// camera matrix: fx, 0, cx, 0, fy, cy, 0, 0, 1
	f64						m_pose_matrix[9];				// [mm]->[calibed pixel]
	f64						m_inv_board_matrix[9];			// [perspectived pixel]->[mm]
	f64						m_inv_cam_matrix[9];			// [image pixel]->[camera internal pixel]
	f64						m_inv_pose_board_matrix[9];		// [calibed pixel]->[mm]->[perspectived pixel]
	f64						m_inv_pose_absolute_matrix[9];	// [calibed pixel]->[mm]->[abs mm]
	f64						m_absolute_pose_matrix[9];		// [abs mm]->[mm]->[calibed pixel]
};

#pragma pack(pop)