#pragma once

#include "struct.h"
#include "struct/camera_calib.h"
#include "find_chessboard.h"
#include "find_circlegrid.h"

#include <opencv2/opencv.hpp>
typedef std::vector<cv::Point2f> cvPointVector2f;
typedef std::vector<cv::Point3f> cvPointVector3f;

enum UndistortType
{
	kUndistCalibPixel = 0,
	kUndistCalibMM,
	kUndistMM,
	kUndistPerspective
};

class CCamCalibration
{
public:
	CCamCalibration();
	virtual ~CCamCalibration();

	bool					initInstance();
	void					exitInstance();

	CameraCalib				getCalibParam();
	void					setCalibParam(CameraCalib* calib_param_ptr, i32 max_w, i32 max_h);
	
	bool					detectBoard(const ImageData* img_data_ptr, CameraCalib* calib_param_ptr, CalibBoard* calib_board_ptr);
	bool					findCorners(const ImageData* img_data_ptr, CameraCalib* calib_param_ptr, CalibBoard* calib_board_ptr);
	bool					buildIndex(const ImageData* img_data_ptr, CalibBoard* calib_board_ptr);

	i32						calibCamera(const i32 w, const i32 h, CalibBoardVector& board_vec);
	i32						calibCameraGrid(const i32 w, const i32 h, CalibBoardVector& board_vec);
	i32						calibCameraScale(f64 sx, f64 sy, i32 w, i32 h);
	bool					undistortImage(ImageData& img_data);
	bool					undistortImage(ImageData& dst_img, const ImageData& src_img);
	bool					undistortImage(u8* img_ptr, i32 w, i32 h, i32 channel);
	bool					undistortImage(u8* dst_img_ptr, u8* src_img_ptr, i32 w, i32 h, i32 channel);
	void					undistortImageGrid(u8* dst_img_ptr, u8* src_img_ptr, i32 w, i32 h, i32 channel);
	void					undistortImageWrap(u8* dst_img_ptr, u8* src_img_ptr, i32 w, i32 h, i32 channel);
	bool					undistortPoint(f64 px, f64 py, f64& cx, f64& cy, i32 mode = kUndistCalibPixel);

	bool					canbeCalib(f32& pixel_per_mm);

private:
	void					freeUndistortMap();
	void					updateUndistortMap(i32 max_w, i32 max_h);

	f64						findExactlyPlatePose(CalibBoardVector& board_vec, cv::Mat camMatrix, cv::Mat distCoeffs, std::vector<cv::Mat> rvecs, std::vector<cv::Mat> tvecs);
	i32						calibPivotPosition(const i32 w, const i32 h);
	i32						calibAxis(CalibBoardVector& board_vec);
	i32						calibAutoAxis(CalibBoardVector& board_vec);
	i32						calibManualAxis();

	void					convRodriguesToMat(f64* r3, f64* r9);
	void					convMatrixToRodrigues(f64* r9, f64* r3);
	void					convertToMat(f64* m9, cv::Mat& rvec, cv::Mat& tvec);
	void					getDistortedCoord(f64 mx, f64 my, f64& px, f64& py, f64* tr);
	void					getDistortedCoord(f64 px, f64 py, f64& dx, f64& dy);
	bool					getPerspectivePixel(f64 px, f64 py, f64& dx, f64& dy);

	void					testCalibPlatePose(CalibBoardVector& board_vec, i32 index);
	void					testCalibBoardConvexHull(cvPointVector2f& hull, i32 w, i32 h);

public:
	CameraCalib*			m_calib_param_ptr;
	CFindChessborad			m_chessboard_finder;
	CFindCircleGrid			m_circlegrid_finder;

#if (1)
	POMutex					m_undist_mutex;
#if !defined(POR_WITH_OVX)
	cv::Mat					m_cv_undist_map1;
	cv::Mat					m_cv_undist_map2;
#else
	void*					m_vx_remap;
#endif
#endif
};
