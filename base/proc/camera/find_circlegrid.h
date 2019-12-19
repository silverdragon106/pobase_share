#pragma once

#include "struct.h"
#include "proc/connected_components.h"

struct CalibBoard;
class CameraCalib;

class CFindCircleGrid
{
public:
	CFindCircleGrid();
	virtual ~CFindCircleGrid();

	bool					findCorners(const ImageData* img_data_ptr, CalibBoard* calib_board_ptr);
	bool					findCornersWithAnchor(const ImageData* img_data_ptr, CameraCalib* calib_param_ptr, CalibBoard* calib_board_ptr);

private:
	CConnectedComponents	m_conn_comp;
};

