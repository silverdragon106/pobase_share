#pragma once
#include "struct.h"

struct CalibBoard;

class CFindChessborad
{
public:
	CFindChessborad();
	virtual ~CFindChessborad();

	bool					findCorners(const ImageData* img_data_ptr, CalibBoard* calib_board_ptr);
};

