#pragma once
#include "base.h"
#pragma pack(push, 4)

enum RunTableFlagType
{
	kRunTableFlagNone = 0x00,
	kRunTableFlagInvert = 0x01,
	kRunTableFlagMaskImg = 0x02,
	kRunTableFlagMaskVal = 0x04
};

struct vxParamConvertImg2RunTable
{
	u16				width;
	u16				height;
	u16				flag;
	i16				mask_val;
};

struct vxParamConvertRunTable2Img
{
	u16				width;
	u16				height;
	u8				value;
};

#pragma pack(pop)