#pragma once
#include "base.h"
#include <QObject>
#include <QThread>
#include <QMutex>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>

#if defined(POR_WITH_OVX)
#include <VX/vx.h>
#include <VX/vxu.h>

#if defined(POR_WITH_CUDA)
#define NVX_MAP_IMAGE(image, mode) \
	void *data_##image = NULL; \
	vx_imagepatch_addressing_t addr_##image; \
	vx_rectangle_t rect_##image; \
	vx_map_id map_id_##image=0; \
	status |= vxGetValidRegionImage(image, &rect_##image); \
	status |= vxMapImagePatch(image, &rect_##image, 0, &map_id_##image, &addr_##image, &data_##image, mode, NVX_MEMORY_TYPE_CUDA, 0);

#define NVX_UNMAP_IMAGE(image) \
	if (map_id_##image > 0) status |= vxUnmapImagePatch(image, map_id_##image);

#define NVX_MAP_ARRAY(array, mode) \
	void *data_##array = NULL; \
	vx_size num_##array=0; \
	vx_map_id map_id_##array=0; \
	vx_size stride_##array=0; \
	status |= vxQueryArray(array, VX_ARRAY_NUMITEMS, &num_##array, sizeof(num_##array)); \
	if (num_##array > 0) status |= vxMapArrayRange(array, 0, num_##array, &map_id_##array, &stride_##array, &data_##array, mode, NVX_MEMORY_TYPE_CUDA, 0);

#define NVX_UNMAP_ARRAY(array) \
	if (map_id_##array > 0) status |= vxUnmapArrayRange(array, map_id_##array);
#endif

#define POVX_RELEASE(p) \
	{ if(p) { vxReleaseReference((vx_reference*)&(p)); (p)=NULL; } }

#define MAP_IMAGE(image, mode) \
	void *data_##image = NULL; \
	vx_imagepatch_addressing_t addr_##image; \
	vx_rectangle_t rect_##image; \
	vx_map_id map_id_##image=0; \
	status |= vxGetValidRegionImage(image, &rect_##image); \
	status |= vxMapImagePatch(image, &rect_##image, 0, &map_id_##image, &addr_##image, &data_##image, mode, VX_MEMORY_TYPE_HOST, 0);

#define UNMAP_IMAGE(image) \
	if (map_id_##image > 0) status |= vxUnmapImagePatch(image, map_id_##image);

#define VX_CHKRET(x)	status |= x; if (status != VX_SUCCESS) { return; }
#define VX_CHKRET_O(x)	status |= x; if (status != VX_SUCCESS) { return status; }
#define VX_CHKRET_T(x)	status |= x; if (status != VX_SUCCESS) { return true; }
#define VX_CHKRET_F(x)	status |= x; if (status != VX_SUCCESS) { return false; }

#define SCAN_WIDTH(image) \
	(vx_int32)(addr_##image.stride_y/addr_##image.stride_x)

#define MAP_ARRAY(array, mode) \
	void *data_##array = NULL; \
	vx_size num_##array=0; \
	vx_map_id map_id_##array=0; \
	vx_size stride_##array=0; \
	status |= vxQueryArray(array, VX_ARRAY_NUMITEMS, &num_##array, sizeof(num_##array)); \
	if (num_##array > 0) status |= vxMapArrayRange(array, 0, num_##array, &map_id_##array, &stride_##array, &data_##array, mode, VX_MEMORY_TYPE_HOST, 0);

#define UNMAP_ARRAY(array) \
	if (map_id_##array > 0) status |= vxUnmapArrayRange(array, map_id_##array);

#define ARRAY_COUNT(array) \
	vx_size num_##array=0; \
	status |= vxQueryArray(array, VX_ARRAY_NUMITEMS, &num_##array, sizeof(num_##array)); 

#define SCALAR(type, val) \
	type data_##val = (type)0;
#define READ_SCALAR(val) \
	status |= vxCopyScalar(val, &data_##val, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
#define WRITE_SCALAR(val) \
	status |= vxCopyScalar(val, &data_##val, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

#define SET_VALID_RECT(image, w, h) \
{ \
	if (image) { \
		vx_rectangle_t rt; \
			rt.start_x = 0; \
			rt.start_y = 0; \
			rt.end_x = w; \
			rt.end_y = h; \
		vxSetImageValidRectangle(image, &rt); \
	} \
}

#define VX_EPSILON		1E-6

#define VX_MIN(a,b) ((a) < (b) ? (a) : (b))
#define VX_MAX(a,b) ((a) > (b) ? (a) : (b))
#define VX_ABS(a) ((a) < 0 ? -(a) : (a))

#define RECT_WIDTH(rect) (rect.end_x - rect.start_x)
#define RECT_HEIGHT(rect) (rect.end_y - rect.start_y)
#define RECTPTR_WIDTH(rect) (rect->end_x - rect->start_x)
#define RECTPTR_HEIGHT(rect) (rect->end_y - rect->start_y)
#define RECT_CENTER(rect, cx, cy) {	cx = (rect.start_x + rect.end_x) / 2; \
									cy = (rect.start_y + rect.end_y) / 2; }

#define RANGE_WIDTH(arr) (arr.x2 - arr.x1)
#define RANGE_HEIGHT(arr) (arr.y2 - arr.y1)
#define RANGEPTR_WIDTH(arr) (arr->x2 - arr->x1)
#define RANGEPTR_HEIGHT(arr) (arr->y2 - arr->y1)
#define RANGE_CENTER(arr, cx, cy) {cx = (arr.x1+arr.x2)/2; cy= (arr.y1+arr.y2)/2;}

class OvxNode;
class OvxHelper
{
public:
	//vx_common
	template<typename T>
	static T					getNodeParameterByIndex(vx_node node, vx_int32 index, T def_val);
	static vx_status			release(vx_reference);

	//vx_node
	static vx_status			setNodeBorder(OvxNode* node_ptr, i32 border_mode, i32 value = 0);

	//vx_image
	static i32					getWidth(vx_image);
	static i32					getHeight(vx_image);
	static i32					getFormat(vx_image);
	static i32					getChannels(vx_image image);
	static Recti				getValidRectangle(vx_image image);
	static vx_status			setValidRectangle(vx_image dst, vx_int32 width, vx_int32 height);
	static vx_status			setValidRectOnly(vx_image dst, vx_int32 width, vx_int32 height);
	static vx_status			copyImage(vx_image dst, vx_image src);
	static vx_status			readImage(void* dst, i32 width, i32 height, vx_image src);
	static vx_status			writeImage(vx_image dst, const void* src, i32 width, i32 height, i32 bpp, i32 pad_size = 0);
	static vx_status			setImage(vx_image dst, void* src, i32 width, i32 height, i32 bpp);
	static vx_status			getImage(vx_image dst, void* src, i32 width, i32 height, i32 bpp);
	static vx_status			clearImage(vx_image dst, u8 val = 0x00);

	static i32					imageFormat(i32 channel);

	//vx_pyramid
	static i32					getWidth(vx_pyramid);
	static i32					getHeight(vx_pyramid);
	static i32					getFormat(vx_pyramid);
	static i32					getScale(vx_pyramid);
	static i32					getLevel(vx_pyramid);
	static vx_status			releasePyramid(vx_pyramid);

	//vx_scalar
	static i32					getFormat(vx_scalar);
	static vx_status			readScalar(void* dst, vx_scalar scalar);
	static vx_status			writeScalar(vx_scalar scalar, void* src);

	//vx_matrix
	static i32					getWidth(vx_matrix);
	static i32					getHeight(vx_matrix);
	static i32					getFormat(vx_matrix);
	static vx_status			writeMatrix(vx_matrix mat, void* src);

	//vx_array
	static i32					getFormat(vx_array);
	static i32					getCapacity(vx_array);
	static i32					getItemSize(vx_array);
	static vx_status			readArray(void* dst, i32 count, i32 stride_bytes, vx_array arr);
	static vx_status			readArray(void* dst, i32 pos, i32 count, i32 stride_bytes, vx_array arr);
	static vx_status			writeArray(vx_array arr, const void* data_ptr, i32 count, i32 stride_bytes);
	static vx_status			appendArray(vx_array arr, const void* data_ptr, i32 count, i32 stride_bytes);
	static vx_status			clearArray(vx_array arr);
	static vx_status			copyArray(vx_array src_arr, vx_array dst_arr);
	static vx_status			makeFullArray(vx_array arr);

	//vx_keypoints
	static vx_status			readKeyPointf32(f32* dst, vx_array src_arr);
	static vx_status			writeKeyPointf32(vx_array dst_arr, f32* src, i32 count);

	//vx_remap
	static i32					getWidth(vx_remap);
	static i32					getHeight(vx_remap);

	static i32					sizeImageElem(i32 type);
	static i32					sizeArrayElem(i32 type);
	static i32					sizeMatrixElem(i32 type);

	//vx_threshold
	static i32					getFormat(vx_threshold);
	static i32					getThreshold(vx_threshold);
	static i32					getThresholdType(vx_threshold);
	static vx_status			writeThreshold(vx_threshold threshold, i32 th, i32 l_value = 0, i32 h_value = 0xFF);
};

template<typename T>
T OvxHelper::getNodeParameterByIndex(vx_node node, vx_int32 index, T def_val)
{
	T val_data;
	vx_scalar val;
	vx_parameter param;
	vx_action act_code = VX_ACTION_CONTINUE;

	param = vxGetParameterByIndex(node, index);
	if (param)
	{
		vxQueryParameter(param, VX_PARAMETER_REF, &val, sizeof(val));
		if (val)
		{
			vxCopyScalar(val, &val_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
		}
	}
	return val_data;
};
#endif
