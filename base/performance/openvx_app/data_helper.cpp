#include "data_helper.h"

#ifdef POR_WITH_OVX
//#include "../openvx_mvs/openvx_define.h"

vx_enum	CDataHelper::type_pointf;
vx_enum	CDataHelper::type_rectf;
vx_enum	CDataHelper::type_recti;

//////////////////////////////////////////////////////////////////////////
vx_status CDataHelper::initInstance(vx_context context)
{
	vx_status status = VX_SUCCESS;

	type_rectf = vxRegisterUserStruct(context, sizeof(Rectf));
	type_recti = vxRegisterUserStruct(context, sizeof(Recti));
	type_pointf = vxRegisterUserStruct(context, sizeof(vector2df));
	return status;
}

vx_array CDataHelper::createRectf(vx_context context)
{
	return vxCreateArray(context, type_rectf, 1);
}

vx_array CDataHelper::createRecti(vx_context context)
{
	return vxCreateArray(context, type_recti, 1);
}

vx_status CDataHelper::setRectfData(vx_array dst, Rectf& src)
{
	vx_size dst_num = 0;
	vx_status status = VX_SUCCESS;
	status |= vxQueryArray(dst, VX_ARRAY_NUMITEMS, &dst_num, sizeof(vx_size));
	if (dst_num == 0)
	{
		status |= vxAddArrayItems(dst, 1, &src, sizeof(Rectf));
	}
	else
	{
		status |= vxCopyArrayRange(dst, 0, 1, sizeof(Rectf), &src, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
	}
	return status;
}

vx_status CDataHelper::setRectiData(vx_array dst, Recti& src)
{
	vx_size dst_num = 0;
	vx_status status = VX_SUCCESS;
	status |= vxQueryArray(dst, VX_ARRAY_NUMITEMS, &dst_num, sizeof(vx_size));
	if (dst_num == 0)
	{
		status |= vxAddArrayItems(dst, 1, &src, sizeof(Recti));
	}
	else
	{
		status |= vxCopyArrayRange(dst, 0, 1, sizeof(Recti), &src, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
	}
	return status;
}

vx_status CDataHelper::copyVxImage(void* dst_ptr, vx_image src)
{
	vx_status status = VX_SUCCESS;

	if (!dst_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	MAP_IMAGE(src, VX_READ_ONLY);
	if (data_src)
	{
		vx_int32 width = rect_src.end_x - rect_src.start_x;
		vx_int32 height = rect_src.end_y - rect_src.start_y;
		
		u8* p_src = NULL;
		u8* p_dst = NULL;
		vx_int32 src_stride_bytes = addr_src.stride_y;
		vx_int32 dst_stride_bytes = width* addr_src.stride_x;

		for (vx_int32 i = rect_src.start_y; i < rect_src.end_y; ++i)
		{
			p_dst = (u8*)dst_ptr + i * dst_stride_bytes;
			p_src = (u8*)data_src + i * src_stride_bytes;
			memcpy(p_dst, p_src, dst_stride_bytes);
		}
	}
	UNMAP_IMAGE(src);
	return status;
}

vx_status CDataHelper::copyVxKeyPoints(ptfvector* corner_vec_ptr, vx_array corners)
{
	vx_status status = VX_FAILURE;
	if (corner_vec_ptr)
	{
		MAP_ARRAY(corners, VX_READ_ONLY);
		vx_int32 i, count = (vx_int32)num_corners;
		corner_vec_ptr->resize(count);

		if (count > 0)
		{
			vx_int32 keep_count = 0;
			vx_keypoint_t* keypoint_ptr;
			keypoint_ptr = (vx_keypoint_t*)data_corners;
			
			Pixelf* corner_ptr = corner_vec_ptr->data();
			
			for (i = 0; i < count; ++i)
			{
				//check tracking error keypoint
				if (!keypoint_ptr[i].tracking_status)
				{
					continue;
				}
				
				corner_ptr[keep_count].x = keypoint_ptr[i].x;
				corner_ptr[keep_count].y = keypoint_ptr[i].y;
				keep_count++;
			}

			corner_vec_ptr->resize(keep_count);
			status = (keep_count > 0) ? VX_SUCCESS : status;
		}
		UNMAP_ARRAY(corners);
	}
	return status;
}

vx_status CDataHelper::copyImageData(vx_image dst, void* data_ptr, vx_int32 width, vx_int32 height, vx_int32 bpp)
{
	vx_status status = VX_SUCCESS;

	if (!data_ptr || !dst)
	{
		return status;
	}

	vx_imagepatch_addressing_t addr;
	vx_rectangle_t rect;
	vx_map_id map_id = 0;
	void* dst_data;

	rect.start_x = 0;
	rect.start_y = 0;
	rect.end_x = width;
	rect.end_y = height;

	//status |= vxGetValidRegionImage(dst, &rect);
	status |= vxSetImageValidRectangle(dst, &rect);
	status |= vxMapImagePatch(dst, &rect, 0, &map_id, &addr, &dst_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

	for (vx_int32 y = 0; y < addr.dim_y; y++)
	{
		void* p_dst = vxFormatImagePatchAddress2d(dst_data, 0, y, &addr);
		u8* p_src = (u8*)data_ptr + y * width * bpp;
		memcpy(p_dst, p_src, addr.dim_x * addr.stride_x);
	}

	vxUnmapImagePatch(dst, map_id);
	return status;
}

vx_status CDataHelper::copyImageData(vx_image dst, Recti& rect, void* data_ptr, vx_int32 width, vx_int32 height, vx_int32 bpp)
{
	vx_status status = VX_SUCCESS;

	if (!data_ptr || !dst)
	{
		return status;
	}

	vx_imagepatch_addressing_t addr;
	vx_rectangle_t vx_rect;
	vx_map_id map_id = 0;
	void* dst_data;
	
	vx_rect.start_x = 0;
	vx_rect.start_y = 0;
	vx_rect.end_x = rect.getWidth();
	vx_rect.end_y = rect.getHeight();

	//status |= vxGetValidRegionImage(dst, &rect);
	status |= vxSetImageValidRectangle(dst, &vx_rect);
	status |= vxMapImagePatch(dst, &vx_rect, 0, &map_id, &addr, &dst_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

	i32 x1 = rect.x1;
	i32 y1 = rect.y1;
	for (vx_int32 y = 0; y < addr.dim_y; y++)
	{
		void* p_dst = vxFormatImagePatchAddress2d(dst_data, 0, y, &addr);
		u8* p_src = (u8*)data_ptr + ((y + y1)*width + x1) * bpp;
		memcpy(p_dst, p_src, addr.dim_x);
	}

	vxUnmapImagePatch(dst, map_id);
	return status;
}

vx_status CDataHelper::copyMatrixData(vx_matrix dst, vx_float32* transform_ptr)
{
	return vxCopyMatrix(dst, (void*)transform_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
}

vx_status CDataHelper::setValidImageRect(vx_image dst, vx_int32 width, vx_int32 height)
{
	vx_status status = VX_SUCCESS;
	vx_rectangle_t rect;
	u8* tmp_dst_ptr;

	rect.start_x = 0;
	rect.start_y = 0;
	rect.end_x = width;
	rect.end_y = height;

	status |= vxSetImageValidRectangle(dst, &rect);
	if (status != VX_SUCCESS)
	{
		return status;
	}

	MAP_IMAGE(dst, VX_WRITE_ONLY);
	for (vx_int32 y = 0; y < height; y++)
	{
		tmp_dst_ptr = (u8*)data_dst + addr_dst.stride_y*y;
		memset(tmp_dst_ptr, 0, width);
	}
	UNMAP_IMAGE(dst);
	return status;
}

vx_status CDataHelper::setValidRectOnly(vx_image dst, vx_int32 width, vx_int32 height)
{
	vx_status status = VX_SUCCESS;
	vx_rectangle_t rect;

	rect.start_x = 0;
	rect.start_y = 0;
	rect.end_x = width;
	rect.end_y = height;

	status |= vxSetImageValidRectangle(dst, &rect);
	return status;
}

vx_status CDataHelper::getScalarValue(void* dst_ptr, vx_scalar src)
{
	if (!dst_ptr)
	{
		return VX_FAILURE;
	}
	return vxCopyScalar(src, dst_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
}

vx_status CDataHelper::setScalarValue(vx_scalar dst, void* src_ptr)
{
	if (!src_ptr)
	{
		return VX_FAILURE;
	}
	return vxCopyScalar(dst, src_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
}
#endif
