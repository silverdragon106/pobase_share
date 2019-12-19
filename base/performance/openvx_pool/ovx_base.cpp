#include "ovx_base.h"
#include "ovx_node.h"

#if defined(POR_WITH_OVX)

vx_status OvxHelper::release(vx_reference obj)
{
	return vxReleaseReference(&obj);
}

vx_status OvxHelper::setNodeBorder(OvxNode* node_ptr, i32 border_mode, i32 value)
{
	if (!node_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_border_t border;
	border.mode = border_mode;
	border.constant_value.U32 = value;
	return vxSetNodeAttribute(node_ptr->getVxNode(), VX_NODE_BORDER, &border, sizeof(vx_border_t));
}

i32 OvxHelper::getWidth(vx_image image)
{
	if (!image)
	{
		return 0;
	}

	vx_int32 width = 0;
	vxQueryImage(image, VX_IMAGE_WIDTH, (void*)&width, sizeof(vx_int32));
	return width;
}

i32 OvxHelper::getHeight(vx_image image)
{
	if (!image)
	{
		return 0;
	}

	vx_int32 height = 0;
	vxQueryImage(image, VX_IMAGE_HEIGHT, (void*)&height, sizeof(vx_int32));
	return height;
}

i32 OvxHelper::getFormat(vx_image image)
{
	if (!image)
	{
		return VX_DF_IMAGE_U8;
	}

	vx_int32 format = 0;
	vxQueryImage(image, VX_IMAGE_FORMAT, (void*)&format, sizeof(vx_int32));
	return format;
}

i32 OvxHelper::getChannels(vx_image image)
{
	if (!image)
	{
		return VX_DF_IMAGE_U8;
	}

	vx_size planes = 0;
	vxQueryImage(image, VX_IMAGE_PLANES, (void*)&planes, sizeof(vx_int32));
	return (i32)planes;
}

Recti OvxHelper::getValidRectangle(vx_image image)
{
	Recti rt;
	if (!image)
	{
		return rt;
	}
	vx_rectangle_t vx_rect;
	vxGetValidRegionImage(image, &vx_rect);
	return Recti(vx_rect.start_x, vx_rect.start_y, vx_rect.end_x, vx_rect.end_y);
}

vx_status OvxHelper::setValidRectangle(vx_image dst, vx_int32 width, vx_int32 height)
{
	vx_status status = VX_SUCCESS;
	vx_rectangle_t rect;

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
	if (status == VX_SUCCESS)
	{
		memset(data_dst, 0, addr_dst.stride_y*RECT_HEIGHT(rect_dst));
	}
	UNMAP_IMAGE(dst);
	return status;
}

vx_status OvxHelper::setValidRectOnly(vx_image dst, vx_int32 width, vx_int32 height)
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

vx_status OvxHelper::copyImage(vx_image dst, vx_image src)
{
	return VX_FAILURE;
}

vx_status OvxHelper::readImage(void* dst, i32 width, i32 height, vx_image src)
{
	vx_status status = VX_SUCCESS;

	if (!dst)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	MAP_IMAGE(src, VX_READ_ONLY);
	if (status == VX_SUCCESS && data_src != NULL)
	{
		void* p_src = NULL;
		void* p_dst = NULL;
		vx_int32 w = po::_min(width, rect_src.end_x - rect_src.start_x);
		vx_int32 h = po::_min(height, rect_src.end_y - rect_src.start_y);
		vx_int32 src_stride_bytes = addr_src.stride_y;
		vx_int32 dst_stride_bytes = w * addr_src.stride_x;

		for (vx_int32 i = 0; i < h; ++i)
		{
			p_dst = (u8*)dst +  i * dst_stride_bytes;
			p_src = (u8*)data_src + (rect_src.start_y + i) * src_stride_bytes;
			memcpy(p_dst, p_src, dst_stride_bytes);
		}
	}
	UNMAP_IMAGE(src);
	return status;
}

vx_status OvxHelper::writeImage(vx_image dst, const void* src, i32 width, i32 height, i32 bpp, i32 padding_size)
{
	vx_status status = VX_SUCCESS;

	if (!src || !dst)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_imagepatch_addressing_t addr;
	vx_rectangle_t rect;
	vx_map_id map_id = 0;
	void* dst_data;

	//clear and copy image
	i32 iw = OvxHelper::getWidth(dst);
	i32 ih = OvxHelper::getHeight(dst);
	i32 w = po::_min(iw, width);
	i32 h = po::_min(ih, height);
	i32 pad_x_size = po::_min(iw - w, padding_size);
	i32 pad_y_size = po::_min(ih - h, padding_size);
	
	rect.start_x = 0;
	rect.start_y = 0;
	rect.end_x = iw;
	rect.end_y = ih;
	VX_CHKRET_O(vxSetImageValidRectangle(dst, &rect));
	status |= vxMapImagePatch(dst, &rect, 0, &map_id, &addr, &dst_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
	if (status != VX_SUCCESS)
	{
		printlog_lvs2(QString("OVX WriteImage failed. error:%1").arg(status), LOG_SCOPE_OVX);
		return status;
	}

	i32 i, y;
	i32 stride_x = addr.stride_x, stride_y = addr.stride_y;
	memset(dst_data, 0, stride_y*RECT_HEIGHT(rect));

	//x-padding process
	u8* p_dst; u8* p_src;
	switch (bpp)
	{
		case 8:
		{
			u8 val; u8* p_val;
			for (y = 0; y < h; y++)
			{
				p_dst = (u8*)dst_data + y * stride_y;
				p_src = (u8*)src + y * width * bpp / 8;
				memcpy(p_dst, p_src, w * stride_x);
				
				p_val = (u8*)(p_dst + (w - 1) * stride_x);
				val = *p_val;
				for (i = 0; i < pad_x_size; i++)
				{
					*(++p_val) = val;
				}
			}
			break;
		}
		case 16:
		{
			u16 val; u16* p_val;
			for (y = 0; y < h; y++)
			{
				p_dst = (u8*)dst_data + y * stride_y;
				p_src = (u8*)src + y * width * bpp / 8;
				memcpy(p_dst, p_src, w * stride_x);

				p_val = (u16*)(p_dst + (w - 1) * stride_x);
				val = *p_val;
				for (i = 0; i < pad_x_size; i++)
				{
					*(++p_val) = val;
				}
			}
			break;
		}
		case 24:
		{
			for (y = 0; y < h; y++)
			{
				p_dst = (u8*)dst_data + y * stride_y;
				p_src = (u8*)src + y * width * bpp / 8;
				memcpy(p_dst, p_src, w * stride_x);

				p_src = p_dst + (w - 1) * stride_x;
				for (i = 0; i < pad_x_size; i++)
				{
					p_dst = p_src + stride_x;
					memcpy(p_dst,p_src, stride_x);
				}
			}
			break;
		}
		case 32:
		{
			u32 val; u32* p_val;
			for (y = 0; y < h; y++)
			{
				p_dst = (u8*)dst_data + y * stride_y;
				p_src = (u8*)src + y * width * bpp / 8;
				memcpy(p_dst, p_src, w * stride_x);

				p_val = (u32*)(p_dst + (w - 1) * stride_x);
				val = *p_val;
				for (i = 0; i < pad_x_size; i++)
				{
					*(++p_val) = val;
				}
			}
			break;
		}
		default:
		{
			VX_CHKRET_O(vxUnmapImagePatch(dst, map_id));
		}
	}

	//y-padding process
	i32 w_stride = (w + pad_x_size) * stride_x;
	p_src = (u8*)dst_data + (h - 1) * stride_y;
	p_dst = p_src;
	for (i = 0; i < pad_y_size; i++)
	{
		p_dst = p_dst + stride_y;
		memcpy(p_dst, p_src, w_stride);
	}
	VX_CHKRET_O(vxUnmapImagePatch(dst, map_id));

	rect.start_x = 0;
	rect.start_y = 0;
	rect.end_x = w + pad_x_size;
	rect.end_y = h + pad_y_size;
	VX_CHKRET_O(vxSetImageValidRectangle(dst, &rect));
	return status;
}

vx_status OvxHelper::setImage(vx_image dst, void* src, i32 width, i32 height, i32 bpp)
{
	vx_status status = VX_SUCCESS;

	if (!src || !dst)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_rectangle_t img_region;
	img_region.start_x = 0;
	img_region.start_y = 0;
	img_region.end_x = width;
	img_region.end_y = height;

	vx_imagepatch_addressing_t img_layout;
	img_layout.stride_x = bpp/8;
	img_layout.stride_y = width*bpp/8;

	status = vxCopyImagePatch(dst, &img_region, 0, &img_layout, src, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
	if (status != VX_SUCCESS)
	{
		printlog_lvs2(QString("OVX SetImage failed. error:%1").arg(status), LOG_SCOPE_OVX);
		return status;
	}

	status |= vxSetImageValidRectangle(dst, &img_region);
	return status;
}

vx_status OvxHelper::getImage(vx_image dst, void* src, i32 width, i32 height, i32 bpp)
{
	vx_status status = VX_SUCCESS;

	if (!src || !dst)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_rectangle_t img_region;
	img_region.start_x = 0;
	img_region.start_y = 0;
	img_region.end_x = width;
	img_region.end_y = height;

	vx_imagepatch_addressing_t img_layout;
	img_layout.stride_x = bpp/8;
	img_layout.stride_y = width*bpp/8;

	status = vxCopyImagePatch(dst, &img_region, 0, &img_layout, src, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
	if (status != VX_SUCCESS)
	{
		printlog_lvs2(QString("OVX GetImage failed. error:%1").arg(status), LOG_SCOPE_OVX);
		return status;
	}
	return status;
}

vx_status OvxHelper::clearImage(vx_image dst, u8 val)
{
	vx_status status = VX_SUCCESS;
	status |= vxSetImageValidRectangle(dst, NULL);

	MAP_IMAGE(dst, VX_WRITE_ONLY);
	if (status == VX_SUCCESS && data_dst != NULL)
	{
		memset(data_dst, val, addr_dst.stride_y*RECT_HEIGHT(rect_dst));
	}
	UNMAP_IMAGE(dst);
	return status;
}

i32 OvxHelper::getWidth(vx_pyramid pyr)
{
	if (!pyr)
	{
		return 0;
	}

	vx_uint32 width = 0;
	vxQueryPyramid(pyr, VX_PYRAMID_WIDTH, (void*)&width, sizeof(vx_uint32));
	return (i32)width;
}

i32 OvxHelper::getHeight(vx_pyramid pyr)
{
	if (!pyr)
	{
		return 0;
	}

	vx_uint32 height = 0;
	vxQueryPyramid(pyr, VX_PYRAMID_HEIGHT, (void*)&height, sizeof(vx_uint32));
	return (i32)height;
}

i32 OvxHelper::getFormat(vx_pyramid pyr)
{
	if (!pyr)
	{
		return 0;
	}

	vx_df_image format = VX_DF_IMAGE_VIRT;
	vxQueryPyramid(pyr, VX_PYRAMID_HEIGHT, (void*)&format, sizeof(vx_df_image));
	return (i32)format;
}

i32 OvxHelper::getScale(vx_pyramid pyr)
{
	if (!pyr)
	{
		return 0;
	}

	vx_float32 scale = 0;
	vxQueryPyramid(pyr, VX_PYRAMID_HEIGHT, (void*)&scale, sizeof(vx_float32));
	return (i32)scale;
}

i32 OvxHelper::getLevel(vx_pyramid pyr)
{
	if (!pyr)
	{
		return 0;
	}

	vx_size level = 0;
	vxQueryPyramid(pyr, VX_PYRAMID_LEVELS, (void*)&level, sizeof(vx_size));
	return (i32)level;
}

vx_status OvxHelper::releasePyramid(vx_pyramid pyramid)
{
	return vxReleasePyramid(&pyramid);
}

i32 OvxHelper::getWidth(vx_matrix mat)
{
	if (!mat)
	{
		return 0;
	}

	vx_size width = 0;
	vxQueryMatrix(mat, VX_MATRIX_COLUMNS, (void*)&width, sizeof(vx_size));
	return (i32)width;
}

i32 OvxHelper::getHeight(vx_matrix mat)
{
	if (!mat)
	{
		return 0;
	}

	vx_size height = 0;
	vxQueryMatrix(mat, VX_MATRIX_ROWS, (void*)&height, sizeof(vx_size));
	return (i32)height;
}

i32 OvxHelper::getFormat(vx_matrix mat)
{
	if (!mat)
	{
		return VX_TYPE_ENUM;
	}

	vx_int32 format = 0;
	vxQueryMatrix(mat, VX_MATRIX_TYPE, (void*)&format, sizeof(vx_int32));
	return format;
}

vx_status OvxHelper::writeMatrix(vx_matrix mat, void* src)
{
	vx_status status = VX_SUCCESS;

	if (!src || !mat)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
	return vxCopyMatrix(mat, src, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
}

i32 OvxHelper::getFormat(vx_array arr)
{
	if (!arr)
	{
		return VX_TYPE_ARRAY;
	}

	vx_int32 type;
	vxQueryArray(arr, VX_ARRAY_ITEMTYPE, (void*)&type, sizeof(vx_int32));
	return type;
}

i32 OvxHelper::getCapacity(vx_array arr)
{
	if (!arr)
	{
		return 0;
	}

	vx_size size;
	vxQueryArray(arr, VX_ARRAY_CAPACITY, (void*)&size, sizeof(vx_size));
	return (i32)size;
}

i32 OvxHelper::getItemSize(vx_array arr)
{
	if (!arr)
	{
		return 0;
	}

	vx_size item_size;
	vxQueryArray(arr, VX_ARRAY_NUMITEMS, (void*)&item_size, sizeof(vx_size));
	return (i32)item_size;
}

vx_status OvxHelper::readArray(void* dst, i32 count, i32 stride_bytes, vx_array arr)
{
	vx_status status = VX_SUCCESS;
	if (!dst || !arr || count < 1)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	status |= vxCopyArrayRange(arr, 0, count, stride_bytes, dst, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
	return status;
}

vx_status OvxHelper::readArray(void* dst, i32 pos, i32 count, i32 stride_bytes, vx_array arr)
{
	vx_status status = VX_SUCCESS;
	if (!dst || !arr || count < 1)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	status |= vxCopyArrayRange(arr, pos, pos + count, stride_bytes, dst, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
	return status;
}

vx_status OvxHelper::writeArray(vx_array arr, const void* data_ptr, i32 count, i32 stride_bytes)
{
	vx_status status = VX_SUCCESS;
	if (!arr || !data_ptr || count < 1)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
	status |= vxTruncateArray(arr, (vx_size)0);
	status |= vxAddArrayItems(arr, count, data_ptr, stride_bytes);
	return status;
}

vx_status OvxHelper::appendArray(vx_array arr, const void* data_ptr, i32 count, i32 stride_bytes)
{
	vx_status status = VX_SUCCESS;
	if (!arr || !data_ptr || count < 1)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
	status |= vxAddArrayItems(arr, count, data_ptr, stride_bytes);
	return status;
}

vx_status OvxHelper::clearArray(vx_array arr)
{
	vx_status status = VX_SUCCESS;
	if (!arr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
	status |= vxTruncateArray(arr, (vx_size)0);
	return status;
}

vx_status OvxHelper::copyArray(vx_array src_arr, vx_array dst_arr)
{
	if (!src_arr || !dst_arr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_status status = VX_SUCCESS;
	MAP_ARRAY(src_arr, VX_READ_ONLY);
	status |= vxTruncateArray(dst_arr, 0);
	status |= vxAddArrayItems(dst_arr, num_src_arr, data_src_arr, stride_src_arr);
	UNMAP_ARRAY(src_arr);
	return status;
}

vx_status OvxHelper::makeFullArray(vx_array arr)
{
	vx_size num_size;
	vx_size cap_size;
	vx_size item_size;

	vx_status status = VX_SUCCESS;
	vxQueryArray(arr, VX_ARRAY_NUMITEMS, (void*)&num_size, sizeof(vx_size));
	vxQueryArray(arr, VX_ARRAY_CAPACITY, (void*)&cap_size, sizeof(vx_size));
	if (num_size != cap_size)
	{
		vxQueryArray(arr, VX_ARRAY_ITEMSIZE, (void*)&item_size, sizeof(vx_size));
		u8* tmp_buffer_ptr = po_new u8[cap_size*item_size];

		status |= vxTruncateArray(arr, 0);
		status |= vxAddArrayItems(arr, cap_size, tmp_buffer_ptr, item_size);
		POSAFE_DELETE_ARRAY(tmp_buffer_ptr);
	}
	return status;
}

vx_status OvxHelper::readKeyPointf32(f32* dst, vx_array src_arr)
{
	if (!dst || !src_arr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_status status = VX_SUCCESS;
	MAP_ARRAY(src_arr, VX_READ_ONLY);
	
	vx_keypoint_t* pt_ptr = (vx_keypoint_t*)data_src_arr;
	i32 i, count = (vx_int32)num_src_arr;
	for (i = 0; i < count; i++)
	{
		*dst = pt_ptr->x; dst++;
		*dst = pt_ptr->y; dst++;
		pt_ptr++;
	}
	UNMAP_ARRAY(src_arr);
	return VX_SUCCESS;
}

vx_status OvxHelper::writeKeyPointf32(vx_array dst_arr, f32* src, i32 count)
{
	if (!dst_arr || !src || !CPOBase::isCount(count))
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_status status = VX_SUCCESS;
	status |= vxTruncateArray(dst_arr, (vx_size)0);

	vx_keypoint_t key_point;
	memset(&key_point, 0, sizeof(vx_keypoint_t));
	key_point.tracking_status = 1;

	for (i32 i = 0; i < count; i++)
	{
		key_point.x = *src; src++;
		key_point.y = *src; src++;
		status |= vxAddArrayItems(dst_arr, 1, &key_point, sizeof(key_point));
	}
	return VX_SUCCESS;
}

i32 OvxHelper::getWidth(vx_remap remap)
{
	if (!remap)
	{
		return 0;
	}

	vx_uint32 width = 0;
	vxQueryRemap(remap, VX_REMAP_DESTINATION_WIDTH, (void*)&width, sizeof(vx_uint32));
	return (i32)width;
}

i32 OvxHelper::getHeight(vx_remap remap)
{
	if (!remap)
	{
		return 0;
	}

	vx_uint32 height = 0;
	vxQueryRemap(remap, VX_REMAP_DESTINATION_HEIGHT, (void*)&height, sizeof(vx_uint32));
	return (i32)height;
}

i32 OvxHelper::getFormat(vx_scalar scalar)
{
	if (!scalar)
	{
		return VX_TYPE_SCALAR;
	}

	vx_int32 type;
	vxQueryScalar(scalar, VX_SCALAR_TYPE, (void*)&type, sizeof(vx_int32));
    return type;
}

vx_status OvxHelper::readScalar(void* dst, vx_scalar scalar)
{
	vx_status status = VX_SUCCESS;
	if (!scalar || !dst)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	status = vxCopyScalar(scalar, dst, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
	return status;
}

vx_status OvxHelper::writeScalar(vx_scalar scalar, void* src)
{
	vx_status status = VX_SUCCESS;
	if (!scalar)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	status = vxCopyScalar(scalar, src, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
	return status;
}

i32 OvxHelper::getFormat(vx_threshold threshold)
{
	if (!threshold)
	{
		return VX_TYPE_ENUM;
	}

	vx_int32 type;
	vxQueryThreshold(threshold, VX_THRESHOLD_DATA_TYPE, (void*)&type, sizeof(vx_int32));
	return type;
}

i32 OvxHelper::getThreshold(vx_threshold threshold)
{
	if (!threshold)
	{
		return VX_TYPE_ENUM;
	}

	vx_int32 value;
	vxQueryThreshold(threshold, VX_THRESHOLD_THRESHOLD_VALUE, (void*)&value, sizeof(vx_int32));
	return value;
}

i32 OvxHelper::getThresholdType(vx_threshold threshold)
{
	if (!threshold)
	{
		return VX_TYPE_ENUM;
	}

	vx_int32 type;
	vxQueryThreshold(threshold, VX_THRESHOLD_TYPE, (void*)&type, sizeof(vx_int32));
	return type;
}

i32 OvxHelper::writeThreshold(vx_threshold threshold, i32 th, i32 l_value, i32 h_value)
{
	if (!threshold)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_status status = VX_SUCCESS;
	status |= vxSetThresholdAttribute(threshold, VX_THRESHOLD_THRESHOLD_VALUE, &th, sizeof(th));
	status |= vxSetThresholdAttribute(threshold, VX_THRESHOLD_FALSE_VALUE, &l_value, sizeof(l_value));
	status |= vxSetThresholdAttribute(threshold, VX_THRESHOLD_TRUE_VALUE, &h_value, sizeof(h_value));
	return status;
}

i32 OvxHelper::imageFormat(i32 channel)
{
	switch (channel)
	{
		case 1: return VX_DF_IMAGE_U8;
		case 2: return VX_DF_IMAGE_YUV4; 
		case 3: return VX_DF_IMAGE_RGB;
	}
	return VX_DF_IMAGE_U8;
}

i32 OvxHelper::sizeImageElem(i32 type)
{
	switch (type)
	{
		case VX_DF_IMAGE_U8:
		{
			return 1;
		}
		case VX_DF_IMAGE_U16:
		case VX_DF_IMAGE_S16:
		{
			return 2;
		}
		case VX_DF_IMAGE_RGB:
		case VX_DF_IMAGE_NV12:
		case VX_DF_IMAGE_NV21:
		{
			return 3;
		}
	}
	return 4;
}

i32 OvxHelper::sizeArrayElem(i32 type)
{
	switch (type)
	{
		case VX_TYPE_INT16:
		case VX_TYPE_UINT16:
		{
			return 2;
		}
		case VX_TYPE_INT32:
		case VX_TYPE_UINT32:
		case VX_TYPE_FLOAT32:
		{
			return 4;
		}
		case VX_TYPE_INT64:
		case VX_TYPE_UINT64:
		case VX_TYPE_FLOAT64:
		{
			return 8;
		}
	}
	return 1;
}

i32 OvxHelper::sizeMatrixElem(i32 type)
{
	return sizeArrayElem(type);
}

#endif
