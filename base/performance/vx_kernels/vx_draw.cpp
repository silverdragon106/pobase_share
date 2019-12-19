#include "vx_draw.h"
#if defined(POR_WITH_OVX)

#include "performance/openvx_pool/ovx_base.h"

#ifdef POR_WITH_CUDA
#include "performance/cuda_kernels/nvx_draw.h"
#endif

vx_status _vxDrawCircle(u8* img_ptr, vx_rectangle_t rect, i32 stride_width, f32 cx, f32 cy, f32 r, u8 value)
{
	if (!img_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	i32 x0 = rect.start_x;
	i32 y0 = rect.start_y;
	i32 x1 = VX_MAX(cx - r - 1, rect.start_x);
	i32 x2 = VX_MIN(cx + r + 1, rect.end_x);
	i32 y1 = VX_MAX(cy - r - 1, rect.start_y);
	i32 y2 = VX_MIN(cy + r + 1, rect.end_y);

	//draw circle
	i32 x, y;
	f32 dx, dy, r2 = r*r;
	u8* scan_img_ptr = NULL;

	for (y = y1; y < y2; y++)
	{
		scan_img_ptr = img_ptr + (y - y0)*stride_width - x0;
		for (x = x1; x < x2; x++)
		{
			dx = x - cx;
			dy = y - cy;
			if (dx*dx + dy*dy <= r2)
			{
				scan_img_ptr[x] = value;
			}
		}
	}
	return VX_SUCCESS;
}

vx_status drawCircleKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_array param = (vx_array)parameters[kDrawCircleKernelInParamArray];
	vx_image draw_image = (vx_image)parameters[kDrawCircleKernelOutImage];

	if (count == kDrawCircleKernelParamCount)
	{
		status = VX_SUCCESS;

#ifdef POR_WITH_CUDA
		MAP_ARRAY(param, VX_READ_ONLY);
		NVX_MAP_IMAGE(draw_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && num_param >= 4)
		{
			f32 cx, cy, r;
			u8 value;
			cx = ((f32*)data_param)[0];
			cy = ((f32*)data_param)[1];
			r = ((f32*)data_param)[2];
			value = VX_MIN((u8)((f32*)data_param)[3], 0xFF);
			status |= cuDrawCircle((u8*)data_draw_image, rect_draw_image, SCAN_WIDTH(draw_image), cx, cy, r, value);
		}
		NVX_UNMAP_ARRAY(param);
		NVX_UNMAP_IMAGE(draw_image);
#else
		MAP_ARRAY(param, VX_READ_ONLY);
		MAP_IMAGE(draw_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && num_param >= 4)
		{
			f32 cx, cy, r;
			u8 value;
			cx = ((f32*)data_param)[0];
			cy = ((f32*)data_param)[1];
			r = ((f32*)data_param)[2];
			value = VX_MIN((u8)((f32*)data_param)[3], 0xFF);
			status |= _vxDrawCircle((u8*)data_draw_image, rect_draw_image, SCAN_WIDTH(draw_image), cx, cy, r, value);
		}
		UNMAP_ARRAY(param);
		UNMAP_IMAGE(draw_image);
#endif
	}
	return status;
}

vx_status drawCircleKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status drawCircleKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status drawCircleKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kDrawCircleKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kDrawCircleKernelOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kDrawCircleKernelOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kDrawCircleKernelOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kDrawCircleKernelOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kDrawCircleKernelOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kDrawCircleKernelOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status _vxDrawEllipse(u8* img_ptr, vx_rectangle_t rect, i32 stride_width, f32 cx, f32 cy, f32 r1, f32 r2, u8 value)
{
	if (!img_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	i32 x0 = rect.start_x;
	i32 y0 = rect.start_y;
	i32 x1 = VX_MAX(cx - r1 - 1, rect.start_x);
	i32 x2 = VX_MIN(cx + r1 + 1, rect.end_x);
	i32 y1 = VX_MAX(cy - r2 - 1, rect.start_y);
	i32 y2 = VX_MIN(cy + r2 + 1, rect.end_y);

	i32 x, y;
	f32 dx, dy;
	f32 r12 = r1 * r1, r22 = r2 * r2, cov_r2 = r12 * r22;
	u8* scan_img_ptr = NULL;

	//draw ellipse
	for (y = y1; y < y2; y++)
	{
		scan_img_ptr = img_ptr + (y - y0)*stride_width - x0;
		for (x = x1; x < x2; x++)
		{
			dx = x - cx;
			dy = y - cy;
			if (dx*dx*r22 + dy * dy*r12 <= cov_r2)
			{
				scan_img_ptr[x] = value;
			}
		}
	}
	return VX_SUCCESS;
}

vx_status drawEllipseKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_array param = (vx_array)parameters[kDrawEllipseKernelInParamArray];
	vx_image draw_image = (vx_image)parameters[kDrawEllipseKernelOutImage];

	if (count == kDrawEllipseKernelParamCount)
	{
		status = VX_SUCCESS;

#ifdef POR_WITH_CUDA
		MAP_ARRAY(param, VX_READ_ONLY);
		NVX_MAP_IMAGE(draw_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && num_param >= 5)
		{
			f32 cx, cy, r1, r2;
			u8 value;
			cx = ((f32*)data_param)[0];
			cy = ((f32*)data_param)[1];
			r1 = ((f32*)data_param)[2] / 2;
			r2 = ((f32*)data_param)[3] / 2;
			value = VX_MIN((u8)((f32*)data_param)[4], 0xFF);
			status |= cuDrawEllipse((u8*)data_draw_image, rect_draw_image, SCAN_WIDTH(draw_image), cx, cy, r1, r2, value);
		}
		NVX_UNMAP_ARRAY(param);
		NVX_UNMAP_IMAGE(draw_image);
#else
		MAP_ARRAY(param, VX_READ_ONLY);
		MAP_IMAGE(draw_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && num_param >= 5)
		{
			f32 cx, cy, r1, r2;
			u8 value;
			cx = ((f32*)data_param)[0];
			cy = ((f32*)data_param)[1];
			r1 = ((f32*)data_param)[2] / 2;
			r2 = ((f32*)data_param)[3] / 2;
			value = VX_MIN((u8)((f32*)data_param)[4], 0xFF);
			status |= _vxDrawEllipse((u8*)data_draw_image, rect_draw_image, SCAN_WIDTH(draw_image), cx, cy, r1, r2, value);
		}
		UNMAP_ARRAY(param);
		UNMAP_IMAGE(draw_image);
#endif
	}
	return status;
}

vx_status drawEllipseKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status drawEllipseKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status drawEllipseKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kDrawEllipseKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kDrawEllipseKernelOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kDrawEllipseKernelOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kDrawEllipseKernelOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kDrawEllipseKernelOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kDrawEllipseKernelOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kDrawEllipseKernelOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status _vxDrawPolygon(u8* img_ptr, vx_rectangle_t rect, i32 stride_width,
					vector2df* poly_ptr, i32 poly_pt_count, u8 value)
{
	if (!img_ptr || !poly_ptr || !CPOBase::isPositive(poly_pt_count))
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	//see more detail... http://alienryderflex.com/polygon/
	i32 i, ni, x, y;
	f32 dx, dy, dtmp;
	f32* poly_mul_ptr = po_new f32[poly_pt_count];
	f32* poly_con_ptr = po_new f32[poly_pt_count];

	for (i = 0; i < poly_pt_count; i++)
	{
		ni = (i + 1) % poly_pt_count;
		if (poly_ptr[i].y == poly_ptr[ni].y)
		{
			poly_mul_ptr[i] = 0;
			poly_con_ptr[i] = poly_ptr[i].x;
		}
		else
		{
			dx = poly_ptr[ni].x - poly_ptr[i].x;
			dy = poly_ptr[ni].y - poly_ptr[i].y;

			dtmp = dx / dy;
			poly_mul_ptr[i] = dtmp;
			poly_con_ptr[i] = poly_ptr[i].x - poly_ptr[i].y * dtmp;
		}
	}

	//draw polygon
	bool is_in_polygon;
	i32 x1 = rect.start_x;
	i32 y1 = rect.start_y;
	i32 x2 = rect.end_x;
	i32 y2 = rect.end_y;
	u8* scan_img_ptr = NULL;

	for (y = y1; y < y2; y++)
	{
		scan_img_ptr = img_ptr + (y - y1)*stride_width - x1;
		for (x = x1; x < x2; x++)
		{
			is_in_polygon = false;
			for (i = 0; i < poly_pt_count; i++)
			{
				ni = (i + 1) % poly_pt_count;
				if ((poly_ptr[i].y < y && poly_ptr[ni].y >= y) || (poly_ptr[ni].y < y && poly_ptr[i].y >= y))
				{
					is_in_polygon ^= (y*poly_mul_ptr[i] + poly_con_ptr[i] < x);
				}
			}

			//check point inside polygon
			if (is_in_polygon)
			{
				scan_img_ptr[x] = value;
			}
		}
	}

	//free buffer
	POSAFE_DELETE_ARRAY(poly_mul_ptr);
	POSAFE_DELETE_ARRAY(poly_con_ptr);
	return VX_SUCCESS;
}

vx_status drawPolygonKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_array param = (vx_array)parameters[kDrawPolygonKernelInParamArray];
	vx_image draw_image = (vx_image)parameters[kDrawPolygonKernelOutImage];

	if (count == kDrawPolygonKernelParamCount)
	{
		status = VX_SUCCESS;

#ifdef POR_WITH_CUDA
		MAP_ARRAY(param, VX_READ_ONLY);
		NVX_MAP_IMAGE(draw_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && num_param > 4)
		{
			i32 pt_count = (i32)((f32*)data_param)[0];
			u8 value = VX_MIN((u8)((f32*)data_param)[1], 0xFF);
			status |= cuDrawPolygon((u8*)data_draw_image, rect_draw_image, SCAN_WIDTH(draw_image),
								(cuVector2df*)data_param + 1, pt_count, value);
		}
		UNMAP_ARRAY(param);
		NVX_UNMAP_IMAGE(draw_image);
#else
		MAP_ARRAY(param, VX_READ_ONLY);
		MAP_IMAGE(draw_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && num_param > 4)
		{
			i32 pt_count = (i32)((f32*)data_param)[0];
			u8 value = VX_MIN((u8)((f32*)data_param)[1], 0xFF);
			status |= _vxDrawPolygon((u8*)data_draw_image, rect_draw_image, SCAN_WIDTH(draw_image),
								(vector2df*)data_param + 1, pt_count, value);
		}
		UNMAP_ARRAY(param);
		UNMAP_IMAGE(draw_image);
#endif
	}
	return status;
}

vx_status VX_CALLBACK drawPolygonKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status VX_CALLBACK drawPolygonKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status VX_CALLBACK drawPolygonKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kDrawPolygonKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kDrawPolygonKernelOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kDrawPolygonKernelOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kDrawPolygonKernelOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kDrawPolygonKernelOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kDrawPolygonKernelOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kDrawPolygonKernelOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}

//////////////////////////////////////////////////////////////////////////
vx_status _vxDrawRing(u8* img_ptr, vx_rectangle_t rect, i32 stride_width,
				f32 cx, f32 cy, f32 min_r, f32 max_r, f32 st_angle, f32 ed_angle, u8 value)
{
	if (!img_ptr)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	i32 x0 = rect.start_x;
	i32 y0 = rect.start_y;
	i32 x1 = VX_MAX(cx - max_r - 1, rect.start_x);
	i32 x2 = VX_MIN(cx + max_r + 1, rect.end_x);
	i32 y1 = VX_MAX(cy - max_r - 1, rect.start_y);
	i32 y2 = VX_MIN(cy + max_r + 1, rect.end_y);

	i32 x, y;
	f32 dx, dy, dist2;
	f32 min_r2 = min_r * min_r, max_r2 = max_r * max_r;
	f32 angle_len = ed_angle - st_angle;
	f32 angle_diff;
	u8* scan_img_ptr = NULL;

	//draw ring
	for (y = y1; y < y2; y++)
	{
		scan_img_ptr = img_ptr + (y - y0)*stride_width - x0;
		for (x = x1; x < x2; x++)
		{
			dx = x - cx;
			dy = y - cy;
			dist2 = dx * dx + dy * dy;
			if (dist2 < min_r2 || dist2 > max_r2)
			{
				continue;
			}

			angle_diff = CPOBase::getAngleRegDiff(CPOBase::getVectorAngle(dx, dy), st_angle);
			if (angle_diff < angle_len)
			{
				scan_img_ptr[x] = value;
			}
		}
	}
	return VX_SUCCESS;
}

vx_status drawRingKernel(vx_node node, const vx_reference *parameters, vx_uint32 count)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_array param = (vx_array)parameters[kDrawRingKernelInParamArray];
	vx_image draw_image = (vx_image)parameters[kDrawRingKernelOutImage];

	if (count == kDrawRingKernelParamCount)
	{
		status = VX_SUCCESS;

#ifdef POR_WITH_CUDA
		MAP_ARRAY(param, VX_READ_ONLY);
		NVX_MAP_IMAGE(draw_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && num_param >= 7)
		{
			f32 cx, cy, min_r, max_r, st_angle, ed_angle;
			u8 value;
			cx = ((f32*)data_param)[0];
			cy = ((f32*)data_param)[1];
			min_r = ((f32*)data_param)[2];
			max_r = ((f32*)data_param)[3];
			st_angle = ((f32*)data_param)[4];
			ed_angle = ((f32*)data_param)[5];
			value = VX_MIN((u8)((f32*)data_param)[6], 0xFF);
			status |= cuDrawRing((u8*)data_draw_image, rect_draw_image, SCAN_WIDTH(draw_image),
							cx, cy, min_r, max_r, st_angle, ed_angle, value);
		}

		NVX_UNMAP_ARRAY(param);
		NVX_UNMAP_IMAGE(draw_image);

#else
		MAP_ARRAY(param, VX_READ_ONLY);
		MAP_IMAGE(draw_image, VX_WRITE_ONLY);

		if (status == VX_SUCCESS && num_param >= 7)
		{
			f32 cx, cy, min_r, max_r, st_angle, ed_angle;
			u8 value;
			cx = ((f32*)data_param)[0];
			cy = ((f32*)data_param)[1];
			min_r = ((f32*)data_param)[2];
			max_r = ((f32*)data_param)[3];
			st_angle = ((f32*)data_param)[4];
			ed_angle = ((f32*)data_param)[5];
			value = VX_MIN((u8)((f32*)data_param)[6], 0xFF);
			status |= _vxDrawRing((u8*)data_draw_image, rect_draw_image, SCAN_WIDTH(draw_image),
							cx, cy, min_r, max_r, st_angle, ed_angle, value);
		}
		UNMAP_ARRAY(param);
		UNMAP_IMAGE(draw_image);
#endif
	}
	return status;
}

vx_status drawRingKernelInitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status drawRingKernelDeinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	/* CensusTransformInitialize requires no initialization of memory or resources */
	return VX_SUCCESS;
}

vx_status drawRingKernelValidator(vx_node node, const vx_reference parameters[], vx_uint32 count, vx_meta_format metas[])
{
	vx_status status = VX_SUCCESS;
	if (count != kDrawRingKernelParamCount)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}

	/* output validation */
	vx_uint32 width = OvxHelper::getWidth((vx_image)parameters[kDrawRingKernelOutImage]);
	vx_uint32 height = OvxHelper::getHeight((vx_image)parameters[kDrawRingKernelOutImage]);
	vx_df_image format = OvxHelper::getFormat((vx_image)parameters[kDrawRingKernelOutImage]);

	status |= vxSetMetaFormatAttribute(metas[kDrawRingKernelOutImage], VX_IMAGE_WIDTH, &width, sizeof(width));
	status |= vxSetMetaFormatAttribute(metas[kDrawRingKernelOutImage], VX_IMAGE_HEIGHT, &height, sizeof(height));
	status |= vxSetMetaFormatAttribute(metas[kDrawRingKernelOutImage], VX_IMAGE_FORMAT, &format, sizeof(format));
	return status;
}
#endif
