#include "fit_shape.h"
#include "base.h"
#include "proc/image_proc.h"

template <>
void CFitLine<u16>::addPixels(Pixel3<u16>* pixel_ptr, i32 count, bool keep_points)
{
	i64 x = 0, y = 0;
	i64 x2 = 0, y2 = 0, xy = 0;
	Pixel3<u16>* tmp_pixel_ptr = pixel_ptr;
	i32 px, py;

	for (i32 i = 0; i < count; i++, tmp_pixel_ptr++)
	{
		px = tmp_pixel_ptr->x;
		py = tmp_pixel_ptr->y;
		x += px;
		y += py;
		x2 += px*px;
		y2 += py*py;
		xy += px*py;
	}

	u.id.m_sx += x; u.id.m_sy += y;
	u.id.m_sx2 += x2; u.id.m_sy2 += y2; u.id.m_sxy += xy;

	if (keep_points)
	{
		if (m_pixels + count > m_max_pixels)
		{
			m_pixels = 0;
			initBuffer(m_pixels + count);
		}
		CPOBase::memCopy(m_pixel_ptr + m_pixels, pixel_ptr, count);
		m_pixels += count;
		m_is_external_alloc = false;
	}
	else
	{
		freeBuffer();
		m_pixel_ptr = pixel_ptr;
		m_pixels = count;
		m_is_external_alloc = true;
	}
}

template <>
void CFitLine<f32>::addPixels(Pixel3<f32>* pixel_ptr, i32 count, bool keep_points)
{
	f64 x = 0, y = 0;
	f64 x2 = 0, y2 = 0, xy = 0;
	Pixel3<f32>* tmp_pixel_ptr = pixel_ptr;
	f32 px, py;

	for (i32 i = 0; i < count; i++, tmp_pixel_ptr++)
	{
		px = tmp_pixel_ptr->x;
		py = tmp_pixel_ptr->y;
		x += px;
		y += py;
		x2 += px*px;
		y2 += py*py;
		xy += px*py;
	}

	u.fd.m_sx += x; u.fd.m_sy += y;
	u.fd.m_sx2 += x2; u.fd.m_sy2 += y2; u.fd.m_sxy += xy;

	if (keep_points)
	{
		if (m_pixels + count > m_max_pixels)
		{
			m_pixels = 0;
			initBuffer(m_pixels + count);
		}
		CPOBase::memCopy(m_pixel_ptr + m_pixels, pixel_ptr, count);
		m_pixels += count;
		m_is_external_alloc = false;
	}
	else
	{
		freeBuffer();
		m_pixel_ptr = pixel_ptr;
		m_pixels = count;
		m_is_external_alloc = true;
	}
}

template <>
bool CFitLine<u16>::fitLineLSM()
{
	if (m_pixels < FITLINE_MIN_POINTS)
	{
		return false;
	}

	f64 cx = (f64)u.id.m_sx / m_pixels;
	f64 cy = (f64)u.id.m_sy / m_pixels;
	f64 cx2 = (f64)u.id.m_sx2 / m_pixels;
	f64 cy2 = (f64)u.id.m_sy2 / m_pixels;
	f64 cxy = (f64)u.id.m_sxy / m_pixels;

	f64 ddx = cx2 - cx*cx;
	f64 ddy = cy2 - cy*cy;
	f64 ddxy = cxy - cx*cy;
	f32 dt = atan2(2 * ddxy, ddx - ddy) / 2;

	m_center = vector2df(cx, cy);
	m_dir = vector2df(cosf(dt), sinf(dt));
	return true;
}

template <>
bool CFitLine<f32>::fitLineLSM()
{
	if (m_pixels < FITLINE_MIN_POINTS)
	{
		return false;
	}

	f64 cx = (f64)u.fd.m_sx / m_pixels;
	f64 cy = (f64)u.fd.m_sy / m_pixels;
	f64 cx2 = (f64)u.fd.m_sx2 / m_pixels;
	f64 cy2 = (f64)u.fd.m_sy2 / m_pixels;
	f64 cxy = (f64)u.fd.m_sxy / m_pixels;

	f64 ddx = cx2 - cx*cx;
	f64 ddy = cy2 - cy*cy;
	f64 ddxy = cxy - cx*cy;
	f32 dt = atan2(2 * ddxy, ddx - ddy) / 2;

	m_center = vector2df(cx, cy);
	m_dir = vector2df(cosf(dt), sinf(dt));
	return true;
}

//////////////////////////////////////////////////////////////////////////

template <>
void CFitCircle<u16>::addPixels(Pixel3u* pixel_ptr, i32 count, bool keep_points)
{
	i64 px, py, pxy, px2, py2;
	i64 x = 0, y = 0;
	i64 x2 = 0, y2 = 0, xy = 0;
	i64 x3 = 0, y3 = 0, xy2 = 0, x2y = 0;
	Pixel3u* tmp_pixel_ptr = pixel_ptr;

	for (i32 i = 0; i < count; i++, tmp_pixel_ptr++)
	{
		px = tmp_pixel_ptr->x;
		py = tmp_pixel_ptr->y;
		pxy = px*py;
		px2 = px*px;
		py2 = py*py;

		x += px; y += py;
		x2 += px2; y2 += py2; xy += pxy;
		x3 += px2*px; y3 += py2*py; xy2 += pxy*py; x2y += pxy*px;
	}

	u.id.m_sx += x; u.id.m_sy += y;
	u.id.m_sx2 += x2; u.id.m_sy2 += y2; u.id.m_sxy += xy;
	u.id.m_sx3 += x3; u.id.m_sy3 += y3; u.id.m_sx2y += x2y; u.id.m_sxy2 += xy2;

	if (keep_points)
	{
		if (m_pixels + count > m_max_pixels)
		{
			m_pixels = 0;
			initBuffer(m_pixels + count);
		}
		CPOBase::memCopy(m_pixel_ptr + m_pixels, pixel_ptr, count);
		m_pixels += count;
		m_is_external_alloc = false;
	}
	else
	{
		freeBuffer();
		m_pixel_ptr = pixel_ptr;
		m_pixels = count;
		m_is_external_alloc = true;
	}
}

template <>
void CFitCircle<f32>::addPixels(Pixel3f* pixel_ptr, i32 count, bool keep_points)
{
	f64 px, py, pxy, px2, py2;
	f64 x = 0, y = 0;
	f64 x2 = 0, y2 = 0, xy = 0;
	f64 x3 = 0, y3 = 0, xy2 = 0, x2y = 0;
	Pixel3f* tmp_pixel_ptr = pixel_ptr;

	for (i32 i = 0; i < count; i++, tmp_pixel_ptr++)
	{
		px = tmp_pixel_ptr->x;
		py = tmp_pixel_ptr->y;
		pxy = px*py;
		px2 = px*px;
		py2 = py*py;

		x += px; y += py;
		x2 += px2; y2 += py2; xy += pxy;
		x3 += px2*px; y3 += py2*py; xy2 += pxy*py; x2y += pxy*px;
	}

	u.fd.m_sx += x; u.fd.m_sy += y;
	u.fd.m_sx2 += x2; u.fd.m_sy2 += y2; u.fd.m_sxy += xy;
	u.fd.m_sx3 += x3; u.fd.m_sy3 += y3; u.fd.m_sx2y += x2y; u.fd.m_sxy2 += xy2;

	if (keep_points)
	{
		if (m_pixels + count > m_max_pixels)
		{
			m_pixels = 0;
			initBuffer(m_pixels + count);
		}
		CPOBase::memCopy(m_pixel_ptr + m_pixels, pixel_ptr, count);
		m_pixels += count;
		m_is_external_alloc = false;
	}
	else
	{
		freeBuffer();
		m_pixel_ptr = pixel_ptr;
		m_pixels = count;
		m_is_external_alloc = true;
	}
}

template <>
bool CFitCircle<u16>::fitCircleLSM()
{
	if (m_pixels < FITCIRCLE_MIN_POINTS)
	{
		return false;
	}

	f64 A = m_pixels*u.id.m_sx2 - u.id.m_sx*u.id.m_sx;
	f64 B = m_pixels*u.id.m_sxy - u.id.m_sx*u.id.m_sy;
	f64 C = m_pixels*u.id.m_sy2 - u.id.m_sy*u.id.m_sy;
	f64 D = 0.5f*(m_pixels*u.id.m_sxy2 - u.id.m_sx*u.id.m_sy2 + m_pixels*u.id.m_sx3 - u.id.m_sx*u.id.m_sx2);
	f64 E = 0.5f*(m_pixels*u.id.m_sx2y - u.id.m_sy*u.id.m_sx2 + m_pixels*u.id.m_sy3 - u.id.m_sy*u.id.m_sy2);

	f64 cond = A*C - B*B;
	if (std::abs(cond) < PO_EPSILON)
	{
		return false;
	}

	f32 cx = (D*C - B*E) / cond;
	f32 cy = (A*E - B*D) / cond;
	f32 dx, dy, dr = 0;
	Pixel3u* tmp_pixel_ptr = m_pixel_ptr;

	for (i32 i = 0; i < m_pixels; i++)
	{
		dx = tmp_pixel_ptr->x - cx;
		dy = tmp_pixel_ptr->y - cy;
		dr += dx*dx + dy*dy;
		tmp_pixel_ptr++;
	}
	m_center = vector2df(cx, cy);
	m_radius = sqrt(dr / m_pixels);
	return true;
}

template <>
bool CFitCircle<f32>::fitCircleLSM()
{
	if (m_pixels < FITCIRCLE_MIN_POINTS)
	{
		return false;
	}

	f64 A = m_pixels*u.fd.m_sx2 - u.fd.m_sx*u.fd.m_sx;
	f64 B = m_pixels*u.fd.m_sxy - u.fd.m_sx*u.fd.m_sy;
	f64 C = m_pixels*u.fd.m_sy2 - u.fd.m_sy*u.fd.m_sy;
	f64 D = 0.5f*(m_pixels*u.fd.m_sxy2 - u.fd.m_sx*u.fd.m_sy2 + m_pixels*u.fd.m_sx3 - u.fd.m_sx*u.fd.m_sx2);
	f64 E = 0.5f*(m_pixels*u.fd.m_sx2y - u.fd.m_sy*u.fd.m_sx2 + m_pixels*u.fd.m_sy3 - u.fd.m_sy*u.fd.m_sy2);

	f64 cond = A*C - B*B;
	if (std::abs(cond) < PO_EPSILON)
	{
		return false;
	}

	f32 cx = (D*C - B*E) / cond;
	f32 cy = (A*E - B*D) / cond;
	f32 dx, dy, dr = 0;
	Pixel3f* tmp_pixel_ptr = m_pixel_ptr;

	for (i32 i = 0; i < m_pixels; i++)
	{
		dx = tmp_pixel_ptr->x - cx;
		dy = tmp_pixel_ptr->y - cy;
		dr += dx*dx + dy*dy;
		tmp_pixel_ptr++;
	}
	m_center = vector2df(cx, cy);
	m_radius = sqrt(dr / m_pixels);
	return true;
}
