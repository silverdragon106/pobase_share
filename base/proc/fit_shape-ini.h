#pragma once

#include "fit_shape.h"
#include "base.h"
#include "opencv2/opencv.hpp"

//////////////////////////////////////////////////////////////////////////
template <class T>
CFitLine<T>::CFitLine()
{
	memset(this, 0, sizeof(CFitLine));
}

template <class T>
CFitLine<T>::~CFitLine()
{
	freeBuffer();
}

template <class T>
void CFitLine<T>::initBuffer(i32 max_pixels)
{
	freeBuffer();
	m_max_pixels = max_pixels;
	m_pixel_ptr = po_new Pixel3<T>[max_pixels];
	m_is_external_alloc = false;
}

template <class T>
void CFitLine<T>::checkBuffer(i32 max_pixels)
{
	if (!m_is_external_alloc && max_pixels <= m_max_pixels)
	{
		return;
	}
	initBuffer(max_pixels);
}

template <class T>
void CFitLine<T>::freeBuffer()
{
	if (!m_is_external_alloc)
	{
		POSAFE_DELETE_ARRAY(m_pixel_ptr);
	}
	m_max_pixels = 0;
	m_pixel_ptr = NULL;
}

template <class T>
void CFitLine<T>::initFitLine()
{
	m_pixels = 0;
	memset(&u, 0, sizeof(u));
}

template <class T>
void CFitLine<T>::setPixelData(Pixel3<T>* pixel_ptr, i32 count)
{
	m_pixels = count;
	m_pixel_ptr = pixel_ptr;
}

template <class T>
f32 CFitLine<T>::fitLine()
{
	if (!fitLineLSM())
	{
		return PO_MAXINT;
	}
	return calcLSMFitLineError();
}

template <class T>
f32 CFitLine<T>::calcLSMFitLineError()
{
	if (m_pixels < FITLINE_MIN_POINTS)
	{
		return PO_MAXINT;
	}

	f32 cx = m_center.x;
	f32 cy = m_center.y;
	f32 dx, dy, fit_error = 0;
	Pixel3<T>* tmp_pixel_ptr = m_pixel_ptr;

	for (i32 i = 0; i < m_pixels; i++, tmp_pixel_ptr++)
	{
		dx = (f32)tmp_pixel_ptr->x - cx;
		dy = (f32)tmp_pixel_ptr->y - cy;
		fit_error += CPOBase::distPt2Line(dx, dy, m_dir);
	}
	m_fit_error = fit_error / m_pixels;
	return m_fit_error;
}

template <class T>
f32 CFitLine<T>::calcEdgeContrast()
{
	if (m_pixels <= 0)
	{
		return 0;
	}

	f32 contrast = 0;
	Pixel3<T>* tmp_pixel_ptr = m_pixel_ptr;
	for (i32 i = 0; i < m_pixels; i++, tmp_pixel_ptr++)
	{
		contrast += (f32)tmp_pixel_ptr->g;
	}
	return contrast / m_pixels;
}

template <class T>
void CFitLine<T>::calcEdgeEndPoints(vector2df& st_point, vector2df& ed_point)
{
	st_point = m_center;
	ed_point = m_center;
	if (m_pixels <= 0)
	{
		return;
	}

	f32 cx = m_center.x;
	f32 cy = m_center.y;
	f32 dx, dy, len, min_len = PO_MAXINT, max_len = -PO_MAXINT;

	Pixel3<T>* tmp_pixel_ptr = m_pixel_ptr;
	vector2df pt_dir;

	for (i32 i = 0; i < m_pixels; i++, tmp_pixel_ptr++)
	{
		dx = (f32)tmp_pixel_ptr->x - cx;
		dy = (f32)tmp_pixel_ptr->y - cy;
		pt_dir = vector2df(dx, dy);

		pt_dir.normalize(len);
		len *= pt_dir.dotProduct(m_dir);
		if (len > max_len)
		{
			max_len = len;
		}
		if (len < min_len)
		{
			min_len = len;
		}
	}
	st_point = m_center + m_dir*min_len;
	ed_point = m_center + m_dir*max_len;
}

template <class T>
f32 CFitLine<T>::refitLineExcludeOutlier(f32 clip_factor, f32 clip_default)
{
	if (m_pixels < FITLINE_MIN_POINTS)
	{
		return PO_MAXINT;
	}

	i32 i, count = 0;
	f32 cx = m_center.x;
	f32 cy = m_center.y;
	f32 dx, dy, d0;
	f64 de = 0, de2 = 0;
	Pixel3<T>* tmp_pixel_ptr = m_pixel_ptr;

	for (i = 0; i < m_pixels; i++)
	{
		dx = (f32)tmp_pixel_ptr->x - cx;
		dy = (f32)tmp_pixel_ptr->y - cy;
		d0 = CPOBase::distPt2Line(dx, dy, m_dir);
		de += d0; de2 = d0*d0;
	}

	de /= m_pixels;
	f32 clip_error = clip_factor * 
				po::_max(std::sqrt(po::_max(de2 / m_pixels - de*de, 0)), clip_default);

	//remove outlier sample points
	std::vector<Pixel3<T>> sample_vec;
	sample_vec.resize(m_pixels);
	tmp_pixel_ptr = m_pixel_ptr;

	for (i = 0; i < m_pixels; i++, tmp_pixel_ptr++)
	{
		dx = (f32)tmp_pixel_ptr->x - cx;
		dy = (f32)tmp_pixel_ptr->y - cy;
		d0 = CPOBase::distPt2Line(dx, dy, m_dir);
		if (d0 < clip_error)
		{
			sample_vec[count] = *tmp_pixel_ptr;
			count++;
		}
	}

	//refit line
	initFitLine();
	addPixels(sample_vec.data(), count, true);
	return fitLine();
}

//////////////////////////////////////////////////////////////////////////
template <class T>
CFitCircle<T>::CFitCircle()
{
	memset(this, 0, sizeof(CFitCircle));
}

template <class T>
CFitCircle<T>::~CFitCircle()
{
	freeBuffer();
}

template <class T>
void CFitCircle<T>::initBuffer(i32 max_pixels)
{
	freeBuffer();
	m_max_pixels = max_pixels;
	m_pixel_ptr = po_new Pixel3<T>[max_pixels];
	m_is_external_alloc = false;
}

template <class T>
void CFitCircle<T>::checkBuffer(i32 max_pixels)
{
	if (!m_is_external_alloc && max_pixels <= m_max_pixels)
	{
		return;
	}
	initBuffer(max_pixels);
}

template <class T>
void CFitCircle<T>::freeBuffer()
{
	if (!m_is_external_alloc)
	{
		POSAFE_DELETE_ARRAY(m_pixel_ptr);
	}
	m_max_pixels = 0;
	m_pixel_ptr = NULL;
}

template <class T>
void CFitCircle<T>::initFitCircle()
{
	m_pixels = 0;
	memset(&u, 0, sizeof(u));
}

template <class T>
void CFitCircle<T>::setPixelData(Pixel3<T>* pixel_ptr, i32 count)
{
	m_pixels = count;
	m_pixel_ptr = pixel_ptr;
}

template <class T>
f32 CFitCircle<T>::fitCircle()
{
	if (!fitCircleLSM())
	{
		return PO_MAXINT;
	}
	return calcLSMFitCircleError();
}

template <class T>
f32 CFitCircle<T>::calcLSMFitCircleError()
{
	if (m_pixels < FITCIRCLE_MIN_POINTS)
	{
		return PO_MAXINT;
	}

	f32 cx = m_center.x;
	f32 cy = m_center.y;
	f32 dx, dy, fit_error = 0;
	Pixel3<T>* tmp_pixel_ptr = m_pixel_ptr;

	for (i32 i = 0; i < m_pixels; i++, tmp_pixel_ptr++)
	{
		dx = (f32)tmp_pixel_ptr->x - cx;
		dy = (f32)tmp_pixel_ptr->y - cy;
		fit_error += std::abs(sqrt(dx*dx + dy*dy) - m_radius);
	}
	m_fit_error = fit_error / m_pixels;
	return m_fit_error;
}

template <class T>
f32 CFitCircle<T>::refitCircleExcludeOutlier(f32 clip_factor, f32 clip_default)
{
	i32 i, count = 0;
	if (m_pixels < FITCIRCLE_MIN_POINTS)
	{
		return PO_MAXINT;
	}

	//calc sample dev and clip threshold
	f32 cx = m_center.x;
	f32 cy = m_center.y;
	f32 dx, dy, dr, de = 0, de2 = 0;
	Pixel3<T>* tmp_pixel_ptr = m_pixel_ptr;
	for (i = 0; i < m_pixels; i++, tmp_pixel_ptr++)
	{
		dx = (f32)tmp_pixel_ptr->x - cx;
		dy = (f32)tmp_pixel_ptr->y - cy;
		dr = std::abs(sqrtf(dx*dx + dy*dy) - m_radius);
		de += dr; de2 += dr*dr;
	}

	de /= m_pixels;
	f32 clip = clip_factor * 
			po::_max(std::sqrt(po::_max(de2 / m_pixels - de*de, 0)), clip_default);
	f32 min_clip = po::_max(m_radius - clip, 0);
	f32 max_clip = m_radius + clip;
	min_clip *= min_clip;
	max_clip *= max_clip;

	//remove outlier sample points
	std::vector<Pixel3<T>> sample_vec;
	sample_vec.resize(m_pixels);
	tmp_pixel_ptr = m_pixel_ptr;

	for (i = 0; i < m_pixels; i++, tmp_pixel_ptr++)
	{
		dx = (f32)tmp_pixel_ptr->x - cx;
		dy = (f32)tmp_pixel_ptr->y - cy;
		de = dx*dx + dy*dy;
		if (CPOBase::checkRange(de, min_clip, max_clip))
		{
			sample_vec[count] = *tmp_pixel_ptr;
			count++;
		}
	}

	//refit circle
	initFitCircle();
	checkBuffer(count);
	addPixels(sample_vec.data(), count, true);
	return fitCircle();
}

template <class T>
f32 CFitCircle<T>::calcEdgeContrast()
{
	if (m_pixels <= 0)
	{
		return 0;
	}

	f32 contrast = 0;
	Pixel3<T>* tmp_pixel_ptr = m_pixel_ptr;
	for (i32 i = 0; i < m_pixels; i++, tmp_pixel_ptr++)
	{
		contrast += tmp_pixel_ptr->g;
	}

	return contrast / m_pixels;
}

template <class T>
f32 CFitCircle<T>::getCoverage()
{
	if (m_radius <= PO_EPSILON)
	{
		return 0;
	}
	return (f32)m_pixels / (PO_PI2*m_radius);
}

//////////////////////////////////////////////////////////////////////////
template <class T>
CFitShape<T>::CFitShape()
{
}

template <class T>
CFitShape<T>::~CFitShape()
{
}

template <class T>
f32 CFitShape<T>::fitLine(Pixel3<T>* pixel_ptr, i32 count, CFitLine<T>& line)
{
	if (count < FITLINE_MIN_POINTS)
	{
		return PO_MAXINT;
	}

	line.initFitLine();
	line.addPixels(pixel_ptr, count, false);
	line.fitLine();
	line.setPixelData(NULL, 0);
	return line.getFitError();
}

template <class T>
bool CFitShape<T>::fitLineOnly(Pixel3<T>* pixel_ptr, i32 count, CFitLine<T>& line)
{
	if (count < FITLINE_MIN_POINTS)
	{
		return false;
	}

	line.initFitLine();
	line.addPixels(pixel_ptr, count, false);
	line.fitLineLSM();
	line.setPixelData(NULL, 0);
	return true;
}

template <class T>
f32 CFitShape<T>::fitCircle(Pixel3<T>* pixel_ptr, i32 count, CFitCircle<T>& circle)
{
	if (count < FITCIRCLE_MIN_POINTS)
	{
		return PO_MAXINT;
	}

	circle.initFitCircle();
	circle.addPixels(pixel_ptr, count, false);
	circle.fitCircle();
	circle.setPixelData(NULL, 0);
	return circle.getFitError();
}

template <class T>
bool CFitShape<T>::fitCircleOnly(Pixel3<T>* pixel_ptr, i32 count, CFitCircle<T>& circle)
{
	if (count < FITCIRCLE_MIN_POINTS)
	{
		return false;
	}

	circle.initFitCircle();
	circle.addPixels(pixel_ptr, count, false);
	circle.fitCircleLSM();
	circle.setPixelData(NULL, 0);
	return true;
}

template <class T>
f32 CFitShape<T>::fitEllipse(Pixel<T>* pixel_ptr, i32 st_pos, i32 ed_pos, FittedEllipse& fit_ellipse)
{
	//It is least square ellipse fitting method in opencv. http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#fitline
	//New fitellipse algorithm, contributed by Dr. Daniel Weiss
	f64 cx = 0;
	f64 cy = 0;
	i32 i, count = ed_pos - st_pos;
	if (count < FITELLIPSE_MIN_POINTS)
	{
		return PO_MAXINT;
	}

	for (i = st_pos; i < ed_pos; i++)
	{
		cx += pixel_ptr[i].x;
		cy += pixel_ptr[i].y;
	}
	cx /= count;
	cy /= count;

	f64 dx, dy, t;
	f64 xp[5], rp[5];
	f64* Ad = po_new f64[5 * count];
	f64* bd = po_new f64[count];
	const f64 min_eps = PO_EPSILON;

	//first fit for parameters A-E
	cv::Mat A(count, 5, CV_64F, Ad);
	cv::Mat b(count, 1, CV_64F, bd);
	cv::Mat x(5, 1, CV_64F, xp);

	count = 0;
	for (i = st_pos; i < ed_pos; i++)
	{
		dx = (f64)pixel_ptr[i].x - cx;
		dy = (f64)pixel_ptr[i].y - cy;

		bd[count] = 10000.0; //1.0?
		Ad[count * 5] = -dx * dx; //A-C signs inverted as proposed by APP
		Ad[count * 5 + 1] = -dy * dy;
		Ad[count * 5 + 2] = -dx * dy;
		Ad[count * 5 + 3] = dx;
		Ad[count * 5 + 4] = dy;
		count++;
	}

	cv::solve(A, b, x, cv::DECOMP_SVD);

	//now use general-form parameters A - E to find the ellipse center:
	//differentiate general form wrt x/y to get two equations for cx and cy
	A = cv::Mat(2, 2, CV_64F, Ad);
	b = cv::Mat(2, 1, CV_64F, bd);
	x = cv::Mat(2, 1, CV_64F, rp);
	Ad[0] = 2 * xp[0];
	Ad[1] = Ad[2] = xp[2];
	Ad[3] = 2 * xp[1];
	bd[0] = xp[3];
	bd[1] = xp[4];
	cv::solve(A, b, x, cv::DECOMP_SVD);

	//re-fit for parameters A-C with those center coordinates
	A = cv::Mat(count, 3, CV_64F, Ad);
	b = cv::Mat(count, 1, CV_64F, bd);
	x = cv::Mat(3, 1, CV_64F, xp);

	count = 0;
	for (i = st_pos; i < ed_pos; i++)
	{
		dx = (f64)pixel_ptr[i].x - cx;
		dy = (f64)pixel_ptr[i].y - cy;

		bd[count] = 1.0;
		Ad[count * 3] = (dx - rp[0]) * (dx - rp[0]);
		Ad[count * 3 + 1] = (dy - rp[1]) * (dy - rp[1]);
		Ad[count * 3 + 2] = (dx - rp[0]) * (dy - rp[1]);
		count++;
	}
	cv::solve(A, b, x, cv::DECOMP_SVD);

	//store angle and radii
	rp[4] = -0.5 * atan2(xp[2], xp[1] - xp[0]); //convert from APP angle usage

	if (std::abs(xp[2]) > min_eps)
	{
		t = xp[2] / sin(-2.0 * rp[4]);
	}
	else //ellipse is rotated by an integer multiple of pi/2
	{
		t = xp[1] - xp[0];
	}

	rp[2] = std::abs(xp[0] + xp[1] - t);
	if (rp[2] > min_eps)
	{
		rp[2] = std::sqrt(2.0 / rp[2]);
	}

	rp[3] = std::abs(xp[0] + xp[1] + t);
	if (rp[3] > min_eps)
	{
		rp[3] = std::sqrt(2.0 / rp[3]);
	}

	if (rp[2] > rp[3])
	{
		fit_ellipse.r1 = rp[2];
		fit_ellipse.r2 = rp[3];
	}
	else
	{
		fit_ellipse.r1 = rp[3];
		fit_ellipse.r2 = rp[2];
	}
	fit_ellipse.an = rp[4];
	fit_ellipse.center = vector2df(cx + rp[0], cy + rp[1]);

	POSAFE_DELETE_ARRAY(Ad);
	POSAFE_DELETE_ARRAY(bd);
	return getFitEllipseError(pixel_ptr, st_pos, ed_pos, fit_ellipse);
}

template <class T>
f32 CFitShape<T>::getFitEllipseError(Pixel<T>* pixel_ptr, i32 st_pos, i32 ed_pos, FittedEllipse& fit_ellipse)
{
	i32 i, count = ed_pos - st_pos;
	if (count <= 0)
	{
		return PO_MAXINT;
	}

	f32 dx, dy, dan, d1, d2;
	f32 cx = fit_ellipse.center.x;
	f32 cy = fit_ellipse.center.y;
	f32 r1 = fit_ellipse.r1;
	f32 r2 = fit_ellipse.r2;
	f32 an = fit_ellipse.an;
	f32 fit_error = 0;

	for (i = st_pos; i < ed_pos; i++)
	{
		dx = (f32)pixel_ptr[i].x - cx;
		dy = (f32)pixel_ptr[i].y - cy;
		dan = atan2(dy, dx) - an;
		d1 = r1*cosf(dan);
		d2 = r2*sinf(dan);
		fit_error += std::fabs(CPOBase::length(dx, dy) - CPOBase::length(d1, d2));
	}
	return fit_error / count;
}

template <class T>
f32 CFitShape<T>::getFitCircleError(std::vector<Pixel3<T>>& pt_vec, vector2df center, f32 radius)
{
	i32 i, count = (i32)pt_vec.size();
	if (count <= 0)
	{
		return 0;
	}

	f32 cx = center.x;
	f32 cy = center.y;
	f32 dist, fit_error = 0;
	for (i = 0; i < count; i++)
	{
		Pixel3<T>& tmp = pt_vec[i];
		dist = CPOBase::distance((f32)tmp.x, (f32)tmp.y, cx, cy);
		fit_error += std::abs(dist - radius);
	}
	return fit_error / count;
}

template <class T>
f32 CFitShape<T>::updateRadiusinCircle(std::vector<Pixel3<T>>& pt_vec, vector2df center, f32& fit_error)
{
	f32 dr2 = 0;
	f32 cx = center.x;
	f32 cy = center.y;
	i32 i, count = (i32)pt_vec.size();
	if (count <= 0)
	{
		return 0;
	}

	for (i = 0; i < count; i++)
	{
		Pixel3<T>& tmp = pt_vec[i];
		dr2 += CPOBase::distanceSQ((f32)tmp.x, (f32)tmp.y, cx, cy);
	}
	f32 new_r = std::sqrt(dr2 / count);
	fit_error = CFitShape::getFitCircleError(pt_vec, center, new_r);
	return new_r;
}

template <class T>
vector2df CFitShape<T>::updateCenterinCircle(std::vector<Pixel3<T>>& pt_vec, f32 radius, f32& fit_error)
{
	return vector2df();
}
