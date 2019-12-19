#pragma once
#include "define.h"
#include "struct.h"

#define FITLINE_MIN_POINTS			2
#define FITCIRCLE_MIN_POINTS		3
#define FITELLIPSE_MIN_POINTS		5

struct FittedEllipse
{
	//ellipse data
	f32						r1;
	f32						r2;
	f32						an;
	vector2df				center;
	u16						contrast;
};

struct FittedCircle
{
	//circle data
	f32						radius;
	vector2df				center;
	u16						contrast;
};

//it is least square line fitting method in use opencv.
//http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#fitline
template <class T>
class CFitLine
{
public:
	CFitLine();
	~CFitLine();

	void					initFitLine();
	void					initBuffer(i32 max_pixels);
	void					checkBuffer(i32 max_pixels);
	void					freeBuffer();

	void					addPixels(Pixel3<T>* pixel_ptr, i32 count, bool keep_point = true);
	void					setPixelData(Pixel3<T>* pixel_ptr, i32 count);
	bool					fitLineLSM();
	f32						fitLine();
	f32						calcLSMFitLineError();
	f32						calcEdgeContrast();
	void					calcEdgeEndPoints(vector2df& st_point, vector2df& ed_point);
	f32						refitLineExcludeOutlier(f32 clip_factor, f32 clip_default);

	inline vector2df		getCenterPoint() { return m_center; };
	inline vector2df		getNormalizedDir() { return m_dir; };
	inline i32				getPixelCount() { return m_pixels; };
	inline f32				getFitError() { return m_fit_error; };

public:
	//accumlation data
	union 
	{
		struct {
			i64				m_sx, m_sy;
			i64				m_sx2, m_sy2, m_sxy;
		} id;
		struct {
			f64				m_sx, m_sy;
			f64				m_sx2, m_sy2, m_sxy;
		} fd;
	} u;

	//tempoary line pixel data
	i32						m_pixels;
	Pixel3<T>*				m_pixel_ptr;
	i32						m_max_pixels;
	bool					m_is_external_alloc;

	//line data
	vector2df				m_center;
	vector2df				m_dir;
	f32						m_fit_error;
};
typedef CFitLine<u16>		CFitLineu;
typedef CFitLine<f32>		CFitLinef;

//it is least square circle fitting method in 
//http://www.cs.bsu.edu/homepages/kerryj/kjones/circles.pdf
template <class T>
class CFitCircle
{
public:
	CFitCircle();
	~CFitCircle();

	void					initFitCircle();
	void					initBuffer(i32 max_pixels);
	void					checkBuffer(i32 max_pixels);
	void					freeBuffer();

	void					addPixels(Pixel3<T>* pixel_ptr, i32 count, bool keep_points);
	void					setPixelData(Pixel3<T>* pixel_ptr, i32 count);
	bool					fitCircleLSM();
	f32						fitCircle();
	f32						calcLSMFitCircleError();
	f32						calcEdgeContrast();
	f32						getCoverage();
	f32						refitCircleExcludeOutlier(f32 clip_factor, f32 clip_default);

	inline f32				getRadius() { return m_radius; };
	inline f32				getFitError() { return m_fit_error; };
	inline i32				getPixelCount() { return m_pixels; };
	inline vector2df		getCenterPoint() { return m_center; };

public:
	//accumlation data
	union
	{
		struct {
			i64				m_sx, m_sy;
			i64				m_sx2, m_sy2, m_sxy;
			i64				m_sx3, m_sy3, m_sx2y, m_sxy2;
		} id;
		struct {
			f64				m_sx, m_sy;
			f64				m_sx2, m_sy2, m_sxy;
			f64				m_sx3, m_sy3, m_sx2y, m_sxy2;
		} fd;
	} u;
	
	//tempoary circle pixel data
	i32						m_pixels;
	Pixel3<T>*				m_pixel_ptr;
	i32						m_max_pixels;
	bool					m_is_external_alloc;

	//circle data
	vector2df				m_center;
	f32						m_radius;
	f32						m_fit_error;
};
typedef CFitCircle<u16>		CFitCircleu;
typedef CFitCircle<f32>		CFitCirclef;

template <class T>
class CFitShape
{
public:
	CFitShape();
	~CFitShape();

	static f32				fitLine(Pixel3<T>* pixel_ptr, i32 count, CFitLine<T>& line);
	static bool				fitLineOnly(Pixel3<T>* pixel_ptr, i32 count, CFitLine<T>& line);

	static f32				fitCircle(Pixel3<T>* pixel_ptr, i32 count, CFitCircle<T>& circle);
	static bool				fitCircleOnly(Pixel3<T>* pixel_ptr, i32 count, CFitCircle<T>& circle);

	static f32				getFitCircleError(std::vector<Pixel3<T>>& pt_vec, vector2df center, f32 radius);
	static f32				updateRadiusinCircle(std::vector<Pixel3<T>>& pt_vec, vector2df center, f32& fit_error);
	static vector2df		updateCenterinCircle(std::vector<Pixel3<T>>& pt_vec, f32 radius, f32& fit_error);

	static f32				fitEllipse(Pixel<T>* pixel_ptr, i32 st_pos, i32 ed_pos, FittedEllipse& fit_ellipse);
	static f32				getFitEllipseError(Pixel<T>* pixel_ptr, i32 st_pos, i32 ed_pos, FittedEllipse& fit_ellipse);
};
typedef CFitShape<u16>		CFitShapeu;
typedef CFitShape<f32>		CFitShapef;

#include "fit_shape-ini.h"