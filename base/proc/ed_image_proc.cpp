#include "ed_image_proc.h"
#include "image_proc.h"
#include "base.h"
#include "logger/logger.h"

#if defined(POR_WITH_OVX)
#include "openvx/sc_graph_pool.h"
#endif

//#define POR_TESTMODE

i32 compareAnchor(const void* p1, const void* p2)
{
	u16 grad1 = ((Pixel3u*)p1)->g;
	u16 grad2 = ((Pixel3u*)p2)->g;

	if (grad1 < grad2)
	{
		return 1;
	}
	else if (grad1 > grad2)
	{
		return -1;
	}
	return 0;
}

i32 compareEdgeSegment(const void* p1, const void* p2)
{
	CEDEdgeSegment* seg1_ptr = (CEDEdgeSegment*)p1;
	CEDEdgeSegment* seg2_ptr = (CEDEdgeSegment*)p2;

	f32 dist2_1 = CPOBase::distanceSQ(seg1_ptr->m_point_st, seg1_ptr->m_point_ed);
	f32 dist2_2 = CPOBase::distanceSQ(seg2_ptr->m_point_st, seg2_ptr->m_point_ed);

	if (dist2_1 < dist2_2)
	{
		return 1;
	}
	else if (dist2_1 > dist2_2)
	{
		return -1;
	}
	return 0;
}

i32 compareEdgeMergeCand(const void* p1, const void* p2)
{
	EDEdgeMergeCand* cand1_ptr = (EDEdgeMergeCand*)p1;
	EDEdgeMergeCand* cand2_ptr = (EDEdgeMergeCand*)p2;

	f32 dist2_1 = cand1_ptr->dist;
	f32 dist2_2 = cand2_ptr->dist;

	if (dist2_1 < dist2_2)
	{
		return -1;
	}
	else if (dist2_1 > dist2_2)
	{
		return 1;
	}
	return 0;
}

i32 compareArcSegment(const void* p1, const void* p2)
{
	CEDArcSegment* seg1_ptr = (CEDArcSegment*)p1;
	CEDArcSegment* seg2_ptr = (CEDArcSegment*)p2;

	f32 bound_len1 = seg1_ptr->m_bound_len;
	f32 bound_len2 = seg2_ptr->m_bound_len;
	f32 coverage1 = bound_len1 * bound_len1 / (seg1_ptr->m_radius * PO_PI2);
	f32 coverage2 = bound_len2 * bound_len2 / (seg2_ptr->m_radius * PO_PI2);

	if (coverage1 < coverage2)
	{
		return 1;
	}
	else if (coverage1 > coverage2)
	{
		return -1;
	}
	return 0;
}

i32 compareCircles(const void* p1, const void* p2)
{
	f32 coverage1 = ((CEDCircle*)p1)->m_coverage;
	f32 coverage2 = ((CEDCircle*)p2)->m_coverage;

	if (coverage1 < coverage2)
	{
		return 1;
	}
	else if (coverage1 > coverage2)
	{
		return -1;
	}
	return 0;
}

//////////////////////////////////////////////////////////////////////////
EDContourParam::EDContourParam()
{
	edge_strength = ED_CONTOUR_EDGETH;
	anchor_threshold = ED_CONTOUR_ANCHTH;
	anchor_interval = ED_CONTOUR_ANCHINTERVAL;
	min_contour_points = ED_CONTOUR_MINPTNUM;
	param_mode = kEDParamContour;
}

EDLineParam::EDLineParam()
{
	min_line_segment = ED_LINE_MIN_SEGMENT;
	max_line_segment = ED_LINE_MAX_SEGMENT;
	fit_line_error = ED_LINE_FITERROR;
	param_mode = kEDParamLine;
}

EDEdgeParam::EDEdgeParam()
{
	max_gap_size = ED_EDGE_MAX_GAPSIZE;
	base_cover_pixels = -1;
	fit_angle_error = ED_EDGE_MAX_ANGLERANGE;
	base_line_dir = vector2df(1, 0);
	base_tolerance_angle = -1;

	is_keep_points = false;
	remove_outlier = false;
	remove_clip_factor = 2.0f;
	param_mode = kEDParamEdge;
}

EDCircleParam::EDCircleParam()
{
	min_radius = 0;
	max_radius = PO_MAXINT;
	min_coverage = ED_CIRCLE_MINCOVERAGE;
	fit_circle_error = ED_CIRCLE_FITERROR;
	fit_circle_err_ratio = ED_CIRCLE_FITERRRATE;
	arc_rextend_rate = ED_CIRCLE_EXTRRATE;
	arc_cextend_rate = ED_CIRCLE_EXTCRATE;
	arc_dextend_rate = ED_CIRCLE_EXTDRATE;
	low_angle_cos = cosf(CPOBase::degToRad(ED_ARCLINE_LOWANGLE));
	high_angle_cos = cosf(CPOBase::degToRad(ED_ARCLINE_HIGHANGLE));
	
	is_keep_points = false;
	remove_outlier = false;
	remove_clip_factor = 2.0f;
	param_mode = kEDParamCircle;
}

//////////////////////////////////////////////////////////////////////////
CEDEdge::CEDEdge()
{
	m_center = vector2df();
	m_dir = vector2df();
	m_point_st = vector2df();
	m_point_ed = vector2df();

	m_coverage = 0;
	m_contrast = 0;
	m_fit_error = 0;
	m_reserved = 0;

	m_point_vec.clear();
}

void CEDEdge::setFitEdge(CFitLineu& fit_line, const vector2df& st_point, const vector2df& ed_point, bool keep_point)
{
	m_center = fit_line.getCenterPoint();
	m_dir = fit_line.getNormalizedDir();
	m_contrast = fit_line.calcEdgeContrast();
	m_fit_error = fit_line.getFitError();

	m_point_st = st_point;
	m_point_ed = ed_point;
	m_coverage = CPOBase::distance(st_point, ed_point);
	m_reserved = 0;

	m_point_vec.clear();

	if (keep_point)
	{
		m_point_vec.resize(fit_line.getPixelCount());
		CPOBase::memCopy(m_point_vec.data(), fit_line.m_pixel_ptr, fit_line.m_pixels);
	}
}

void CEDEdge::freeBuffer()
{
	m_point_vec.clear();
}

//////////////////////////////////////////////////////////////////////////
CEDCircle::CEDCircle()
{
	m_center = vector2df();
	m_radius = 0;

	m_coverage = 0;
	m_contrast = 0;
	m_fit_error = 0;
	
	m_is_used = false;
	m_point_vec.clear();
}

void CEDCircle::setFitCircle(CFitCircleu& fit_circle, bool keep_point)
{
	m_center = fit_circle.getCenterPoint();
	m_radius = fit_circle.getRadius();

	m_fit_error = fit_circle.getFitError();
	m_contrast = fit_circle.calcEdgeContrast();
	m_coverage = po::_min(1.0f, fit_circle.getPixelCount() / (PO_PI2*m_radius));
	
	m_is_used = false;
	m_point_vec.clear();

	if (keep_point)
	{
		m_point_vec.resize(fit_circle.getPixelCount());
		CPOBase::memCopy(m_point_vec.data(), fit_circle.m_pixel_ptr, fit_circle.m_pixels);
	}
}

void CEDCircle::setCoverage(f32 coverage)
{
	m_coverage = coverage;
}

void CEDCircle::freeBuffer()
{
	m_point_vec.clear();
}

//////////////////////////////////////////////////////////////////////////
CEDContour::CEDContour()
{
	memset(this, 0, sizeof(CEDContour));
}

CEDContour::~CEDContour()
{
}

void CEDContour::update(Pixel3u* pixel_ptr, i32 pos1, i32 pos2)
{
	if (pos2 - pos1 > 3)
	{
		m_st_pos = pos1;
		m_ed_pos = pos2;

		//check 8-neighborhood connection
		i32 npos = pos2 - 1;
		i32 dx = std::abs(pixel_ptr[npos].x - pixel_ptr[pos1].x);
		i32 dy = std::abs(pixel_ptr[npos].y - pixel_ptr[pos1].y);
		if (po::_max(dx, dy) <= 1)
		{
			m_is_closed = true;
		}
	}
}

CEDContourSet::CEDContourSet()
{
	memset(this, 0, sizeof(CEDContourSet));
}

CEDContourSet::~CEDContourSet()
{
	freeContours();
}

void CEDContourSet::initContours(i32 max_anchor_count, i32 max_px_count)
{
	freeContours();
	if (max_anchor_count > 0 && max_px_count > 0)
	{
		m_px_max_count = max_px_count;
		m_contour_max_count = max_anchor_count;
		m_pixel_ptr = po_new Pixel3u[max_px_count];
		m_contour_ptr = po_new CEDContour[max_anchor_count];
		memset(m_pixel_ptr, 0, sizeof(Pixel3u)*max_px_count);
		memset(m_contour_ptr, 0, sizeof(CEDContour)*max_anchor_count);
	}
}

void CEDContourSet::freeContours()
{
	m_px_max_count = 0;
	m_contour_max_count = 0;
	m_px_count = 0;
	m_contour_count = 0;
	POSAFE_DELETE_ARRAY(m_pixel_ptr);
	POSAFE_DELETE_ARRAY(m_contour_ptr);
}

void CEDContourSet::setContourPixel(Pixel3u* pixel_ptr, i32& pos1, i32& pos2)
{
	i32 count = pos2 - pos1;
	if (pixel_ptr && count > 0 && (m_px_count + count) <= m_px_max_count)
	{
		CPOBase::memCopy(m_pixel_ptr + m_px_count, pixel_ptr + pos1, count);
		pos1 = m_px_count; m_px_count += count;
		pos2 = m_px_count;
	}
}

CEDContour* CEDContourSet::getNewContour()
{
	if (m_contour_ptr && m_contour_count < m_contour_max_count)
	{
		m_contour_count++;
		return m_contour_ptr + m_contour_count - 1;
	}
	return NULL;
}

i32 CEDContourSet::getContourCount()
{
	return m_contour_count;
}

//////////////////////////////////////////////////////////////////////////
CEDLineSegment::CEDLineSegment()
{
	memset(this, 0, sizeof(CEDLineSegment));
}

CEDLineSegSet::CEDLineSegSet()
{
	memset(this, 0, sizeof(CEDLineSegSet));
}

CEDLineSegSet::~CEDLineSegSet()
{
	freeLines();
}

void CEDLineSegSet::initLines(i32 max_count, Pixel3u* pixel_ptr)
{
	if (max_count > 0 && pixel_ptr)
	{
		m_line_max_count = max_count;
		m_line_count = 0;
		m_pixel_ptr = pixel_ptr;
		m_line_seg_ptr = po_new CEDLineSegment[max_count];
		memset(m_line_seg_ptr, 0, sizeof(CEDLineSegment));
	}
}

void CEDLineSegSet::freeLines()
{
	if (m_is_embedd)
	{
		m_is_embedd = false;
		POSAFE_DELETE_ARRAY(m_pixel_ptr);
	}
	POSAFE_DELETE_ARRAY(m_line_seg_ptr);
	m_line_count = 0;
	m_line_max_count = 0;
}

CEDLineSegment* CEDLineSegSet::getLineSegData()
{
	return m_line_seg_ptr;
}

i32 CEDLineSegSet::getLineSegCount()
{
	return m_line_count;
}

CEDLineSegment* CEDLineSegSet::getNewLine()
{
	if (m_line_seg_ptr && m_line_count < m_line_max_count)
	{
		m_line_count++;
		return m_line_seg_ptr + m_line_count - 1;
	}
	return NULL;
}

//////////////////////////////////////////////////////////////////////////
CEDEdgeSegment::CEDEdgeSegment()
{
	m_center = vector2df();
	m_dir = vector2df();
	m_fit_error = 0;

	m_point_st = vector2df();
	m_point_ed = vector2df();
	m_line_vec.clear();

	m_index = -1;
	m_is_used = false;
}

f32 CEDEdgeSegment::getLength()
{
	return CPOBase::distance(m_point_st, m_point_ed);
}

CEDEdgeSegSet::CEDEdgeSegSet()
{
	memset(this, 0, sizeof(CEDEdgeSegSet));
}

CEDEdgeSegSet::~CEDEdgeSegSet()
{
	freeEdges();
}

void CEDEdgeSegSet::initEdges(i32 max_count, Pixel3u* pixel_ptr)
{
	freeEdges();
	if (max_count > 0 && pixel_ptr)
	{
		m_edge_max_count = max_count;
		m_edge_count = 0;
		m_pixel_ptr = pixel_ptr;
		m_edge_seg_ptr = po_new CEDEdgeSegment[max_count];
	}
}

void CEDEdgeSegSet::freeEdges()
{
	POSAFE_DELETE_ARRAY(m_edge_seg_ptr);
	m_edge_count = 0;
	m_edge_max_count = 0;
}

CEDEdgeSegment* CEDEdgeSegSet::getNewEdgeSeg()
{
	if (m_edge_seg_ptr && m_edge_count < m_edge_max_count)
	{
		CEDEdgeSegment* edge_seg_ptr = m_edge_seg_ptr + m_edge_count;
		edge_seg_ptr->m_index = m_edge_count;
		m_edge_count++;
		return edge_seg_ptr;
	}
	return NULL;
}

CEDEdgeSegment* CEDEdgeSegSet::getEdgeSegData()
{
	return m_edge_seg_ptr;
}

i32 CEDEdgeSegSet::getEdgeSegCount()
{
	return m_edge_count;
}

//////////////////////////////////////////////////////////////////////////
CEDArcSegment::CEDArcSegment()
{
	memset(this, 0, sizeof(CEDArcSegment));
}

void CEDArcSegment::updateArcSegment(CEDArcSegment& out, CEDLineSegSet& lines, i32 st_line, i32 ed_line)
{
	Pixel3u* pixel_ptr = lines.m_pixel_ptr;
	CEDLineSegment* lines_ptr = lines.m_line_seg_ptr;
	i32 st_index = lines_ptr[st_line].m_st_pos;
	i32 ed_index = lines_ptr[ed_line].m_ed_pos - 1;
	i32 mid_index = (st_index + ed_index) / 2;

	m_center = out.m_center;
	m_radius = out.m_radius;
	m_fit_error = out.m_fit_error;
	m_st_line = st_line;
	m_ed_line = ed_line;
	m_st_pos = pixel_ptr[st_index];
	m_ed_pos = pixel_ptr[ed_index];

	vector2df start_point = vector2df(m_st_pos.x, m_st_pos.y);
	vector2df end_point = vector2df(m_ed_pos.x, m_ed_pos.y);
	vector2df mid_point = vector2df(pixel_ptr[mid_index].x, pixel_ptr[mid_index].y);

	f32 an1 = CPOBase::getUnitVectorAngle((start_point - m_center).normalize());
	f32 an2 = CPOBase::getUnitVectorAngle((end_point - m_center).normalize());
	f32 an3 = CPOBase::getUnitVectorAngle((mid_point - m_center).normalize());
	f32 min_angle = po::_min(an1, an2);
	f32 max_angle = po::_max(an1, an2);

	if (min_angle < an3 && max_angle > an3)
	{
		m_bound_len = (max_angle - min_angle)*m_radius;
	}
	else
	{
		m_bound_len = (PO_PI2 + min_angle - max_angle)*m_radius;
	}
}

CEDArcSegSet::CEDArcSegSet()
{
	memset(this, 0, sizeof(CEDArcSegSet));
}

CEDArcSegSet::~CEDArcSegSet()
{
	freeArcSegments();
}

void CEDArcSegSet::initArcSegments(i32 max_count, Pixel3u* pixel_ptr)
{
	freeArcSegments();
	if (max_count > 0 && pixel_ptr)
	{
		m_arc_max_count = max_count;
		m_arc_count = 0;
		m_pixel_ptr = pixel_ptr;
		m_arc_seg_ptr = po_new CEDArcSegment[max_count];
		memset(m_arc_seg_ptr, 0, sizeof(CEDArcSegment));
	}
}

void CEDArcSegSet::freeArcSegments()
{
	if (m_is_embedd)
	{
		m_is_embedd = false;
		POSAFE_DELETE_ARRAY(m_pixel_ptr);
	}
	POSAFE_DELETE_ARRAY(m_arc_seg_ptr);
	m_arc_count = 0;
	m_arc_max_count = 0;
}

CEDArcSegment* CEDArcSegSet::getNewArcSeg()
{
	if (m_arc_seg_ptr && m_arc_count < m_arc_max_count)
	{
		CEDArcSegment* arc_seg_ptr = m_arc_seg_ptr + m_arc_count;
		arc_seg_ptr->m_index = m_arc_count;
		m_arc_count++;
		return arc_seg_ptr;
	}
	return NULL;
}

CEDArcSegment* CEDArcSegSet::getArcSegData()
{
	return m_arc_seg_ptr;
}

i32 CEDArcSegSet::getArcSegCount()
{
	return m_arc_count;
}

//////////////////////////////////////////////////////////////////////////
CEDCircleCand::CEDCircleCand()
{
	m_center = vector2df(0,0);
	m_radius  = 0;
	m_fit_error = 0;
	m_bound_len = 0;
	m_corner_vec.clear();
	m_line_vec.clear();
}

CEDCircleCand::~CEDCircleCand()
{
	m_corner_vec.clear();
	m_line_vec.clear();
}

void CEDCircleCand::setBasedData(CEDArcSegment* arc_seg_ptr)
{
	if (!arc_seg_ptr)
	{
		return;
	}

	m_center = arc_seg_ptr->m_center;
	m_radius = arc_seg_ptr->m_radius;
	m_fit_error = arc_seg_ptr->m_fit_error;
	m_bound_len = 0;
	m_corner_vec.clear();
	m_line_vec.clear();
	updateExtend(arc_seg_ptr);
}

f32 CEDCircleCand::getCoverage()
{
	if (m_radius <= PO_EPSILON)
	{
		return 0;
	}
	return po::_min(1.0f, m_bound_len / (PO_PI2*m_radius));
}

f32 CEDCircleCand::getDistance(CEDArcSegment* arc_seg_ptr)
{
	if (!arc_seg_ptr)
	{
		return PO_MAXINT;
	}

	Pixel3u* pixel_ptr;
	Pixel3u* pixel_data_ptr = m_corner_vec.data();
	Pixel3u st_pos = arc_seg_ptr->m_st_pos;
	Pixel3u ed_pos = arc_seg_ptr->m_ed_pos;

	i32 i, dist1, dist2;
	i32 min_dist = PO_MAXINT;
	i32 count = (i32)m_corner_vec.size();

	for (i = 0; i < count; i++)
	{
		pixel_ptr = pixel_data_ptr + i;
		dist1 = CPOBase::distanceSQ(pixel_ptr->x, pixel_ptr->y, st_pos.x, st_pos.y);
		dist2 = CPOBase::distanceSQ(pixel_ptr->x, pixel_ptr->y, ed_pos.x, ed_pos.y);
		min_dist = po::_min(min_dist, dist1);
		min_dist = po::_min(min_dist, dist2);
	}

    return std::sqrt((f32)min_dist);
}

f32 CEDCircleCand::getRoughError(CEDArcSegment* arc_seg_ptr, CEDLineSegSet& lines)
{
	if (!arc_seg_ptr)
	{
		return PO_MAXINT;
	}

	i32 st_line = arc_seg_ptr->m_st_line;
	i32 ed_line = arc_seg_ptr->m_ed_line;
	CEDLineSegment* line_ptr;
	CEDLineSegment* lines_ptr = lines.m_line_seg_ptr;
	Pixel3u* pixel_ptr = lines.m_pixel_ptr;
	Pixel3u* tmp_pixel_ptr;

	i32 i, line_count = 0, err = 0;
	i32 st_index, ed_index, dist2;
	i32 radius2 = m_radius*m_radius;
	u32 cx = CPOBase::int_cast(m_center.x);
	u32 cy = CPOBase::int_cast(m_center.y);

	for (i = st_line; i <= ed_line; i++)
	{
		line_count++;
		line_ptr = lines_ptr + i;
		st_index = line_ptr->m_st_pos;
		ed_index = (line_ptr->m_st_pos + line_ptr->m_ed_pos) / 2;

		tmp_pixel_ptr = pixel_ptr + st_index;
		dist2 = CPOBase::distanceSQ((u32)tmp_pixel_ptr->x, (u32)tmp_pixel_ptr->y, cx, cy);
		err += std::abs(dist2 - radius2);

		tmp_pixel_ptr = pixel_ptr + ed_index;
		dist2 = CPOBase::distanceSQ((u32)tmp_pixel_ptr->x, (u32)tmp_pixel_ptr->y, cx, cy);
		err += std::abs(dist2 - radius2);
	}

	if (line_count <= 0)
	{
		return PO_MAXINT;
	}
	err = err/line_count;
	return err;
}

CEDCircle CEDCircleCand::getCircle()
{
	CEDCircle circle;
	circle.m_radius = m_radius;
	circle.m_center = m_center;
	circle.m_fit_error = m_fit_error;
	circle.m_coverage = m_bound_len / (PO_PI2*m_radius);
	circle.m_is_used = false;
	return circle;
}

void CEDCircleCand::setCircle(CFitCircleu& fitcircle)
{
	m_center = fitcircle.getCenterPoint();
	m_radius = fitcircle.getRadius();
	m_fit_error = fitcircle.getFitError();
}

void CEDCircleCand::updateExtend(CEDArcSegment* arc_seg_ptr)
{
	if (arc_seg_ptr)
	{
		m_bound_len += arc_seg_ptr->m_bound_len;

		i32 st_line = arc_seg_ptr->m_st_line;
		i32 ed_line = arc_seg_ptr->m_ed_line;
		for (i32 i = st_line; i <= ed_line; i++)
		{
			m_line_vec.push_back(i);
		}
		m_corner_vec.push_back(arc_seg_ptr->m_st_pos);
		m_corner_vec.push_back(arc_seg_ptr->m_ed_pos);
	}
}

//////////////////////////////////////////////////////////////////////////
CEDContourExtractor::CEDContourExtractor()
{
	m_edcontour_param_ptr = NULL;

	m_grad_ptr = NULL;
	m_abs_gradx_ptr = NULL;
	m_abs_grady_ptr = NULL;
	m_edge_img_ptr = NULL;
	m_mask_img_ptr = NULL;
	m_tmp_contour_ptr = NULL;
	m_tmp_size = 0;
}

CEDContourExtractor::~CEDContourExtractor()
{
	m_grad_ptr = NULL;
	m_abs_gradx_ptr = NULL;
	m_abs_grady_ptr = NULL;
	m_edge_img_ptr = NULL;
	m_mask_img_ptr = NULL;

	freeBuffer();
}

void CEDContourExtractor::initBuffer(i32 wh)
{
	if (wh > m_tmp_size)
	{
		freeBuffer();
		m_tmp_size = wh;
		m_tmp_contour_ptr = po_new Pixel3u[wh];
		m_edge_img_ptr = po_new u8[wh];
	}
	memset(m_edge_img_ptr, 0, wh);
}

void CEDContourExtractor::freeBuffer()
{
	m_tmp_size = 0;
	POSAFE_DELETE_ARRAY(m_edge_img_ptr);
	POSAFE_DELETE_ARRAY(m_tmp_contour_ptr);
}

i32 CEDContourExtractor::getEDContourSegment(u8* img_ptr, u8* mask_img_ptr, Recti& range, CEDContourSet& contours, EDContourParam* param)
{
	i32 w = range.getWidth();
	i32 h = range.getHeight();
	if (!img_ptr || w <= 0 || h <= 0)
	{
		return 0;
	}

	m_range = range;
	m_mask_img_ptr = mask_img_ptr;
	m_edcontour_param_ptr = param;

	//check buffer
	i32 wh = w*h;
	initBuffer(wh);

	i32 i, count, px_count;
	Pixel3u* anchor_ptr;
	Pixel3u* anchor_data_ptr = NULL;
	bool is_processed = false;
	
	cv::Mat cv_grad(h, w, CV_16SC1);
	cv::Mat cv_abs_gradx(h, w, CV_16SC1);
	cv::Mat cv_abs_grady(h, w, CV_16SC1); 
	m_grad_ptr = (u16*)cv_grad.data;
	m_abs_gradx_ptr = (u16*)cv_abs_gradx.data;
	m_abs_grady_ptr = (u16*)cv_abs_grady.data;
	
#if defined(POR_WITH_OVX)
	CGEDEdgePrepare* graph_ptr = (CGEDEdgePrepare*)g_vx_graph_pool.fetchGraph(
			kGEDEdgePrepare, m_edcontour_param_ptr, img_ptr, w, h, mask_img_ptr, m_grad_ptr,
			m_abs_gradx_ptr, m_abs_grady_ptr, m_tmp_contour_ptr, &count, &px_count);
	if (graph_ptr)
	{
		is_processed = graph_ptr->process();
		g_vx_graph_pool.releaseGraph(graph_ptr);

		if (is_processed)
		{
			anchor_data_ptr = updateAnchor(count, w, h);
		}
	}
#endif

	if (!is_processed)
	{
		//gaussian blur for reduce noise
		cv::Mat cv_blur_img;
		cv::Mat cv_img(h, w, CV_8UC1, img_ptr);
		cv::GaussianBlur(cv_img, cv_blur_img, cv::Size(3, 3), 0);

		//calc gradent and normal vector
		cv::Mat cv_gradx, cv_grady;
		cv::Sobel(cv_blur_img, cv_gradx, CV_16S, 1, 0);
		cv::Sobel(cv_blur_img, cv_grady, CV_16S, 0, 1);

		cv_abs_gradx = cv::abs(cv_gradx);
		cv_abs_grady = cv::abs(cv_grady);
		cv::addWeighted(cv_abs_gradx, 0.5, cv_abs_grady, 0.5, 0, cv_grad);

		//find anchor
		anchor_data_ptr = findAnchor(count, px_count);
	}
	if (!anchor_data_ptr || count <= 0)
	{
		POSAFE_DELETE_ARRAY(anchor_data_ptr);
		return 0;
	}
 
 	//find contour by walking algorithm
 	contours.initContours(count, px_count);
 
 	i32 index, pos1, pos2;
 	for (i = 0; i < count; i++)
 	{
 		anchor_ptr = anchor_data_ptr + i;
 		index = anchor_ptr->y*w + anchor_ptr->x;
 		if (m_edge_img_ptr[index] == 0)
 		{
 			if (traceContour(anchor_ptr, pos1, pos2) > param->min_contour_points)
 			{
 				pos2 = pos2 + 1;
 				contours.setContourPixel(m_tmp_contour_ptr, pos1, pos2);
 
 				CEDContour* contour_ptr = contours.getNewContour();
 				contour_ptr->update(contours.m_pixel_ptr, pos1, pos2);
 			}
 		}
 	}
 
 	testEDContourResult(contours);
 	POSAFE_DELETE_ARRAY(anchor_data_ptr);

	m_grad_ptr = NULL;
	m_abs_gradx_ptr = NULL;
	m_abs_grady_ptr = NULL;
	return contours.getContourCount();
}

void CEDContourExtractor::testEDContourResult(CEDContourSet& contour)
{
#if defined(POR_TESTMODE)
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	cv::Mat drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::RNG rng(12345);
	cv::Vec3b color;

	//draw each contour with individual color
	i32 i, j, st_pos, ed_pos;
	i32 contour_count = contour.getContourCount();
	Pixel3u* pixel_ptr = contour.m_pixel_ptr;
	CEDContour* contour_ptr;

	for (i = 0; i < contour_count; i++)
	{
		color = cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		contour_ptr = contour.m_contour_ptr + i;
		st_pos = contour_ptr->m_st_pos;
		ed_pos = contour_ptr->m_ed_pos;

		for (j = st_pos; j < ed_pos; j++)
		{
			drawing.at<cv::Vec3b>(cv::Point(pixel_ptr[j].x, pixel_ptr[j].y)) = color;
		}
	}

	//write whole contour and edge image for testing
	cv::imwrite(PO_LOG_PATH"EDCountour.bmp", drawing);
	CImageProc::saveBinImgOpenCV(PO_LOG_PATH"EDContour_EdgeImg.bmp", m_edge_img_ptr, w, h);
#endif
}

void CEDContourExtractor::testEDAnchorResult(Pixel3u* anchor_ptr, i32 count)
{
#if defined(POR_TESTMODE)
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	i32 i, index, wh = w*h;

	u8* img_ptr = po_new u8[wh];
	memset(img_ptr, 0, wh);

	for (i = 0; i < count; i++)
	{
		index = anchor_ptr[i].y*w + anchor_ptr[i].x;
		img_ptr[index] = 255;
	}
	CImageProc::saveImgOpenCV(PO_LOG_PATH"EDContour_Anchor.bmp", img_ptr, w, h);
#endif
}

Recti CEDContourExtractor::getEDBoundingBox(CEDContour* contour_ptr, Pixel3u* pixel_ptr)
{
	Recti rt;
	if (pixel_ptr && contour_ptr)
	{
		i32 st_pos = contour_ptr->m_st_pos;
		i32 ed_pos = contour_ptr->m_ed_pos;
		for (i32 i = st_pos; i < ed_pos; i++)
		{
			rt.insertPoint(pixel_ptr[i].x, pixel_ptr[i].y);
		}
	}
	return rt;
}

u8* CEDContourExtractor::getEDEdgeImage(i32& w, i32& h)
{
	w = m_range.getWidth();
	h = m_range.getHeight();
	return m_edge_img_ptr;
}

Pixel3u* CEDContourExtractor::findAnchor(i32& count, i32& px_count)
{
	count = 0;
	px_count = 0;
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();

	u16 low_grad_th = m_edcontour_param_ptr->edge_strength;
	u16 high_grad_th = m_edcontour_param_ptr->edge_strength * 1.2f;
	u16 anchor_th = m_edcontour_param_ptr->anchor_threshold;
	u16 anchor_interval = po::_max(1, m_edcontour_param_ptr->anchor_interval);

	//limit grad value, if grad value is less than gradient threshold or out of mask region
	px_count = 0;
	i32 x, y, wh = w*h;
	u16* tmp_grad_ptr = m_grad_ptr;

	if (!m_mask_img_ptr)
	{
		//without mask image
		for (x = 0; x < wh; x++)
		{
			if (tmp_grad_ptr[x] >= low_grad_th)
			{
				px_count++;
				continue;
			}
			tmp_grad_ptr[x] = 0;
		}
	}
	else
	{
		//with mask image
		u8* tmp_mask_img_ptr = m_mask_img_ptr;
		for (x = 0; x < wh; x++)
		{
			if (tmp_grad_ptr[x] >= low_grad_th && tmp_mask_img_ptr[x] != kPOBackPixel)
			{
				px_count++;
				continue;
			}
			tmp_grad_ptr[x] = 0;
		}
	}

	//if pixel has high gradient value is nothing, return
	if (px_count == 0)
	{
		return NULL;
	}

	//find anchor with anchor interval
	i32 index, grad_value;
	i32 diff1, diff2;
	Pixel3u* pixel_ptr;
	bool is_horizontal = false;

	for (y = 1; y < h - 1; y += anchor_interval)
	{
		for (x = 1; x < w - 1; x++)
		{
			index = y*w + x;
			grad_value = m_grad_ptr[index];
			if (grad_value < high_grad_th) //high gradient value like as canny's high threshold
			{
				continue;
			}

			if (m_abs_gradx_ptr[index] < m_abs_grady_ptr[index]) //horizontal
			{
				is_horizontal = true;
				diff1 = grad_value - m_grad_ptr[index - w];
				diff2 = grad_value - m_grad_ptr[index + w];
			}
			else //vertical
			{
				diff1 = grad_value - m_grad_ptr[index - 1];
				diff2 = grad_value - m_grad_ptr[index + 1];
			}

			//local maxima and non-maxium suppression
			if (diff1 >= anchor_th && diff2 >= anchor_th)
			{
				if (is_horizontal)
				{
					diff1 = grad_value - m_grad_ptr[index - 1];
					diff2 = grad_value - m_grad_ptr[index + 1];
				}
				else
				{
					diff1 = grad_value - m_grad_ptr[index - w];
					diff2 = grad_value - m_grad_ptr[index + w];
				}
				if (diff1 > 0 && diff2 >= 0)
				{
					pixel_ptr = m_tmp_contour_ptr + count;
					pixel_ptr->x = x;
					pixel_ptr->y = y;
					pixel_ptr->g = grad_value;
					count++;
				}
			}
		}
	}
	return updateAnchor(count, w, h);
}

Pixel3u* CEDContourExtractor::updateAnchor(i32 count, i32 w, i32 h)
{
	//copy anchor from tmp buffer
	Pixel3u* anchor_ptr = po_new Pixel3u[count];
	Pixel3u* scan_anchor_ptr = anchor_ptr;
	Pixel3u* contour_ptr = m_tmp_contour_ptr;
	
	i32 i, x, y;
	for (i = 0; i < count; i++)
	{
		x = contour_ptr->x;
		y = contour_ptr->y;
		scan_anchor_ptr->x = x;
		scan_anchor_ptr->y = y;
		scan_anchor_ptr->g = m_grad_ptr[y*w + x];
		scan_anchor_ptr++;
		contour_ptr++;
	}

	//sort by gradient value
	std::qsort(anchor_ptr, count, sizeof(Pixel3u), compareAnchor);
	testEDAnchorResult(anchor_ptr, count);
	return anchor_ptr;
}

i32 CEDContourExtractor::traceContour(Pixel3u* pixel_ptr, i32& pos1, i32& pos2)
{
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	i32 pos = w*h / 2;
	i32 index = pixel_ptr->y*w + pixel_ptr->x;

	pos1 = pos;
	pos2 = pos;
	m_tmp_contour_ptr[pos].x = pixel_ptr->x;
	m_tmp_contour_ptr[pos].y = pixel_ptr->y;
	m_tmp_contour_ptr[pos].g = pixel_ptr->g;
	m_edge_img_ptr[index] = 1;
		
	if (m_abs_gradx_ptr[index] < m_abs_grady_ptr[index]) //horizontal
	{
		pos1 = subTraceContour(kEDWalk_Left, pixel_ptr->x, pixel_ptr->y, pos, -1);
		pos2 = subTraceContour(kEDWalk_Right, pixel_ptr->x, pixel_ptr->y, pos, 1);
	}
	else
	{
		pos1 = subTraceContour(kEDWalk_Up, pixel_ptr->x, pixel_ptr->y, pos, -1);
		pos2 = subTraceContour(kEDWalk_Down, pixel_ptr->x, pixel_ptr->y, pos, 1);
	}

	return pos2 - pos1 + 1;
}

i32 CEDContourExtractor::subTraceContour(i32 mode, u16 px, u16 py, i32 pos, i32 acc)
{
	i32 npos = pos;
	u16 nx = px, ny = py;
	while (mode != kEDWalk_Stop)
	{
		switch (mode)
		{
			case kEDWalk_Left:
			{
				mode = walkLeft(nx, ny, npos, acc);
				break;
			}
			case kEDWalk_Right:
			{
				mode = walkRight(nx, ny, npos, acc);
				break;
			}
			case kEDWalk_Up:
			{
				mode = walkUp(nx, ny, npos, acc);
				break;
			}
			case kEDWalk_Down:
			{
				mode = walkDown(nx, ny, npos, acc);
				break;
			}
		}
	}
	return npos;
}

i32 CEDContourExtractor::walkLeft(u16& px, u16& py, i32& pos, i32 acc)
{
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	i32 lu, ld, ll, indl, indu, indd;
	i32 index = py*w + px;
	Pixel3u* pixel_ptr;

	while (true)
	{
		indl = index - 1;
		indu = indl - w;
		indd = indl + w;
		lu = m_grad_ptr[indu];
		ld = m_grad_ptr[indd];
		ll = m_grad_ptr[indl];

		if (lu > ld && lu > ll)
		{
			py--; px--; ll = lu; index = indu; //up-left
		}
		else if (ld > lu && ld > ll)
		{
			py++; px--; ll = ld; index = indd; //down-left
		}
		else
		{
			px--; index = indl; //left-only
		}
		
		if (px <= 0 || py <= 0 || py >= h-1)
		{
			return kEDWalk_Stop;
		}

		if (ll == 0 || m_edge_img_ptr[index])
		{
			return kEDWalk_Stop;
		}
		m_edge_img_ptr[index] = 1;

		//add point to contour segment
		pos += acc;
		pixel_ptr = m_tmp_contour_ptr + pos;
		pixel_ptr->x = px;
		pixel_ptr->y = py;
		pixel_ptr->g = m_grad_ptr[index];

		//check walk direction
		if (m_abs_gradx_ptr[index] > m_abs_grady_ptr[index])
		{
			return (lu >= ld) ? kEDWalk_Up : kEDWalk_Down;
		}
	}
	return kEDWalk_Stop;
}

i32 CEDContourExtractor::walkRight(u16& px, u16& py, i32& pos, i32 acc)
{
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	i32 ru, rd, rr, indr, indu, indd;
	i32 index = py*w + px;
	Pixel3u* pixel_ptr;

	while (true)
	{
		indr = index + 1;
		indu = indr - w;
		indd = indr + w;
		ru = m_grad_ptr[indu];
		rd = m_grad_ptr[indd];
		rr = m_grad_ptr[indr];

		if (ru > rd && ru > rr)
		{
			py--; px++; rr = ru; index = indu; //up-right
		}
		else if (rd > ru && rd > rr)
		{
			py++; px++; rr = rd; index = indd; //down-right
		}
		else
		{
			px++; index = indr; //right-only
		}

		if (px >= w-1 || py <= 0 || py >= h - 1)
		{
			return kEDWalk_Stop;
		}

		if (rr == 0 || m_edge_img_ptr[index])
		{
			return kEDWalk_Stop;
		}
		m_edge_img_ptr[index] = 1;

		//add point to contour segment
		pos += acc;
		pixel_ptr = m_tmp_contour_ptr + pos;
		pixel_ptr->x = px;
		pixel_ptr->y = py;
		pixel_ptr->g = m_grad_ptr[index];

		//check walk direction
		if (m_abs_gradx_ptr[index] > m_abs_grady_ptr[index])
		{
			return (ru >= rd) ? kEDWalk_Up : kEDWalk_Down;
		}
	}
	return kEDWalk_Stop;
}

i32 CEDContourExtractor::walkDown(u16& px, u16& py, i32& pos, i32 acc)
{
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	i32 dl, dr, dd, indl, indr, indd;
	i32 index = py*w + px;
	Pixel3u* pixel_ptr;

	while (true)
	{
		indd = index + w;
		indl = indd - 1;
		indr = indd + 1;
		dl = m_grad_ptr[indl];
		dr = m_grad_ptr[indr];
		dd = m_grad_ptr[indd];

		if (dl > dr && dl > dd)
		{
			px--; py++; dd = dl; index = indl; //down-left
		}
		else if (dr > dl && dr > dd)
		{
			px++; py++; dd = dr; index = indr; //down-right
		}
		else
		{
			py++; index = indd; //down-only
		}

		if (px <= 0 || px >= w-1 || py >= h - 1)
		{
			return kEDWalk_Stop;
		}

		if (dd == 0 || m_edge_img_ptr[index])
		{
			return kEDWalk_Stop;
		}
		m_edge_img_ptr[index] = 1;

		//add point to contour segment
		pos += acc;
		pixel_ptr = m_tmp_contour_ptr + pos;
		pixel_ptr->x = px;
		pixel_ptr->y = py;
		pixel_ptr->g = m_grad_ptr[index];

		//check walk direction
		if (m_abs_gradx_ptr[index] < m_abs_grady_ptr[index])
		{
			return (dl >= dr) ? kEDWalk_Left : kEDWalk_Right;
		}
	}
	return kEDWalk_Stop;
}

i32 CEDContourExtractor::walkUp(u16& px, u16& py, i32& pos, i32 acc)
{
	i32 w = m_range.getWidth();
	i32 ul, ur, uu, indl, indr, indu;
	i32 index = py*w + px;
	Pixel3u* pixel_ptr;

	while (true)
	{
		indu = index - w;
		indl = indu - 1;
		indr = indu + 1;
		ul = m_grad_ptr[indl];
		ur = m_grad_ptr[indr];
		uu = m_grad_ptr[indu];

		if (ul > ur && ul > uu)
		{
			px--; py--; uu = ul; index = indl; //up-left
		}
		else if (ur > ul && ur > uu)
		{
			px++; py--; uu = ur; index = indr; //up-right
		}
		else
		{
			py--; index = indu; //up-only
		}

		if (px <= 0 || px >= w - 1 || py <= 0)
		{
			return kEDWalk_Stop;
		}

		if (uu == 0 || m_edge_img_ptr[index])
		{
			return kEDWalk_Stop;
		}
		m_edge_img_ptr[index] = 1;

		//add point to contour segment
		pos += acc;
		pixel_ptr = m_tmp_contour_ptr + pos;
		pixel_ptr->x = px;
		pixel_ptr->y = py;
		pixel_ptr->g = m_grad_ptr[index];

		//check walk direction
		if (m_abs_gradx_ptr[index] < m_abs_grady_ptr[index])
		{
			return (ul >= ur) ? kEDWalk_Left : kEDWalk_Right;
		}
	}
	return kEDWalk_Stop;
}

//////////////////////////////////////////////////////////////////////////
CEDLineDetector::CEDLineDetector()
{
	m_edline_param_ptr = NULL;
}

CEDLineDetector::~CEDLineDetector()
{
}

i32 CEDLineDetector::getEDLineSegment(u8* img_ptr, u8* mask_img_ptr, Recti& range, CEDLineSegSet& lines, EDLineParam* param)
{
	i32 w = range.getWidth();
	i32 h = range.getHeight();
	if (!img_ptr || w <= 0 || h <= 0)
	{
		return 0;
	}

	CEDContourSet contour;
	m_edline_param_ptr = param;

	//extract all contour in current image and return contour count
	if (m_edcontour.getEDContourSegment(img_ptr, mask_img_ptr, range, contour, param->getEDContourParam()) <= 0)
	{
		return 0;
	}

	//extract all line segment from contour
	i32 count = getEDLineSegment(contour, range, lines, param);
	if (count <= 0)
	{
		return 0;
	}

	//line segment set has pixel data, if extract line segment from image directly
	lines.m_is_embedd = true;
	contour.m_pixel_ptr = NULL;
	return count;
}

i32 CEDLineDetector::getEDLineSegment(CEDContourSet& contours, Recti& range, CEDLineSegSet& lines, EDLineParam* param)
{
	m_range = range;
	m_edline_param_ptr = param;
	i32 max_count = (i32)((f32)contours.m_px_count / m_edline_param_ptr->min_line_segment + 1);
	lines.initLines(max_count, contours.m_pixel_ptr);

	CEDContour* contour_ptr;
	Pixel3u* pixel_ptr = contours.m_pixel_ptr;
	i32 count = contours.getContourCount();
	i32 i, pos, ed_pos;
	bool is_continue;
		
	for (i = 0; i < count; i++)
	{
		contour_ptr = contours.m_contour_ptr + i;
		if (contour_ptr->m_is_used)
		{
			continue;
		}

		//fit line segmentations from contour
		is_continue = false;
		pos = contour_ptr->m_st_pos;
		ed_pos = contour_ptr->m_ed_pos;
		while (pos >= 0)
		{
			pos = fitLineFromContour(pixel_ptr, pos, ed_pos, is_continue, lines);
			is_continue = true;
		}
		contour_ptr->m_is_used = true;
	}

	testEDLineResult(lines);
	return lines.getLineSegCount();
}

void CEDLineDetector::testEDLineResult(CEDLineSegSet& lines)
{
#if defined(POR_TESTMODE)
	if (lines.getLineSegCount() <= 0)
	{
		return;
	}

	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	cv::Mat pixel_drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::Mat line_drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::RNG rng(12345);
	cv::Vec3b color;

	//draw each contour with individual color
	i32 i, j, px_count;
	i32 count = lines.getLineSegCount();
	Pixel3u* pixel_ptr = lines.m_pixel_ptr;

	for (i = 0; i < count; i++)
	{
		color = cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		CEDLineSegment& line = lines.m_line_seg_ptr[i];
		px_count = line.m_ed_pos - line.m_st_pos;

		for (j = line.m_st_pos; j < line.m_ed_pos; j++)
		{
			pixel_drawing.at<cv::Vec3b>(cv::Point(pixel_ptr[j].x, pixel_ptr[j].y)) = color;
		}

		vector2df p1 = line.m_center - line.m_nor_dir * px_count / 2;
		vector2df p2 = line.m_center + line.m_nor_dir * px_count / 2;
		cv::arrowedLine(line_drawing, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), color, 1, 8, 0, 0.4f);

  		postring draw_text = std::to_string(i);
  		cv::putText(line_drawing, draw_text.c_str(), cv::Point(p1.x, p1.y), cv::FONT_HERSHEY_SIMPLEX, 0.3f, color);
	}

	//write whole contour and edge image for testing
	cv::imwrite(PO_LOG_PATH"EDLineSegment.bmp", pixel_drawing);
	cv::imwrite(PO_LOG_PATH"EDLineSegment_line.bmp", line_drawing);
#endif
}

i32 CEDLineDetector::fitLineFromContour(Pixel3u* pixel_data_ptr, i32 st_pos, i32 ed_pos, bool is_continue, CEDLineSegSet& lines)
{
	CFitLineu fit_line;
	Pixel3u* pixel_ptr;
	f32 fit_error = PO_MAXINT;
	f32 ext_fit_error1 = m_edline_param_ptr->fit_line_error;
	f32 ext_fit_error2 = m_edline_param_ptr->fit_line_error;
	i32 i, seg_len = m_edline_param_ptr->min_line_segment;
	i32 max_seg_len = m_edline_param_ptr->max_line_segment;
	bool is_found = false;
	bool is_changed = false;
	
	switch (m_edline_param_ptr->getParamMode())
	{
		case kEDParamEdge:
		{
			ext_fit_error2 *= 1.2f;
			break;
		}
		case kEDParamCircle:
		{
			ext_fit_error2 *= 2.0f;
			break;
		}
	}
	
	//initial line segment detect
	for (i = st_pos; i < ed_pos - seg_len; i++)
	{
		fit_error = CFitShapeu::fitLine(pixel_data_ptr + i, seg_len, fit_line);
		if (fit_error < ext_fit_error1)
		{
			is_found = true;
			st_pos = i;
			break;
		}
	}
	if (!is_found)
	{
		return -1;
	}

	//initial line segment detected. try to extend this line segment
	vector2df cp = fit_line.getCenterPoint();
	vector2df ndir = fit_line.getNormalizedDir();
	f32 fit_error_sum = fit_line.getFitError()*seg_len;
	f32 cx = cp.x;
	f32 cy = cp.y;

	for (i = st_pos + seg_len; i < ed_pos; i++)
	{
		pixel_ptr = pixel_data_ptr + i;
		fit_error = CPOBase::distPt2Line((f32)pixel_ptr->x - cx, (f32)pixel_ptr->y - cy, ndir);
		fit_error_sum += fit_error;
		seg_len++;

		if (fit_error > ext_fit_error2 || fit_error_sum > ext_fit_error1*seg_len || seg_len > max_seg_len)
		{
			break;
		}
		is_changed = true;
	}

	//rebuild line segment
	if (is_changed)
	{
		fit_error = CFitShapeu::fitLine(pixel_data_ptr + st_pos, i - st_pos, fit_line);
	}

	//set all pixel data into line segment
	CEDLineSegment* line_ptr = lines.getNewLine();
	if (!line_ptr)
	{
		return -1;
	}

	line_ptr->m_center = fit_line.getCenterPoint();
	line_ptr->m_nor_dir = fit_line.getNormalizedDir();
	line_ptr->m_fit_error = fit_error;
	line_ptr->m_st_pos = st_pos;
	line_ptr->m_ed_pos = i;
	line_ptr->m_is_continue = is_continue;

	//calibration line segment direction
	Pixel3u& p1 = pixel_data_ptr[st_pos];
	Pixel3u& p2 = pixel_data_ptr[i - 1];
	f32 dx = (f32)p2.x - p1.x;
	f32 dy = (f32)p2.y - p1.y;
	vector2df dir = vector2df(dx, dy).normalize();
	if (dir.dotProduct(line_ptr->m_nor_dir) < 0)
	{
		line_ptr->m_nor_dir *= -1.0f;
	}
	line_ptr->m_len = CPOBase::length(dx, dy);
	return (i < ed_pos) ? i : -1;
}

//////////////////////////////////////////////////////////////////////////
CEDEdgeDetector::CEDEdgeDetector()
{
	m_size = 0;
	m_ededge_param_ptr = NULL;
}

CEDEdgeDetector::~CEDEdgeDetector()
{
	freeBuffer();
}

void CEDEdgeDetector::initBuffer(i32 w, i32 h)
{
	i32 wh = w*h;
	if (m_size < wh)
	{
		freeBuffer();

		m_size = wh;
		m_fitted_line.initBuffer(wh);
		m_edcontour.initBuffer(wh);
	}
}

void CEDEdgeDetector::freeBuffer()
{
	m_size = 0;
	m_fitted_line.freeBuffer();
	m_edcontour.freeBuffer();
}

i32 CEDEdgeDetector::getEDEdges(u8* img_ptr, u8* mask_img_ptr, Recti& range, EDEdgeVector& edge_vec, EDEdgeParam* param_ptr)
{
	i32 w = range.getWidth();
	i32 h = range.getHeight();
	if (!img_ptr || w <= 0 || h <= 0)
	{
		return 0;
	}

	m_range = range;
	m_ededge_param_ptr = param_ptr;
	initBuffer(w, h);

	edge_vec.clear();
	CEDContourSet contour;
	CEDLineSegSet lines;
	CEDEdgeSegSet seg_lines;
	EDEdgeMergeCandVector merge_vec;

	//extract all contour in current image and return contour count
	if (m_edcontour.getEDContourSegment(img_ptr, mask_img_ptr, range, contour, param_ptr->getEDContourParam()) <= 0)
	{
		return 0;
	}

	//extract all line segment in current image and return line segment count
	m_edline_detector.getEDLineSegment(contour, range, lines, param_ptr->getEDLineParam());

	//extract all potential arc from line segments and result arc count
	mergeNeighborLines(lines, seg_lines);
	testEDEdgeSegResult(seg_lines);

	//sort edge segment and extend to candidate circles
	std::qsort(seg_lines.getEdgeSegData(), seg_lines.getEdgeSegCount(), sizeof(CEDEdgeSegment), compareEdgeSegment);

	while (true)
	{
		CEDEdgeSegment* cand_edge_ptr = findLongestLineSegment(seg_lines);
		if (!cand_edge_ptr)
		{
			break;
		}

		//extend selected candiate circle
		if (extendLineSeg2Edge(cand_edge_ptr, seg_lines, merge_vec, lines))
		{
			if (m_ededge_param_ptr->isRemoveOutlier())
			{
				//remove outlier sample points
				m_fitted_line.refitLineExcludeOutlier(m_ededge_param_ptr->remove_clip_factor,
													m_ededge_param_ptr->fit_line_error);
			}

			//check coverage of edge
			vector2df st_point, ed_point;
			m_fitted_line.calcEdgeEndPoints(st_point, ed_point);
			if (CPOBase::distance(st_point, ed_point) > m_ededge_param_ptr->base_cover_pixels)
			{
				CEDEdge* edge_ptr = CPOBase::pushBackNew(edge_vec);
				edge_ptr->setFitEdge(m_fitted_line, st_point, ed_point, m_ededge_param_ptr->is_keep_points);
			}
		}
	}

	//update found circles
	testEDEdgeResult(edge_vec);
	updateEdges(edge_vec);

	//free buffers
	contour.freeContours();
	lines.freeLines();
	seg_lines.freeEdges();
	return (i32)edge_vec.size();
}

u8* CEDEdgeDetector::getEDEdgeImage(i32& w, i32& h)
{
	return m_edcontour.getEDEdgeImage(w, h);
}

void CEDEdgeDetector::mergeNeighborLines(CEDLineSegSet& lines, CEDEdgeSegSet& seg_lines)
{
	i32 i, px_count;
	i32 count = (i32)lines.getLineSegCount();
	if (count <= 0)
	{
		return;
	}

	f32 angle_cos;
	i32 st_pos, ed_pos;
	vector2df prev_dir, dir;
	CFitLineu tmp_fitted_line;
	CEDLineSegment* line_seg_ptr;
	CEDLineSegment* line_data_ptr = lines.getLineSegData();
	Pixel3u* pixel_data_ptr = lines.m_pixel_ptr;
	f32 th_angle_cos = cosf(CPOBase::degToRad(m_ededge_param_ptr->fit_angle_error));
	
	seg_lines.initEdges(count, lines.m_pixel_ptr);

	for (i = 0; i < count; i++)
	{
		line_seg_ptr = line_data_ptr + i;
		px_count = line_seg_ptr->m_ed_pos - line_seg_ptr->m_st_pos;
		st_pos = line_seg_ptr->m_st_pos;
		ed_pos = line_seg_ptr->m_ed_pos - 1;

		m_fitted_line.initFitLine();
		m_fitted_line.addPixels(pixel_data_ptr + line_seg_ptr->m_st_pos, px_count);
		m_fitted_line.m_center = line_seg_ptr->m_center;
		m_fitted_line.m_dir = line_seg_ptr->m_nor_dir;
		m_fitted_line.m_fit_error = line_seg_ptr->m_fit_error;
		prev_dir = line_seg_ptr->m_nor_dir;

		CEDEdgeSegment* edge_seg_ptr = seg_lines.getNewEdgeSeg();
		edge_seg_ptr->m_line_vec.push_back(i);

		i++;
		while (i < count)
		{
			line_seg_ptr = line_data_ptr + i;
			if (!line_seg_ptr->m_is_continue)
			{
				i--; break;
			}

			dir = line_seg_ptr->m_nor_dir;
			angle_cos = prev_dir.dotProduct(dir);
			prev_dir = dir;
			if (angle_cos < th_angle_cos)
			{
				i--; break;
			}

			tmp_fitted_line = m_fitted_line;
			px_count = line_seg_ptr->m_ed_pos - line_seg_ptr->m_st_pos;
			m_fitted_line.addPixels(pixel_data_ptr + line_seg_ptr->m_st_pos, px_count);
			if (m_fitted_line.fitLine() > m_ededge_param_ptr->fit_line_error)
			{
				m_fitted_line = tmp_fitted_line;
				i--; break;
			}

			ed_pos = line_seg_ptr->m_ed_pos - 1;
			edge_seg_ptr->m_line_vec.push_back(i);
			i++;
		}

		//update edge segment
		edge_seg_ptr->m_center = m_fitted_line.getCenterPoint();
		edge_seg_ptr->m_dir = m_fitted_line.getNormalizedDir();
		edge_seg_ptr->m_fit_error = m_fitted_line.getFitError();
		edge_seg_ptr->m_point_st = vector2df(pixel_data_ptr[st_pos].x, pixel_data_ptr[st_pos].y);
		edge_seg_ptr->m_point_ed = vector2df(pixel_data_ptr[ed_pos].x, pixel_data_ptr[ed_pos].y);
	}

	tmp_fitted_line.setPixelData(NULL, 0);
}

CEDEdgeSegment* CEDEdgeDetector::findLongestLineSegment(CEDEdgeSegSet& seg_lines)
{
	i32 i, count = seg_lines.getEdgeSegCount();
	CEDEdgeSegment* edge_seg_ptr = seg_lines.getEdgeSegData();
	CEDEdgeSegment* tmp_edge_seg_ptr;

	f32 th_cover_pixels = m_ededge_param_ptr->base_cover_pixels / 10.0f;
	f32 th_base_angle_tol = cosf(CPOBase::degToRad(m_ededge_param_ptr->base_tolerance_angle));

	for (i = 0; i < count; i++)
	{
		tmp_edge_seg_ptr = edge_seg_ptr + i;
		if (!tmp_edge_seg_ptr->m_is_used)
		{
			tmp_edge_seg_ptr->m_is_used = true;

			//check base coverage
			if (tmp_edge_seg_ptr->getLength() < th_cover_pixels)
			{
				continue;
			}

			//check base angle
			if (m_ededge_param_ptr->base_tolerance_angle > 0)
			{
				vector2df dir = tmp_edge_seg_ptr->m_dir;
				if (std::abs(dir.dotProduct(m_ededge_param_ptr->base_line_dir)) < th_base_angle_tol)
				{
					continue;
				}
			}
			return tmp_edge_seg_ptr;
		}
	}
	return NULL;
}

bool CEDEdgeDetector::extendLineSeg2Edge(CEDEdgeSegment* cand_edge_ptr, CEDEdgeSegSet& seg_lines,
									EDEdgeMergeCandVector& merge_vec, CEDLineSegSet& lines)
{
	if (!cand_edge_ptr)
	{
		return false;
	}

	i32 i, count = seg_lines.getEdgeSegCount();
	CEDEdgeSegment* edge_seg_ptr = seg_lines.getEdgeSegData();
	CEDEdgeSegment* tmp_edge_ptr;
	CFitLineu tmp_fitted_line;

	f32 fit_error, angle_cos, dist;
	vector2df st_point = cand_edge_ptr->m_point_st;
	vector2df ed_point = cand_edge_ptr->m_point_ed;
	vector2df dir = cand_edge_ptr->m_dir;
	
	m_fitted_line.initFitLine();
	updateFittedLine(cand_edge_ptr, lines);

	f32 th_fit_error = m_ededge_param_ptr->fit_line_error;
	f32 th_cover_pixels = m_ededge_param_ptr->base_cover_pixels;
	f32 th_angle_cos = cosf(CPOBase::degToRad(m_ededge_param_ptr->fit_angle_error));
	f32 th_base_angle_tol = m_ededge_param_ptr->base_tolerance_angle;
	f32 th_rough_error = th_fit_error*7.0f;

	bool is_updated = false;
	i32 merge_count = 0;
	merge_vec.resize(count);

	while (true)
	{
		is_updated = false;
		merge_count = 0;
		for (i = 0; i < count; i++)
		{
			tmp_edge_ptr = edge_seg_ptr + i;
			if (tmp_edge_ptr->m_is_used)
			{
				continue;
			}

			angle_cos = std::abs(dir.dotProduct(tmp_edge_ptr->m_dir));
			if (angle_cos < th_angle_cos)
			{
				continue;
			}
			dist = calcEdgeRoughtError(tmp_edge_ptr);
			if (dist > th_rough_error)
			{
				continue;
			}

			merge_vec[merge_count].index = i;
			merge_vec[merge_count].dist = calcEdgeDistance(st_point, ed_point, tmp_edge_ptr);
			merge_count++;
		}

		if (merge_count <= 0)
		{
			break;
		}

		//sort edge segment and extend to candidate circles
		std::qsort(merge_vec.data(), merge_count, sizeof(EDEdgeMergeCand), compareEdgeMergeCand);

		for (i = 0; i < merge_count; i++)
		{
			tmp_edge_ptr = edge_seg_ptr + merge_vec[i].index;
			angle_cos = std::abs(dir.dotProduct(tmp_edge_ptr->m_dir));
			if (angle_cos < th_angle_cos)
			{
				continue;
			}
			dist = calcEdgeDistance(st_point, ed_point, tmp_edge_ptr);
			if (dist > m_ededge_param_ptr->max_gap_size)
			{
				break;
			}

			//backup fitted line
			tmp_fitted_line = m_fitted_line;
			fit_error = updateFittedLine(tmp_edge_ptr, lines);
			if (fit_error > th_fit_error)
			{
				m_fitted_line = tmp_fitted_line;
			}
			else
			{
				is_updated = true;
				tmp_edge_ptr->m_is_used = true;
				updateEdgePoint(st_point, ed_point, tmp_edge_ptr);
			}
		}
		if (!is_updated)
		{
			break;
		}
	}
	
	//prevent tempoary uesd memeory pointer
	tmp_fitted_line.setPixelData(NULL, 0);

	//check fit error and coverage
	if (m_fitted_line.getFitError() > th_fit_error || 
		(m_fitted_line.getPixelCount() < th_cover_pixels && th_cover_pixels > 0))
	{
		return false;
	}

	//check base angle
	if (th_base_angle_tol > 0)
	{
		dir = m_fitted_line.m_dir;
		th_base_angle_tol = cosf(CPOBase::degToRad(th_base_angle_tol));
		if (std::abs(dir.dotProduct(m_ededge_param_ptr->base_line_dir)) < th_base_angle_tol)
		{
			return false;
		}
	}
	return true;
}

void CEDEdgeDetector::updateEdgePoint(vector2df& st_point, vector2df& ed_point, CEDEdgeSegment* edge_seg_ptr)
{
	vector2df pt[4];
	pt[0] = st_point;
	pt[1] = ed_point;
	pt[2] = edge_seg_ptr->m_point_st;
	pt[3] = edge_seg_ptr->m_point_ed;

	i32 i, min_index = 0, max_index = 0;
	f32 dist, max_dist = -PO_MAXINT, min_dist = PO_MAXINT;
	vector2df center = m_fitted_line.m_center;
	vector2df dir = m_fitted_line.m_dir;
	vector2df pt_dir;

	for (i = 0; i < 4; i++)
	{
		pt_dir = pt[i] - center;
		pt_dir.normalize(dist);
		dist *= dir.dotProduct(pt_dir);
		if (dist > max_dist)
		{
			max_dist = dist; max_index = i;
		}
		if (dist < min_dist)
		{
			min_dist = dist; min_index = i;
		}
	}

	st_point = pt[min_index];
	ed_point = pt[max_index];
}

f32 CEDEdgeDetector::calcEdgeDistance(vector2df st_point, vector2df ed_point, CEDEdgeSegment* edge_seg_ptr)
{
	f32 dist11 = CPOBase::distanceSQ(st_point, edge_seg_ptr->m_point_st);
	f32 dist12 = CPOBase::distanceSQ(st_point, edge_seg_ptr->m_point_ed);
	f32 dist21 = CPOBase::distanceSQ(ed_point, edge_seg_ptr->m_point_st);
	f32 dist22 = CPOBase::distanceSQ(ed_point, edge_seg_ptr->m_point_ed);
    return std::sqrt(po::_min(po::_min(dist11, dist12), po::_min(dist21, dist22)));
}

f32 CEDEdgeDetector::calcEdgeRoughtError(CEDEdgeSegment* edge_seg_ptr)
{
	vector2df center = m_fitted_line.m_center;
	vector2df dir = m_fitted_line.m_dir;
	vector2df ca = edge_seg_ptr->m_point_st - center;
	vector2df da = edge_seg_ptr->m_point_ed - center;
	return (CPOBase::distPt2Line(ca.x, ca.y, dir) + CPOBase::distPt2Line(da.x, da.y, dir)) / 2;
}

f32 CEDEdgeDetector::updateFittedLine(CEDEdgeSegment* edge_seg_ptr, CEDLineSegSet& lines)
{
	if (!edge_seg_ptr)
	{
		return PO_MAXINT;
	}

	i32vector line_vec = edge_seg_ptr->m_line_vec;
	i32 i, px_count, count = (i32)line_vec.size();
	Pixel3u* pixel_ptr = lines.m_pixel_ptr;
	CEDLineSegment* line_data_ptr = lines.getLineSegData();
	CEDLineSegment* line_seg_ptr;
	
	for (i = 0; i < count; i++)
	{
		line_seg_ptr = line_data_ptr + line_vec[i];
		px_count = line_seg_ptr->m_ed_pos - line_seg_ptr->m_st_pos;
		m_fitted_line.addPixels(pixel_ptr + line_seg_ptr->m_st_pos, px_count);
	}

	return m_fitted_line.fitLine();
}

void CEDEdgeDetector::updateEdges(EDEdgeVector& edge_vec)
{
	//add offset of circles with range
	i32 i, count = (i32)edge_vec.size();
	for (i = 0; i < count; i++)
	{
		CEDEdge& edge = edge_vec[i];
		edge.m_center += vector2df(m_range.x1, m_range.y1);
		edge.m_point_st += vector2df(m_range.x1, m_range.y1);
		edge.m_point_ed += vector2df(m_range.x1, m_range.y1);
	}
}

void CEDEdgeDetector::testEDEdgeResult(EDEdgeVector& edge_vec)
{
#if defined(POR_TESTMODE)
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	cv::Mat drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::RNG rng(12345);
	cv::Vec3b color;
	vector2df point_st, point_ed;

	i32 i, count = (i32)edge_vec.size();
	for (i = 0; i < count; i++)
	{
		point_st = edge_vec[i].m_point_st;
		point_ed = edge_vec[i].m_point_ed;
		color = cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

		cv::line(drawing, cv::Point(point_st.x, point_st.y), cv::Point(point_ed.x, point_ed.y), color);
	}
	cv::imwrite(PO_LOG_PATH"EDEdge_Lines.bmp", drawing);
#endif
}

void CEDEdgeDetector::testEDEdgeSegResult(CEDEdgeSegSet& seg_lines)
{
#if defined(POR_TESTMODE)
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	cv::Mat drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::RNG rng(12345);
	cv::Vec3b color;
	vector2df st_point, ed_point;

	CEDEdgeSegment* edge_seg_ptr;
	CEDEdgeSegment* edge_data_ptr = seg_lines.getEdgeSegData();
	i32 i, count = (i32)seg_lines.getEdgeSegCount();
	for (i = 0; i < count; i++)
	{
		color = cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		
		edge_seg_ptr = edge_data_ptr + i;
		st_point = edge_seg_ptr->m_point_st;
		ed_point = edge_seg_ptr->m_point_ed;
		
		cv::line(drawing, cv::Point(st_point.x, st_point.y), cv::Point(ed_point.x, ed_point.y), color);

		postring draw_text = std::to_string(edge_seg_ptr->m_index);
		cv::putText(drawing, draw_text.c_str(), cv::Point(st_point.x, st_point.y), cv::FONT_HERSHEY_SIMPLEX, 0.3f, color);
	}
	cv::imwrite(PO_LOG_PATH"EDEdge_EdgeSeg.bmp", drawing);
#endif
}

////////////////////////////////////////////////////////////////////////////
CEDCircleDetector::CEDCircleDetector()
{
	m_size = 0;
	m_edcircle_param_ptr = NULL;
}

CEDCircleDetector::~CEDCircleDetector()
{
	freeBuffer();
}

void CEDCircleDetector::initBuffer(i32 w, i32 h)
{
	i32 wh = w*h;
	if (m_size < wh)
	{
		freeBuffer();

		m_size = wh;
		m_fitted_circle.initBuffer(wh);
		m_edcontour.initBuffer(wh);
	}
}

void CEDCircleDetector::freeBuffer()
{
	m_size = 0;
	m_fitted_circle.freeBuffer();
	m_edcontour.freeBuffer();
}

i32 CEDCircleDetector::getEDCircles(u8* img_ptr, u8* mask_img_ptr, Recti& range, EDCircleVector& circle_vec, EDCircleParam* param)
{
	i32 w = range.getWidth();
	i32 h = range.getHeight();
	if (!img_ptr || w <= 0 || h <= 0)
	{
		return 0;
	}

	m_range = range;
	m_edcircle_param_ptr = param;
	initBuffer(w, h);

	circle_vec.clear();
	CEDContourSet contour;
	CEDLineSegSet lines;
	CEDArcSegSet arcs;

	//extract all contour in current image and return contour count
	if (m_edcontour.getEDContourSegment(img_ptr, mask_img_ptr, range, contour, param->getEDContourParam()) <= 0)
	{
		return 0;
	}

	//find all circle can be representation with closed edge
	findClosedEdgeCircle(contour, circle_vec);

	//extract all line segment in current image and return line segment count
	m_edline_detector.getEDLineSegment(contour, range, lines, param->getEDLineParam());

	//extract all potential arc from line segments and result arc count
	findPotentialArcLines(lines, arcs);
	testEDArcSegmentResult(lines, arcs);

	//sort arc segment and extend to candidate circles
	std::qsort(arcs.getArcSegData(), arcs.getArcSegCount(), sizeof(CEDArcSegment), compareArcSegment);

	while (true)
	{
		CEDArcSegment* cand_arc_ptr = findLongestArcSegment(arcs);
		if (!cand_arc_ptr)
		{
			break;
		}

		//extend selected candiate circle
		if (extendArc2Circle(cand_arc_ptr, arcs, lines))
		{
			if (m_edcircle_param_ptr->isRemoveOutlier())
			{
				//remove outlier sample points
				m_fitted_circle.refitCircleExcludeOutlier(m_edcircle_param_ptr->remove_clip_factor,
														m_edcircle_param_ptr->fit_circle_error);
			}

			if (m_fitted_circle.getCoverage() >= m_edcircle_param_ptr->min_coverage)
			{
				CEDCircle* circle_ptr = CPOBase::pushBackNew(circle_vec);
				circle_ptr->setFitCircle(m_fitted_circle, m_edcircle_param_ptr->is_keep_points);
			}
		}
	}

	//update found circles
	testEDCircleResult(circle_vec);
	updateCircles(circle_vec);

	//free buffers
	contour.freeContours();
	lines.freeLines();
	arcs.freeArcSegments();
	return (i32)circle_vec.size();
}

u8* CEDCircleDetector::getEDEdgeImage(i32& w, i32& h)
{
	return m_edcontour.getEDEdgeImage(w, h);
}

i32 CEDCircleDetector::findClosedEdgeCircle(CEDContourSet& contour, EDCircleVector& circles)
{	
	i32 i, min_r, px_count;
	i32 circle_count = 0;
	i32 count = contour.getContourCount();
	f32 radius, fit_error, th_fit_error;
	Pixel3u* pixel_ptr = contour.m_pixel_ptr;
	Pixel3u* tmp_pixel_ptr;
	CEDContour* contour_ptr;
	CFitCircleu tmp_fitted_circle;

	for (i = 0; i < count; i++)
	{
		contour_ptr = contour.m_contour_ptr + i;
		if (contour_ptr->m_is_used || !contour_ptr->m_is_closed)
		{
			continue;
		}

		Recti rt = m_edcontour.getEDBoundingBox(contour_ptr, pixel_ptr);
		min_r = po::_min(rt.getWidth(), rt.getHeight()) / 2;

		//check radius with rough bounding box condition
		if (min_r < m_edcircle_param_ptr->min_radius*0.7f || min_r > m_edcircle_param_ptr->max_radius*1.5f)
		{
			continue;
		}

		//check circle validation
		tmp_pixel_ptr = pixel_ptr + contour_ptr->m_st_pos;
		px_count = contour_ptr->m_ed_pos - contour_ptr->m_st_pos;
		tmp_fitted_circle.initFitCircle();
		tmp_fitted_circle.addPixels(tmp_pixel_ptr, px_count, false);
		fit_error = tmp_fitted_circle.fitCircle();
		radius = tmp_fitted_circle.getRadius();

		th_fit_error = po::_max(m_edcircle_param_ptr->fit_circle_error, radius * m_edcircle_param_ptr->fit_circle_err_ratio);

		if (fit_error < th_fit_error &&
			radius >= m_edcircle_param_ptr->min_radius &&  radius <= m_edcircle_param_ptr->max_radius)
		{
			if (m_edcircle_param_ptr->isRemoveOutlier())
			{
				//remove outlier sample points
				tmp_fitted_circle.refitCircleExcludeOutlier(m_edcircle_param_ptr->remove_clip_factor,
														m_edcircle_param_ptr->fit_circle_error);
			}

			CEDCircle* circle_ptr = CPOBase::pushBackNew(circles);
			circle_ptr->setFitCircle(tmp_fitted_circle, m_edcircle_param_ptr->is_keep_points);
			circle_ptr->setCoverage(1.0f);
			circle_count++;

			contour_ptr->m_is_used = true;
		}
	}
	return circle_count;
}

i32 CEDCircleDetector::findPotentialArcLines(CEDLineSegSet& lines, CEDArcSegSet& arcs)
{
	i32 i, j;
	i32 count = lines.getLineSegCount();
	if (count < ED_ARCLINE_SEGNUM)
	{
		return 0;
	}

	i32 max_count = (i32)((f32)lines.getLineSegCount() / ED_ARCLINE_SEGMIN + 1);
	arcs.initArcSegments(max_count, lines.m_pixel_ptr);

	bool is_segmented = false;
	i32 ni, pos = 0, isign = 0, jsign = 0;
	f32 prev_angle = 0.0f, angle = 0.0f;
	f32 low_angle_cos = m_edcircle_param_ptr->low_angle_cos;
	f32 high_angle_cos = m_edcircle_param_ptr->high_angle_cos;
	CEDLineSegment* line1_ptr;
	CEDLineSegment* line2_ptr;
	
	for (i = 1; i <= count; i++)
	{
		if (i < count)
		{
			line1_ptr = lines.m_line_seg_ptr + i;
			is_segmented = !line1_ptr->m_is_continue;
		}
		else
		{
			is_segmented = true;
		}

		if (is_segmented)
		{
			//if it has valid line segment set
			if (i >= pos + ED_ARCLINE_SEGNUM)
			{
				ni = i - 1;
				while (pos < ni)
				{
					//find the starting line
					for (; pos < ni; pos++)
					{
						line1_ptr = lines.m_line_seg_ptr + pos;
						line2_ptr = line1_ptr + 1;
						isign = po::_sgn(line1_ptr->m_nor_dir.crossProduct(line2_ptr->m_nor_dir));
						angle = line1_ptr->m_nor_dir.dotProduct(line2_ptr->m_nor_dir);
						prev_angle = angle;

						if (angle <= low_angle_cos && angle >= high_angle_cos)
						{
							break;
						}						
					}
					if (pos == ni) //no seed for arc segment
					{
						break;
					}

					//extent potential arc lines starting at pos
					for (j = pos + 1; j < ni; j++)
					{
						line1_ptr = lines.m_line_seg_ptr + j;
						line2_ptr = line1_ptr + 1;
						angle = line1_ptr->m_nor_dir.dotProduct(line2_ptr->m_nor_dir);

						line1_ptr = line1_ptr - 1; /* angle between before and after lines, it's used for more robustness */
						jsign = po::_sgn(line1_ptr->m_nor_dir.crossProduct(line2_ptr->m_nor_dir));

						if (angle > low_angle_cos || angle < high_angle_cos || isign != jsign ||
							!CPOBase::checkRange(angle, prev_angle / 1.5f, prev_angle * 1.5f))
						{
							break;
						}
						prev_angle = angle;
					}

					//at least minimal lines
					if (j - pos + 1 >= ED_ARCLINE_SEGMIN)
					{
						i32 kpos = pos;
						while (kpos >= 0)
						{
							//insert new arc from lines...
							kpos = extractArcFromLines(lines, kpos, j, arcs);
						}
						pos = j + 1;
					}
					else
					{
						pos = j;
					}
				}
			}
			pos = i;
		}
	}

	return arcs.getArcSegCount();
}

f32 CEDCircleDetector::fitLSMCircleFromLineGroup(CEDLineSegSet& lines, i32 st_line, i32 ed_line,
											CEDArcSegment& arc_seg, bool is_clear)
{
	if (is_clear)
	{
		//clear all tempoary pixel data for fit arc segment
		m_fitted_circle.initFitCircle();
	}

	//copy all pixel data of sub lines to one line buffer
	Pixel3u* pixel_ptr = lines.m_pixel_ptr;
	CEDLineSegment* lines_ptr = lines.m_line_seg_ptr;
	CEDLineSegment* line_ptr;

	i32 i, count;
	for (i = st_line; i <= ed_line; i++)
	{
		line_ptr = lines_ptr + i;
		count = line_ptr->m_ed_pos - line_ptr->m_st_pos;
		m_fitted_circle.addPixels(pixel_ptr + line_ptr->m_st_pos, count, true);
	}

	f32 fit_error = m_fitted_circle.fitCircle();
	arc_seg.m_center = m_fitted_circle.getCenterPoint();
	arc_seg.m_radius = m_fitted_circle.getRadius();
	arc_seg.m_bound_len = m_fitted_circle.getPixelCount();
	arc_seg.m_fit_error = fit_error;
	return fit_error;
}

i32 CEDCircleDetector::extractArcFromLines(CEDLineSegSet& lines, i32 st_line, i32 ed_line, CEDArcSegSet& arcs)
{
	CEDArcSegment arc_seg1;
	CEDArcSegment arc_seg2;
	f32 min_radius = m_edcircle_param_ptr->min_radius*0.5f;
	f32 max_radius = m_edcircle_param_ptr->max_radius*2.0f;
	f32 th_fit_error = m_edcircle_param_ptr->fit_circle_error;
	f32 th_fit_err_rate = m_edcircle_param_ptr->fit_circle_err_ratio;

	//first, try fitting an arc to the pixels of all lines
	f32 fit_error = fitLSMCircleFromLineGroup(lines, st_line, ed_line, arc_seg1);
	f32 th_fit_circle = po::_max(th_fit_error, arc_seg1.m_radius * th_fit_err_rate);

	if (fit_error < th_fit_circle)
	{
		f32 radius = arc_seg1.m_radius;
		f32 bound_len = arc_seg1.m_bound_len;
		if (radius > min_radius && radius < max_radius && bound_len > ED_ARCLINE_MINBOUNDLEN)
		{
			CEDArcSegment* arc_seg_ptr = arcs.getNewArcSeg();
			if (!arc_seg_ptr)
			{
				return -1;
			}
			arc_seg_ptr->updateArcSegment(arc_seg1, lines, st_line, ed_line);
		}
		return -1;
	}
	if (ed_line - st_line + 1 < ED_ARCLINE_SEGNUM)
	{
		return -1;
	}

	i32 i, pos1 = 0, pos2 = 0;
	i32 count = ed_line - st_line + 1;
	bool is_found = false;

	for (i = 0; i <= count - ED_ARCLINE_SEGNUM; i++)
	{
		pos1 = st_line + i;
		pos2 = pos1 + ED_ARCLINE_SEGNUM - 1;
		fit_error = fitLSMCircleFromLineGroup(lines, pos1, pos2, arc_seg1);
		th_fit_circle = po::_max(th_fit_error, arc_seg1.m_radius * th_fit_err_rate);
		if (fit_error < th_fit_circle)
		{
			is_found = true;
			break;
		}
	}
	if (!is_found)
	{
		return -1; //none initial arc
	}

	//initial arc consisting of 3lines. Extend it one line at a time
	while (++pos2 <= ed_line)
	{
		fit_error = fitLSMCircleFromLineGroup(lines, pos2, pos2, arc_seg2, false);
		th_fit_circle = po::_max(th_fit_error, arc_seg2.m_radius * th_fit_err_rate);
		if (fit_error >= th_fit_circle)
		{
			break;
		}

		arc_seg1.m_center = arc_seg2.m_center;
		arc_seg1.m_radius = arc_seg2.m_radius;
		arc_seg1.m_fit_error = arc_seg2.m_fit_error;
	}

	//add current arc consists of the lines
	fit_error = arc_seg1.m_fit_error;
	th_fit_circle = po::_max(th_fit_error, arc_seg1.m_radius * th_fit_err_rate);
	if (fit_error < th_fit_circle && arc_seg1.m_radius > min_radius && arc_seg1.m_radius < max_radius)
	{
		CEDArcSegment* arc_seg_ptr = arcs.getNewArcSeg();
		if (!arc_seg_ptr)
		{
			return -1;
		}
		arc_seg_ptr->updateArcSegment(arc_seg1, lines, st_line, pos2 - 1);
	}
	return (pos2 < ed_line) ? pos2 : -1;
}

CEDArcSegment* CEDCircleDetector::findLongestArcSegment(CEDArcSegSet& arcs)
{
	i32 i, count = arcs.getArcSegCount();
	CEDArcSegment* arc_seg_ptr = arcs.getArcSegData();
	CEDArcSegment* tmp_arc_seg_ptr;

	for (i = 0; i < count; i++)
	{
		tmp_arc_seg_ptr = arc_seg_ptr + i;
		if (!tmp_arc_seg_ptr->m_is_used)
		{
			tmp_arc_seg_ptr->m_is_used = true;
			return tmp_arc_seg_ptr;
		}
	}
	return NULL;
}

bool CEDCircleDetector::extendArc2Circle(CEDArcSegment* cand_arc_ptr, CEDArcSegSet& arcs, CEDLineSegSet& lines)
{
	if (!cand_arc_ptr)
	{
		return false;
	}

	CEDCircleCand cand;
	i32 cand_index = cand_arc_ptr->m_index;
	cand.setBasedData(cand_arc_ptr);
	
	f32 dist, min_rough_err;
	f32 diff, th_merge, avg_radius;
	f32 th_radius = cand.m_radius*m_edcircle_param_ptr->arc_rextend_rate;
	f32 th_center_dist = cand.m_radius*m_edcircle_param_ptr->arc_cextend_rate;

	CEDArcSegment* arc_seg_ptr;
	CEDArcSegment* arc_seg_data_ptr = arcs.m_arc_seg_ptr;

	EDArcSegmentVector cand_vec;
	b8vector is_used;
	i32 i, min_index = 0, count = arcs.getArcSegCount();
	is_used.resize(count, false);
	cand_vec.reserve(count);

	//clear & initialize fitting circle
	updateFitCircle(cand, lines);

	//collect candidate arc segment can be merge with current candidate circle
	for (i = 0; i < count; i++)
	{
		arc_seg_ptr = arc_seg_data_ptr + i;
		if (arc_seg_ptr->m_is_used)
		{
			continue;
		}

		avg_radius = (arc_seg_ptr->m_radius + cand.m_radius) / 2;
		f32 th_radius = avg_radius*m_edcircle_param_ptr->arc_rextend_rate;
		f32 th_center_dist = avg_radius*m_edcircle_param_ptr->arc_cextend_rate;

		diff = std::abs(arc_seg_ptr->m_radius - cand.m_radius);
		dist = arc_seg_ptr->m_center.getDistanceFrom(cand.m_center);

		if (diff < th_radius && dist < th_center_dist)
		{
			cand_vec.push_back(arc_seg_ptr);
		}
	}

	//check over the candidates and join with candidate
	count = (i32)cand_vec.size();
	if (count > 0)
	{
		f32 rough_err = 0;
		while (true)
		{
			min_index = -1;
			min_rough_err = PO_MAXINT;
			th_merge = cand.m_radius* m_edcircle_param_ptr->arc_dextend_rate;
			for (i = 0; i < count; i++)
			{
				if (!is_used[i])
				{
					dist = cand.getDistance(cand_vec[i]);
					if (dist < th_merge)
					{
						rough_err = cand.getRoughError(cand_vec[i], lines);
						if (rough_err < min_rough_err)
						{
							min_index = i;
							min_rough_err = rough_err;
						}
					}
				}
			}
			if (min_index < 0)
			{
				break;
			}

			is_used[min_index] = true;
			updateLSMCircleExtend(cand, cand_vec[min_index], lines);
		}
	}

	//check extracted cadidate circle's radius
	if (cand.m_radius >= m_edcircle_param_ptr->min_radius && 
		cand.m_radius <= m_edcircle_param_ptr->max_radius)
	{
		if (cand.getCoverage() > m_edcircle_param_ptr->min_coverage)
		{
			return checkCandValidation(cand, lines);
		}
	}
	return false;
}

void CEDCircleDetector::updateFitCircle(CEDCircleCand& cand, CEDLineSegSet& lines)
{
	m_fitted_circle.initFitCircle();

	i32vector& fit_lines = cand.m_line_vec;
	Pixel3u* pixel_ptr = lines.m_pixel_ptr;
	CEDLineSegment* line_data_ptr = lines.m_line_seg_ptr;
	CEDLineSegment* line_ptr;

	i32 i, sub_count;
	i32 count = (i32)fit_lines.size();
	for (i = 0; i < count; i++)
	{
		line_ptr = line_data_ptr + fit_lines[i];
		sub_count = line_ptr->m_ed_pos - line_ptr->m_st_pos;
		m_fitted_circle.addPixels(pixel_ptr + line_ptr->m_st_pos, sub_count, true);
	}

	m_fitted_circle.fitCircle();
	cand.setCircle(m_fitted_circle);
}

void CEDCircleDetector::updateLSMCircleExtend(CEDCircleCand& cand, CEDArcSegment* arc_seg_ptr, CEDLineSegSet& lines)
{
	CFitCircleu tmp_fit_circle = m_fitted_circle; //backup fitting circle

	i32 i, count;
	i32 st_line = arc_seg_ptr->m_st_line;
	i32 ed_line = arc_seg_ptr->m_ed_line;
	CEDLineSegment* line_ptr;
	CEDLineSegment* lines_ptr = lines.m_line_seg_ptr;
	Pixel3u* pixel_ptr = lines.m_pixel_ptr;

	for (i = st_line; i <= ed_line; i++)
	{
		line_ptr = lines_ptr + i;
		count = line_ptr->m_ed_pos - line_ptr->m_st_pos;
		m_fitted_circle.addPixels(pixel_ptr + line_ptr->m_st_pos, count, true);
	}

	f32 radius = m_fitted_circle.getRadius();
	f32 fit_error = m_fitted_circle.fitCircle();
	f32 th_fit_circle = po::_max(m_edcircle_param_ptr->fit_circle_error,
								m_edcircle_param_ptr->fit_circle_err_ratio * radius);

	if (fit_error < th_fit_circle)
	{
		arc_seg_ptr->m_is_used = true;
		cand.setCircle(m_fitted_circle);
		cand.updateExtend(arc_seg_ptr);
	}
	else
	{
		//restore fitting circle
		m_fitted_circle = tmp_fit_circle;
	}

	//prevent tempoary uesd memeory pointer
	tmp_fit_circle.setPixelData(NULL, 0);
}

bool CEDCircleDetector::checkCandValidation(CEDCircleCand& cand, CEDLineSegSet& lines)
{
	CEDLineSegment* line_ptr;
	CEDLineSegment* line_data_ptr = lines.m_line_seg_ptr;
	i32vector& fit_lines = cand.m_line_vec;
	vector2df cp = cand.m_center;
	vector2df dir;
	f32 cosan = 0;
	f32 costh = cosf(CPOBase::degToRad(45));

	i32 i, px_count, ccount = 0;
	i32 count = (i32)fit_lines.size();
	for (i = 0; i < count; i++)
	{
		line_ptr = line_data_ptr + fit_lines[i];
		line_ptr->m_is_used = true;
		dir = (line_ptr->m_center - cp).normalize();
		cosan = std::abs(dir.crossProduct(line_ptr->m_nor_dir));
		if (cosan >= costh)
		{
			ccount++;
		}
	}

	if ((f32)ccount / count < 0.8f)
	{
		return false;
	}
	
	f32 dist;
	f32 min_dist2 = (cand.m_radius - m_edcircle_param_ptr->fit_circle_error);
	f32 max_dist2 = (cand.m_radius + m_edcircle_param_ptr->fit_circle_error);
	bool is_changed = false;

	count = lines.m_line_count;
	min_dist2 = min_dist2*min_dist2;
	max_dist2 = max_dist2*max_dist2;
	
	//collect lines expected arc segment merge and refit to circle
	for (i = 0; i < count; i++)
	{
		line_ptr = line_data_ptr + i;
		if (line_ptr->m_is_used)
		{
			continue;
		}

		dir = line_ptr->m_center - cp;
		dist = dir.getLengthSQ();
		if (dist < min_dist2 || dist > max_dist2)
		{
			continue;
		}

		line_ptr->m_is_used = true;
		dir.normalize();
		cosan = std::abs(dir.crossProduct(line_ptr->m_nor_dir));
		if (cosan >= costh)
		{
			is_changed = true;
			fit_lines.push_back(i);
		}
	}

	if (is_changed)
	{
		count = (i32)fit_lines.size();
		m_fitted_circle.initFitCircle();
		Pixel3u* pixel_ptr = lines.m_pixel_ptr;

		for (i = 0; i < count; i++)
		{
			line_ptr = lines.m_line_seg_ptr + fit_lines[i];
			px_count = line_ptr->m_ed_pos - line_ptr->m_st_pos;
			m_fitted_circle.addPixels(pixel_ptr + line_ptr->m_st_pos, px_count, true);
		}

		f32 radius = m_fitted_circle.getRadius();
		f32 fit_error = m_fitted_circle.fitCircle();
		f32 th_fit_error = po::_max(m_edcircle_param_ptr->fit_circle_error,
								radius * m_edcircle_param_ptr->fit_circle_err_ratio);

		if (fit_error < th_fit_error)
		{
			cand.m_fit_error = fit_error;
			cand.m_radius = m_fitted_circle.getRadius();
			cand.m_center = m_fitted_circle.getCenterPoint();
			cand.m_bound_len = m_fitted_circle.getPixelCount();
		}
	}
	return true;
}

void CEDCircleDetector::updateCircles(EDCircleVector& circles)
{
	//add offset of circles with range
	i32 i, count = (i32)circles.size();
	for (i = 0; i < count; i++)
	{
		CEDCircle& circle = circles[i];
		circle.m_center += vector2df(m_range.x1, m_range.y1);
	}
}

void CEDCircleDetector::testEDCircleResult(EDCircleVector circles)
{
#if defined(POR_TESTMODE)
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	cv::Mat drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::RNG rng(12345);
	cv::Vec3b color;
	vector2df center;

	i32 i, radius;
	i32 count = (i32)circles.size();
	for (i = 0; i < count; i++)
	{
		center = circles[i].m_center;
		radius = circles[i].m_radius;
		color = cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

		cv::circle(drawing, cv::Point(center.x, center.y), radius, color);
	}
	cv::imwrite(PO_LOG_PATH"EDCircle_Circles.bmp", drawing);
#endif
}

void CEDCircleDetector::testEDArcSegmentResult(CEDLineSegSet& lines, CEDArcSegSet& arcs)
{
#if defined(POR_TESTMODE)
	i32 w = m_range.getWidth();
	i32 h = m_range.getHeight();
	cv::Mat circle_drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::Mat pixel_drawing = cv::Mat::zeros(h, w, CV_8UC3);
	cv::RNG rng(12345);
	cv::Vec3b color;
	vector2df center;

	i32 i, j, k, radius;
	i32 st_line, ed_line;
	i32 st_pos, ed_pos;
	i32 seg_count = arcs.getArcSegCount();
	CEDArcSegment* arc_seg_ptr;
	CEDArcSegment* arc_seg_data_ptr = arcs.m_arc_seg_ptr;
	CEDLineSegment* line_ptr;
	CEDLineSegment* line_data_ptr = lines.m_line_seg_ptr;
	Pixel3u* pixel_ptr;
	Pixel3u* pixel_data_ptr = lines.m_pixel_ptr;

	for (i = 0; i < seg_count; i++)
	{
		arc_seg_ptr = arc_seg_data_ptr + i;
		center = arc_seg_ptr->m_center;
		radius = arc_seg_ptr->m_radius;
		color = cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

		cv::circle(circle_drawing, cv::Point(center.x, center.y), radius, color);

		bool is_draw = false;
		st_line = arc_seg_ptr->m_st_line;
		ed_line = arc_seg_ptr->m_ed_line;
		for (j = st_line; j <= ed_line; j++)
		{
			line_ptr = line_data_ptr + j;
			st_pos = line_ptr->m_st_pos;
			ed_pos = line_ptr->m_ed_pos;
			for (k = st_pos; k < ed_pos; k++)
			{
				pixel_ptr = pixel_data_ptr + k;
				pixel_drawing.at<cv::Vec3b>(cv::Point(pixel_ptr->x, pixel_ptr->y)) = color;

				if (!is_draw)
				{
					is_draw = true;
					postring draw_text = std::to_string(i);
					cv::putText(pixel_drawing, draw_text.c_str(), cv::Point(pixel_ptr->x, pixel_ptr->y),
							cv::FONT_HERSHEY_SIMPLEX, 0.3f, color);
				}
			}
		}
	}
	cv::imwrite(PO_LOG_PATH"EDCircle_ArcSegment.bmp", circle_drawing);
	cv::imwrite(PO_LOG_PATH"EDCircle_ArcPixels.bmp", pixel_drawing);
#endif
}
