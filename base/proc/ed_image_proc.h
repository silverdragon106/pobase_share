#pragma once

#include "define.h"
#include "fit_shape.h"

#define ED_CONTOUR_EDGETH			40
#define ED_CONTOUR_ANCHTH			8
#define ED_CONTOUR_ANCHINTERVAL		4
#define ED_CONTOUR_MINPTNUM			6
#define ED_LINE_ANCHINTERVAL		1
#define ED_LINE_FITERROR			1.5f
#define ED_LINE_MIN_SEGMENT			6
#define ED_LINE_MAX_SEGMENT			50
#define ED_EDGE_MAX_GAPSIZE			6
#define ED_EDGE_MAX_ANGLERANGE		30
#define ED_CIRCLE_FITERROR			2.0f
#define ED_CIRCLE_FITERRRATE		0.01f
#define ED_ARCLINE_SEGMIN			2
#define ED_ARCLINE_SEGNUM			3
#define ED_ARCLINE_LOWANGLE			3
#define ED_ARCLINE_HIGHANGLE		70
#define ED_ARCLINE_MINBOUNDLEN		15
#define ED_CIRCLE_EXTRRATE			0.7f
#define ED_CIRCLE_EXTCRATE			0.8f
#define ED_CIRCLE_EXTDRATE			2.0f
#define ED_CIRCLE_MINCOVERAGE		0.5f

enum EDWalkType
{
	kEDWalk_Stop = -1,
	kEDWalk_Left = 0,
	kEDWalk_Right,
	kEDWalk_Down,
	kEDWalk_Up
};

enum EDParamType
{
	kEDParamContour = 0,
	kEDParamLine,
	kEDParamEdge,
	kEDParamCircle,

	kEDParamTypeCount
};

struct EDContourParam
{
	u8						param_mode;
	u16						edge_strength;
	u8						anchor_threshold;
	u16						anchor_interval;
	u16						min_contour_points;

public:
	EDContourParam();
	inline i32				getParamMode() { return param_mode; };
};

struct EDLineParam : public EDContourParam
{
	u16						min_line_segment;
	u16						max_line_segment;
	f32						fit_line_error;

public:
	EDLineParam();

	inline EDContourParam*	getEDContourParam() { return (EDContourParam*)this; };
};

struct EDEdgeParam : public EDLineParam
{
	f32						max_gap_size;
	f32						base_cover_pixels;
	vector2df				base_line_dir;
	f32						base_tolerance_angle;
	f32						fit_angle_error;

	bool					is_keep_points;
	bool					remove_outlier;
	f32						remove_clip_factor;

public:
	EDEdgeParam();

	inline bool				isKeepPoints() { return is_keep_points; };
	inline bool				isRemoveOutlier() { return remove_outlier; };

	inline EDLineParam*		getEDLineParam() { return (EDLineParam*)this; };
	inline EDContourParam*	getEDContourParam() { return (EDContourParam*)this; };
};

struct EDCircleParam : public EDLineParam
{
	f32						min_radius;
	f32						max_radius;
	f32						min_coverage;
	f32						fit_circle_err_ratio;
	f32						fit_circle_error;
	f32						low_angle_cos;
	f32						high_angle_cos;
	f32						arc_rextend_rate;
	f32						arc_cextend_rate;
	f32						arc_dextend_rate;

	bool					is_keep_points;
	bool					remove_outlier;
	f32						remove_clip_factor;

public:
	EDCircleParam();
	
	inline bool				isKeepPoints() { return is_keep_points; };
	inline bool				isRemoveOutlier() { return remove_outlier; };

	inline EDLineParam*		getEDLineParam() { return (EDLineParam*)this; };
	inline EDContourParam*	getEDContourParam() { return (EDContourParam*)this; };
};

struct EDEdgeMergeCand
{
	i32		index;
	f32		dist;
};
typedef std::vector<EDEdgeMergeCand> EDEdgeMergeCandVector;

//////////////////////////////////////////////////////////////////////////
class CEDEdge
{
public:
	CEDEdge();
	
	void					freeBuffer();
	void					setFitEdge(CFitLineu& fit_line, const vector2df& st_point,
									const vector2df& ed_point, bool keep_point);

public:
	vector2df				m_center;
	vector2df				m_dir;
	vector2df				m_point_st;
	vector2df				m_point_ed;

	f32						m_coverage;
	f32						m_contrast;
	f32						m_fit_error;
	u8						m_reserved;

	pt3uvector				m_point_vec;
};
typedef std::vector<CEDEdge> EDEdgeVector;

class CEDCircle
{
public:
	CEDCircle();

	void					freeBuffer();
	void					setFitCircle(CFitCircleu& fit_circle, bool keep_point);
	void					setCoverage(f32 coverage);

public:
	vector2df				m_center;
	f32						m_radius;

	f32						m_coverage;
	f32						m_contrast;
	f32						m_fit_error;

	bool					m_is_used;
	pt3uvector				m_point_vec;
};
typedef std::vector<CEDCircle> EDCircleVector;

class CEDContour
{
public:
	CEDContour();
	~CEDContour();

	void					update(Pixel3u* pixel_ptr, i32 pos1, i32 pos2);

public:
	bool					m_is_closed;
	bool					m_is_used;
	i32						m_st_pos;
	i32						m_ed_pos;
};

class CEDContourSet
{
public:
	CEDContourSet();
	~CEDContourSet();

	void					initContours(i32 max_anchor_count, i32 max_px_count);
	void					freeContours();

	void					setContourPixel(Pixel3u* pixel_ptr, i32& pos1, i32& pos2);
	CEDContour*				getNewContour();
	i32						getContourCount();

public:
	i32						m_px_max_count;
	i32						m_contour_max_count;
	i32						m_px_count;
	i32						m_contour_count;
	Pixel3u*				m_pixel_ptr;
	CEDContour*				m_contour_ptr;
};

class CEDLineSegment
{
public:
	CEDLineSegment();

public:
	vector2df				m_center;
	vector2df				m_nor_dir; //normalized direction
	f32						m_fit_error;
	f32						m_len;
	i32						m_st_pos;
	i32						m_ed_pos;

	bool					m_is_used;
	bool					m_is_continue;
};

class CEDLineSegSet
{
public:
	CEDLineSegSet();
	~CEDLineSegSet();

	void					initLines(i32 max_count, Pixel3u* pixel_ptr);
	void					freeLines();

	CEDLineSegment*			getNewLine();
	CEDLineSegment*			getLineSegData();
	i32						getLineSegCount();

public:
	i32						m_line_max_count;
	i32						m_line_count;
	bool					m_is_embedd;
	Pixel3u*				m_pixel_ptr;
	CEDLineSegment*			m_line_seg_ptr;
};

class CEDEdgeSegment
{
public:
	CEDEdgeSegment();

	f32						getLength();

public:
	vector2df				m_center;
	vector2df				m_dir;
	f32						m_fit_error;

	vector2df				m_point_st;
	vector2df				m_point_ed;
	i32vector				m_line_vec;

	i32						m_index;
	bool					m_is_used;
};

class CEDEdgeSegSet
{
public:
	CEDEdgeSegSet();
	~CEDEdgeSegSet();

	void					initEdges(i32 max_count, Pixel3u* pixel_ptr);
	void					freeEdges();

	CEDEdgeSegment*			getNewEdgeSeg();
	CEDEdgeSegment*			getEdgeSegData();
	i32						getEdgeSegCount();

public:
	i32						m_edge_max_count;
	i32						m_edge_count;
	Pixel3u*				m_pixel_ptr;
	CEDEdgeSegment*			m_edge_seg_ptr;
};

class CEDArcSegment
{
public:
	CEDArcSegment();

	void					updateArcSegment(CEDArcSegment& out_arc_seg, CEDLineSegSet& lines, i32 st_line, i32 ed_line);

public:
	i32						m_index;
	vector2df				m_center;
	f32						m_radius;
	f32						m_fit_error;
	f32						m_bound_len;
	Pixel3u					m_st_pos;
	Pixel3u					m_ed_pos;
	i32						m_st_line;
	i32						m_ed_line;

	bool					m_is_used;
};
typedef std::vector<CEDArcSegment*>	EDArcSegmentVector;

class CEDArcSegSet
{
public:
	CEDArcSegSet();
	~CEDArcSegSet();

	void					initArcSegments(i32 max_count, Pixel3u* pixel_ptr);
	void					freeArcSegments();

	CEDArcSegment*			getNewArcSeg();
	CEDArcSegment*			getArcSegData();
	i32						getArcSegCount();

public:
	i32						m_arc_max_count;
	i32						m_arc_count;
	bool					m_is_embedd;
	Pixel3u*				m_pixel_ptr;
	CEDArcSegment*			m_arc_seg_ptr;
};

class CEDCircleCand
{
public:
	CEDCircleCand();
	~CEDCircleCand();

	void					setCircle(CFitCircleu& fitcircle);
	void					setBasedData(CEDArcSegment* arc_seg_ptr);
	void					updateExtend(CEDArcSegment* arc_seg_ptr);

	CEDCircle				getCircle();
	f32						getCoverage();
	f32						getDistance(CEDArcSegment* arc_seg_ptr);
	f32						getRoughError(CEDArcSegment* arc_seg_ptr, CEDLineSegSet& lines);

public:
	vector2df				m_center;
	f32						m_radius;
	f32						m_fit_error;
	f32						m_bound_len;
	pt3uvector				m_corner_vec;
	i32vector				m_line_vec;
};
typedef std::vector<CEDCircleCand> EDCircleCandVector;

//////////////////////////////////////////////////////////////////////////
class CEDContourExtractor
{
public:
	CEDContourExtractor();
	~CEDContourExtractor();

	void					initBuffer(i32 wh);
	void					freeBuffer();

	i32						getEDContourSegment(u8* img_ptr, u8* mask_img_ptr, Recti& range, CEDContourSet& contour, EDContourParam* param);
	Recti					getEDBoundingBox(CEDContour* contour_ptr, Pixel3u* pixel_ptr);
	u8*						getEDEdgeImage(i32& w, i32& h);

	Pixel3u*				findAnchor(i32& count, i32& px_count);
	Pixel3u*				updateAnchor(i32 count, i32 w, i32 h);
	i32						traceContour(Pixel3u* ppixel, i32& pos1, i32& pos2);
	i32						subTraceContour(i32 mode, u16 px, u16 py, i32 pos, i32 acc);
	i32						walkLeft(u16& px, u16& py, i32& pos, i32 acc);
	i32						walkRight(u16& px, u16& py, i32& pos, i32 acc);
	i32						walkDown(u16& px, u16& py, i32& pos, i32 acc);
	i32						walkUp(u16& px, u16& py, i32& pos, i32 acc);
	void					testEDContourResult(CEDContourSet& contour);
	void					testEDAnchorResult(Pixel3u* anchor_ptr, i32 count);

public:
	Recti					m_range;
	EDContourParam*			m_edcontour_param_ptr;

	u16*					m_grad_ptr;
	u16*					m_abs_gradx_ptr;
	u16*					m_abs_grady_ptr;
	u8*						m_mask_img_ptr;

	i32						m_tmp_size;
	Pixel3u*				m_tmp_contour_ptr;
	u8*						m_edge_img_ptr;
};

class CEDLineDetector
{
public:
	CEDLineDetector();
	~CEDLineDetector();

	i32						getEDLineSegment(CEDContourSet& contour, Recti& range, CEDLineSegSet& lines, EDLineParam* param);
	i32						getEDLineSegment(u8* img_ptr, u8* mask_img_ptr, Recti& range, CEDLineSegSet& lines, EDLineParam* param);

	i32						fitLineFromContour(Pixel3u* pixel_ptr, i32 st_pos, i32 ed_pos, bool bcontinue, CEDLineSegSet& lines);
	void					testEDLineResult(CEDLineSegSet& lines);

public:
	Recti					m_range;
	CEDContourExtractor		m_edcontour;
	EDLineParam*			m_edline_param_ptr;
};

class CEDEdgeDetector
{
public:
	CEDEdgeDetector();
	~CEDEdgeDetector();

	void					initBuffer(i32 w, i32 h);
	void					freeBuffer();

	u8*						getEDEdgeImage(i32& w, i32& h);
	i32						getEDEdges(u8* img_ptr, u8* mask_img_ptr, Recti& range, EDEdgeVector& edge_vec, EDEdgeParam* param);

	void					mergeNeighborLines(CEDLineSegSet& lines, CEDEdgeSegSet& seg_lines);
	CEDEdgeSegment*			findLongestLineSegment(CEDEdgeSegSet& seg_lines);
	bool					extendLineSeg2Edge(CEDEdgeSegment* line_ptr, CEDEdgeSegSet& seg_lines, EDEdgeMergeCandVector& merge_vec, CEDLineSegSet& lines);
	void					updateEdges(EDEdgeVector& edge_vec);
	void					updateEdgePoint(vector2df& st_point, vector2df& ed_point, CEDEdgeSegment* edge_seg_ptr);
	f32						updateFittedLine(CEDEdgeSegment* edge_seg_ptr, CEDLineSegSet& lines);
	f32						calcEdgeDistance(vector2df st_point, vector2df ed_point, CEDEdgeSegment* edge_seg_ptr);
	f32						calcEdgeRoughtError(CEDEdgeSegment* edge_seg_ptr);

	void					testEDEdgeSegResult(CEDEdgeSegSet& seg_lines);
	void					testEDEdgeResult(EDEdgeVector& edge_vec);

public:
	i32						m_size;
	Recti					m_range;
	CFitLineu				m_fitted_line;

	CEDContourExtractor		m_edcontour;
	CEDLineDetector			m_edline_detector;
	EDEdgeParam*			m_ededge_param_ptr;
};

class CEDCircleDetector
{
public:
	CEDCircleDetector();
	~CEDCircleDetector();

	void					initBuffer(i32 w, i32 h);
	void					freeBuffer();

	u8*						getEDEdgeImage(i32& w, i32& h);
	i32						getEDCircles(u8* img_ptr, u8* mask_img_ptr, Recti& range, EDCircleVector& circle_vec, EDCircleParam* param);

	i32						findClosedEdgeCircle(CEDContourSet& contour, EDCircleVector& circles);
 	i32						findPotentialArcLines(CEDLineSegSet& lines, CEDArcSegSet& arcs);
	CEDArcSegment*			findLongestArcSegment(CEDArcSegSet& arcs);
	i32						extractArcFromLines(CEDLineSegSet& lines, i32 stline, i32 edline, CEDArcSegSet& arcs);
	bool					extendArc2Circle(CEDArcSegment* cand_arc_ptr, CEDArcSegSet& arcs, CEDLineSegSet& lines);
 	bool					checkCandValidation(CEDCircleCand& cand, CEDLineSegSet& lines);
	void					updateCircles(EDCircleVector& circles);
	void					updateFitCircle(CEDCircleCand& cand, CEDLineSegSet& lines);
	void					updateLSMCircleExtend(CEDCircleCand& cand, CEDArcSegment* arc_seg_ptr, CEDLineSegSet& lines);
	f32						fitLSMCircleFromLineGroup(CEDLineSegSet& lines, i32 stline, i32 edline, CEDArcSegment& arcsegment, bool bclear = true);

	void					testEDCircleResult(EDCircleVector circles);
	void					testEDArcSegmentResult(CEDLineSegSet& lines, CEDArcSegSet& arcs);

public:
	i32						m_size;
	Recti					m_range;
	CFitCircleu				m_fitted_circle;

	CEDContourExtractor		m_edcontour;
	CEDLineDetector			m_edline_detector;
	EDCircleParam*			m_edcircle_param_ptr;
};

