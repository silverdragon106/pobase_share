#pragma once

#include "define.h"
#include "struct.h"

#define CALIPER_MAX_EDGE		16

enum FindCaliperTypes
{
	kFindCaliperEdge = 0,
	kFindCaliperEdgePair,
	kFindCaliperEdgeCands
};

enum CaliperFlagTypes
{
	kCaliperCommon			= 0x00,
	kCaliperStartCaliper	= 0x01,
	kCaliperEndCaliper		= 0x02,
	kCaliperBrokenCaliper	= 0x04
};

enum CaliperEdgeRuleTypes
{
	kCaliperEdgeRuleFirst,
	kCaliperEdgeRuleStrong,
	kCaliperEdgeRulePair,

	kCaliperEdgeRuleCount
};

enum BeadEdgeTransitionMode
{
	kBeadEdgeEither = 0,		//same as kEdgeEither
	kBeadEdgeBlackToWhite,		//same as kEdgeBlackToWhite
	kBeadEdgeWhiteToBlack,		//same as kEdgeWhiteToBlack

	kBeadEdgeTransitionModeCount
};

enum BeadEdgeDirectionTypes
{
	kBeadEdgeDirectionLeftToRight = 0,
	kBeadEdgeDirectionRightToLeft,
	kBeadEdgeDirectionOutsideToIn,
	kBeadEdgeDirectionInToOutside,

	kBeadEdgeDirectionCount
};

enum DefectFilterTypes
{
	kDefectFilterNone = 0,
	kDefectFilterLeftPart,
	kDefectFilterRightPart,

	kDefectFilterModeCount
};

struct CaliperParam
{
	u8						caliper_mode;
	u16						edge_filter_size;
	u8						edge_min_contrast;
	u8						edge_max_contrast;
	u8						edge_threshold;
	u8						edge_direction;
	u8						edge0_transition;
	u8						edge1_transition;
	u8						edge_rule;
	u16						edge_pair_size;
};

struct Caliper
{
	vector2di				caliper_pos;
	vector2df				caliper_dir;
	u16						caliper_width;
	u16						caliper_height;
	
	u8						caliper_type;
	u16						edge_pt_count;
	vector2df				edge_pt_ptr[CALIPER_MAX_EDGE];

public:
	Caliper();

	static i32				paramSize();
	static i32				resultSize();

	void					initEdges();
	inline i32				getEdgePtCount() const { return edge_pt_count; }
	inline vector2df		getEdgePoint(i32 index) const { return edge_pt_ptr[index]; }
};
typedef std::vector<Caliper> CaliperVec;

class CFindCaliper
{
public:
	CFindCaliper();
	~CFindCaliper();

	static bool				findCalipers(const u8* img_ptr, u16 w, u16 h, CaliperVec& caliper_vec, const CaliperParam& param);

	static bool				findEdgesInCaliper(const u8* img_ptr, u16 w, u16 h, u16 stride_w, Caliper& caliper, const CaliperParam& param);

	static void				edgeFilterInCaliper(f32* edge_score_ptr, i32 size, i32 filter_size, i32 min_contrast, i32 max_contrast);
	static void				updateCaliperEdges(f32* edge_score_ptr, i32 caliper_size, Caliper& caliper, const CaliperParam& param);
	static void				updateCaliperOneEdge(const vector2df* cand_ptr, i32 count, const CaliperParam& param, ptvector2df& edge_vec);
	static void				updateCaliperEdgePair(const vector2df* cand_ptr, i32 count, const CaliperParam& param, ptvector2df& edge_vec);
	static void				updateCaliperEdgeCands(const vector2df* cand_ptr, i32 count, const CaliperParam& param, ptvector2df& edge_vec);
	static void				findOneEdgeInCaliperCand(const CaliperVec& caliper_vec, const CaliperParam& param, ptvector2df& edge_pt_vec);
	static void				findEdgePairInCaliperCand(const CaliperVec& caliper_vec, const CaliperParam& param, ptvector2df& edge_pt1_vec, ptvector2df& edge_pt2_vec);
	static void				rescoreEdgeInCaliperCand(CaliperVec& caliper_vec, POShapeVec& shape_vec);

	static Recti			calcBoundingBox(const Caliper& caliper);
	static bool				checkTransition(i32 score, i32 edge_transition);
	static f32				getDirectionScore(f32 pos, const CaliperParam& param);
	static f32				getDirectionScore(f32 pos1, f32 pos2, const CaliperParam& param);
};