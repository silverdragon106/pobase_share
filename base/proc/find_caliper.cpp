#include "find_caliper.h"
#include "image_proc.h"
#include "base.h"
#include "logger/logger.h"

#if defined(POR_WITH_OVX)
#include "openvx/sc_graph_pool.h"
#endif

//#define POR_TESTMODE

Caliper::Caliper()
{
	memset(this, 0, sizeof(Caliper));
}

//////////////////////////////////////////////////////////////////////////
CFindCaliper::CFindCaliper()
{
}

CFindCaliper::~CFindCaliper()
{
}

bool CFindCaliper::findCalipers(const u8* img_ptr, u16 w, u16 h, CaliperVec& caliper_vec, const CaliperParam& param)
{
	if (!img_ptr || w*h <= 0)
	{
		return false;
	}
	i32 i, count = (i32)caliper_vec.size();
	if (!CPOBase::isPositive(count))
	{
		return true;
	}

	bool is_processed = false;
#if defined(POR_WITH_OVX)
	CGFindCalipers* graph_ptr = (CGFindCalipers*)g_vx_graph_pool.fetchGraph(
			kGFindCalipers, img_ptr, w, h, caliper_vec.data(), (i32)caliper_vec.size(), &param);
	if (graph_ptr)
	{
		is_processed = graph_ptr->process();
		g_vx_graph_pool.releaseGraph(graph_ptr);
	}
#endif

	if (!is_processed)
	{
		for (i = 0; i < count; i++)
		{
			findEdgesInCaliper(img_ptr, w, h, w, caliper_vec[i], param);
		}
	}
	return true;
}

bool CFindCaliper::findEdgesInCaliper(const u8* img_ptr, u16 w, u16 h, u16 stride_w, Caliper& caliper, const CaliperParam& param)
{
	i32 i, j;
	i32 caliper_width = caliper.caliper_width;
	i32 caliper_height = caliper.caliper_height;
	i32 caliper_size = caliper_height * 2 + 1;
	
	//check bounding box
	vector2df dir2d, tmp2d;
	Recti caliper_bound = calcBoundingBox(caliper);
	if (!caliper_bound.isInRect(Recti(w, h), 1))
	{
		caliper.caliper_type = kCaliperBrokenCaliper;
		return false;
	}

	f32* edge_score_ptr = po_new f32[caliper_size];
	i32* img_offset_ptr = po_new i32[caliper_width];

	//prepare image offsets
	dir2d = caliper.caliper_dir.getOrthogonal();
	for (i = 0; i < caliper_width; i++)
	{
		tmp2d = i*dir2d;
		img_offset_ptr[i] = (i32)(tmp2d.y + 0.5f)*stride_w + (i32)(tmp2d.x + 0.5f);
	}

	//build edge average
	i32 sx, sy, px_pos, px_tmp;
	dir2d = caliper.caliper_dir;
	for (i = -caliper_height; i <= caliper_height; i++)
	{
		tmp2d = i*dir2d;
		sx = (i32)(caliper.caliper_pos.x + tmp2d.x + 0.5f);
		sy = (i32)(caliper.caliper_pos.y + tmp2d.y + 0.5f);
		px_pos = sy*stride_w + sx;
		px_tmp = 0;

		for (j = 0; j < caliper_width; j++)
		{
			px_tmp += img_ptr[px_pos + img_offset_ptr[j]];
		}
		edge_score_ptr[i + caliper_height] = (f32)px_tmp / caliper_width;
	}

	//edge filter
	edgeFilterInCaliper(edge_score_ptr, caliper_size, param.edge_filter_size,
					param.edge_min_contrast, param.edge_max_contrast);

	//search edge and edge pair with caliper parameters
	updateCaliperEdges(edge_score_ptr, caliper_size, caliper, param);

	POSAFE_DELETE_ARRAY(edge_score_ptr);
	POSAFE_DELETE_ARRAY(img_offset_ptr);
	return true;
}

void CFindCaliper::edgeFilterInCaliper(f32* edge_score_ptr, i32 size, i32 filter_size,
								i32 min_contrast, i32 max_contrast)
{
	if (!CPOBase::isPositive(size))
	{
		return;
	}

	static i32 edge_kernel[7][15] = {
										{ -1, 0, 1 },
										{ -3, -2, 0, 2, 3 },
										{ -5, -4, -3, 0, 3, 4, 5 },
										{ -6, -5, -4, -3, 0, 3, 4, 5, 6 },
										{ -8, -7, -6, -5, -4, 0, 4, 5, 6, 7, 8 },
										{ -9, -8, -7, -6, -5, -4, 0, 4, 5, 6, 7, 8, 9 },
										{ -11, -10, -9, -8, -7, -6, -5, 0, 5, 6, 7, 8, 9, 10, 11 }
									};

	static i32 edge_kernel_coeff[7] = { 1, 5, 12, 18, 30, 39, 56 };

	filter_size = po::_max(po::_min(filter_size, 7), 1);
	i32 filter_coeff = edge_kernel_coeff[filter_size - 1];
	i32* filter_kernel = edge_kernel[filter_size - 1];

	//edge filter
	i32 i, j, count;
	i32 filter_count = filter_size * 2 + 1;
	f32 tmp, tmp1;

	count = size + 2 * filter_size + size;
	f32* mem_score_ptr = po_new f32[count];
	memset(mem_score_ptr, 0, sizeof(f32)*count);
	memcpy(mem_score_ptr + filter_size, edge_score_ptr, size*sizeof(f32));

	tmp = edge_score_ptr[0];
	tmp1 = edge_score_ptr[size - 1];
	f32* tmp_score_ptr = mem_score_ptr;
	f32* tmp1_score_ptr = mem_score_ptr + filter_size + size;
	for (i = 0; i < filter_size; i++) //padding
	{
		tmp_score_ptr[i] = tmp;
		tmp1_score_ptr[i] = tmp1;
	}

	f32* tmp2_score_ptr;
	tmp_score_ptr = mem_score_ptr + filter_size;
	tmp1_score_ptr = mem_score_ptr + 2 * filter_size + size;
	for (i = 0; i < size; i++)
	{
		tmp = 0;
		tmp2_score_ptr = tmp_score_ptr + (i - filter_size);
		for (j = 0; j < filter_count; j++)
		{
			tmp += tmp2_score_ptr[j] * filter_kernel[j];
		}
		tmp1_score_ptr[i] = tmp / filter_coeff;
	}

	//non-maximum suppression
	i32 index = -1, cur_sign, prev_sign = 0;
	f32 th_contrast = min_contrast / 2;
	f32 tmp_edge_score = min_contrast;
	memset(edge_score_ptr, 0, sizeof(f32)*size);

	for (i = 0; i < size; i++)
	{
		tmp = fabs(tmp1_score_ptr[i]);
		cur_sign = po::_sgn(tmp1_score_ptr[i]);
		if (cur_sign != prev_sign || tmp < th_contrast)
		{
			if (index >= 0)
			{
				edge_score_ptr[index] = tmp1_score_ptr[index];
			}
			index = -1;
			tmp_edge_score = min_contrast;
		}
		
		prev_sign = cur_sign;
		if (tmp > tmp_edge_score)
		{
			index = i;
			tmp_edge_score = tmp;
		}
	}
	if (index >= 0)
	{
		edge_score_ptr[index] = tmp1_score_ptr[index];
	}

	//free buffer
	POSAFE_DELETE_ARRAY(mem_score_ptr);
}

void CFindCaliper::updateCaliperEdges(f32* edge_score_ptr, i32 caliper_size, Caliper& caliper, const CaliperParam& param)
{
	if (!edge_score_ptr || !CPOBase::isPositive(caliper_size))
	{
		return;
	}

	i32 i, count;
	i32 min_contrast = param.edge_min_contrast;
	i32 caliper_height = caliper.caliper_height;
	ptvector2df cand_vec, edge_vec;
	cand_vec.reserve(caliper_size);

	//make edge candidate vector 
	for (i = 0; i < caliper_size; i++)
	{
		if (fabs(edge_score_ptr[i]) > min_contrast)
		{
			cand_vec.push_back(vector2df(i - caliper_height, edge_score_ptr[i]));
		}
	}

	//caliper edge's type to find
	switch (param.caliper_mode)
	{
		case kFindCaliperEdge:
		{
			updateCaliperOneEdge(cand_vec.data(), (i32)cand_vec.size(), param, edge_vec);
			break;
		}
		case kFindCaliperEdgePair:
		{
			updateCaliperEdgePair(cand_vec.data(), (i32)cand_vec.size(), param, edge_vec);
			break;
		}
		case kFindCaliperEdgeCands:
		{
			updateCaliperEdgeCands(cand_vec.data(), (i32)cand_vec.size(), param, edge_vec);
			break;
		}
	}

	//update caliper's edge
	count = 0;
	switch (param.caliper_mode)
	{
		case kFindCaliperEdge:
		case kFindCaliperEdgePair:
		{
			f32 px = caliper.caliper_pos.x;
			f32 py = caliper.caliper_pos.y;
			f32 caliper_halfw = caliper.caliper_width / 2;
			vector2df v_dir = caliper.caliper_dir;
			vector2df h_dir = v_dir.getOrthogonal();

			count = po::_min((i32)edge_vec.size(), CALIPER_MAX_EDGE);
			caliper.edge_pt_count = count;
			vector2df* edge_cand_ptr = caliper.edge_pt_ptr;
			for (i = 0; i < count; i++)
			{
				edge_cand_ptr[i] = vector2df(px, py) + v_dir*edge_vec[i].x + h_dir*caliper_halfw;
			}
			break;
		}
		case kFindCaliperEdgeCands:
		{
			count = (i32)edge_vec.size();
			if (count > CALIPER_MAX_EDGE)
			{
				//sortby edge strength
				f32vector edge_score;
				edge_score.resize(count);
				for (i = 0; i < count; i++)
				{
					edge_score[i] = fabs(edge_vec[i].y);
				}

				i32 j;
				for (i = 0; i < CALIPER_MAX_EDGE; i++)
				{
					for (j = i + 1; j < count; j++)
					{
						if (edge_score[i] < edge_score[j])
						{
							CPOBase::swap(edge_score[i], edge_score[j]);
							CPOBase::swap(edge_vec[i], edge_vec[j]);
						}
					}
				}
			}

			count = po::_min(count, CALIPER_MAX_EDGE);
			caliper.edge_pt_count = count;
			vector2df* edge_cand_ptr = caliper.edge_pt_ptr;
			for (i = 0; i < count; i++)
			{
				edge_cand_ptr[i] = edge_vec[i];
			}
			break;
		}
	}
	caliper.caliper_type = count ? kCaliperCommon : kCaliperBrokenCaliper;
}

void CFindCaliper::findOneEdgeInCaliperCand(const CaliperVec& caliper_vec, const CaliperParam& param,
										ptvector2df& edge_pt_vec)
{
	i32 i, count = (i32)caliper_vec.size();
	edge_pt_vec.resize(count);
	vector2df* edge_pt_ptr = edge_pt_vec.data();
	
	ptvector2df edge_vec;
	vector2df v_dir, h_dir;
	f32 px, py, caliper_halfw;

	for (i = 0; i < count; i++)
	{
		const Caliper& caliper = caliper_vec[i];
		if (!CPOBase::bitCheck(caliper.caliper_type, kCaliperBrokenCaliper))
		{
			updateCaliperOneEdge(caliper.edge_pt_ptr, caliper.edge_pt_count, param, edge_vec);
			if (edge_vec.size() >= 1)
			{
				px = caliper.caliper_pos.x;
				py = caliper.caliper_pos.y;
				v_dir = caliper.caliper_dir;
				h_dir = v_dir.getOrthogonal();
				caliper_halfw = caliper.caliper_width / 2;

				edge_pt_vec[i] = vector2df(px, py) + v_dir*edge_vec[0].x + h_dir*caliper_halfw;
				continue;
			}
		}
		edge_pt_ptr[i] = vector2df(PO_MINVAL, PO_MINVAL);
	}
}

void CFindCaliper::findEdgePairInCaliperCand(const CaliperVec& caliper_vec, const CaliperParam& param,
										ptvector2df& edge_pt1_vec, ptvector2df& edge_pt2_vec)
{
	i32 i, count = (i32)caliper_vec.size();
	edge_pt1_vec.resize(count);
	edge_pt2_vec.resize(count);
	vector2df* edge_pt1_ptr = edge_pt1_vec.data();
	vector2df* edge_pt2_ptr = edge_pt2_vec.data();

	f32 px, py, caliper_halfw;
	vector2df v_dir, h_dir, tmp2d;
	ptvector2df edge_vec;

	for (i = 0; i < count; i++)
	{
		const Caliper& caliper = caliper_vec[i];
		if (!CPOBase::bitCheck(caliper.caliper_type, kCaliperBrokenCaliper))
		{
			updateCaliperEdgePair(caliper.edge_pt_ptr, caliper.edge_pt_count, param, edge_vec);
			if (edge_vec.size() >= 2)
			{
				px = caliper.caliper_pos.x;
				py = caliper.caliper_pos.y;
				v_dir = caliper.caliper_dir;
				h_dir = v_dir.getOrthogonal();
				caliper_halfw = caliper.caliper_width / 2;

				tmp2d = vector2df(px, py) + h_dir*caliper_halfw;
				edge_pt1_vec[i] = tmp2d + v_dir*edge_vec[0].x;
				edge_pt2_vec[i] = tmp2d + v_dir*edge_vec[1].x;
				continue;
			}
		}
		edge_pt1_ptr[i] = vector2df(PO_MINVAL, PO_MINVAL);
		edge_pt2_ptr[i] = vector2df(PO_MINVAL, PO_MINVAL);
	}
}

void CFindCaliper::rescoreEdgeInCaliperCand(CaliperVec& caliper_vec, POShapeVec& shape_vec)
{
	i32 i, j, k, pt_count;
	i32 count = (i32)caliper_vec.size();
	i32 shape_count = (i32)shape_vec.size();
	if (!CPOBase::isPositive(count) || !CPOBase::isPositive(shape_count))
	{
		return;
	}

	CPOShape* shape_ptr = shape_vec.data();
	vector2df v_dir, h_dir, tmp2d, pos2d;
	vector2df* edge_pt_ptr;
	f32 px, py, caliper_halfw, caliper_h, min_dist, scale;

	for (i = 0; i < count; i++)
	{
		Caliper& caliper = caliper_vec[i];

		px = caliper.caliper_pos.x;
		py = caliper.caliper_pos.y;
		v_dir = caliper.caliper_dir;
		h_dir = v_dir.getOrthogonal();
		caliper_halfw = caliper.caliper_width / 2;
		caliper_h = caliper.caliper_height * 2 + 1;
		tmp2d = vector2df(px, py) + h_dir*caliper_halfw;

		pt_count = caliper.edge_pt_count;
		edge_pt_ptr = caliper.edge_pt_ptr;
		for (j = 0; j < pt_count; j++)
		{
			min_dist = 1E+10;
			pos2d = tmp2d + v_dir*edge_pt_ptr[j].x;
			for (k = 0; k < shape_count; k++)
			{
				min_dist = po::_min(shape_ptr[k].distance(pos2d), min_dist);
			}

			scale = ((f32)po::_max(caliper_h - min_dist, 0) / (2 * caliper_h) + 1.0f);
			edge_pt_ptr[j].y *= scale;
		}
	}
}

void CFindCaliper::updateCaliperOneEdge(const vector2df* cand_ptr, i32 count, const CaliperParam& param, ptvector2df& edge_vec)
{
	edge_vec.clear();
	if (!CPOBase::isPositive(count))
	{
		return;
	}

	vector2df pt;
	f32 abs_score, max_score = 0;
	f32 px, py, dscore, max_dscore = -1E+10;
	i32 i, max_index = -1;
	i32 threshold = param.edge_threshold;
	i32 edge_transition = param.edge0_transition;
	
	switch (param.edge_rule)
	{
		case kCaliperEdgeRuleFirst:
		{
			for (i = 0; i < count; i++)
			{
				pt = cand_ptr[i];
				px = pt.x; py = pt.y;
				abs_score = fabs(py);
				if (checkTransition(py, edge_transition) && abs_score > threshold)
				{
					dscore = getDirectionScore(px, param);
					if (dscore > max_dscore)
					{
						max_index = i;
						max_dscore = dscore;
					}
				}
			}
			break;
		}
		default:
		{
			for (i = 0; i < count; i++)
			{
				pt = cand_ptr[i];
				px = pt.x; py = pt.y;
				abs_score = fabs(py);
				if (checkTransition(py, edge_transition) && abs_score > threshold)
				{
					dscore = getDirectionScore(px, param);
					if ((dscore >= max_dscore && abs_score > max_score * 0.8f) ||
						(dscore < max_dscore && abs_score > max_score * 1.2f))
					{
						max_index = i;
						max_dscore = dscore;
						max_score = abs_score;
					}
				}
			}
			break;
		}
	}

	if (max_index >= 0)
	{
		edge_vec.push_back(cand_ptr[max_index]);
	}
}

void CFindCaliper::updateCaliperEdgeCands(const vector2df* cand_ptr, i32 count, const CaliperParam& param, ptvector2df& edge_vec)
{
	edge_vec.clear();
	i32 i, threshold = param.edge_threshold;
	if (!CPOBase::isPositive(count))
	{
		return;
	}
	
	//cut edge candidate with threshold
	vector2df tmp2d;
	edge_vec.reserve(count);
	for (i = 0; i < count; i++)
	{
		tmp2d = cand_ptr[i];
		if (fabs(tmp2d.y) > threshold)
		{
			edge_vec.push_back(tmp2d);
		}
	}
}

void CFindCaliper::updateCaliperEdgePair(const vector2df* cand_ptr, i32 count, const CaliperParam& param, ptvector2df& edge_vec)
{
	edge_vec.clear();
	i32 i, j;
	f32 score1, score2, pos1, pos2;
	if (!CPOBase::isPositive(count))
	{
		return;
	}

	vector2df pt1, pt2;
	f32 dscore, abs_score, max_score = 0, max_dscore = -1E+10;
	i32 max_index_i = -1;
	i32 max_index_j = -1;
	i32 threshold = param.edge_threshold;
	i32 edge0_transition = param.edge0_transition;
	i32 edge1_transition = param.edge1_transition;

	switch (param.edge_rule)
	{
		case kCaliperEdgeRuleFirst:
		{
			for (i = 0; i < count; i++)
			{
				pt1 = cand_ptr[i];
				pos1 = pt1.x; score1 =pt1.y;
				if (!checkTransition(score1, edge0_transition))
				{
					continue;
				}

				for (j = i + 1; j < count; j++)
				{
					pt2 = cand_ptr[j];
					pos2 = pt2.x; score2 = pt2.y;
					abs_score = (fabs(score1) + fabs(score2)) / 2;
					if (checkTransition(score2, edge1_transition) && abs_score > threshold)
					{
						dscore = getDirectionScore(pos1, pos2, param);
						if (dscore > max_dscore)
						{
							max_index_i = i;
							max_index_j = j;
							max_dscore = dscore;
						}
					}
				}
			}
			break;
		}
		case kCaliperEdgeRuleStrong:
		{
			for (i = 0; i < count; i++)
			{
				pt1 = cand_ptr[i];
				pos1 = pt1.x; score1 = pt1.y;
				if (!checkTransition(score1, edge0_transition))
				{
					continue;
				}

				for (j = i + 1; j < count; j++)
				{
					pt2 = cand_ptr[j];
					pos2 = pt2.x; score2 = pt2.y;
					abs_score = (f32)(fabs(score1) + fabs(score2)) / 2;
					if (checkTransition(score2, edge1_transition) && abs_score > threshold)
					{
						dscore = getDirectionScore(pos1, pos2, param);
						if ((dscore >= max_dscore && abs_score > max_score * 0.8f) ||
							(dscore < max_dscore && abs_score > max_score * 1.2f))
						{
							max_index_i = i;
							max_index_j = j;
							max_dscore = dscore;
							max_score = abs_score;
						}
					}
				}
			}
			break;
		}
		case kCaliperEdgeRulePair:
		{
			f32 pair_rate, max_pair_rate = 0;
			f32 pair_size = param.edge_pair_size;

			for (i = 0; i < count; i++)
			{
				pt1 = cand_ptr[i];
				pos1 = pt1.x;  score1 = pt1.y;
				if (!checkTransition(score1, edge0_transition))
				{
					continue;
				}

				for (j = i + 1; j < count; j++)
				{
					pt2 = cand_ptr[j];
					pos2 = pt2.x; score2 = pt2.y;
					abs_score = (fabs(score1) + fabs(score2)) / 2;
					pair_rate = CPOBase::ratio((f32)fabs(cand_ptr[i].x - cand_ptr[j].x), pair_size);
					if (checkTransition(score2, edge1_transition) && abs_score > threshold)
					{
						dscore = getDirectionScore(pos1, pos2, param);
						if ((dscore >= max_dscore && pair_rate > max_pair_rate * 0.8f) ||
							(dscore < max_dscore && pair_rate > max_pair_rate * 1.2f))
						{
							max_index_i = i;
							max_index_j = j;
							max_dscore = dscore;
							max_pair_rate = pair_rate;
						}
					}
				}
			}
			break;
		}
	}

	//resort caliper edge
	if (max_index_i >= 0 && max_index_j >= 0)
	{
		edge_vec.push_back(cand_ptr[max_index_i]);
		edge_vec.push_back(cand_ptr[max_index_j]);
		if (edge_vec[0].x > edge_vec[1].x)
		{
			vector2df tmp2d;
			tmp2d = edge_vec[0]; edge_vec[0] = edge_vec[1]; edge_vec[1] = tmp2d;
		}
	}
}

bool CFindCaliper::checkTransition(i32 score, i32 edge_transition)
{
	if (edge_transition == kBeadEdgeEither)
	{
		return true;
	}
	return ((edge_transition == kBeadEdgeBlackToWhite && score >= 0) ||
			(edge_transition == kBeadEdgeWhiteToBlack && score <= 0));
}

Recti CFindCaliper::calcBoundingBox(const Caliper& caliper)
{
	vector2df dir, dx, dy;
	dir = caliper.caliper_dir;
	dy = dir*caliper.caliper_height;
	dx = dir.getOrthogonal()*caliper.caliper_width;

	vector2di pt[4];
	i32 px = caliper.caliper_pos.x;
	i32 py = caliper.caliper_pos.y;
	pt[0] = vector2di(px + (i32)(dy.x + 0.5f), py + (i32)(dy.y + 0.5f));
	pt[1] = vector2di(px - (i32)(dy.x - 0.5f), py - (i32)(dy.y - 0.5f));
	pt[2] = vector2di(px + (i32)(dy.x + dx.x + 0.5f), py + (i32)(dy.y + dx.y + 0.5f));
	pt[3] = vector2di(px - (i32)(dy.x - dx.x - 0.5f), py - (i32)(dy.y - dx.y - 0.5f));

	Recti rt;
	rt.x1 = po::_min(po::_min(pt[0].x, pt[1].x), po::_min(pt[2].x, pt[3].x));
	rt.x2 = po::_max(po::_max(pt[0].x, pt[1].x), po::_max(pt[2].x, pt[3].x));
	rt.y1 = po::_min(po::_min(pt[0].y, pt[1].y), po::_min(pt[2].y, pt[3].y));
	rt.y2 = po::_max(po::_max(pt[0].y, pt[1].y), po::_max(pt[2].y, pt[3].y));
	return rt;
}

f32 CFindCaliper::getDirectionScore(f32 pos, const CaliperParam& param)
{
	switch (param.edge_direction)
	{
		case kBeadEdgeDirectionLeftToRight:
		{
			return pos;
		}
		case kBeadEdgeDirectionRightToLeft:
		{
			return -pos;
		}
		case kBeadEdgeDirectionOutsideToIn:
		{
			return abs(pos);
		}
		case kBeadEdgeDirectionInToOutside:
		{
			return -abs(pos);
		}
	}
	return 0;
}

f32 CFindCaliper::getDirectionScore(f32 pos1, f32 pos2, const CaliperParam& param)
{
	switch (param.edge_direction)
	{
		case kBeadEdgeDirectionLeftToRight:
		{
			return (pos1 + pos2);
		}
		case kBeadEdgeDirectionRightToLeft:
		{
			return -(pos1 + pos2);
		}
		case kBeadEdgeDirectionOutsideToIn:
		{
			f32 w = po::_max(pos1, pos2) - po::_min(pos1, pos2);
			return w - abs(pos1 + pos2) / 2;
		}
		case kBeadEdgeDirectionInToOutside:
		{
			f32 w = po::_max(pos1, pos2) - po::_min(pos1, pos2);
			return -w - abs(pos1 + pos2) / 2;
		}
	}
	return 0;
}
