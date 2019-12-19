#pragma once
#include "define.h"

struct SSIMMap
{
public:
	SSIMMap();
	~SSIMMap();

	void				initBuffer(i32 w, i32 h, bool is_external = false);
	void				freeBuffer();

	inline bool			hasSSIMMap() { return mean_lum_ptr != NULL; };

public:
	u8*					mean_lum_ptr;
	u16*				mean_struct_sq_ptr;
	u16*				edge_img_ptr;
	f32*				ssim_map_ptr;

	bool				is_external;
};

class CSSIMProcessor
{
public:
	CSSIMProcessor();
	~CSSIMProcessor();

	void				initInstance();
	void				exitInstance();

	void				makeSSIMMap(SSIMMap& ssim_map, u8* img_ptr, i32 w, i32 h);
	f32					checkStructSimilar(SSIMMap& ssim_map1, SSIMMap& ssim_map2, i32 w, i32 h);
};