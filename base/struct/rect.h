#pragma once
#include "types.h"
#include "define.h"
#include "vector2d.h"

#pragma pack(push, 4)
template <class T>
class TRect
{
public:
	//constructor
	TRect()
	{
		reset();
	}

	//constructor
	template <typename U>
	TRect(const TRect<U>& rt)
	{
		this->x1 = (T)rt.x1;
		this->y1 = (T)rt.y1;
		this->x2 = (T)rt.x2;
		this->y2 = (T)rt.y2;
	}

	//constructor
	template <typename U>
	TRect(U x1, U y1, U x2, U y2)
	{
		this->x1 = (T)x1;
		this->y1 = (T)y1;
		this->x2 = (T)x2;
		this->y2 = (T)y2;
	}

	//constructor
	template <typename U>
	TRect(U w, U h)
	{
		this->x1 = (T)0;
		this->y1 = (T)0;
		this->x2 = (T)w;
		this->y2 = (T)h;
	}

	inline void reset()
	{
		x1 = 0; y1 = 0;
		x2 = 0; y2 = 0;
	}

	template <typename U>
    inline void insertPoint(vector2d<U> pt)
	{
		insertPoint(pt.x, pt.y);
	}

	template <typename U>
	void set(U x1, U y1, U x2, U y2)
	{
		this->x1 = (T)x1;
		this->y1 = (T)y1;
		this->x2 = (T)x2;
		this->y2 = (T)y2;
	}

	template <typename U>
	inline void insertPoint(U x, U y)
	{
		if (x1 == 0 && x2 == 0 && y1 == 0 && y2 == 0)
		{
			x1 = (T)x; x2 = x1 + 1;
			y1 = (T)y; y2 = y1 + 1;
			return;
		}
        x1 = po::_min(x1, (T)x);
        y1 = po::_min(y1, (T)y);
        x2 = po::_max(x2, (T)x + 1);
        y2 = po::_max(y2, (T)y + 1);
	}

	template <typename U>
	inline bool pointInRect(U x, U y)
	{
		return (x >= x1 && x < x2 && y >= y1 && y < y2);
	}

	template <typename U>
	void unionRect(TRect<U>& rt)
	{
		x1 = po::_min(x1, (T)rt.x1);
		y1 = po::_min(y1, (T)rt.y1);
		x2 = po::_max(x2, (T)rt.x2);
		y2 = po::_max(y2, (T)rt.y2);
	}

	template <typename U>
	void unionRect(U x1, U y1, U x2, U y2)
	{
		this->x1 = po::_min(this->x1, (T)x1);
		this->y1 = po::_min(this->y1, (T)y1);
		this->x2 = po::_max(this->x2, (T)x2);
		this->y2 = po::_max(this->y2, (T)y2);
	}

	template <typename U>
	inline T getDistancePoint(U x, U y)
	{
		T tx = (T)x;
		T ty = (T)y;
		T dd = PO_MAXINT;
		dd = po::_min(dd, x2 - tx); dd = po::_min(dd, tx - x1);
		dd = po::_min(dd, y2 - ty); dd = po::_min(dd, ty - y1);
		return dd;
	}

	template <typename U>
	inline void inflateRect(U dd)
	{
		T tdd = (T)dd;
		x1 -= tdd; x2 += tdd;
		y1 -= tdd; y2 += tdd;
	}

	template <typename U>
	inline void deflateRect(U dd)
	{
		T tdd = (T)dd;
		x1 += tdd; x2 -= tdd;
		y1 += tdd; y2 -= tdd;
        x1 = po::_min(x1, x2);
        y1 = po::_min(y1, y2);
	}

	template <typename U>
	inline TRect<T> substract(TRect<U> rt1)
	{
		T tx1 = (T)rt1.x1;
		T ty1 = (T)rt1.y1;

		TRect<T> rt = *this;
		rt.x1 -= tx1; rt.y1 -= ty1;
		rt.x2 -= tx1; rt.y2 -= ty1;
		return rt;
	}

	template <typename U>
    inline void crop(U min_x, U min_y, U max_x, U max_y)
	{
		T min_tx = (T)min_x;
		T min_ty = (T)min_y;
		T max_tx = (T)max_x;
		T max_ty = (T)max_y;
		x1 = po::_max(min_tx, x1);
		y1 = po::_max(min_ty, y1);
		x2 = po::_min(max_tx, x2);
		y2 = po::_min(max_ty, y2);
	}

	inline TRect<i32> getRange()
	{
		TRect<i32> rt;
		rt.x1 = (i32)x1;
		rt.y1 = (i32)y1;
		rt.x2 = (i32)(x2 + 0.5f);
		rt.y2 = (i32)(y2 + 0.5f);
		return rt;
	}

	template <typename U>
	inline TRect<i32> intersectRect(U x1, U y1, U x2, U y2) const
	{
		TRect<i32> rt;
		rt.x1 = (i32)po::_max(this->x1, x1);
		rt.y1 = (i32)po::_max(this->y1, y1);
		rt.x2 = (i32)po::_min(this->x2 + 0.5f, x2);
		rt.y2 = (i32)po::_min(this->y2 + 0.5f, y2);

		if (rt.x1 > rt.x2 || rt.y1 > rt.y2)
		{
			rt.reset();
		}
		return rt;
	}

	template <typename U>
    inline TRect<i32> intersectRect(TRect<U> rt1) const
	{
		TRect<i32> rt;
        rt.x1 = (i32)po::_max(x1, rt1.x1);
        rt.y1 = (i32)po::_max(y1, rt1.y1);
        rt.x2 = (i32)po::_min(x2 + 0.5f, rt1.x2 + 0.5f);
        rt.y2 = (i32)po::_min(y2 + 0.5f, rt1.y2 + 0.5f);

		if (rt.x1 > rt.x2 || rt.y1 > rt.y2)
		{
			rt.reset();
		}
		return rt;
	}
	
	template <typename U>
	inline bool isInRect(TRect<U> rt, i32 border_size = 0) const
	{
		if (x1 >= rt.x1 + border_size && y1 >= rt.y1 + border_size &&
			x2 <= rt.x2 - border_size && y2 <= rt.y2 - border_size)
		{
			return true;
		}
		return false;
	}

	template <typename U>
	inline void translate(U dx, U dy)
	{
		T tdx = (T)dx;
		T tdy = (T)dy;
		x1 += tdx; y1 += tdy;
		x2 += tdx; y2 += tdy;
	}

	template <typename U>
	inline void getCenterPoint(U& cx, U& cy)
	{
		cx = (x1 + x2) / 2;
		cy = (y1 + y2) / 2;
	}
	
	inline vector2df getCenterPointf32()
	{
		f32 cx = (x1 + x2) / 2;
		f32 cy = (y1 + y2) / 2;
		return vector2df(cx, cy);
	}

	inline vector2di getCenterPointi32()
	{
		i32 cx = (x1 + x2) / 2;
		i32 cy = (y1 + y2) / 2;
		return vector2di(cx, cy);
	}

	inline bool isEmpty()
	{
		if (x2 - x1 <= 0 || y2 - y1 <= 0)
		{
			return true;
		}
		return false;
	}

	inline bool operator !=(const TRect<T>& rt)
	{
		return (rt.x1 != x1 || rt.x2 != x2 || rt.y1 != y1 || rt.y2 != y2);
	}

	inline T getWidth() const	{ return x2 - x1; }
	inline T getHeight() const	{ return y2 - y1; }
	inline T getArea() const	{ return (x2 - x1)*(y2 - y1); }
	inline T getMinSize() const	{ return po::_min(x2 - x1, y2 - y1); }
	inline T getMaxSize() const	{ return po::_max(x2 - x1, y2 - y1); }
	
public:
	T	x1, y1, x2, y2;
};

typedef TRect<i32> Recti;
typedef TRect<f32> Rectf;
typedef TRect<f64> Rectd;

#pragma pack(pop)
