#pragma once
#include "types.h"

#if defined(POR_WITH_CUDA_DLL)
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#pragma pack(push, 4)

template<typename T>
class cuRect
{
public:
	T x1, y1, x2, y2;
	CUDA_CALLABLE cuRect() { x1 = 0; y1 = 0; x2 = 0; y2 = 0; }

	//constructor
	template <typename U>
	CUDA_CALLABLE cuRect(U x1, U y1, U x2, U y2)
	{
		this->x1 = (T)x1;
		this->y1 = (T)y1;
		this->x2 = (T)x2;
		this->y2 = (T)y2;
	}

	//constructor
	template <typename U>
	CUDA_CALLABLE cuRect(U w, U h)
	{
		this->x1 = (T)0;
		this->y1 = (T)0;
		this->x2 = (T)w;
		this->y2 = (T)h;
	}

	template <typename U>
	CUDA_CALLABLE inline bool isInRect(cuRect<U> rt, i32 border_size = 0) const
	{
		if (x1 >= rt.x1 + border_size && y1 >= rt.y1 + border_size &&
			x2 <= rt.x2 - border_size && y2 <= rt.y2 - border_size)
		{
			return true;
		}
		return false;
	}
};
typedef cuRect<float> cuRectf;
typedef cuRect<int> cuRecti;

template<typename T>
class cuVector2d
{
public:
	T x, y;

	CUDA_CALLABLE cuVector2d() : x(0), y(0) {}
	CUDA_CALLABLE cuVector2d(T nx, T ny) : x(nx), y(ny) {}
	explicit CUDA_CALLABLE cuVector2d(T n) : x(n), y(n) {}

	CUDA_CALLABLE cuVector2d(const cuVector2d<T>& other) : x(other.x), y(other.y) {}

	CUDA_CALLABLE cuVector2d<T> getOrthogonal() { return cuVector2d<T>(-y, x);	}
	CUDA_CALLABLE T getLength() const { return sqrt(x*x + y*y); }

	CUDA_CALLABLE cuVector2d<T>& normalize()
	{
		f32 length = (f32)(x*x + y*y);
		if (length == 0)
			return *this;
        length = 1 / sqrt(length);
		x = (T)(x * length);
		y = (T)(y * length);
		return *this;
	}

	CUDA_CALLABLE cuVector2d<T>& normalize(T& len)
	{
		len = x*x + y*y;
		if (len == 0)
			return *this;

		len = sqrt(len);
		T invlen = 1 / len;
		x = (T)(x * invlen);
		y = (T)(y * invlen);
		return *this;
	}

	CUDA_CALLABLE T dotProduct(const cuVector2d<T>& other) const
	{
		return x*other.x + y*other.y;
	}

	//CUDA_CALLABLE cuVector2d<T> operator*(const T v) const { return cuVector2d<T>(x * v, y * v); }
	//CUDA_CALLABLE cuVector2d<T> operator+(const cuVector2d<T>& other) const { return cuVector2d<T>(x + other.x, y + other.y); }

	//CUDA_CALLABLE cuVector2d<T> operator-() const { return cuVector2d<T>(-x, -y); }
	//CUDA_CALLABLE cuVector2d<T> operator-(const cuVector2d<T>& other) const { return cuVector2d<T>(x - other.x, y - other.y); }
	//CUDA_CALLABLE cuVector2d<T> operator-(const T v) const { return cuVector2d<T>(x - v, y - v); }


	CUDA_CALLABLE cuVector2d<T> operator-() const { return cuVector2d<T>(-x, -y); }

	CUDA_CALLABLE cuVector2d<T>& operator=(const cuVector2d<T>& other) { x = other.x; y = other.y; return *this; }

	CUDA_CALLABLE cuVector2d<T> operator+(const cuVector2d<T>& other) const { return cuVector2d<T>(x + other.x, y + other.y); }
	CUDA_CALLABLE cuVector2d<T>& operator+=(const cuVector2d<T>& other) { x += other.x; y += other.y; return *this; }
	CUDA_CALLABLE cuVector2d<T> operator+(const T v) const { return cuVector2d<T>(x + v, y + v); }
	CUDA_CALLABLE cuVector2d<T>& operator+=(const T v) { x += v; y += v; return *this; }

	CUDA_CALLABLE cuVector2d<T> operator-(const cuVector2d<T>& other) const { return cuVector2d<T>(x - other.x, y - other.y); }
	CUDA_CALLABLE cuVector2d<T>& operator-=(const cuVector2d<T>& other) { x -= other.x; y -= other.y; return *this; }
	CUDA_CALLABLE cuVector2d<T> operator-(const T v) const { return cuVector2d<T>(x - v, y - v); }
	CUDA_CALLABLE cuVector2d<T>& operator-=(const T v) { x -= v; y -= v; return *this; }

	CUDA_CALLABLE cuVector2d<T> operator*(const cuVector2d<T>& other) const { return cuVector2d<T>(x * other.x, y * other.y); }
	CUDA_CALLABLE cuVector2d<T>& operator*=(const cuVector2d<T>& other) { x *= other.x; y *= other.y; return *this; }
	CUDA_CALLABLE cuVector2d<T> operator*(const T v) const { return cuVector2d<T>(x * v, y * v); }
	CUDA_CALLABLE cuVector2d<T>& operator*=(const T v) { x *= v; y *= v; return *this; }

	CUDA_CALLABLE cuVector2d<T> operator/(const cuVector2d<T>& other) const { return cuVector2d<T>(x / other.x, y / other.y); }
	CUDA_CALLABLE cuVector2d<T>& operator/=(const cuVector2d<T>& other) { x /= other.x; y /= other.y; return *this; }
	CUDA_CALLABLE cuVector2d<T> operator/(const T v) const { return cuVector2d<T>(x / v, y / v); }
	CUDA_CALLABLE cuVector2d<T>& operator/=(const T v) { x /= v; y /= v; return *this; }

	//! sort in order X, Y. Equality with rounding tolerance.
	CUDA_CALLABLE bool operator<=(const cuVector2d<T>&other) const
	{
		return 	(x<other.x || equals(x, other.x)) ||
			(equals(x, other.x) && (y<other.y || equals(y, other.y)));
	}

	//! sort in order X, Y. Equality with rounding tolerance.
	CUDA_CALLABLE bool operator>=(const cuVector2d<T>&other) const
	{
		return 	(x>other.x || equals(x, other.x)) ||
			(equals(x, other.x) && (y>other.y || equals(y, other.y)));
	}

	//! sort in order X, Y. Difference must be above rounding tolerance.
	CUDA_CALLABLE bool operator<(const cuVector2d<T>&other) const
	{
		return 	(x<other.x && !equals(x, other.x)) ||
			(equals(x, other.x) && y<other.y && !equals(y, other.y));
	}

	//! sort in order X, Y. Difference must be above rounding tolerance.
	CUDA_CALLABLE bool operator>(const cuVector2d<T>&other) const
	{
		return 	(x>other.x && !equals(x, other.x)) ||
			(equals(x, other.x) && y>other.y && !equals(y, other.y));
	}

	CUDA_CALLABLE bool operator==(const cuVector2d<T>& other) const { return equals(other); }
	CUDA_CALLABLE bool operator!=(const cuVector2d<T>& other) const { return !equals(other); }

	// functions

	//! Checks if this vector equals the other one.
	/** Takes floating point rounding errors into account.
	\param other Vector to compare with.
	\return True if the two vector are (almost) equal, else false. */
	CUDA_CALLABLE bool equals(const cuVector2d<T>& other) const
	{
		return equals(x, other.x) && equals(y, other.y);
	}

};
template<class S, class T>
CUDA_CALLABLE cuVector2d<T> operator*(const S scalar, const cuVector2d<T>& vector) { return vector*scalar; }

typedef cuVector2d<float> cuVector2df;
typedef cuVector2d<int> cuVector2di;
typedef std::vector<cuVector2df> cuPtVector2df;
typedef std::vector<cuVector2di> cuPtVector2di;

template <typename T>
struct cuPixel
{
	T					x;
	T					y;
};

template <typename T>
struct cuPixel3
{
	T					x;
	T					y;
	T					g;
};
typedef cuPixel3<u16> cuPixel3u;

#pragma pack(pop)
