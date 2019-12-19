#pragma once

#include <math.h>

#pragma pack(push, 4)

const f32 ROUNDING_ERROR_f32 = 0.000001f;
const f64 ROUNDING_ERROR_f64 = 0.00000001;

//! returns if a equals b, taking possible rounding errors into account
inline bool equals(const f64 a, const f64 b, const f64 tolerance = ROUNDING_ERROR_f64)
{
	return (a + tolerance >= b) && (a - tolerance <= b);
}

//! returns if a equals b, taking possible rounding errors into account
inline bool equals(const f32 a, const f32 b, const f32 tolerance = ROUNDING_ERROR_f32)
{
	return (a + tolerance >= b) && (a - tolerance <= b);
}

inline bool isZero(const f32 a)
{
	return abs(a) < ROUNDING_ERROR_f32;
}

inline bool isZero(const f64 a)
{
	return abs(a) < ROUNDING_ERROR_f64;
}

//! 2d vector template class with lots of operators and methods.
template <class T>
class vector2d
{
public:
	vector2d() : x(0), y(0) {}
	vector2d(T nx, T ny) : x(nx), y(ny) {}
	explicit vector2d(T n) : x(n), y(n) {}

	//! Copy constructor
	vector2d(const vector2d<T>& other) : x(other.x), y(other.y) {}

	// operators
	vector2d<T> operator-() const { return vector2d<T>(-x, -y); }

	vector2d<T>& operator=(const vector2d<T>& other) { x = other.x; y = other.y; return *this; }

	vector2d<T> operator+(const vector2d<T>& other) const { return vector2d<T>(x + other.x, y + other.y); }
	vector2d<T>& operator+=(const vector2d<T>& other) { x+=other.x; y+=other.y; return *this; }
	vector2d<T> operator+(const T v) const { return vector2d<T>(x + v, y + v); }
	vector2d<T>& operator+=(const T v) { x+=v; y+=v; return *this; }

	vector2d<T> operator-(const vector2d<T>& other) const { return vector2d<T>(x - other.x, y - other.y); }
	vector2d<T>& operator-=(const vector2d<T>& other) { x-=other.x; y-=other.y; return *this; }
	vector2d<T> operator-(const T v) const { return vector2d<T>(x - v, y - v); }
	vector2d<T>& operator-=(const T v) { x-=v; y-=v; return *this; }

	vector2d<T> operator*(const vector2d<T>& other) const { return vector2d<T>(x * other.x, y * other.y); }
	vector2d<T>& operator*=(const vector2d<T>& other) { x*=other.x; y*=other.y; return *this; }
	vector2d<T> operator*(const T v) const { return vector2d<T>(x * v, y * v); }
	vector2d<T>& operator*=(const T v) { x*=v; y*=v; return *this; }

	vector2d<T> operator/(const vector2d<T>& other) const { return vector2d<T>(x / other.x, y / other.y); }
	vector2d<T>& operator/=(const vector2d<T>& other) { x/=other.x; y/=other.y; return *this; }
	vector2d<T> operator/(const T v) const { return vector2d<T>(x / v, y / v); }
	vector2d<T>& operator/=(const T v) { x/=v; y/=v; return *this; }

	//! sort in order X, Y. Equality with rounding tolerance.
	bool operator<=(const vector2d<T>&other) const
	{
		return 	(x<other.x || equals(x, other.x)) ||
				(equals(x, other.x) && (y<other.y || equals(y, other.y)));
	}

	//! sort in order X, Y. Equality with rounding tolerance.
	bool operator>=(const vector2d<T>&other) const
	{
		return 	(x>other.x || equals(x, other.x)) ||
				(equals(x, other.x) && (y>other.y || equals(y, other.y)));
	}

	//! sort in order X, Y. Difference must be above rounding tolerance.
	bool operator<(const vector2d<T>&other) const
	{
		return 	(x<other.x && !equals(x, other.x)) ||
				(equals(x, other.x) && y<other.y && !equals(y, other.y));
	}

	//! sort in order X, Y. Difference must be above rounding tolerance.
	bool operator>(const vector2d<T>&other) const
	{
		return 	(x>other.x && !equals(x, other.x)) ||
				(equals(x, other.x) && y>other.y && !equals(y, other.y));
	}

	bool operator==(const vector2d<T>& other) const { return equals(other); }
	bool operator!=(const vector2d<T>& other) const { return !equals(other); }

	// functions

	//! Checks if this vector equals the other one.
	/** Takes floating point rounding errors into account.
	\param other Vector to compare with.
	\return True if the two vector are (almost) equal, else false. */
	bool equals(const vector2d<T>& other) const
	{
		return equals(x, other.x) && equals(y, other.y);
	}

	vector2d<T>& set(T nx, T ny) {x=nx; y=ny; return *this; }
	vector2d<T>& set(const vector2d<T>& p) { x=p.x; y=p.y; return *this; }

	//! Gets the length of the vector.
	/** \return The length of the vector. */
	T getLength() const { return sqrt(x*x + y*y); }

	//! Get the squared length of this vector
	/** This is useful because it is much faster than getLength().
	\return The squared length of the vector. */
	T getLengthSQ() const { return x*x + y*y; }

	//! Get the dot product of this vector with another.
	/** \param other Other vector to take dot product with.
	\return The dot product of the two vectors. */
	T dotProduct(const vector2d<T>& other) const
	{
		return x*other.x + y*other.y;
	}

	T crossProduct(const vector2d<T>& other) const
	{
		return x*other.y - y*other.x;
	}

	//! Gets distance from another point.
	/** Here, the vector is interpreted as a point in 2-dimensional space.
	\param other Other vector to measure from.
	\return Distance from other point. */
	T getDistanceFrom(const vector2d<T>& other) const
	{
		return vector2d<T>(x - other.x, y - other.y).getLength();
	}

	//! Returns squared distance from another point.
	/** Here, the vector is interpreted as a point in 2-dimensional space.
	\param other Other vector to measure from.
	\return Squared distance from other point. */
	T getDistanceFromSQ(const vector2d<T>& other) const
	{
		return vector2d<T>(x - other.x, y - other.y).getLengthSQ();
	}

	vector2d<T> getOrthogonal()
	{
		return vector2d<T>(-y, x);
	}

	//! Normalize the vector.
	/** The null vector is left untouched.
	\return Reference to this vector, after normalization. */
	vector2d<T>& normalize()
	{
		f32 length = (f32)(x*x + y*y);
		if (length == 0)
			return *this;
		length = 1 / sqrt(length);
		x = (T)(x * length);
		y = (T)(y * length);
		return *this;
	}

	vector2d<T>& normalize(T& len)
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

	bool isEmpty() const
	{
		return isZero(x) && isZero(y);
	}

	//! Returns if this vector interpreted as a point is on a line between two other points.
	/** It is assumed that the point is on the line.
	\param begin Beginning vector to compare between.
	\param end Ending vector to compare between.
	\return True if this vector is between begin and end, false if not. */
	bool isBetweenPoints(const vector2d<T>& begin, const vector2d<T>& end) const
	{
		if (begin.x != end.x)
		{
			return ((begin.x <= x && x <= end.x) ||
				(begin.x >= x && x >= end.x));
		}
		else
		{
			return ((begin.y <= y && y <= end.y) ||
				(begin.y >= y && y >= end.y));
		}
	}

	//! Creates an interpolated vector between this vector and another vector.
	/** \param other The other vector to interpolate with.
	\param d Interpolation value between 0.0f (all the other vector) and 1.0f (all this vector).
	Note that this is the opposite direction of interpolation to getInterpolated_quadratic()
	\return An interpolated vector.  This vector is not modified. */
	vector2d<T> getInterpolated(const vector2d<T>& other, f64 d) const
	{
		f64 inv = 1.0f - d;
		return vector2d<T>((T)(other.x*inv + x*d), (T)(other.y*inv + y*d));
	}

	//! Creates a quadratically interpolated vector between this and two other vectors.
	/** \param v2 Second vector to interpolate with.
	\param v3 Third vector to interpolate with (maximum at 1.0f)
	\param d Interpolation value between 0.0f (all this vector) and 1.0f (all the 3rd vector).
	Note that this is the opposite direction of interpolation to getInterpolated() and interpolate()
	\return An interpolated vector. This vector is not modified. */
	vector2d<T> getInterpolated_quadratic(const vector2d<T>& v2, const vector2d<T>& v3, f64 d) const
	{
		// this*(1-d)*(1-d) + 2 * v2 * (1-d) + v3 * d * d;
		const f64 inv = 1.0f - d;
		const f64 mul0 = inv * inv;
		const f64 mul1 = 2.0f * d * inv;
		const f64 mul2 = d * d;

		return vector2d<T> ( (T)(x * mul0 + v2.x * mul1 + v3.x * mul2),
					(T)(y * mul0 + v2.y * mul1 + v3.y * mul2));
	}

	//! Sets this vector to the linearly interpolated vector between a and b.
	/** \param a first vector to interpolate with, maximum at 1.0f
	\param b second vector to interpolate with, maximum at 0.0f
	\param d Interpolation value between 0.0f (all vector b) and 1.0f (all vector a)
	Note that this is the opposite direction of interpolation to getInterpolated_quadratic()
	*/
	vector2d<T>& interpolate(const vector2d<T>& a, const vector2d<T>& b, f64 d)
	{
		x = (T)((f64)b.x + ( ( a.x - b.x ) * d ));
		y = (T)((f64)b.y + ( ( a.y - b.y ) * d ));
		return *this;
	}

	//! X coordinate of vector.
	T x;

	//! Y coordinate of vector.
	T y;
};

//! Typedef for f32 2d vector.
typedef vector2d<f32> vector2df;

//! Typedef for integer 2d vector.
typedef vector2d<i32> vector2di;

//! Typedef for double 2d vector.
typedef vector2d<f64> vector2dd;

template<class S, class T>
vector2d<T> operator*(const S scalar, const vector2d<T>& vector) { return vector*scalar; }

#pragma pack(pop)