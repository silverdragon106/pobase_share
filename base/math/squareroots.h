#pragma once
#include "config.h"

#ifndef POR_IA64
#if defined(POR_WINDOWS)

#include <windows.h>

namespace POMath
{
#define SQRT_MAGIC_F 0x5f3759df

	//		[speed]			[precision]
	//0		100				100
	//1		127.868			100
	//2		141.718			99.9045
	//3		196.337			97.1726
	//4		23.454			100
	//5		3.64049			100
	//6		61.1251			99.9998
	//7		235.007			97.1726
	//8		44.2157			100
	//9		20.1051			100
	//10	36.5731			100
	//11	27.9527			99.4641
	//12	0.124521		100
	//13	202.134			100
	//14	406.752			100


	//http://www.lomont.org/Math/Papers/2003/InvSqrt.pdf
	//For sqrt1,sqrt2,and sqrt3 all Credit goes to http://ilab.usc.edu/wiki/index.php/Fast_Square_Root
	//For sqrt5 all credit goes to Sanchit Karve(A.K.A born2c0de), He can be contacted at born2c0de@hotmail.com
	//Functions sqrt9,sqrt12 and sqrt11 only compute integer inputs 
	//enjoy :)


	inline f32 sqrt1(const f32 x)
	{
		union
		{
			i32 i;
			f32 x;
		} u;
		u.x = x;
		u.i = (1 << 29) + (u.i >> 1) - (1 << 22);

		// Two Babylonian Steps (simplified from:)
		// u.x = 0.5f * (u.x + x/u.x);
		// u.x = 0.5f * (u.x + x/u.x);
		u.x = u.x + x / u.x;
		u.x = 0.25f*u.x + x / u.x;

		return u.x;
	}

	inline f32 sqrt2(const f32 x)
	{
		const f32 xhalf = 0.5f*x;

		union // get bits for floating value
		{
			f32 x;
			i32 i;
		} u;
		u.x = x;
		u.i = SQRT_MAGIC_F - (u.i >> 1);  // gives initial guess y0
		return x*u.x*(1.5f - xhalf*u.x*u.x);// Newton step, repeating increases accuracy
	}

	inline f32 sqrt3(const f32 x)
	{
		union
		{
			i32 i;
			f32 x;
		} u;

		u.x = x;
		u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
		return u.x;
	}

	inline f32 sqrt4(const f32 m)
	{
		i32 i = 0;
		while ((i*i) <= m)
		{
			i++;
		}

		i--;
		f32 d = m - i*i;
		f32 p = d / (2 * i);
		f32 a = i + p;
		return a - (p*p) / (2 * a);
	}

	inline f32 sqrt5(const f32 m)
	{
		f32 i = 0;
		f32 x1, x2;
		while ((i*i) <= m)
		{
			i += 0.1f;
		}

		x1 = i;
		for (i32 j = 0; j < 10; j++)
		{
			x2 = m;
			x2 /= x1;
			x2 += x1;
			x2 /= 2;
			x1 = x2;
		}
		return x2;
	}

	//http://www.azillionmonkeys.com/qed/sqroot.html#calcmeth
	inline f64 sqrt6(f64 y)
	{
		f64 x, z, tempf;
		unsigned long *tfptr = ((unsigned long *)&tempf) + 1;

		tempf = y;
		*tfptr = (0xbfcdd90a - *tfptr) >> 1;
		x = tempf;
		z = y*0.5;
		x = (1.5*x) - (x*x)*(x*z);
		x = (1.5*x) - (x*x)*(x*z);

		return x*y;
	}

	//http://bits.stephan-brumme.com/squareRoot.html
	inline f32 sqrt7(f32 x)
	{
		u32 i = *(u32*)&x;

		//adjust bias
		i += 127 << 23;

		//approximation of square root
		i >>= 1;

		return *(f32*)&i;
	}

	//http://forums.techarena.in/software-development/1290144.htm
	inline f64 sqrt8(const f64 x)
	{
		f64 n = x / 2.0;
		f64 lstX = 0.0;

		while (n != lstX)
		{
			lstX = n;
			n = (n + x / n) / 2.0;
		}
		return n;
	}


	//http://www.functionx.com/cpp/examples/squareroot.htm
	inline f64 abs_for_sqrt9(f64 Nbr)
	{
		if (Nbr >= 0)
		{
			return Nbr;
		}
		else
		{
			return -Nbr;
		}
	}

	inline f64 sqrt9(f64 Nbr)
	{
		f64 Number = Nbr / 2;
		const f64 Tolerance = 1.0e-7;
		do
		{
			Number = (Number + Nbr / Number) / 2;
		} while (abs_for_sqrt9(Number * Number - Nbr) > Tolerance);

		return Number;
	}

	//http://www.cs.uni.edu/~jacobson/C++/newton.html
	inline f64 sqrt10(const f64 x)
	{
		const f64 ACCURACY = 0.001;
		f64 lower, upper, guess;

		if (x < 1)
		{
			lower = x;
			upper = 1;
		}
		else
		{
			lower = 1;
			upper = x;
		}

		while ((upper - lower) > ACCURACY)
		{
			guess = (lower + upper) / 2;
			if (guess*guess > x)
			{
				upper = guess;
			}
			else
			{
				lower = guess;
			}
		}
		return (lower + upper) / 2;
	}

	//http://www.drdobbs.com/184409869;jsessionid=AIDFL0EBECDYLQE1GHOSKH4ATMY32JVN
	inline f64 sqrt11(unsigned long x)
	{
		f64 n, p, low, high;
		if (2 > x)
		{
			return x;
		}

		low = 0;
		high = x;
		while (high > low + 1)
		{
			n = (high + low) / 2;
			p = n * n;
			if (x < p)
			{
				high = n;
			}
			else if (x > p)
			{
				low = n;
			}
			else
			{
				break;
			}
		}
		return (x == p ? n : low);
	}

	//http://cjjscript.q8ieng.com/?p=32
	inline f64 sqrt12(i32 n)
	{
		// double a = (eventually the main method will plug values into a)
		f64 a = (f64)n;
		f64 x = 1;

		// For loop to get the square root value of the entered number.
		for (i32 i = 0; i < n; i++)
		{
			x = 0.5 * (x + a / x);
		}

		return x;
	}

	inline f64 sqrt13(f64 n)
	{
		__asm{
			fld n
				fsqrt
		}
	}

	f64 inline __declspec (naked) __fastcall sqrt14(f64 n)
	{
		_asm fld qword ptr[esp + 4]
			_asm fsqrt
		_asm ret 8
	}
}

#endif
#endif