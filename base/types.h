#pragma once

#include "config.h"

#include <map>
#include <list>
#include <memory>
#include <vector>
#include <string>
#include <atomic>

#if defined(POR_WINDOWS)
	using u8		= unsigned __int8;
	using u16		= unsigned __int16;
	using u32		= unsigned __int32;
	using u64		= unsigned __int64;

	using i8		= __int8;
	using i16		= __int16;
	using i32		= __int32;
	using i64		= __int64;

	using f32		= float;
	using f64		= double;

#elif defined(POR_LINUX)
	using u8		= uint8_t;
	using u16		= uint16_t;
	using u32		= uint32_t;
	using u64		= uint64_t;

	using i8		= int8_t;
	using i16		= int16_t;
	using i32		= int32_t;
	using i64		= int64_t;

	using f32		= float;
	using f64		= double;
#else	 
#endif

using b8vector = std::vector<bool>; //note: each value is stored in a single bit. don't call data()
using u8vector = std::vector<u8>;
using u16vector = std::vector<u16>;
using u32vector = std::vector<u32>;
using u64vector = std::vector<u64>;
using i8vector = std::vector<i8>;
using i16vector = std::vector<i16>;
using i32vector = std::vector<i32>;
using i64vector = std::vector<i64>;
using f32vector = std::vector<f32>;
using f64vector = std::vector<f64>;
using strvector	= std::vector<std::string>;
using strwvector = std::vector<std::wstring>;

using i32map = std::map<i32, i32>;
using strmap = std::map<std::string, i32>;

using postring = std::string;
using powstring = std::wstring;

#if defined(POR_SUPPORT_UNICODE)
using potstring = powstring;
#else
using potstring = postring;
#endif
