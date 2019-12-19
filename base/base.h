#pragma once
#include <cmath>
#include "define.h"
#include "struct.h"
#include "logger/logger.h"
#include <assert.h>
#include <codecvt>

#if defined(POR_WINDOWS)
#undef CDECL
#define CDECL __cdecl
#elif defined(POR_LINUX)
#define CDECL
#endif

i32 cmp_azfind_i32(const void* a, const i32 b);
i64 cmp_azfind_i64(const void* a, const i64 b);
i32 cmp_zafind_i32(const void* a, const i32 b);
i64 cmp_zafind_i64(const void* a, const i64 b);

i32 cmp_azsort_i32(const void* a, const void* b);
i32 cmp_azsort_i64(const void* a, const void* b);
i32 cmp_zasort_i32(const void* a, const void* b);
i32 cmp_zasort_i64(const void* a, const void* b);

class CPOBase
{
public:
	CPOBase();
	virtual ~CPOBase();

public:
	static i32			random();
	static f32			randomUnit();
	static f32			randomPlus();
	static void			randomInit();
	static void			getNowTime(DateTime& dtm);
	static postring		getDateTimeFileName(i32 id = -1);

	static bool			isNumber(const postring& str);
	static bool			isString(const postring& str);
	static bool			removeQuote(postring& str_expression);
	static bool			removeParenthesis(postring& str_expression);
	static void			spiltToVector(const postring& path, const postring& delimiter, strvector& str_vec);
	static void			splitToPath(const postring& path, const postring& delimiter, postring& split_path, postring& split_name);
	static void			toLowerExpression(postring& str_expression);
	static postring		toString(f32 value, i32 decimal = -1);
	static postring		toString(f64 value, i32 decimal = -1);
	static powstring	toWString(f32 value, i32 decimal = -1);
	static powstring	toWString(f64 value, i32 decimal = -1);
	static postring		getTermiator(i32 terminator);

	static i32			convertIPAddress(const postring& ip_string);
	static void			convertIPAddress(const i32 ip, postring& ip_string);
	
	static bool			stoi(const postring& str, i32& tmp);
	static bool			stoi(const powstring& str, i32& tmp);
	static bool			stoll(const postring& str, i64& tmp);
	static bool			stoll(const powstring& str, i64& tmp);
	static bool			stoul(const postring& str, u32& tmp);
	static bool			stoul(const powstring& str, u32& tmp);
	static bool			stoull(const postring& str, u64& tmp);
	static bool			stoull(const powstring& str, u64& tmp);
	static bool			stof(const postring& str, f32& tmp);
	static bool			stof(const powstring& str, f32& tmp);
	static bool			stod(const postring& str, f64& tmp);
	static bool			stod(const powstring& str, f64& tmp);

	
	//////////////////////////////////////////////////////////////////////////
	// inline string functions
	static inline bool reportOverWrite()
	{
        printlog_lv0("OverWrite Detection...");
		assert(false);
		return false;
	}
	static inline bool reportOverRead()
	{
        printlog_lv0("OverRead Detection...");
		assert(false);
        return false;
	}

	static inline potstring stringToTString(const postring& s)
	{
#if defined(POR_SUPPORT_UNICODE)
		return stringToWString(s);
#else
		return s;
#endif
	}
	static inline postring tstringToString(const potstring& s)
	{
#if defined(POR_SUPPORT_UNICODE)
		return wstringToString(s);
#else
		return s;
#endif
	}
	static inline powstring stringToWString(const postring& s)
	{
#if defined(POR_WINDOWS)
		//setup converter
		typedef std::codecvt_utf8<wchar_t> convert_type;
		std::wstring_convert<convert_type, wchar_t> converter;

		//use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
		return converter.from_bytes(s);
#elif defined(POR_LINUX)
        return std::wstring(s.begin(), s.end());
#endif
	}
	static inline powstring wstringFromBuffer(const char* buffer_ptr)
	{
#if defined(POR_WINDOWS)
		//setup converter
		typedef std::codecvt_utf8<wchar_t> convert_type;
		std::wstring_convert<convert_type, wchar_t> converter;

		//use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
		return converter.from_bytes(buffer_ptr);
#elif defined(POR_LINUX)
        return stringToWString(buffer_ptr);
#endif
	}
	static inline postring wstringToString(const powstring& s)
	{
#if defined(POR_WINDOWS)
		//setup converter
		typedef std::codecvt_utf8<wchar_t> convert_type;
		std::wstring_convert<convert_type, wchar_t> converter;

		//use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
		return converter.to_bytes(s);
#elif defined(POR_LINUX)
        const unsigned wlen = s.length();
        char buf[wlen * sizeof(std::wstring::value_type) + 1];
        const ssize_t res = std::wcstombs(buf, s.c_str(), sizeof(buf));
        return (res >= 0) ? postring(buf) : "?";
#endif
	}
	static inline postring trimLeft(const postring& str, char find_char)
	{
		size_t first = str.find_first_not_of(find_char);
        if (first == postring::npos)
		{
			return str;
		}
		return str.substr(first);
	}
	static inline postring trimRight(const postring& str, char find_char)
	{
		size_t last = str.find_last_not_of(find_char);
		return str.substr(0, last + 1);
	}
	static inline postring trim(const postring& str, char find_char = ' ')
	{
		size_t first = str.find_first_not_of(find_char);
		if (postring::npos == first)
		{
			return str;
		}
		size_t last = str.find_last_not_of(find_char);
		return str.substr(first, (last - first + 1));
	}
	static inline postring cutFindLast(const postring& str, const postring& find_str)
	{
		size_t last = str.find_last_of(find_str);
		if (last == postring::npos)
		{
			return str;
		}
		return str.substr(last + find_str.size());
	}
	static inline char nextChar(postring& str, i32& index)
	{
		if ((index + 1) >= str.size())
		{
			return 0;
		}
		return str[++index];
	}
	static inline void toLower(postring& str)
	{
		std::transform(str.begin(), str.end(), str.begin(), ::tolower);
	}
	static inline postring toLower(const postring& str)
	{
		postring val = str;
		std::transform(val.begin(), val.end(), val.begin(), ::tolower);
		return val;
	}
	static inline void toUpper(postring& str)
	{
		std::transform(str.begin(), str.end(), str.begin(), ::toupper);
	}
	
	//////////////////////////////////////////////////////////////////////////
	// inline static functions
    static inline i32 round(i32 value, i32 base)
    {
        return (i32)((value / base) + ((value % base)? 1 : 0))*base;
    }
	static inline i32 fastPow10(i32 fmt_number)
	{
		if (fmt_number > 10)
		{
			return -1;
		}

		static i32 pow10[10] = { 1, 10, 100, 1000, 10000,
								100000, 1000000, 10000000, 100000000, 1000000000};
		return pow10[fmt_number];
	}
	static inline i64 factorial(i32 val)
	{
		i64 result = 1;
		for (i32 i = 2; i <= val; i++)
		{
			result *= i;
		}
		return result;
	}
	static inline bool isAscii(char ch)
	{
		return (ch >= 0x20);
	}
	static inline bool isNonAscii(char ch)
	{
		return (ch < 0x20);
	}
	static inline bool checkIndex(i32 val, i32 max_val)
	{
		return (val >= 0 && val < max_val);
	}
	static inline bool checkIndex(i32 val, i32 min_val, i32 max_val)
	{
		return (val >= min_val && val < max_val);
	}
	static inline bool checkCount(i32 val, i32 max_val)
	{
		return (val > 0 && val <= max_val);
	}
	static inline bool checkCount(i32 val, i32 min_val, i32 max_val)
	{
		return (val > min_val && val <= max_val);
	}
	static inline bool checkRange(i32 val, i32 max_val)
	{
		return (val >= 0 && val <= max_val);
	}
	static inline bool checkRange(f32 val, f32 max_val)
	{
		return (val >= 0 && val <= max_val);
	}
	static inline bool checkRange(i32 val, i32 min_val, i32 max_val)
	{
		return (val >= min_val && val <= max_val);
	}
	static inline bool checkRange(f32 val, f32 min_val, f32 max_val)
	{
		return (val >= min_val && val <= max_val);
	}
	static inline bool maskBitCheck(i32 val, i32 index)
	{
		i32 tmp = (1 << index);
		return (val & tmp) == tmp;
	}
	static inline void maskBitAdd(i32& val, i32 index)
	{
		val |= (1 << index);
	}
	static inline void maskBitRemove(i32& val, i32 index)
	{
		val &= ~(1 << index);
	}
	static inline bool bitAvailable(i32 val, i32 chk)
	{
		return (val & chk) != 0;
	}
	static inline bool bitCheck(i32 val, i32 chk)
	{
		return (val & chk) == chk;
	}
	static inline u8 bitAdd(u8& val, i32 add)
	{
		val |= add;
		return val;
	}
	static inline u8 bitRemove(u8& val, i32 del)
	{
		val &= ~del;
		return val;
	}
	static inline u16 bitAdd(u16& val, i32 add)
	{
		val |= add;
		return val;
	}
	static inline u16 bitRemove(u16& val, i32 del)
	{
		val &= ~del;
		return val;
	}
	static inline u32 bitAdd(u32& val, i32 add)
	{
		val |= add;
		return val;
	}
	static inline u32 bitRemove(u32& val, i32 del)
	{
		val &= ~del;
		return val;
	}
	static inline i32 bitAdd(i32& val, i32 add)
	{
		val |= add;
		return val;
	}
	static inline i32 bitRemove(i32& val, i32 del)
	{
		val &= ~del;
		return val;
	}
	static inline i32 int_cast(u8* buffer_ptr)
	{
		return *((i32*)buffer_ptr);
	}
	static inline void fileCurSeek(FILE* fp, uint fpos)
	{
		if (!fp)
		{
			return;
		}
		fseek(fp, fpos, SEEK_CUR);
	}
	static inline void fileSeek(FILE* fp, uint fpos)
	{
		if (!fp)
		{
			return;
		}
		fseek(fp, fpos, SEEK_SET);
	}
	static inline u64 filePos(FILE* fp)
	{
		return ftell(fp);
	}
	static inline u64 fileSize(FILE* fp)
	{
		if (!fp)
		{
			return 0;
		}

		fseek(fp, 0, SEEK_END);
		u64 len = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		return len;
	}
	static inline bool fileRead(FILE* fp, postring& str, i32 maxlen = PO_MAXSTRING)
	{
		if (!fp)
		{
			str = "";
			return false;
		}

		i32 len = 0;
		fread(&len, sizeof(i32), 1, fp);

		if (len >= 0 && len < maxlen)
		{
			str.resize(len, 0);
			if (len > 0)
			{
				void* buffer_ptr = (void*)str.data();
				fread(buffer_ptr, len, 1, fp);
			}
			return true;
		}
		return false;
	}
	static inline bool fileRead(FILE* fp, powstring& str, i32 maxlen = PO_MAXSTRING)
	{
		if (!fp)
		{
			str = L"";
			return false;
		}

		i32 len, size;
		fread(&len, sizeof(i32), 1, fp);
		
		str.clear();
		if (len > 0 && len < maxlen)
		{
			str.resize(len);
			u8* buffer_ptr = (u8*)str.data();
			i32 strsize = len*sizeof(wchar_t);

			fread(&size, sizeof(i32), 1, fp);
			if (size == strsize)
			{
				fread(buffer_ptr, size, 1, fp);
			}
			else if (size > 0)
			{
				memset(buffer_ptr, 0, strsize);

				i32 stp1 = size / len;
				i32 stp2 = sizeof(wchar_t);
				i32 stpmin = po::_min(stp1, stp2);
				for (i32 i = 0; i < len; i++)
				{
					fread(buffer_ptr, stpmin, 1, fp);
					buffer_ptr += stp2;
				}
			}
		}
		return true;
	}
	static inline bool fileSignRead(FILE* fp, i32 sign_code)
	{
		i32 code = 0;
		fread(&code, sizeof(i32), 1, fp);
		return (code == sign_code);
	}
	static inline void fileWrite(FILE* fp, postring& str)
	{
		if (!fp)
		{
			return;
		}

		i32 len = (i32)str.size();
		fwrite(&len, sizeof(i32), 1, fp);

		if (len > 0 && len < PO_MAXINT)
		{
			fwrite(str.data(), len, 1, fp);
		}
	}
	static inline void fileWrite(FILE* fp, powstring& str)
	{
		if (!fp)
		{
			return;
		}

		i32 len = (i32)str.size();
		fwrite(&len, sizeof(i32), 1, fp);

		if (len > 0 && len < PO_MAXINT)
		{
			i32 size = len*sizeof(wchar_t);
			fwrite(&size, sizeof(i32), 1, fp);
			fwrite(str.data(), size, 1, fp);
		}
	}
	static inline void fileSignWrite(FILE* fp, i32 sign_code)
	{
		fwrite(&sign_code, sizeof(i32), 1, fp);
	}
	static inline void fileWriteImage(u8* img_ptr, i32 w, i32 h, FILE* fp)
	{
		if (!fp)
		{
			return;
		}

		CPOBase::fileWrite(w, fp);
		CPOBase::fileWrite(h, fp);
		if (!img_ptr || w*h <= 0)
		{
			return;
		}
		CPOBase::fileWrite(img_ptr, w*h, fp);
	}
	static inline void fileWriteImage(u8* img_ptr, i32 w, i32 h, i32 channel, FILE* fp)
	{
		if (!fp)
		{
			return;
		}

		CPOBase::fileWrite(w, fp);
		CPOBase::fileWrite(h, fp);
		CPOBase::fileWrite(channel, fp);
		if (!img_ptr || w*h*channel <= 0)
		{
			return;
		}
		CPOBase::fileWrite(img_ptr, w*h*channel, fp);
	}
	static inline bool fileReadImage(u8*& img_ptr, i32& w, i32& h, FILE* fp)
	{
		if (!fp)
		{
			return false;
		}

		w = h = 0;
		POSAFE_DELETE_ARRAY(img_ptr);

		CPOBase::fileRead(w, fp);
		CPOBase::fileRead(h, fp);
		if (w*h <= 0)
		{
			return false;
		}

		img_ptr = po_new u8[w*h];
		CPOBase::fileRead(img_ptr, w*h, fp);
		return true;
	}
	static inline bool fileReadImage(u8*& img_ptr, i32& w, i32& h, i32& channel, FILE* fp)
	{
		if (!fp)
		{
			return false;
		}

		w = h = channel = 0;
		POSAFE_DELETE_ARRAY(img_ptr);

		CPOBase::fileRead(w, fp);
		CPOBase::fileRead(h, fp);
		CPOBase::fileRead(channel, fp);
		if (w*h*channel <= 0)
		{
			return false;
		}

		img_ptr = po_new u8[w*h*channel];
		CPOBase::fileRead(img_ptr, w*h*channel, fp);
		return true;
	}

	static inline i32 getImageSize(i32 w, i32 h)
	{
		i32 len = 0;
		len += 2 * sizeof(i32);
		len += (w > 0 && h > 0) ? w*h : 0;
		return len;
	}
	static inline i32 getImageSize(i32 w, i32 h, i32 channel)
	{
		i32 len = 0;
		len += 3 * sizeof(i32);
		len += (w > 0 && h > 0 && channel > 0) ? w*h*channel : 0;
		return len;
	}
	static inline i32 getStringMemSize(const postring& str)
	{
		return sizeof(i32) + (i32)str.size();
	}
	static inline i32 getStringMemSize(const powstring& str)
	{
		return 2 * sizeof(i32) + (i32)str.size()*sizeof(wchar_t);
	}
	static inline i32 getStrVectorMemSize(strvector& str_vec)
	{
		i32 len = sizeof(i32);
		i32 i, count = (i32)str_vec.size();
		for (i = 0; i < count; i++)
		{
			postring& str = str_vec[i];
			len += getStringMemSize(str);
		}
		return len;
	}
	
	static inline void fileWriteStrVector(strvector& str_vec, FILE* fp)
	{
		if (!fp)
		{
			return;
		}

		i32 i, count = (i32)str_vec.size();
		fileWrite(count, fp);
		for (i = 0; i < count; i++)
		{
			postring& str = str_vec[i];
			fileWrite(fp, str);
		}
	}
	static inline bool fileReadStrVector(strvector& str_vec, FILE* fp)
	{
		if (!fp)
		{
			return false;
		}

		i32 i, count;
		postring str;
		str_vec.clear();
		fileRead(count, fp);
		if (!CPOBase::isCount(count))
		{
			return false;
		}

		for (i = 0; i < count; i++)
		{
			fileRead(fp, str);
			str_vec.push_back(str);
		}
		return true;
	}
	static u16 makeCRCCode(u8* buffer_ptr, u32 len)
	{
		u16 crc_code = 0xFFFF;
		u16 crc_const_param = 0xA001;
		u16 data_word;
		u32 byte_index;
		i32 loop = 8;

		if (!buffer_ptr || len <= 0)
		{
			return crc_code;
		}

		for (byte_index = 0; byte_index < len; byte_index++)
		{
			data_word = buffer_ptr[byte_index];
			crc_code ^= data_word;
			loop = 8;
			while (loop-- != 0)
			{
				if ((crc_code & 0x01) == 0)
				{
					crc_code >>= 1;
				}
				else
				{
					crc_code >>= 1;
					crc_code ^= crc_const_param;
				}
			}
		}
		return crc_code;
	}

	//////////////////////////////////////////////////////////////////////////
	// template inline static functions
	template <typename T>
	static inline bool isPositive(T val)
	{
		return (val > 0 && val < PO_MAXINT);
	}
	template <typename T>
	static inline bool isCount(T val)
	{
		return (val >= 0 && val < PO_MAXINT);
	}
	template <typename T>
	static inline i32 int_cast(T val)
	{
		return (i32)(val + 0.5f);
	}

	template <typename T>
	static inline bool isEvenNumber(T val)
	{
		return ((i32)val % 2) == 0;
	}

	template <typename T>
	static inline bool isOddNumber(T val)
	{
		return ((i32)val % 2) == 1;
	}

	template <typename T>
	static inline T interpol(T val1, T val2, f32 interpolation)
	{
		return val1*interpolation + val2*(1.0f - interpolation);
	}

	template <typename T>
	static inline bool updateRange(T& val, T min_val, T max_val)
	{
		bool is_valid = true;
		if (val < min_val || val > max_val)
		{
			is_valid = false;
		}
		val = po::_max(min_val, po::_min(val, max_val));
		return is_valid;
	}

	template <typename T>
	static inline bool getRangeOverlap(T st1, T ed1, T st2, T ed2)
	{
		return po::_max(po::_min(ed1, ed2) - po::_max(st1, st2), 0);
	}

	template <typename T>
	static inline bool checkRangeOverlap(T st1, T ed1, T st2, T ed2)
	{
		return (po::_min(ed1, ed2) - po::_max(st1, st2)) <= 0;
	}

	template <typename T>
	static inline void init2x2(T* tr, f32 angle = 0, f32 scale = 1.0f)
	{
		f32 cosan = cosf(angle);
		f32 sinan = sinf(angle);

		tr[0] = (T)(cosan*scale);
		tr[1] = (T)(-sinan);
		tr[2] = (T)(sinan);
		tr[3] = (T)(cosan*scale);
	}

	template <typename T>
	static inline void init2x3(T* tr, f32 angle = 0, f32 scale = 1.0f, f32 tx = 0, f32 ty = 0)
	{
		f32 cosan = cosf(angle);
		f32 sinan = sinf(angle);

		tr[0] = (T)(cosan*scale);
		tr[1] = (T)(-sinan);
		tr[2] = (T)tx;
		tr[3] = (T)(sinan);
		tr[4] = (T)(cosan*scale);
		tr[5] = (T)ty;
	}

	template <typename T>
	static inline void init2x3(T* tr, f32 angle, f32 sx, f32 sy, f32 tx, f32 ty)
	{
		f32 cosan = cosf(angle);
		f32 sinan = sinf(angle);

		tr[0] = (T)(cosan*sx);
		tr[1] = (T)(-sinan);
		tr[2] = (T)tx;
		tr[3] = (T)(sinan);
		tr[4] = (T)(cosan*sy);
		tr[5] = (T)ty;
	}

	template <typename T>
	static inline void init3x3(T* tr, f32 angle = 0, f32 scale = 1.0f, f32 tx = 0, f32 ty = 0)
	{
		init2x3(tr, angle, scale, tx, ty);
		tr[6] = (T)0;
		tr[7] = (T)0;
		tr[8] = (T)1;
	}

	template <typename T>
	static inline void init3x3(T* tr, f32 angle, f32 sx, f32 sy, f32 tx, f32 ty)
	{
		init2x3(tr, angle, sx, sy, tx, ty);
		tr[6] = (T)0;
		tr[7] = (T)0;
		tr[8] = (T)1;
	}

	template <typename T>
	static inline bool isIdentifyMat3x3(T* mat)
	{
		return (std::abs(mat[0] - 1.0f) < PO_EPSILON &&
			std::abs(mat[1] - 0.0f) < PO_EPSILON &&
			std::abs(mat[2] - 0.0f) < PO_EPSILON &&
			std::abs(mat[3] - 0.0f) < PO_EPSILON &&
			std::abs(mat[4] - 1.0f) < PO_EPSILON &&
			std::abs(mat[5] - 0.0f) < PO_EPSILON &&
			std::abs(mat[6] - 0.0f) < PO_EPSILON &&
			std::abs(mat[7] - 0.0f) < PO_EPSILON &&
			std::abs(mat[8] - 1.0f) < PO_EPSILON);
	}

	template <typename T, typename U>
	static inline void transpose(T* dst_mat, U* src_mat, i32 w, i32 h)
	{
		if (!dst_mat || !src_mat || w*h <= 0)
		{
			return;
		}

		i32 x, y, src_index, dst_index;
		for (y = 0; y < h; y++)
		{
			src_index = y*w;
			dst_index = y;
			for (x = 0; x < w; x++)
			{
				dst_mat[dst_index] = (T)src_mat[src_index];
				dst_index += h;
				src_index++;
			}
		}
	}

	template <typename T>
	static inline i32 qFind(void* void_data_ptr, i32 count, i32 size, T reference, T(CDECL *comp)(const void *, const T))
	{
		T ret;
		i32 i, n1, n2, n3;
		u8* u8_data_ptr = (u8*)void_data_ptr;

		if (count <= 0)
		{
			return -1;
		}
		else if (count < 4)
		{
			for (i = 0; i < count; i++)
			{
				if (comp(u8_data_ptr + size*i, reference) == 0)
				{
					return i;
				}
			}
			return -1;
		}

		n1 = 0;
		n2 = count - 1;
		while (n2 >= n1)
		{
			n3 = (n1 + n2) / 2;
			ret = comp(u8_data_ptr + size*n3, reference);
			if (ret == 0)
			{
				return n3;
			}
			if (ret > 0)
			{
				n1 = n3 + 1;
			}
			else
			{
				n2 = n3 - 1;
			}
		}
		return -1;
	}

	template <typename T>
	static inline T transDet2x3(T* tr)
	{
		return tr[0] * tr[4] - tr[1] * tr[3];
	}

	template <typename T, typename U>
	static inline vector2d<T> trans2x2(T* tr, U x, U y)
	{
		vector2d<T> pt;
		pt.x = tr[0] * x + tr[1] * y;
		pt.y = tr[2] * x + tr[3] * y;
		return pt;
	}

	template <typename T, typename U>
	static inline void trans2x2(T* tr, U x, U y, U& tx, U& ty)
	{
		tx = tr[0] * x + tr[1] * y;
		ty = tr[2] * x + tr[3] * y;
	}

	template <typename T, typename U>
	static inline void trans2x3(T* tr, U& x, U& y)
	{
		U px = x;
		U py = y;
		x = tr[0] * px + tr[1] * py + tr[2];
		y = tr[3] * px + tr[4] * py + tr[5];
	}

	template <typename T, typename U>
	static inline void trans2x3(T* tr, U x, U y, U& tx, U& ty)
	{
		tx = tr[0] * x + tr[1] * y + tr[2];
		ty = tr[3] * x + tr[4] * y + tr[5];
	}

	template <typename T, typename U>
	static inline void trans2x3(T* tr, vector2d<U>& pt)
	{
		U tx = tr[0] * pt.x + tr[1] * pt.y + tr[2];
		U ty = tr[3] * pt.x + tr[4] * pt.y + tr[5];
		pt.x = tx;
		pt.y = ty;
	}

	template <typename T, typename U>
	static inline void rotate2x3(T* tr, U x, U y, U& tx, U& ty)
	{
		tx = tr[0] * x + tr[1] * y;
		ty = tr[3] * x + tr[4] * y;
	}

	template <typename T, typename U>
    static inline vector2d<U> rotate2x3(T* tr, const vector2d<U>& pt)
	{
		vector2d<U> pt1;
		pt1.x = tr[0] * pt.x + tr[1] * pt.y;
		pt1.y = tr[3] * pt.x + tr[4] * pt.y;
		return pt1;
	}

	template <typename T, typename U>
	static inline void trans3x3(T* tr, U& x, U& y)
	{
		U x1, y1, z1;
		x1 = tr[0] * x + tr[1] * y + tr[2];
		y1 = tr[3] * x + tr[4] * y + tr[5];
		z1 = tr[6] * x + tr[7] * y + tr[8];
		x = x1 / z1; y = y1 / z1;
	}

	template <typename T, typename U>
	static inline void trans3x3(T* tr, U x, U y, U& tx, U& ty)
	{
		U x1, y1, z1;
		x1 = tr[0] * x + tr[1] * y + tr[2];
		y1 = tr[3] * x + tr[4] * y + tr[5];
		z1 = tr[6] * x + tr[7] * y + tr[8];
		tx = x1 / z1; ty = y1 / z1;
	}

	template <typename T, typename U >
	static inline void perspective2d(U* tr, T x, T y, T& px, T& py)
	{
		px = tr[0] * x + tr[1] * y + tr[2];
		py = tr[3] * x + tr[4] * y + tr[5];
		T pz = tr[6] * x + tr[7] * y + tr[8];
		if (pz != 0)
		{
			px /= pz; py /= pz;
		}
	}

	template <typename T, typename U>
	static inline Rectf getTrBoundingRect(T* tr_ptr, TRect<U>& rect)
	{
		U px[4], py[4];
		CPOBase::trans2x3(tr_ptr, rect.x1, rect.y1, px[0], py[0]);
		CPOBase::trans2x3(tr_ptr, rect.x1, rect.y2, px[1], py[1]);
		CPOBase::trans2x3(tr_ptr, rect.x2, rect.y1, px[2], py[2]);
		CPOBase::trans2x3(tr_ptr, rect.x2, rect.y2, px[3], py[3]);
		
		Rectf rt(PO_MAXINT, PO_MAXINT, 0, 0);
		for (i32 i = 0; i < 4; i++)
		{
			rt.x1 = po::_min(rt.x1, px[i]);
			rt.y1 = po::_min(rt.y1, py[i]);
			rt.x2 = po::_max(rt.x2, px[i]);
			rt.y2 = po::_max(rt.y2, py[i]);
		}
		return rt;
	}

	template <typename T>
	static inline void fileRead(T& x, FILE* fp)
	{
		if (!fp)
		{
			return;
		}
		fread(&x, sizeof(T), 1, fp);
	}

	template <typename T>
	static inline void fileRead(T* x, FILE* fp)
	{
		if (!fp)
		{
			return;
		}
		fread(x, sizeof(T), 1, fp);
	}

	template <typename T>
	static inline void fileRead(T* x, i32 count, FILE* fp)
	{
		if (!fp)
		{
			return;
		}
		fread(x, sizeof(T)*count, 1, fp);
	}

	template <class T>
	static inline void filePackRead(T*& x, i32 count, FILE* fp)
	{
		i32 len;
		if (count <= 0)
		{
			return;
		}
		POSAFE_DELETE_ARRAY(x);

		fread(&len, sizeof(len), 1, fp);
		if (len == count)
		{
			x = po_new T[count];
			fread(x, sizeof(T)*count, 1, fp);
		}
	}

	template <typename T>
	static inline bool fileReadVector(std::vector<T>& vec, FILE* fp)
	{
		if (!fp)
		{
			return false;
		}

		i32 count = 0;
		vec.clear();
		fileRead(count, fp);
		if (!CPOBase::isCount(count))
		{
			return false;
		}
		vec.resize(count);
		fileRead((T*)vec.data(), count, fp);
		return true;
	}

	template <typename T>
	static inline void fileWrite(const T& x, FILE* fp)
	{
		if (!fp)
		{
			return;
		}
		fwrite(&x, sizeof(T), 1, fp);
	}

	template <typename T>
	static inline void fileWrite(T* x, FILE* fp)
	{
		if (!fp)
		{
			return;
		}
		fwrite(x, sizeof(T), 1, fp);
	}

	template <typename T>
	static inline void fileWrite(T* x, i32 count, FILE* fp)
	{
		if (!fp)
		{
			return;
		}
		fwrite(x, sizeof(T)*count, 1, fp);
	}

	template <class T>
	static inline void filePackWrite(T* x, i32 count, FILE* fp)
	{
		if (!fp)
		{
			return;
		}
		if (!x || count <= 0)
		{
			count = 0;
			fwrite(&count, sizeof(count), 1, fp);
			return;
		}
		fwrite(&count, sizeof(count), 1, fp);
		fwrite(x, sizeof(T)*count, 1, fp);
	}

	template <typename T>
	static inline void fileWriteVector(const std::vector<T>& vec, FILE* fp)
	{
		if (!fp)
		{
			return;
		}

		i32 count = (i32)vec.size();
		T* data_ptr = (T*)vec.data();
		fileWrite(count, fp);
		if (CPOBase::isCount(count))
		{
			fileWrite(data_ptr, count, fp);
		}
	}

	static inline bool memRead(u8*& buffer_ptr, i32& buffer_size, postring& str, i32 maxlen = PO_MAXSTRING)
	{
		if (!buffer_ptr || buffer_size < sizeof(i32))
		{
			str = "";
			return false;
		}

		i32 len = 0;
		if (!CPOBase::memRead(len, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!CPOBase::checkRange(len, maxlen) || buffer_size < len)
		{
			return CPOBase::reportOverRead();
		}

		str.resize(len);
		u8* tmp_buffer_ptr = (u8*)str.data();
		memcpy(tmp_buffer_ptr, buffer_ptr, len);
		buffer_ptr += len;
		buffer_size -= len;
		return true;
	}

	static inline bool memRead(u8*& buffer_ptr, i32& buffer_size, powstring& str, i32 maxlen = PO_MAXSTRING)
	{
		if (!buffer_ptr || buffer_size < sizeof(i32) * 2)
		{
			str = L"";
			return false;
		}

		i32 len = 0, size = 0;
		if (!CPOBase::memRead(len, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!CPOBase::memRead(size, buffer_ptr, buffer_size))
		{
			return false;
		}

		if (!CPOBase::checkRange(len, maxlen) || buffer_size < size)
		{
			return CPOBase::reportOverRead();
		}

		str.resize(len);
		i32 str_size = len*sizeof(wchar_t);
		u8* str_buffer_ptr = (u8*)str.data();

		if (size == str_size)
		{
			memcpy(str_buffer_ptr, buffer_ptr, size);
			buffer_ptr += size;
			buffer_size -= size;
		}
		else if (size > 0 && len > 0)
		{
			memset(str_buffer_ptr, 0, str_size);

			i32 stp1 = size / len;
			i32 stp2 = sizeof(wchar_t);
			i32 stp_min = po::_min(stp1, stp2);
			for (i32 i = 0; i < len; i++)
			{
				memcpy(str_buffer_ptr, buffer_ptr, stp_min);
				str_buffer_ptr += stp2;
				buffer_ptr += stp1;
				buffer_size -= stp1;
			}
		}
		return true;
	}
	
	static inline bool memReadImage(u8*& img_ptr, i32& w, i32& h, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		w = h = 0;
		POSAFE_DELETE_ARRAY(img_ptr);
		if (!CPOBase::memRead(w, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!CPOBase::memRead(h, buffer_ptr, buffer_size))
		{
			return false;
		}

		if (w <= 0 || h <= 0 || buffer_size < w*h)
		{
			return false;
		}

		img_ptr = po_new u8[w*h];
		return memRead(img_ptr, w*h, buffer_ptr, buffer_size);
	}

	static inline bool memReadImage(u8*& img_ptr, i32& w, i32& h, i32& channel, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		w = h = 0;
		POSAFE_DELETE_ARRAY(img_ptr);
		if (!CPOBase::memRead(w, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!CPOBase::memRead(h, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!CPOBase::memRead(channel, buffer_ptr, buffer_size))
		{
			return false;
		}

		if (w <= 0 || h <= 0 || channel <=0 || buffer_size < w*h*channel)
		{
			return false;
		}

		img_ptr = po_new u8[w*h*channel];
		return memRead(img_ptr, w*h*channel, buffer_ptr, buffer_size);
	}

	static inline bool memRead(u8*& img_ptr, i32& w, i32& h, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr || buffer_size <= 0)
		{
			return false;
		}

		w = 0;
		h = 0;
		POSAFE_DELETE_ARRAY(img_ptr);

		if (!CPOBase::memRead(w, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!CPOBase::memRead(h, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (w <= 0 || h <= 0 || buffer_size < w*h)
		{
			return false;
		}

		img_ptr = po_new u8[w*h];
		return memRead(img_ptr, w*h, buffer_ptr, buffer_size);
	}

	template <typename T>
	static inline bool memRead(T& x, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		i32 type_size = sizeof(T);
		if (buffer_size < type_size)
		{
			return CPOBase::reportOverRead();
		}

		memcpy(&x, buffer_ptr, type_size);
		buffer_ptr += type_size;
		buffer_size -= type_size;
		return true;
	}

	template <typename T>
	static inline bool memRead(T* x, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr || !x)
		{
			return false;
		}

		i32 type_size = sizeof(T);
		if (buffer_size < type_size)
		{
			return CPOBase::reportOverRead();
		}

		memcpy(x, buffer_ptr, type_size);
		buffer_ptr += type_size;
		buffer_size -= type_size;
		return true;
	}

	template <typename T>
	static inline bool memRead(T* x, i32 count, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr || !x)
		{
			return false;
		}

		i32 type_size = sizeof(T)*count;
		if (buffer_size < type_size)
		{
			return CPOBase::reportOverRead();
		}
		if (type_size > 0)
		{
			memcpy(x, buffer_ptr, type_size);
			buffer_ptr += type_size;
			buffer_size -= type_size;
		}
		return true;
	}

	template <typename T>
	static inline bool memReadVector(std::vector<T>& vec, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		i32 count = 0;
		vec.clear();
		if (!memRead(count, buffer_ptr, buffer_size))
		{
			return false;
		}

		if (!CPOBase::isCount(count))
		{
			return false;
		}

		vec.resize(count);
		return memRead((T*)vec.data(), count, buffer_ptr, buffer_size);
	}

	static inline bool memReadStrVector(strvector& str_vec, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		i32 i, count;
		postring str;
		str_vec.clear();
		if (!memRead(count, buffer_ptr, buffer_size))
		{
			return false;
		}

		if (!CPOBase::isCount(count))
		{
			return false;
		}

		for (i = 0; i < count; i++)
		{
			if (!memRead(buffer_ptr, buffer_size, str))
			{
				return false;
			}
			str_vec.push_back(str);
		}
		return true;
	}
	static inline bool memSignRead(u8*& buffer_ptr, i32& buffer_size, i32 sign_code)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		i32 code = 0;
		if (!memRead(code, buffer_ptr, buffer_size))
		{
			return false;
		}
		return (code == sign_code);
	}

	static inline void memReadStrVector(strvector& str_vec, u8*& buffer_ptr)
	{
		if (!buffer_ptr)
		{
			return;
		}

		i32 i, count;
		postring str;
		str_vec.clear();
		memRead(count, buffer_ptr);
		if (!CPOBase::isCount(count))
		{
			return;
		}

		for (i = 0; i < count; i++)
		{
			memRead(buffer_ptr, str);
			str_vec.push_back(str);
		}
	}

 	static inline void memRead(u8*& buffer_ptr, postring& str, i32 maxlen = PO_MAXSTRING)
 	{
 		if (!buffer_ptr)
 		{
 			str = "";
 			return;
 		}
 
 		i32 len;
 		memcpy(&len, buffer_ptr, sizeof(i32));
 		buffer_ptr += sizeof(i32);
 
 		str.clear();
 		if (len > 0 && len < maxlen)
 		{
 			str.resize(len);
 			u8* pbuff = (u8*)str.data();
 			memcpy(pbuff, buffer_ptr, len);
 			buffer_ptr += len;
 		}
 	}

 	static inline void memRead(u8*& buffer_ptr, powstring& str, i32 maxlen = PO_MAXSTRING)
 	{
 		if (!buffer_ptr)
 		{
 			str = L"";
 			return;
 		}
 
 		i32 len, size;
 		memcpy(&len, buffer_ptr, sizeof(i32));
 		buffer_ptr += sizeof(i32);
 		memcpy(&size, buffer_ptr, sizeof(i32));
 		buffer_ptr += sizeof(i32);
 
 		str.clear();
 		if (len > 0 && len < maxlen)
 		{
 			str.resize(len);
 			u8* pbuff = (u8*)str.data();
 			i32 strsize = len*sizeof(wchar_t);
 
 			if (size == strsize)
 			{
 				memcpy(pbuff, buffer_ptr, size);
 				buffer_ptr += size;
 			}
 			else if (size > 0)
 			{
 				memset(pbuff, 0, strsize);
 
 				i32 stp1 = size / len;
 				i32 stp2 = sizeof(wchar_t);
 				i32 stpmin = po::_min(stp1, stp2);
 				for (i32 i = 0; i < len; i++)
 				{
 					memcpy(pbuff, buffer_ptr, stpmin);
 					pbuff += stp2;
 					buffer_ptr += stp1;
 				}
 			}
 		}
 	}
 
 	static inline bool memReadImage(u8*& img_ptr, i32& w, i32& h, u8*& buffer_ptr)
 	{
		if (!buffer_ptr)
		{
			return false;
		}

 		w = h = 0;
 		POSAFE_DELETE_ARRAY(img_ptr);
 
 		CPOBase::memRead(w, buffer_ptr);
 		CPOBase::memRead(h, buffer_ptr);
 		if (w*h <= 0)
 		{
 			return false;
 		}
 
 		img_ptr = po_new u8[w*h];
 		CPOBase::memRead(img_ptr, w*h, buffer_ptr);
 		return true;
 	}
 
	static inline bool memRead(u8*& img_ptr, i32& w, i32& h, u8*& buffer_ptr)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		if (!buffer_ptr)
		{
			return false;
		}

		w = 0;
		h = 0;
		POSAFE_DELETE_ARRAY(img_ptr);

		CPOBase::memRead(w, buffer_ptr);
		CPOBase::memRead(h, buffer_ptr);
		if (w <= 0 || h <= 0)
		{
			return false;
		}

		img_ptr = po_new u8[w*h];
		CPOBase::memRead(img_ptr, w*h, buffer_ptr);
		return true;
	}

 	template <typename T>
 	static inline void memRead(T& x, u8*& buffer_ptr)
 	{
		if (!buffer_ptr)
		{
			return;
		}

 		memcpy(&x, buffer_ptr, sizeof(T));
 		buffer_ptr += sizeof(T);
 	}
 
 	template <typename T>
 	static inline void memRead(T* x, u8*& buffer_ptr)
 	{
		if (!buffer_ptr || !x)
		{
			return;
		}

 		memcpy(x, buffer_ptr, sizeof(T));
 		buffer_ptr += sizeof(T);
 	}
 
 	template <typename T>
 	static inline void memRead(T* x, i32 count, u8*& buffer_ptr)
 	{
		if (!buffer_ptr || !x)
		{
			return;
		}

 		memcpy(x, buffer_ptr, sizeof(T)*count);
 		buffer_ptr += sizeof(T)*count;
 	}
 
 	template <typename T>
 	static inline bool memReadVector(std::vector<T>& vec, u8*& buffer_ptr)
 	{
		if (!buffer_ptr)
		{
			return false;
		}

 		i32 count = 0;
 		vec.clear();
 		memRead(count, buffer_ptr);
 		if (!CPOBase::isCount(count))
 		{
 			return false;
 		}
 
 		vec.resize(count);
 		memRead((T*)vec.data(), count, buffer_ptr);
 		return true;
 	}

	template <typename T>
    static inline bool memWrite(const T& x, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		i32 type_size = sizeof(T);
		if (buffer_size < type_size)
		{
			return CPOBase::reportOverWrite();
		}

		memcpy(buffer_ptr, &x, type_size);
		buffer_ptr += type_size;
		buffer_size -= type_size;
		return true;
	}

	template <typename T>
	static inline bool memWrite(T* x, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr || !x)
		{
			return false;
		}

		i32 type_size = sizeof(T);
		if (buffer_size < type_size)
		{
			return CPOBase::reportOverWrite();
		}

		memcpy(buffer_ptr, x, type_size);
		buffer_ptr += type_size;
		buffer_size -= type_size;
		return true;
	}

	template <typename T>
	static inline bool memWrite(T* x, i32 count, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr || !x)
		{
			return false;
		}

		i32 type_size = sizeof(T)*count;
		if (buffer_size < type_size)
		{
			return CPOBase::reportOverWrite();
		}
		if (type_size > 0)
		{
			memcpy(buffer_ptr, x, type_size);
			buffer_ptr += type_size;
			buffer_size -= type_size;
		}
		return true;
	}

	template <typename T>
	static inline bool memWriteVector(const std::vector<T>& vec, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		i32 count = (i32)vec.size();
		T* vec_data_ptr = (T*)vec.data();
		if (!memWrite(count, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!CPOBase::isCount(count))
		{
			return false;
		}
		return memWrite(vec_data_ptr, count, buffer_ptr, buffer_size);
	}

	static inline bool memWrite(u8*& buffer_ptr, i32& buffer_size, const postring& str)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		i32 len = (i32)str.size();
		if (!memWrite(len, buffer_ptr, buffer_size))
		{
			return false;
		}

		if (!CPOBase::checkRange(len, PO_MAXINT))
		{
			return false;
		}
		return memWrite((u8*)str.data(), len, buffer_ptr, buffer_size);
	}

	static inline bool memWrite(u8*& buffer_ptr, i32& buffer_size, const powstring& str)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		i32 len = (i32)str.size();
		i32 size = len*sizeof(wchar_t);
		if (!memWrite(len, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!memWrite(size, buffer_ptr, buffer_size))
		{
			return false;
		}
		return memWrite((u8*)str.data(), size, buffer_ptr, buffer_size);
	}

	static inline bool memWriteImage(u8* img_ptr, i32 w, i32 h, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		if (!CPOBase::memWrite(w, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!CPOBase::memWrite(h, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!img_ptr || w*h <= 0)
		{
			return false;
		}

		return CPOBase::memWrite(img_ptr, w*h, buffer_ptr, buffer_size);
	}

	static inline bool memWriteImage(u8* img_ptr, i32 w, i32 h, i32 channel, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		if (!CPOBase::memWrite(w, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!CPOBase::memWrite(h, buffer_ptr, buffer_size))
		{
			return false;
		}
		if (!CPOBase::memWrite(channel, buffer_ptr, buffer_size))
		{
			return false;
		}

		if (!img_ptr || w*h*channel <= 0)
		{
			return false;
		}
		return CPOBase::memWrite(img_ptr, w*h*channel, buffer_ptr, buffer_size);
	}

	static inline bool memWriteStrVector(strvector& str_vec, u8*& buffer_ptr, i32& buffer_size)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		i32 i, count = (i32)str_vec.size();
		if (!memWrite(count, buffer_ptr, buffer_size))
		{
			return false;
		}

		for (i = 0; i < count; i++)
		{
			postring& str = str_vec[i];
			if (!memWrite(buffer_ptr, buffer_size, str))
			{
				return false;
			}
		}
		return true;
	}
	
	static inline bool memSignWrite(u8*& buffer_ptr, i32& buffer_size, i32 sign_code)
	{
		if (!buffer_ptr)
		{
			return false;
		}

		return memWrite(sign_code, buffer_ptr, buffer_size);
	}

	static inline void memWrite(u8*& buffer_ptr, const postring& str)
	{
		if (!buffer_ptr)
		{
			return;
		}

		i32 len = (i32)str.size();
		memcpy(buffer_ptr, &len, sizeof(i32));
		buffer_ptr += sizeof(i32);

		if (len > 0 && len < PO_MAXINT)
		{
			memcpy(buffer_ptr, str.data(), len);
			buffer_ptr += len;
		}
	}

	static inline void memWrite(u8*& buffer_ptr, const powstring& str)
	{
		if (!buffer_ptr)
		{
			return;
		}

		i32 len = (i32)str.size();
		i32 size = len*sizeof(wchar_t);
		memcpy(buffer_ptr, &len, sizeof(i32));
		buffer_ptr += sizeof(i32);
		memcpy(buffer_ptr, &size, sizeof(i32));
		buffer_ptr += sizeof(i32);

		memcpy(buffer_ptr, (u8*)str.data(), size);
		buffer_ptr += size;
	}

	static inline void memWriteImage(u8* img_ptr, i32 w, i32 h, u8*& buffer_ptr)
	{
		if (!buffer_ptr)
		{
			return;
		}

		CPOBase::memWrite(w, buffer_ptr);
		CPOBase::memWrite(h, buffer_ptr);

		if (!img_ptr || w*h <= 0)
		{
			return;
		}

		CPOBase::memWrite(img_ptr, w*h, buffer_ptr);
	}

	static inline void memWriteStrVector(strvector& str_vec, u8*& buffer_ptr)
	{
		if (!buffer_ptr)
		{
			return;
		}

		i32 i, count = (i32)str_vec.size();
		memWrite(count, buffer_ptr);
		for (i = 0; i < count; i++)
		{
			postring& str = str_vec[i];
			memWrite(buffer_ptr, str);
		}
	}

	template <typename T>
    static inline void memWrite(const T& x, u8*& buffer_ptr)
	{
		if (!buffer_ptr)
		{
			return;
		}

		memcpy(buffer_ptr, &x, sizeof(T));
		buffer_ptr += sizeof(T);
	}

	template <typename T>
	static inline void memWrite(T* x, u8*& buffer_ptr)
	{
		if (!buffer_ptr)
		{
			return;
		}

		memcpy(buffer_ptr, x, sizeof(T));
		buffer_ptr += sizeof(T);
	}

	template <typename T>
	static inline void memWrite(T* x, i32 count, u8*& buffer_ptr)
	{
		if (!buffer_ptr)
		{
			return;
		}

		memcpy(buffer_ptr, x, sizeof(T)*count);
		buffer_ptr += sizeof(T)*count;
	}

	template <typename T>
	static inline void memWriteVector(const std::vector<T>& vec, u8*& buffer_ptr)
	{
		if (!buffer_ptr)
		{
			return;
		}

		i32 count = vec.size();
		T* vec_data_ptr = (T*)vec.data();
		memWrite(count, buffer_ptr);
		if (CPOBase::isCount(count))
		{
			memWrite(vec_data_ptr, count, buffer_ptr);
		}
	}

	template <typename T>
	static inline void memZero(T& data)
	{
		memset(&data, 0, sizeof(T));
	}

	template <typename T>
	static inline void memZero(T* data_ptr)
	{
		memset(data_ptr, 0, sizeof(T));
	}

	template <typename T>
	static inline void memCopy(void* dst, T* src_array, i32 count = 1)
	{
		memcpy(dst, src_array, sizeof(T)*count);
	}

	template <typename T>
	static inline T* getLast(std::vector<T*>& vec)
	{
		return vec[vec.size() - 1];
	}

	template <typename T>
	static inline T* getLast(std::vector<T>& vec)
	{
		return vec.data() + vec.size() - 1;
	}

	template <typename T>
	static inline T* pushBackNew(std::vector<T*>& vec)
	{
		vec.push_back(po_new T);
		return getLast(vec);
	}

	template <typename T>
	static inline T* pushBackNew(std::vector<T>& vec)
	{
		vec.push_back(T());
		return getLast(vec);
	}

	template <typename T>
	static inline T* pushBackNew(std::vector<T>& vec, i32& index)
	{
		vec.push_back(T());
		index = vec.size() - 1;
		return getLast(vec);
	}

	template <class T>
	static inline void pushFrontNew(std::vector<T>& vec, T& value)
	{
		vec.insert(vec.begin(), value);
	}

	template <typename T>
	static inline void eraseInVector(std::vector<T*>& vec, i32 index)
	{
		vec.erase(vec.begin() + index);
	}

	template <typename T>
	static inline void eraseInVector(std::vector<T*>& vec, i32 spos, i32 epos)
	{
		vec.erase(vec.begin() + spos, vec.begin() + epos);
	}

	template <typename T>
	static inline void eraseInVector(std::vector<T>& vec, i32 index)
	{
		vec.erase(vec.begin() + index);
	}

	template <typename T>
	static inline void eraseInVector(std::vector<T>& vec, i32 spos, i32 epos)
	{
		vec.erase(vec.begin() + spos, vec.begin() + epos);
	}

	template <typename T>
	static inline void normalize(T& x, T& y)
	{
		T dis = sqrt(x*x + y*y);
		if (dis > 0)
		{
			x /= dis;
			y /= dis;
			return;
		}
		x = y = 0;
	}

	template <typename T>
	static inline void normalize(T& x, T& y, T& dd)
	{
		dd = sqrt(x*x + y*y);
		if (dd > 0)
		{
			x /= dd;
			y /= dd;
			return;
		}
		x = y = 0;
	}

	template <typename T>
	static inline void normalize(vector2d<T> &v)
	{
		normalize(v.x, v.y);
	}

	template <typename T>
	static inline T dot(T x1, T y1, T x2, T y2)
	{
		return x1*x2 + y1*y2;
	}

	template <typename T>
	static inline void swap(T& x1, T& x2)
	{
		T xtmp;
		xtmp = x1; x1 = x2; x2 = xtmp;
	}

	template <typename T>
	static inline void swap(T* x1, T* x2)
	{
		T xtmp;
		xtmp = *x1; *x1 = *x2; *x2 = xtmp;
	}

	template <typename T>
	static inline i32 getVectorMemSize(const std::vector<T>& vec)
	{
		return (i32)(sizeof(i32) + vec.size()*sizeof(T));
	}

	template <typename T>
	static inline T cutAngle(T angle, T cut)
	{
		angle -= ((i32)(angle / PO_PI2))*PO_PI2;
		if (angle > PO_PI)
		{
			angle -= PO_PI2;
		}
		else if (angle < -PO_PI)
		{
			angle += PO_PI2;
		}

		if (angle >= -cut && angle <= cut)
		{
			return angle;
		}
		else if (angle > cut)
		{
			return cut;
		}
		return -cut;
	}

	template <typename T>
	static inline T getRegAngle(T angle)
	{
		angle -= ((i32)(angle / PO_PI2))*PO_PI2;
		if (angle < 0)
		{
			angle += PO_PI2;
		}
		if (angle > PO_PI)
		{
			return angle - PO_PI2;
		}
		return angle;
	}
	
	template <typename T>
	static inline T getAngleRegDiff(T angle, T base_angle)
	{
		angle = angle - base_angle;
		return angle < 0 ? PO_PI2 + angle : angle;
	}

	template <typename T>
	static inline T getVectorAngle(T dx, T dy)
	{
		T angle = atan2(dy, dx);
		return angle < 0 ? PO_PI2 + angle : angle;
	}

	template <typename T>
	static inline T getUnitVectorAngle(T x, T y)
	{
		T angle = 0;
		if (x >= 1.0)	return 0;
		if (x <= -1.0)	return PO_PI;

		angle = acos(x);
		if (y < 0)
		{
			angle = 2 * PO_PI - angle;
		}
		return angle;
	}

	template <typename T>
	static inline T getUnitVectorAngle180(T x, T y)
	{
		T angle = getUnitVectorAngle(x, y);
		if (angle > PO_PI)
		{
			angle -= PO_PI;
		}
		return angle;
	}

	template <typename T>
	static inline T getUnitVectorAngle(vector2d<T> p)
	{
		T angle = 0;
		if (p.x >= 1.0)  return 0;
		if (p.x <= -1.0) return PO_PI;

		angle = acos(p.x);
		if (p.y < 0)
		{
			angle = PO_PI2 - angle;
		}
		return angle;
	}

	template <typename T>
	static inline T getAngleDiff(T angle1, T angle2)
	{
		T min_angle = po::_min(angle1, angle2);
		T max_angle = po::_max(angle1, angle2);
		T diff_angle = max_angle - min_angle;
		while (diff_angle > PO_PI2)
		{
			diff_angle -= PO_PI2;
		}
		return diff_angle > PO_PI ? PO_PI2 - diff_angle : diff_angle;
	}

	template <typename T>
	static inline T getAngleDiff180(T an1, T an2)
	{
		T min_an = po::_min(an1, an2);
		T max_an = po::_max(an1, an2);
		return po::_min(max_an - min_an, min_an + PO_PI - max_an);
	}

	template <typename T>
	static inline T getAngleLen(T min_al, T max_al)
	{
		if (max_al < min_al)
		{
			return PO_PI2 + max_al - min_al;
		}
		return max_al - min_al;
	}

	template <typename T>
	static inline void getMinMaxAngle(T& dmin, T& dmax, T al)
	{
		if (dmin < 0 || dmax < 0)
		{
			dmin = al;
			dmax = al;
			return;
		}
		f32 dd1, dd2;
		dd1 = al - dmax;
		dd2 = dmin - al;

		if (dd1 < 0)
		{
			dd1 += PO_PI2;
		}
		if (dd2 < 0)
		{
			dd2 += PO_PI2;
		}
		if (dd1 > PO_PI && dd2 > PO_PI)
		{
			return;
		}

		if (dd1 < dd2)
		{
			dmax = al;
		}
		else
		{
			dmin = al;
		}
	}

	template <typename T>
	static inline T getAbsAngle(T angle)
	{
		while (angle < 0)
		{
			angle += PO_PI2;
		}
		while (angle >= PO_PI2)
		{
			angle -= PO_PI2;
		}
		return angle;
	}

	template <typename T>
	static inline T getAngle(vector2d<T> pos2d)
	{
		T angle = atan2(pos2d.y, pos2d.x);
		if (angle < 0)
		{
			angle += PO_PI2;
		}
		return angle;
	}

	template <typename T>
	static inline T getAngle(vector2d<T> v1, vector2d<T> v2)
	{
		normalize(v1);
		normalize(v2);

		T cosal = v1.x * v2.x + v1.y * v2.y;
		if (cosal < -1)	cosal = -1;
		if (cosal > 1)	cosal = 1;
		T angle = acos(cosal);

		if (angle > PO_PI)
		{
			angle = PO_PI2 - angle;
		}
		if (angle < 0)
		{
			angle = PO_PI + angle;
		}
		return angle;
	}

	template <typename T>
	static inline T getAbsAngle(vector2d<T> v1, vector2d<T> v2)
	{
		normalize(v1);
		normalize(v2);

		T cosa = v1.x * v2.x + v1.y * v2.y;
		if (cosa < -1)
		{
			cosa = -1;
		}
		if (cosa > 1)
		{
			cosa = 1;
		}

		T angle = acos(cosa);
		return angle;
	}

	template <typename T>
	static inline T getAngle(vector2d<T> p01, vector2d<T> p02, vector2d<T> p11, vector2d<T> p12)
	{
		return getAngle(p02 - p01, p12 - p11);
	}

	template <typename T>
	static inline T length(T x, T y)
	{
		return sqrt(x*x + y*y);
	}

	template <typename T>
	static inline T distance(T x1, T y1, T x2, T y2)
	{
		T x = x1 - x2;
		T y = y1 - y2;
		return sqrt(x*x + y*y);
	}

	template <typename T, typename U>
	static inline T distance(vector2d<T> pt1, vector2d<U> pt2)
	{
		T x = pt1.x - pt2.x;
		T y = pt1.y - pt2.y;
		return sqrt(x*x + y*y);
	}

	template <typename T>
	static inline T distanceSQ(T x1, T y1, T x2, T y2)
	{
		T x = x1 - x2;
		T y = y1 - y2;
		return x*x + y*y;
	}

	template <typename T>
	static inline T distanceSQ(vector2d<T>& pt1, vector2d<T>& pt2)
	{
		T x = pt2.x - pt1.x;
		T y = pt2.y - pt1.y;
		return x*x + y*y;
	}

	template <typename T>
	static inline f64 ratio(T x1, T x2)
	{
		f64 mx1 = po::_min(x1, x2);
		f64 mx2 = po::_max(x1, x2);
		return (mx2 > PO_EPSILON) ? (mx1 / mx2) : 0;
	}

	template <typename T>
	static inline void rotate(T& x, T& y, T sinth, T costh)
	{
		T xx = x;
		T yy = y;
		x = costh * xx - sinth * yy;
		y = sinth * xx + costh * yy;
	}

	template <typename T>
	static inline vector2d<T> rotate(vector2d<T> pt, T sinth, T costh)
	{
		vector2d<T> rpt;
		rpt.x = costh * pt.x - sinth * pt.y;
		rpt.y = sinth * pt.x + costh * pt.y;
		return rpt;
	}

	template <typename T>
	static inline T distPt2Line(T cx, T cy, T ax, T ay)
	{
		return std::abs(cx*ay - cy*ax);
	}

	template <typename T>
	static inline T distPt2Line(vector2d<T> ca, vector2d<T> normalize_ba)
	{
		return std::abs(ca.x*normalize_ba.y - ca.y*normalize_ba.x);
	}

	template <typename T>
	static inline T distPt2Line(T cax, T cay, vector2d<T> normalize_ba)
	{
		return std::abs(cax*normalize_ba.y - cay*normalize_ba.x);
	}

 	template <typename T>
 	static inline T distPt2Line(vector2d<T> pt, vector2d<T> line_pt1, vector2d<T> line_pt2)
 	{
		vector2d<T> normalized_ba = line_pt2 - line_pt1;
		vector2d<T> ca = line_pt1 - pt;

		normalized_ba.normalize();
 		return std::abs(ca.crossProduct(normalized_ba));
 	}

	template <typename T>
	static inline T signedDisPt2Line(T cx, T cy, T ax, T ay)
	{
		return cx*ay - cy*ax;
	}

	template <typename T>
	static inline T signedDisPt2Line(vector2d<T> ca, vector2d<T> normalize_ba)
	{
		return ca.x*normalize_ba.y - ca.y*normalize_ba.x;
	}

	template <typename T>
	static inline T signedDisPt2Line(T cax, T cay, vector2d<T> normalize_ba)
	{
		return cax*normalize_ba.y - cay*normalize_ba.x;
	}

	template <typename T>
	static inline T signedDisPt2Line(vector2d<T> pt, vector2d<T> line_pt1, vector2d<T> line_pt2)
	{
		vector2d<T> normalized_ba = line_pt2 - line_pt1;
		vector2d<T> ca = line_pt1 - pt;

		normalized_ba.normalize();
		return ca.crossProduct(normalized_ba);
	}

	template <typename T>
	static inline T distPtInLine(T cax, T cay, vector2d<T> normalize_ba)
	{
		T len;
		vector2d<T> ca(cax, cay);
		vector2d<T> normalize_ca = ca.normalize(len);
		return normalize_ca.dotProduct(normalize_ba)*len;
	}

	template <typename T>
	static inline T det(T cx, T cy, T ax, T ay)
	{
		return cx*ay - cy*ax;
	}

	template <typename T, typename U>
	static inline T distL1Norm(T x1, T y1, U x2 = 0, U y2 = 0)
	{
		return std::abs(x2 - x1) + std::abs(y2 - y1);
	}

	template <typename T, typename U>
	static inline T distL2Norm(T x1, T y1, U x2 = 0, U y2 = 0)
	{
		T x0 = x2 - x1;
		T y0 = y2 - y1;
		return x0*x0 + y0*y0;
	}
	
	template <typename T>
	static inline vector2d<T> getRightPoint(vector2d<T> p, vector2d<T> s, vector2d<T> e)
	{
		vector2df line_dir = e - s;
		normalize(line_dir);

		vector2df pt_vec = p - s;
		vector2df pt_dir = pt_vec;
		normalize(pt_dir);

		f32 cosq = pt_dir.dotProduct(line_dir);
		f32 len = cosq * pt_vec.getLength();
		return s + len*line_dir;
	}

	template <typename T>
	static inline bool getCircleCenter(vector2d<T> p0, vector2d<T> p1, vector2d<T> p2, vector2d<T>& center)
	{
		vector2d<T> R;
		vector2d<T> n01 = p1 - p0;
		vector2d<T> n02 = p2 - p0;
		n01 = vector2d<T>(n01.y, -n01.x);
		n02 = vector2d<T>(n02.y, -n02.x);

		if (n01.getLength() < 5 || n02.getLength() < 5 ||
			!getLineIntersect(p1, p1 + n01 * 10, p2, p2 + n02 * 10, R))
		{
			center = vector2d<T>(-1, -1);
			return false;
		}
		
		center = (p0 + R) / 2.0;
		return true;
	}

	template <typename T>
	static inline bool getLineIntersect(vector2d<T> s0, vector2d<T> e0, vector2d<T> s1, vector2d<T> e1, vector2d<T>& cross)
	{
		vector2df x = s1 - s0;
		vector2df d1 = e0 - s0;
		vector2df d2 = e1 - s1;

		f32 cross_val = d1.crossProduct(d2);
		if (abs(cross_val) < PO_EPSILON)
		{
			cross = vector2d<T>(0, 0);
			return false;
		}

		f64 t1 = (x.x * d2.y - x.y * d2.x) / cross_val;
		cross = s0 + d1 * t1;
		return true;
	}

	template <typename T>
	static inline bool pointInLine(vector2d<T> s0, vector2d<T> e0, vector2d<T> point)
	{
		vector2df dir1 = point - s0;
		vector2df dir2 = point - e0;
		vector2df dir3 = e0 - s0;
		vector2df nor = dir3;
		normalize(nor);

		f32 d = abs(dir1.crossProduct(nor));
		if (d > PO_DELTA)
		{
			return false;
		}

		if (dir1.getLengthSQ() > dir3.getLengthSQ())
		{
			return false;
		}
		if (dir2.getLengthSQ() > dir3.getLengthSQ())
		{
			return false;
		}
		return true;
	}

	template <typename T>
	static inline bool getInLineIntersect(vector2d<T> s0, vector2d<T> e0, vector2d<T> s1, vector2d<T> e1, vector2d<T>& cross)
	{
		vector2df p0;
		cross = vector2d<T>(-1, -1);
		if (!getLineIntersect(s0, e0, s1, e1, p0))
		{
			return false;
		}

		if (pointInLine(s0, e0, p0))
		{
			cross = p0;
			return true;
		}
		if (pointInLine(s1, e1, p0))
		{
			cross = p0;
			return true;
		}
		cross = vector2df(-1, -1);
		return false;
	}

	template <typename T>
	static inline T distInLine(T cx, T cy, T ax, T ay)
	{
		return cx*ax + cy*ay;
	}
	
	static inline f32 degToRad(f32 value)
	{
		return value*PO_PI / 180;
	}

	static inline vector2df degToRad(const vector2df& angle_vec2d)
	{
		return angle_vec2d*PO_PI / 180;
	}

	static inline f32 radToDeg(f32 value)
	{
		return value * 180 / PO_PI;
	}

	static inline vector2df radToDeg(const vector2df& angle_vec2d)
	{
		return angle_vec2d * 180 / PO_PI;
	}

	static inline f32 radToHalfDeg(f32 value)
	{ 
		value = value * 180 / PO_PI;
		return (value > 180) ? (value - 180) : value;
	}

	static inline f32 percent(f32 value)
	{
		return value*100.0f;
	}

	static inline f32 percentToRate(f32 value)
	{
		return value/100.0f;
	}

	template <typename T>
	static inline void crossProduct(T* v1, T* v2, T* v3)
	{
		v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
		v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
		v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
	}

	template <typename T>
	static inline T getPeakValueFromHist(T* buffer, i32 count, f64 dstp, f64 low, f64 high)
	{
		if (count <= 0)
		{
			return 0;
		}

		i32 i;
		T hmin = 1E+10;
		T hmax = -1E+10;
		T hmean = 0;
		T result = -1;

		// calc min & max & mean 
		for (i = 0; i < count; i++)
		{
			hmin = po::_min(hmin, buffer[i]);
			hmax = po::_max(hmax, buffer[i]);
			hmean += buffer[i];
		}

		hmean /= count;
		hmin = po::_max(hmin, hmean*0.2f);
		hmax = po::_min(hmax, hmean*5.0f);

		i32 index = 0;
		i32 hist_count = (i32)((hmax - hmin) / dstp + 2);
		i32* hist = po_new i32[hist_count];
		T* histmean = po_new T[hist_count];
		memset(hist, 0, sizeof(i32)*hist_count);
		memset(histmean, 0, sizeof(T)*hist_count);

		for (i = 0; i < count; i++)
		{
			if (buffer[i] < hmin || buffer[i] > hmax)
			{
				continue;
			}

			index = (i32)((buffer[i] - hmin) / dstp + 0.5);
			hist[index]++;
			histmean[index] += buffer[i];
		}

		index = -1;
		i32 histmax = 0;
		for (i = 0; i < hist_count; i++)
		{
			if (hist[i] > histmax)
			{
				index = i;
				histmax = hist[i];
			}
		}
		if (index >= 0)
		{
			i32 hn = 0;
			T hsum = 0;
			T hval = histmean[index] / hist[index];
			T lowval = (T)(low*hval);
			T highval = (T)(high*hval);

			for (i = 0; i < count; i++)
			{
				if (buffer[i] > lowval && buffer[i] < highval)
				{
					hn++;
					hsum += buffer[i];
				}
			}
			result = hsum / hn;
		}

		POSAFE_DELETE_ARRAY(hist);
		POSAFE_DELETE_ARRAY(histmean);
		return result;
	}

	template <typename T>
	static inline bool findNearest(std::vector<vector2d<T>> points, vector2d<T> seed, i32 roi_count, i32* indices_ptr, f32* dist_ptr)
	{
		i32 point_count = (i32)points.size();
		if (roi_count > point_count)
		{
			return false;
		}

		std::vector<bool> is_used_vec;
		is_used_vec.resize(point_count);
		for (i32 i = 0; i < point_count; i++)
		{
			is_used_vec[i] = false;
		}

		std::vector<f32> dvec;
		dvec.resize(point_count);
		for (i32 i = 0; i < point_count; i++)
		{
			dvec[i] = (points[i] - seed).getLengthSQ();
		}

		for (i32 k = 0; k < roi_count; k++)
		{
			indices_ptr[k] = -1;
			dist_ptr[k] = 10240 * 10240;

			// Find k-th Nearest point
			for (i32 i = 0; i < point_count; i++)
			{
				if (is_used_vec[i])
				{
					continue;
				}
				if (dist_ptr[k] > dvec[i])
				{
					indices_ptr[k] = i;
					dist_ptr[k] = dvec[i];
				}

			}

			if (indices_ptr[k] != -1)
			{
				is_used_vec[indices_ptr[k]] = true;
			}
		}
		return true;
	}

	template <typename T>
	static inline void clearList(std::list<T*> list)
	{
		typename std::list<T*>::iterator iter = list.begin();
		for (; iter != list.end(); iter++)
		{
			if (*iter)
			{
				delete (*iter);
			}
		}
		list.clear();
	}

	template <typename T>
	static inline void popFront(std::list<T>& list, T& value)
	{
		if (list.size() <= 0)
		{
			return;
		}
		value = list.front();
		list.pop_front();
	}

	template<size_t count, class X>  static inline i32 arraySize(X(&arr)[count])
	{
		return (i32)count;
	}
};
