#include "base.h"
#include <ctime>
#include <time.h> 
#include <iomanip>	// setprecision
#include <sstream>	// stringstream
#include <string>	// std::wstring, std::to_wstring
#include "os/os_support.h"

//////////////////////////////////////////////////////////////////
i32 cmp_azfind_i32(const void* a, const i32 b)
{
	return b - *((i32*)a);
}

i64 cmp_azfind_i64(const void* a, const i64 b)
{
	return b - *((i64*)a);
}

i32 cmp_zafind_i32(const void* a, const i32 b)
{
	return *((i32*)a) - b;
}

i64 cmp_zafind_i64(const void* a, const i64 b)
{
	return *((i64*)a) - b;
}

i32 cmp_azsort_i32(const void* pa, const void* pb)
{
	i32 a = *(i32*)pa;
	i32 b = *(i32*)pb;
	return a < b ? 1 : (a > b ? -1 : 0);
}

i32 cmp_azsort_i64(const void* pa, const void* pb)
{
	i64 a = *(i64*)pa;
	i64 b = *(i64*)pb;
	return a < b ? 1 : (a > b ? -1 : 0);
}

i32 cmp_zasort_i32(const void* pa, const void* pb)
{
	i32 a = *(i32*)pa;
	i32 b = *(i32*)pb;
	return a > b ? 1 : (a < b ? -1 : 0);
}

i32 cmp_zasort_i64(const void* pa, const void* pb)
{
	i64 a = *(i64*)pa;
	i64 b = *(i64*)pb;
	return a > b ? 1 : (a < b ? -1 : 0);
}

//////////////////////////////////////////////////////////////////////////
CPOBase::CPOBase()
{
}

CPOBase::~CPOBase()
{
}
postring CPOBase::getDateTimeFileName(i32 id)
{
	DateTime tm;
	CPOBase::getNowTime(tm);
	return tm.toString(id);
}

void CPOBase::getNowTime(DateTime& dtm)
{
#if defined(POR_DEVICE)
	QDateTime q_dtm = QDateTime::currentDateTime();
	QDate dm = q_dtm.date();
	QTime tm = q_dtm.time();
	dtm.setDateTime(dm.year(), dm.month(), dm.day(), tm.hour(), tm.minute(), tm.second(), tm.msec());
#else
	dtm.init();
#endif
}

void CPOBase::randomInit()
{
	srand((unsigned)time(NULL));
}

i32 CPOBase::random()
{
	return rand();
}

f32 CPOBase::randomPlus()
{
	return (f32)(rand() % 1000) / 1000;
}

f32 CPOBase::randomUnit()
{
	return (f32)(rand() % 1000 - 500) / 500;
}

bool CPOBase::removeParenthesis(postring& str_expression)
{
	i32 i, size = (i32)str_expression.size();
	if (size <= 0 || str_expression[0] != '(')
	{
		return true;
	}

	char ch;
	bool quote_block = false;
	i32 level = 0, spos = -1, epos = -1;
	for (i = 0; i < size; i++)
	{
		ch = str_expression[i];

		//check quote block
		if (ch == '"')
		{
			quote_block = !quote_block;
		}
		if (quote_block)
		{
			continue;
		}

		//check parenthesis
		if (ch == '(')
		{
			if (level == 0)
			{
				spos = i;
			}
			level++;
		}
		else if (ch == ')')
		{
			level--;
			if (level == 0)
			{
				epos = i;
			}
			else if (level < 0)
			{
				return false;
			}
		}
	}
	if (level > 0)
	{
		return false;
	}
	if (spos == 0 && epos == size - 1)
	{
		str_expression = str_expression.substr(1, size - 2);
	}
	return true;
}

bool CPOBase::removeQuote(postring& str_expression)
{
	if (!isString(str_expression))
	{
		return false;
	}
	str_expression = str_expression.substr(1, str_expression.size() - 2);
	return true;
}

bool CPOBase::isString(const postring& str)
{
	i32 count = (i32)str.size();
	if (count < 3)
	{
		return false;
	}

	if (str[0] != '"' || str[count - 1] != '"')
	{
		return false;
	}
	return true;
}

bool CPOBase::isNumber(const postring& str)
{
	std::size_t char_pos(0);

	// skip the whilespaces
	char_pos = str.find_first_not_of(' ');
	if (char_pos == str.size())
	{
		return false;
	}
	
	// check the significand
	if (str[char_pos] == '+' || str[char_pos] == '-')
	{
		++char_pos; // skip the sign if exist
	}

	i32 n_nm, n_pt;
	for (n_nm = 0, n_pt = 0; (str[char_pos] >= '0' && str[char_pos] <= '9') || str[char_pos] == '.'; ++char_pos)
	{
		str[char_pos] == '.' ? ++n_pt : ++n_nm;
	}
	if (n_pt > 1 || n_nm < 1) // no more than one point, at least one digit
	{
		return false;
	}

	// skip the trailing whitespaces
	while (str[char_pos] == ' ')
	{
		++char_pos;
	}
	return char_pos == str.size();  // must reach the ending 0 of the string
}

void CPOBase::spiltToVector(const postring& path, const postring& delimiter, strvector& str_vec)
{
	size_t pos = 0;
	postring str_token;
	postring str_path = path;
	str_vec.clear();

	while ((pos = str_path.find(delimiter)) != std::string::npos)
	{
		str_token = str_path.substr(0, pos);
		str_vec.push_back(str_token);
		str_path.erase(0, pos + delimiter.length());
	}
	str_vec.push_back(str_path);
}

void CPOBase::splitToPath(const postring& path, const postring& delimiter, postring& split_path, postring& split_name)
{
	size_t last_pos = path.find_last_of(delimiter);
	if (last_pos == postring::npos)
	{
		split_path = "";
		split_name = path;
		return;
	}
	
	if (last_pos > 0)
	{
		split_path = path.substr(0, last_pos);
	}
	else
	{
		split_path = "";
	}

	last_pos = last_pos + delimiter.size();
	if (last_pos < path.size())
	{
		split_name = path.substr(last_pos);
	}
	else
	{
		split_name = "";
	}
}

void CPOBase::toLowerExpression(postring& str)
{
	bool is_blocking = false;
	i32 bpos = 0, bend = 0;
	i32 i, count = (i32)str.size();
	
	for (i = 0; i < count; i++)
	{
		if (str[i] == '"')
		{
			is_blocking = !is_blocking;
			if (!is_blocking)
			{
				bpos = i + 1;
				continue;
			}
			if (i > bpos)
			{
				bend = i - 1;
				std::transform(str.begin() + bpos, str.begin() + bend, str.begin() + bpos, ::tolower);
			}
		}
	}
}

i32 CPOBase::convertIPAddress(const postring& ip_string)
{
	std::stringstream s(ip_string);
	i32 a, b, c, d; //to store the 4 ints
	char ch; //to temporarily store the '.'
	s >> a >> ch >> b >> ch >> c >> ch >> d;
	return (a << 24) | (b << 16) | (c << 8) | d;
}

void CPOBase::convertIPAddress(const i32 ip, postring& ip_string)
{
	i32 a, b, c, d;
	a = (ip >> 24) & 0xFF;
	b = (ip >> 16) & 0xFF;
	c = (ip >> 8) & 0xFF;
	d = ip & 0xFF;

	char str_ip_address[16];
	po_sprintf(str_ip_address, 16, "%d.%d.%d.%d", a, b, c, d);
	ip_string = str_ip_address;
}

bool CPOBase::stoi(const postring& str, i32& tmp)
{
	try
	{
		tmp = std::stoi(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stoi(const powstring& str, i32& tmp)
{
	try
	{
		tmp = std::stoi(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stoll(const postring& str, i64& tmp)
{
	try
	{
		tmp = std::stoll(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stoll(const powstring& str, i64& tmp)
{
	try
	{
		tmp = std::stoll(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stoul(const postring& str, u32& tmp)
{
	try
	{
		tmp = std::stoul(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stoul(const powstring& str, u32& tmp)
{
	try
	{
		tmp = std::stoul(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stoull(const postring& str, u64& tmp)
{
	try
	{
		tmp = std::stoull(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stoull(const powstring& str, u64& tmp)
{
	try
	{
		tmp = std::stoull(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stof(const postring& str, f32& tmp)
{
	try
	{
		tmp = std::stof(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stof(const powstring& str, f32& tmp)
{
	try
	{
		tmp = std::stof(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stod(const postring& str, f64& tmp)
{
	try
	{
		tmp = std::stod(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

bool CPOBase::stod(const powstring& str, f64& tmp)
{
	try
	{
		tmp = std::stod(str);
	}
	catch (...)
	{
		return false;
	}
	return true;
}

postring CPOBase::getTermiator(i32 terminator)
{
	postring str;
	switch (terminator)
	{
		case kPOTermCRLF:
		{
			str.resize(2); str[0] = 0x0D; str[1] = 0x0A;
			break;
		}
		case kPOTermCR:
		{
			str.resize(1); str[0] = 0x0D;
			break;
		}
		case kPOTermLF:
		{
			str.resize(1); str[0] = 0x0A;
			break;
		}
	}
	return str;
}

postring CPOBase::toString(f32 value, i32 decimal)
{
	if (decimal < 0)
	{
		return std::to_string(value);
	}
	std::stringstream ss;
	ss << std::fixed << std::setprecision(decimal) << value;
	return ss.str();
}

postring CPOBase::toString(f64 value, i32 decimal)
{
	if (decimal < 0)
	{
		return std::to_string(value);
	}
	std::stringstream ss;
	ss << std::fixed << std::setprecision(decimal) << value;
	return ss.str();
}

powstring CPOBase::toWString(f32 value, i32 decimal)
{
	if (decimal < 0)
	{
		return std::to_wstring(value);
	}
	std::wstringstream ss;
	ss << std::fixed << std::setprecision(decimal) << value;
	return ss.str();
}

powstring CPOBase::toWString(f64 value, i32 decimal)
{
	if (decimal < 0)
	{
		return std::to_wstring(value);
	}
	std::wstringstream ss;
	ss << std::fixed << std::setprecision(decimal) << value;
	return ss.str();
}