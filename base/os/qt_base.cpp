#include "qt_base.h"
#include "logger/logger.h"
#include <QFileInfo>
#include <QDir>
#include <QDirIterator>
#include <QMutex>
#include <QMutexLocker>
#include <QSharedMemory>
#include <QNetworkInterface>

//////////////////////////////////////////////////////////////////////////
bool QTBase::addDir(const postring& str_path, i32 sub_id)
{
	if (sub_id < 0)
	{
		return false;
	}
	
	QDir dir(str_path.c_str());
	QString sub_dir_name = QString::number(sub_id);
	if (!dir.exists())
	{
		return false;
	}
	return dir.mkdir(sub_dir_name);
}

bool QTBase::copyDir(const QString& src_file_path, const QString& dst_file_path)
{
	QString dir_name;
	QFileInfo src_file_info(src_file_path);

	if (src_file_info.isDir())
	{
		QDir target_dir(dst_file_path);
		target_dir.cdUp();

		dir_name = QFileInfo(dst_file_path).fileName();
		if (!target_dir.exists(dir_name) && !target_dir.mkdir(dir_name))
		{
			return false;
		}

		QDir source_dir(src_file_path);
		QStringList file_vec = source_dir.entryList(QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System);
		foreach(const QString &filename, file_vec)
		{
			const QString new_src_file_path = src_file_path + QLatin1Char('/') + filename;
			const QString new_tgt_file_path = dst_file_path + QLatin1Char('/') + filename;
			if (!copyDir(new_src_file_path, new_tgt_file_path))
			{
				return false;
			}
		}
	}
	else
	{
		return QTBase::copyFile(src_file_path, dst_file_path);
	}
	return true;
}

bool QTBase::copyFile(const QString& src_file_name, const QString& dst_file_name)
{
	if (QFile::exists(dst_file_name))
	{
		QFile::remove(dst_file_name);
	}
	if (!QFile::copy(src_file_name, dst_file_name))
	{
		return false;
	}
	return true;
}

i32 QTBase::getFileCount(const QString& dir_path, QString file_pattern)
{
	QDir current_dir(dir_path);
	QStringList file_vec;

	file_vec = current_dir.entryList(QStringList(file_pattern), QDir::Files | QDir::NoSymLinks);
	return file_vec.size();
}

bool QTBase::clearContents(const postring& dir_path)
{
	return clearContents(QString::fromStdString(dir_path));
}

bool QTBase::clearContents(const powstring& dir_path)
{
	return clearContents(QString::fromStdWString(dir_path));
}

bool QTBase::clearContents(const QString& dir_path)
{
	//rebuild high-level path
	QDir dir(dir_path);
	dir.removeRecursively();
	if (!dir.mkpath(dir_path))
	{
		printlog_lv2(QString("Can't clear directory. %1").arg(dir_path));
		return false;
	}
	return true;
}

QString QTBase::convertToString(const DateTime& dtm)
{
	return QString("%1-%2-%3T%4:%5:%6.%7Z")
			.arg(dtm.yy, 4, 10, QLatin1Char('0'))
			.arg(dtm.mm, 2, 10, QLatin1Char('0'))
			.arg(dtm.dd, 2, 10, QLatin1Char('0'))
			.arg(dtm.h, 2, 10, QLatin1Char('0'))
			.arg(dtm.m, 2, 10, QLatin1Char('0'))
			.arg(dtm.s, 2, 10, QLatin1Char('0'))
			.arg(dtm.ms, 3, 10, QLatin1Char('0'));
}

QString QTBase::convertToString(const QDateTime& dtm)
{
	return dtm.toString("yyyy-MM-ddTHH:mm:ss.zzzZ");
}

DateTime QTBase::convertToDateTime(const QString& string)
{
	QDateTime dtm = QDateTime::fromString(string, "yyyy-MM-ddTHH:mm:ss.zzzZ");
	QDate dm = dtm.date();
	QTime tm = dtm.time();
	return DateTime(dm.year(), dm.month(), dm.day(), tm.hour(), tm.minute(), tm.second(), tm.msec());
}

DateTime QTBase::currentDateTime()
{
	QDateTime dtm = QDateTime::currentDateTime();
	QDate dm = dtm.date();
	QTime tm = dtm.time();

	DateTime ptm;
	ptm.yy = dm.year();
	ptm.mm = dm.month();
	ptm.dd = dm.day();
	ptm.h = tm.hour();
	ptm.m = tm.minute();
	ptm.s = tm.second();
	ptm.ms = tm.msec();
	return ptm;
}

QDateTime QTBase::convertToQDateTime(const QString& string)
{
	return QDateTime::fromString(string, "yyyy-MM-ddTHH:mm:ss.zzzZ");
}

QDateTime QTBase::convertToQDateTime(const DateTime& dtm)
{
	QDate dm;
	QTime tm;
	dm.setDate(dtm.yy, dtm.mm, dtm.dd);
	tm.setHMS(dtm.h, dtm.m, dtm.s, dtm.ms);
	return QDateTime(dm, tm);
}

DateTime QTBase::convertToDateTime(const QDateTime& dtm)
{
	QDate dm = dtm.date();
	QTime tm = dtm.time();
	return DateTime(dm.year(), dm.month(), dm.day(), tm.hour(), tm.minute(), tm.second(), tm.msec());
}

//////////////////////////////////////////////////////////////////////////
bool QTBase::getNetworkAdapters(NetAdapterArray& adapter_vec)
{
	i32 i, j;
	NetAdapter adapter;
	adapter_vec.clear();
		
	QHostAddress hostaddr;
	QList<QNetworkAddressEntry> addrentries;
	QList<QNetworkInterface> ifaces = QNetworkInterface::allInterfaces();

	if (!ifaces.isEmpty())
	{
		for (i = 0; i < ifaces.size(); i++)
		{
			u32 flags = ifaces[i].flags();
			bool is_loop_back = (bool)(flags & QNetworkInterface::IsLoopBack);
			bool is_p2p = (bool)(flags & QNetworkInterface::IsPointToPoint);
			//bool is_running = (bool)(flags & QNetworkInterface::IsRunning);

			//If this interface isn't running, we don't care about it
			//if (!is_running)
			//{
			//	continue;
			//}

			//We only want valid interfaces that aren't loopback/virtual and not point to point
			if (!ifaces[i].isValid() || is_p2p)
			{
				continue;
			}

			adapter.adapter_name = ifaces[i].humanReadableName().toStdString();
			adapter.mac_address = ifaces[i].hardwareAddress().toStdString();
			adapter.is_conf_dhcp = true;

			addrentries = ifaces[i].addressEntries();
			for (j = 0; j < addrentries.size(); j++)
			{
				hostaddr = addrentries[j].ip();
				if (hostaddr.toIPv4Address() > 0)
				{
					adapter.ip_address = hostaddr.toIPv4Address();
					adapter.ip_subnet = addrentries[j].netmask().toIPv4Address();
					adapter.is_loopback = hostaddr.isLoopback();
					adapter_vec.push_back(adapter);
				}
			}
		}
	}
	return true;
}

bool QTBase::deleteMultiFiles(const postring& str_path, const postring& str_file_ext, i32 st_index)
{
	i32 pos, index;
	QString str_filename;
	QDir dir(str_path.c_str());
	QDirIterator it(str_path.c_str(), QStringList() << str_file_ext.c_str(),
				QDir::Files | QDir::NoDotAndDotDot | QDir::NoSymLinks);
	
	while (it.hasNext())
	{
		QFileInfo file_info(it.next());
		str_filename = file_info.fileName();
		pos = str_filename.indexOf(".");
		if (pos <= 0)
		{
			continue;
		}

		index = str_filename.left(pos).toInt();
		if (index >= st_index)
		{
			dir.remove(str_filename);
		}
	}
	return true;
}

bool QTBase::deleteMultiFiles(const postring& str_path, const postring& str_file_ext, i32 st_index, i32 base_pow)
{
	i32 pos, index;
	QString str_filename;
	QDir dir(str_path.c_str());
	QDirIterator it(str_path.c_str(), QStringList() << str_file_ext.c_str(),
				QDir::Files | QDir::NoDotAndDotDot | QDir::NoSymLinks);

	while (it.hasNext())
	{
		QFileInfo file_info(it.next());
		str_filename = file_info.fileName();
		pos = str_filename.indexOf(".");
		if (pos <= 0)
		{
			continue;
		}

		index = str_filename.left(pos).toInt() % base_pow;
		if (index >= st_index)
		{
			dir.remove(str_filename);
		}
	}
	return true;
}

bool QTBase::deleteSubDirMultiFiles(const postring& str_path, const postring& str_file_ext,
								i32 st_index, i32 base_pow)
{
	QString str_dir;
	QDirIterator it(str_path.c_str(), QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks);
	while (it.hasNext())
	{
		str_dir = it.next();
		deleteMultiFiles(str_dir.toStdString(), str_file_ext, st_index, base_pow);
	}
	return true;
}

bool QTBase::deleteMultiDirs(const postring& str_path, i32 sub_id)
{
	if (sub_id < 0)
	{
		QString str_dir;
		QDirIterator it(str_path.c_str(), QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks);
		while (it.hasNext())
		{
			str_dir = it.next();
			QDir dir(str_dir);
			dir.removeRecursively();
		}
	}
	else
	{
		QString str_dir = QString(str_path.c_str()) + "/" + QString::number(sub_id);
		QDir dir(str_dir);
		dir.removeRecursively();
	}
	return true;
}

bool QTBase::deleteMultiDirsNotIn(const postring& str_path, i32vector sub_id_vec)
{
	if (sub_id_vec.size() <= 0)
	{
		return true;
	}

	i32 index;
	i32 i, count = (i32)sub_id_vec.size();
	QDirIterator it(str_path.c_str(), QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks);
	while (it.hasNext())
	{
		QFileInfo file_info(it.next());
		index = file_info.fileName().toInt();
		for (i = 0; i < count; i++)
		{
			if (index == sub_id_vec[i])
			{
				break;
			}
		}
		if (i == count)
		{
			QDir dir(file_info.filePath());
			dir.removeRecursively();
		}
	}
	return true;
}
