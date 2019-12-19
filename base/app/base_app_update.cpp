#include "base_app_update.h"
#include "base_app.h"
#include "base.h"
#include "os/qt_base.h"
#include "quazip/JlCompress.h"
#include "app/base_disk.h"
#include "os/os_support.h"
#include "log_config.h"

CBaseAppUpdate::CBaseAppUpdate()
{
	m_app_update.init();
}

CBaseAppUpdate::~CBaseAppUpdate()
{
}

CPOBaseApp* CBaseAppUpdate::getBaseApp()
{
	return dynamic_cast<CPOBaseApp*>(this);
}

void CBaseAppUpdate::initAppUpdate()
{
	if (m_app_update.isUpdate())
	{
		deviceUpdateCancel();
		m_app_update.init();
	}
}

i32 CBaseAppUpdate::deviceUpdateStream(i32 conn, i32 packet_id, u8* buffer_ptr, i32& buffer_size)
{
	if (!buffer_ptr || buffer_size <= 0)
	{
		printlog_lv1("Device update data is not correctly. Please check now.");
		alarmlog1(kPOErrDeviceUpdateData);
		return kPOErrDeviceUpdateData;
	}
	
	DeviceInfo* dev_info_ptr = getBaseApp()->getDeviceInfo();

	if (packet_id == 0)
	{
		//check already updating now
		if (m_app_update.isUpdate())
		{
			printlog_lv1("The device is already updating now.");
			alarmlog1(kPOErrDeviceDupUpdate);
			return kPOErrDeviceDupUpdate;
		}

		m_app_update.init();
		CPOBase::memRead(buffer_ptr, buffer_size, m_app_update.model_name);
		CPOBase::memRead(buffer_ptr, buffer_size, m_app_update.update_version);
		CPOBase::memReadStrVector(m_app_update.extra_key, buffer_ptr, buffer_size);
		CPOBase::memReadStrVector(m_app_update.extra_value, buffer_ptr, buffer_size);
		CPOBase::memReadStrVector(m_app_update.lowlevel_file_vec, buffer_ptr, buffer_size);
		CPOBase::memReadStrVector(m_app_update.lowlevel_dir_vec, buffer_ptr, buffer_size);
		CPOBase::memReadStrVector(m_app_update.highlevel_file_vec, buffer_ptr, buffer_size);
		CPOBase::memReadStrVector(m_app_update.highlevel_dir_vec, buffer_ptr, buffer_size);
		CPOBase::memRead(m_app_update.update_compatibility, buffer_ptr, buffer_size);
		CPOBase::memRead(m_app_update.is_update_highlevel, buffer_ptr, buffer_size);
		CPOBase::memRead(m_app_update.filesize, 2, buffer_ptr, buffer_size);
		
		printlog_lv1("-----Update Information-----");
		printlog_lv1(QString("Model Name: %1").arg(m_app_update.model_name.c_str()));
		printlog_lv1(QString("Update Version: %1").arg(m_app_update.update_version.c_str()));
		printlog_lv1(QString("Compatibility: %1").arg(m_app_update.update_compatibility));
		printlog_lv1(QString("HighLevel Update: %1").arg(m_app_update.is_update_highlevel));
		printlog_lv1(QString("UPdateFileSize LL: %1, HL: %2").arg(m_app_update.filesize[0]).arg(m_app_update.filesize[1]));
		printlog_lv1(QString("LL files: %1, dirs: %2").arg(m_app_update.lowlevel_file_vec.size()).arg(m_app_update.lowlevel_dir_vec.size()));
		printlog_lv1(QString("HL files: %1, dirs: %2").arg(m_app_update.highlevel_file_vec.size()).arg(m_app_update.highlevel_dir_vec.size()));
		printlog_lv1(QString("Extra keys: %1").arg(m_app_update.extra_key.size()));
		printlog_lv1(QString("Connection: %1").arg(conn));
		printlog_lv1("----------------------------");
		
		//check information data
		if (m_app_update.extra_key.size() != m_app_update.extra_value.size())
		{
			printlog_lv1("Extra KeyData invalid.");
			alarmlog1(kPOErrDeviceUpdateData);
			return kPOErrDeviceUpdateData;
		}

		//현재의 하위기버젼과 갱신하려는 하위기버젼이 같은 경우 갱신을 허용한다.(재설치 또는 작은오유들을 수정하여 같은판본으로 등록한 경우)
		if (dev_info_ptr->device_version == m_app_update.update_version)
		{
			printlog_lv1(QString("The update version is same with current device version[%1].")
							.arg(dev_info_ptr->device_version.c_str()));
			alarmlog1(kPOErrDeviceUpdateVer);
			//return kPOErrDeviceUpdateVer;
		}

		//check high-level compatibility
		if (dev_info_ptr->is_hl_embedded != m_app_update.is_update_highlevel)
		{
			printlog_lv1("The update doesn't include embedded highlevel.");
			alarmlog1(kPOErrDeviceUpdateEmbedded);
			return kPOErrDeviceUpdateEmbedded;
		}

		QString ll_data_path = getBaseApp()->getDevicePath();
		QString hl_data_path = getBaseApp()->getDeviceHLPath();
		if (dev_info_ptr->is_hl_embedded && hl_data_path == "")
		{
			printlog_lv1("The LowLevel is not connected with local highlevel in embedded mode.");
			alarmlog1(kPOErrDeviceUpdateEmbedded);
			return kPOErrDeviceUpdateEmbedded;
		}

		m_app_update.is_update = true;
		m_app_update.update_from = conn;
		m_app_update.update_packet_id = 0;
		m_app_update.read_id = 0;
		m_app_update.read_size = m_app_update.filesize[0];

		getBaseApp()->onRequestStop(NULL);
		QFile::remove(ll_data_path+ "/~update0.tmp");
		QFile::remove(ll_data_path+ "/~update1.tmp");
	}
	else
	{
		if (m_app_update.getFrom() != conn)
		{
			//check updating source address
			printlog_lv1("The device is already updating now.");
			alarmlog1(kPOErrDeviceDupUpdate);
			return kPOErrDeviceDupUpdate;
		}
		if (packet_id != m_app_update.getCurPacketID() + 1 || !m_app_update.isUpdate())
		{
			//check next update packet_ptr id and validation updating...
			deviceUpdateCancel();
			printlog_lv1("Device update data is not correctly. Please check now.");
			alarmlog1(kPOErrDeviceUpdateData);
			return kPOErrDeviceUpdateData;
		}

		//recive buffer_ptr data and store to tmp low-level rom file.
		i32 size = 0;
		i32 max_read_count = dev_info_ptr->is_hl_embedded ? 2 : 1;
		char filename[PO_MAXPATH];

		while (m_app_update.read_id < max_read_count && buffer_size > 0)
		{
			size = po::_min(m_app_update.read_size, buffer_size);
			if (size > 0)
			{
				po_sprintf(filename, PO_MAXPATH, "~update%d.tmp", m_app_update.read_id);
				getBaseApp()->appendToFile(filename, buffer_ptr, size);
				m_app_update.read_size -= size;
				buffer_size -= size;
			}

			if (buffer_size > 0)
			{
				m_app_update.read_id++;
				if (m_app_update.read_id >= max_read_count)
				{
					if (m_app_update.read_size != 0)
					{
						//update data is not correctly
						deviceUpdateCancel();

						printlog_lv1(QString("Update data is valid. read[%1], len[%2]")
										.arg(m_app_update.read_id).arg(buffer_size));
						return kPOErrDeviceUpdateData;
					}
				}
				else
				{
					m_app_update.read_size = m_app_update.filesize[m_app_update.read_id];
				}
			}
		}

		m_app_update.updatePacketID();
	}
	return kPOSuccess;
}

bool CBaseAppUpdate::deviceUpdateInternal()
{
	QString data_path = getBaseApp()->getDevicePath();
	QString data_path_hl = getBaseApp()->getDeviceHLPath();

	if (!m_app_update.is_update)
	{
		printlog_lv1("Can't update device");
		return false;
	}
	if (m_app_update.is_update_highlevel && data_path_hl == "")
	{
		printlog_lv1("Can't update with null HL path when update HL also.");
		return false;
	}

	//write actionlog
	actionlog(POAction(kPOActionDevUpdate));

	if (!getBaseApp()->updateDeviceOffline())
	{
		printlog_lv1("Can't update becasue of safe offline is failed.");
		return false;
	}

	//prepare all path for update
	QDir dir = QDir(data_path);
	if (!dir.cdUp())
	{
		printlog_lv1(QString("Can't update because of [%1] cdUp is failed.").arg(data_path));
		return false;
	}
	QString update_path = dir.absolutePath();
	QString lowlevel_path = dir.absolutePath() + PO_UPDATE_LLPATH;
	QString update_lowlevel_filename = data_path + "/~update0.tmp";
	QString update_highlevel_filename = data_path + "/~update1.tmp";

	QString hl_path;
	if (m_app_update.is_update_highlevel)
	{
		dir = QDir(data_path_hl);
		if (!dir.cdUp())
		{
			printlog_lv1(QString("Can't update because of [%1](HL) cdUp is failed.").arg(data_path_hl));
			return false;
		}
		hl_path = dir.absolutePath() + PO_UPDATE_HLPATH;
	}

	//rebuild low-level path
	dir = QDir(lowlevel_path);
	dir.removeRecursively();
	if (!dir.mkpath(lowlevel_path))
	{
		printlog_lv1(QString("Can't update because of [%1] mkdir is failed.").arg(lowlevel_path));
		return false;
	}

	//extract low-level 
	JlCompress::extractDir(update_lowlevel_filename, lowlevel_path);

	if (m_app_update.is_update_highlevel)
	{
		//rebuild high-level path
		QDir dir(hl_path);
		dir.removeRecursively();
		if (!dir.mkpath(hl_path))
		{
			printlog_lv1(QString("Can't update because of [%1](HL) mkdir is failed.").arg(hl_path));
			return false;
		}

		//extract high-level
		JlCompress::extractDir(update_highlevel_filename, hl_path);
	}

	//some file operations in HL
	i32 i, count;
	QString filename1;
	QString filename2;
	if (m_app_update.is_update_highlevel)
	{
		for (i = 0; i < m_app_update.highlevel_file_vec.size(); i++)
		{
			filename1 = data_path_hl + "/" + m_app_update.highlevel_file_vec[i].c_str();
			filename2 = hl_path + "/" + m_app_update.highlevel_file_vec[i].c_str();
			if (!QTBase::copyFile(filename1, filename2))
			{
				printlog_lv1(QString("Can't update because of filecopy(HL) is failed. [%1]->[%2]")
					.arg(filename1).arg(filename2));
				return false;
			}
		}
		for (i = 0; i < m_app_update.highlevel_dir_vec.size(); i++)
		{
			filename1 = data_path_hl + m_app_update.highlevel_dir_vec[i].c_str();
			filename2 = hl_path + m_app_update.highlevel_dir_vec[i].c_str();
			if (!QTBase::copyDir(filename1, filename2))
			{
				printlog_lv1(QString("Can't update because of dircopy(HL) is failed. [%1]->[%2]")
					.arg(filename1).arg(filename2));
				return false;
			}
		}
	}

	//some file operations in LL
	if (m_app_update.getCompatibility() == kPOAppUpdateCPTValid)
	{
		//copy additional data
		for (i = 0; i < m_app_update.lowlevel_file_vec.size(); i++)
		{
			filename1 = data_path + "/" + m_app_update.lowlevel_file_vec[i].c_str();
			filename2 = lowlevel_path + "/" + m_app_update.lowlevel_file_vec[i].c_str();
			if (!QTBase::copyFile(filename1, filename2))
			{
				printlog_lv1(QString("Can't update because of filecopy is failed. [%1]->[%2]")
								.arg(filename1).arg(filename2));
				return false;
			}
		}
		for (i = 0; i < m_app_update.lowlevel_dir_vec.size(); i++)
		{
			filename1 = data_path + m_app_update.lowlevel_dir_vec[i].c_str();
			filename2 = lowlevel_path + m_app_update.lowlevel_dir_vec[i].c_str();
			if (!QTBase::copyDir(filename1, filename2))
			{
				printlog_lv1(QString("Can't update because of dircopy is failed. [%1]->[%2]")
								.arg(filename1).arg(filename2));
				return false;
			}
		}
	}
	else
	{
		count = (i32)m_app_update.extra_key.size();
		for (i = 0; i < count; i++)
		{
			postring extra_key_str = m_app_update.extra_key[i];
			postring extra_value_str = m_app_update.extra_value[i];
			CPOBase::toLower(extra_key_str);

			if (extra_key_str == PODB_SQL_MODE)
			{
				//restore SQLDump file to database
				QString sql_filename;
				sql_filename = lowlevel_path + "/" + extra_value_str.c_str();
				printlog_lv1(QString("Update database MYSQL:%1").arg(sql_filename));

				QFile qf(sql_filename);
				if (qf.open(QIODevice::ReadOnly))
				{
					printlog_lv1("Can't update because of database file is missing.");
					return false;
				}

				QString sql_string(qf.readAll());
				if (!getBaseApp()->executeSQLQuery(sql_string))
				{
					printlog_lv1("Can't update because of rebuild database.");
					qf.close();
					return false;
				}

				qf.close();
				QFile::remove(sql_filename);
			}
			else if (extra_key_str == PODB_SQLITE_MODE)
			{
				//copy SQLite file to database path
				QString sql_filename1 = lowlevel_path + "/" + extra_value_str.c_str();
				QString sql_filename2 = QString(PO_DATABASE_PATH) + "/" + extra_value_str.c_str();

				QDir dir;
				dir.mkpath(PO_DATABASE_PATH);
				if (!QTBase::copyFile(sql_filename1, sql_filename2))
				{
					printlog_lv1(QString("Can't sqlite_db copy because of filecopy is failed. [%1]->[%2]")
									.arg(sql_filename1).arg(sql_filename2));
					return false;
				}
			}
		}
	}

	//update device information file[device.ini]
	if (!getBaseApp()->updateDeviceINISettings(lowlevel_path))
	{
		printlog_lv1("Can't update because of write INI is failed.");
		return false;
	}

	//close self execute file and execute update.exe
	getBaseApp()->exitApplication();

	QString filename;
#if defined(POR_WINDOWS)
{
	filename = update_path;
	filename += PO_UPDATE_MANAGER;
	filename += " " + data_path;
	filename += " " + lowlevel_path;
	filename += " " + getBaseApp()->getLowLevelName();
	if (m_app_update.is_update_highlevel)
	{
		filename += " " + data_path_hl;
		filename += " " + hl_path;
		filename += " " + getBaseApp()->getHighLevelName();
	}

	printlog_lv1(QString("ExcuteCommand: %1").arg(filename));
	printlog_lv1(QString("UpdaterPath: %1").arg(update_path));
	COSBase::executeProcess(filename.toStdString().c_str(), update_path.toStdString().c_str());
}
#elif defined(POR_LINUX)
{
	filename = update_path;
	filename += PO_UPDATE_MANAGER;
	QString low_level_name = getBaseApp()->getLowLevelName();
	QString hl_name = getBaseApp()->getHighLevelName();

    char* argv[8];
    argv[0] = const_cast<char*>(filename.toStdString().c_str());
    argv[1] = const_cast<char*>(data_path.toStdString().c_str());
    argv[2] = const_cast<char*>(lowlevel_path.toStdString().c_str());
    argv[3] = const_cast<char*>(low_level_name.toStdString().c_str());
    argv[4] = const_cast<char*>(data_path_hl.toStdString().c_str());
    argv[5] = const_cast<char*>(hl_path.toStdString().c_str());
    argv[6] = const_cast<char*>(hl_name.toStdString().c_str());
	COSBase::executeProcess((m_app_update.is_update_highlevel ? 7 : 4), argv);
}
#endif

	m_app_update.init();
	getBaseApp()->updateTimerStop();
	return true;
}

void CBaseAppUpdate::deviceUpdateCancel()
{
	getBaseApp()->updateTimerStop();

	QString str_data_path = getBaseApp()->getDevicePath();
	QString strHLDataPath = getBaseApp()->getDeviceHLPath();

	//init application update
	m_app_update.init();

	//delete tempoary files and directory
	QFile::remove(str_data_path+ "/~update0.tmp");
	QFile::remove(str_data_path+ "/~update1.tmp");

	QDir dir = QDir(str_data_path);
	if (dir.cdUp())
	{
		QString str_ll_path = dir.absolutePath() + PO_UPDATE_LLPATH;
		dir = QDir(str_ll_path);
		dir.removeRecursively();
	}

	dir = QDir(strHLDataPath);
	if (dir.cdUp())
	{
		QString strHLPath = dir.absolutePath() + PO_UPDATE_HLPATH;
		dir = QDir(strHLPath);
		dir.removeRecursively();
	}
}

bool CBaseAppUpdate::isUpdateNow(i32 conn)
{
	return (m_app_update.isUpdate() && m_app_update.getFrom() == conn);
}

bool CBaseAppUpdate::isUpdateReady(i32 conn)
{
	return (isUpdateNow(conn) && m_app_update.isUpdateReady());
}

bool CBaseAppUpdate::checkUpdateReady(i32 conn)
{
	if (!m_app_update.isUpdate() || m_app_update.getFrom() != conn)
	{
		return false;
	}

	DeviceInfo* dev_info_ptr = getBaseApp()->getDeviceInfo();
	i32 max_read_count = dev_info_ptr->is_hl_embedded ? 2 : 1;

	if (m_app_update.read_id != max_read_count || m_app_update.read_size != 0)
	{
		printlog_lv1(QString("Can't update... last read id[%1]: read bytes[%2].")
						.arg(m_app_update.read_id).arg(m_app_update.read_size));
		return false;
	}

	m_app_update.is_update_ready = true;
	return true;
}

void CBaseAppUpdate::deviceUpdateConfirm(i32 confirm_delay_ms)
{
	if (m_app_update.isUpdateReady())
	{
		CPOBaseApp* base_app_ptr = getBaseApp();
		base_app_ptr->updateTimerStart(confirm_delay_ms);
	}
}

bool CBaseAppUpdate::deviceUpdate()
{
	i32 conn = m_app_update.getFrom();
	if (isUpdateReady(conn) && deviceUpdateInternal())
	{
		return true;
	}

	printlog_lv1("Update internal operation was failed.");
	deviceUpdateCancel();
	return false;
}
