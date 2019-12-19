#pragma once

#define POR_DEVICE
#define POR_WINDOWS
//#define POR_LINUX

#define POR_WITH_LOG
#define PO_LOG_FILESIZE					4096000	//about 4MB

#define PO_DEVICE_ID					1001
#define PO_DEVICE_NAME					"MVPorIPDev"
#define PO_DEVICE_MODELNAME				"MV2017FD"
#define PO_DEVICE_VERSION				"1.00"
#define PO_DEVICE_CAMPORT				"UNKPORT"

#define PO_MYSQL_HOSTNAME				"localhost"
#define PO_MYSQL_DBFILEPATH				"mvs_sqllite.db"
#define PO_MYSQL_DATABASE				"mvs_db"
#define PO_MYSQL_USERNAME				"root"
#define PO_MYSQL_PASSWORD				"123456"
#define PO_MYSQL_REMOTEUSERNAME			"mvs_user"
#define PO_MYSQL_REMOTEPASSWORD			"123456"
#define PO_MYSQL_PORT					3306
#define PO_MYSQL_RECLIMIT				1000
#define PO_MYSQL_LOGLIMIT				100000

#define PO_IOCOM_ADDR					0
#define PO_IOCOM_PORT					"COM1"
#define PO_IOCOM_BAUDBAND				115200
#define PO_IOCOM_DATABITS				8
#define PO_IOCOM_PARITY					0
#define PO_IOCOM_STOPBITS				1
#define PO_IOCOM_FLOWCTRL				0

#ifdef POR_WINDOWS
#ifdef POR_DEBUG
#pragma comment(lib, "opencv_core340d.lib")
#pragma comment(lib, "opencv_imgproc340d.lib")
#pragma comment(lib, "opencv_highgui340d.lib")
#pragma comment(lib, "opencv_imgcodecs340d.lib")
#pragma comment(lib, "opencv_imgproc340d.lib")
#pragma comment(lib, "opencv_video340d.lib")
#pragma comment(lib, "opencv_calib3d340d.lib")
#else
#pragma comment(lib, "opencv_core340.lib")
#pragma comment(lib, "opencv_imgproc340.lib")
#pragma comment(lib, "opencv_highgui340.lib")
#pragma comment(lib, "opencv_imgcodecs340.lib")
#pragma comment(lib, "opencv_imgproc340.lib")
#pragma comment(lib, "opencv_video340.lib")
#pragma comment(lib, "opencv_calib3d340.lib")
#endif
#endif
