import xml, os
import numpy as np
import time, pickle
try: import commands
except: import subprocess as commands
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.image as mpi
import glob
from scipy.interpolate import griddata

def gettime(x):
	return datetime.fromtimestamp(x).time()

def dist_between(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Source: http://gis.stackexchange.com/a/56589/15183
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    km = 6367 * c
    return km


def q_xml_decode(filelist, step=1):
	'''
	get lon, lat, timestamps for qfit file
	'''
	print (str(len(filelist))+" number of qfit files")
	minlons = np.zeros(len(filelist))/step
	minlats = minlons.copy() #space
	maxlats = minlons.copy()
	maxlons = minlons.copy()
	starts = minlons.copy() #time
	ends = starts.copy()
	for i in range(int(len(filelist)/step)):
		#XML of qi, 50 of these
		tree = ET.parse(filelist[i*step])
		root = tree.getroot()
		starttime = root.find(".//RangeBeginningTime").text
		strpstart = datetime.strptime(starttime, "%H:%M:%S.%f").replace(day=30, month=10, year=2010)
		endtime = root.find(".//RangeEndingTime").text
		strpend = datetime.strptime(endtime, "%H:%M:%S.%f").replace(day=30, month=10, year=2010)

		startsec = time.mktime(strpstart.timetuple())
		endsec = time.mktime(strpend.timetuple())
		starts[i] = startsec
		ends[i] = endsec

		alllons = root.findall(".//PointLongitude")
		alllats = root.findall(".//PointLatitude")
		storelon = np.zeros(len(alllons))
		storelat = storelon.copy()

		for j in range(len(alllons)):
			storelon[j] = float(alllons[j].text)
			storelat[j] = float(alllats[j].text)


		minlons[i] = min(storelon)
		minlats[i] = min(storelat)
		maxlons[i] = max(storelon)
		maxlats[i] = max(storelat)
	qi_data = [minlons, minlats, maxlons, maxlats, starts, ends]

	return qi_data

def im_xml_decode(filelist, datenum, step=1):
	'''
	get lon, lat, timestamps for image xmls. date format is YYYY.MMDD
	'''
	print (str(len(filelist))+ " number of image files")
	lons2 = np.zeros(len(filelist))/step
	lats2 = lons2.copy()
	starts2 = lons2.copy()
	ends2 = starts2.copy()
	inds = starts2.copy()
	lon_start = np.zeros(len(filelist))/step
	lon_stop = lon_start.copy()
	lat_start = lon_start.copy()
	lat_stop = lon_start.copy()
	lonsall2 = np.zeros(( len(filelist), 4))
	latsall2 = np.zeros(( len(filelist), 4))
	for i in range(int(len(filelist)/step)):
		#XML of images, 11000 of these

		tree2 = ET.parse(filelist[i*step])
		root2 = tree2.getroot()
		starttime2 = root2.find(".//TimeofDay").text[:8] #used to have [:-2]#  but why?
		year = int(np.floor(datenum))
		month = int(np.floor(100*(datenum-year)))
		day = int(np.floor(100*(100*(datenum-year)-month)))
		strpstart2 = datetime.strptime(starttime2, "%H:%M:%S").replace(day=day, month=month, year=year)
		startsec2 = time.mktime(strpstart2.timetuple())
		starts2[i] = startsec2

		allpoints = root2.findall(".//Point/*")
		alllats = root2.findall(".//PointLatitude")
		cumsum = 0
		for j in range(len(alllats)):
			value = float(alllats[j].text)
			latsall2[i][j] = value
			cumsum += value
		midpointlat = cumsum/4. 
		lat_start[i] = np.min(latsall2[i])
		lat_stop[i] = np.max(latsall2[i])

		alllons = root2.findall(".//PointLongitude")
		cumsum = 0
		for j in range(len(alllons)):
			value = float(alllons[j].text)
			cumsum += value
			lonsall2[i][j] = value
		midpointlon = cumsum/4. 
		
		lon2 = allpoints[0].text
		lat2 = allpoints[1].text
		lons2[i] = midpointlon
		lats2[i] = midpointlat
		lon_start[i] = np.min(lonsall2[i])
		lon_stop[i] = np.max(lonsall2[i])
	start_start = starts2[0]
	im_data = [lat_start, lon_start, lat_stop, lon_stop, starts2, ends2, start_start, lonsall2, latsall2]
	return im_data