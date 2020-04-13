import numpy as np
import datetime as dt
import struct, os
import subprocess as commands
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def get_data_nsidc(filename):
	year, month, day = int(filename[3:7]), int(filename[7:9]), int(filename[9:11])
	date = dt.date(year, month, day)
	icefile = open(filename)
	contents = icefile.read()
	icefile.close()
	# unpack binary data into a flat tuple z
	source = filename[:2]
	if source=="nt": #nasa
		s="%dB" % (int(width*height),)
		z=struct.unpack_from(s, contents, offset = 300)#was 300
		nsidc = np.array(z).reshape((332,316))
		nsidcdata = np.rot90(nsidc, 2)/2.5
	elif source=="bt": #bootstrap
		icefile = open(filename)
		contents = np.fromfile(icefile, dtype='<i2')
		nsidc = np.array(contents).reshape((332,316))
		nsidcdata = np.rot90(nsidc, 2)/10.
	else:
		print("please cd into the directory of the file. files should start with either bt_ or nt_")
	return nsidcdata, date

def get_data_asi(filename):
	datagroup = Dataset(filename, "r")
	time = datagroup["time"][:][0]
	zz = datagroup["sea_ice_area_fraction"][0]
	lands = datagroup["land"][:]
	date1901 = dt.datetime(1900,1,1,0,0,0)
	hours = dt.timedelta(0,0,0,0,0,time)
	realdate = (date1901+hours).date()
	return zz, realdate



def plot_map_asi(ZZ, date, show=True, contours=True):
	plt.clf()
	m = Basemap(projection='spstere',boundinglat = -60, lon_0=-180,resolution='h')
	m.drawcoastlines(linewidth=3, linestyle = '-', color='w')
	m.drawmapboundary(fill_color='navy')
	m.drawparallels(np.arange(-91,-30,15),labels=[False,False,True,True])
	m.drawmeridians(np.arange(0,177,15),labels=[False,False,False,False], xoffset=1.00, yoffset=1.00)

	#test if the lat/lon grid is downloaded yet, if not then download:
	if os.path.isfile("19911205_median5day.nc")==False:
		commands.getoutput("wget ftp://ftp.icdc.zmaw.de/asi_ssmi_iceconc/ant/1991/19911205_median5day.nc") #get lat/lon grid from ASI

	with Dataset("19911205_median5day.nc", "r") as datagroup:
		lats = datagroup["latitude"][:]
		lons = datagroup["longitude"][:]
	mlons, mlats = m(lons, lats)
	XX, YY = mlons, mlats
	if contours:
		plt.contour(XX,YY,ZZ,alpha=1, cmap = pylab.cm.jet)
	plt.pcolormesh(XX,YY,ZZ, cmap = pylab.cm.YlGnBu, alpha=1)
	plt.tight_layout( rect=[0,0.06,1,1])
	cbar = plt.colorbar(orientation="horizontal", fraction=.03, aspect=46, pad=.07)
	cbar.set_label("% ice cover")
	plt.legend(loc=2)
	if show:
		plt.show()
	else:
		pass

	return

def plot_map_nsidc(ZZ, date, show=True, contours=True):
	plt.clf()
	m = Basemap(projection='spstere',boundinglat = -60, lon_0=-180,resolution='h')
	m.drawcoastlines(linewidth=3, linestyle = '-', color='w')
	m.drawmapboundary(fill_color='navy')
	m.drawparallels(np.arange(-91,-30,15),labels=[False,False,True,True])
	m.drawmeridians(np.arange(0,177,15),labels=[False,False,False,False], xoffset=1.00, yoffset=1.00)

	#test if the lat/lon grid is downloaded yet, if not then download:
	if os.path.isfile("pss25lons_v3.dat")*os.path.isfile("pss25lats_v3.dat")==False:
		commands.getoutput("wget ftp://sidads.colorado.edu/pub/DATASETS/seaice/polar-stereo/tools/pss25lats_v3.dat") #get the lat/lon grid from NSIDC
		commands.getoutput("wget ftp://sidads.colorado.edu/pub/DATASETS/seaice/polar-stereo/tools/pss25lons_v3.dat") #get the lat/lon grid from NSIDC
		commands.getoutput("wget ftp://sidads.colorado.edu/pub/DATASETS/seaice/polar-stereo/tools/pss25area_v3.dat") #get the lat/lon grid from NSIDC
	with open("pss25lons_v3.dat", "rb") as file:
		lons  = np.fromfile(file, dtype= np.dtype("(332,316)i4"))[0]
		lons = np.rot90(lons, 2)/100000.
	with open("pss25lats_v3.dat", "rb") as file:
		lats  = np.fromfile(file, dtype= np.dtype("(332,316)i4"))[0]
		lats = np.rot90(lats,2)/100000.
	with open("pss25area_v3.dat", "rb") as file:
		areas  = np.fromfile(file, dtype= np.dtype("(332,316)i4"))[0]
		areas = np.rot90(areas,2)/1000.
	mlons, mlats = m(lons, lats)
	XX, YY = mlons, mlats
	if contours:
		plt.contour(XX,YY,ZZ,alpha=1, cmap = 'jet')
	plt.pcolormesh(XX,YY,ZZ, cmap = 'YlGnBu', alpha=1)
	plt.tight_layout( rect=[0,0.06,1,1])
	cbar = plt.colorbar(orientation="horizontal", fraction=.03, aspect=46, pad=.07)
	cbar.set_label("% ice cover")
	plt.legend(loc=2)
	if show:
		plt.show()
	else:
		pass

	return

if __name__ == '__main__':
	print("note that the plot_map_i function is specifically for the south pole")
	pass
