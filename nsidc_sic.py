import numpy as np
import datetime as dt
import struct, os
import subprocess as commands
from netCDF4 import Dataset
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import matplotlib.path as mpath

def get_data_nsidc(filename):
    year, month, day = int(filename[3:7]), int(filename[7:9]), int(filename[9:11])
    date = dt.date(year, month, day)
    with open(filename, 'rb') as icefile:
        contents = icefile.read()
    # unpack binary data into a flat tuple z
    source = filename[:2]
    if source=="nt": #nasa
        s="%dB" % (int(width*height),)
        z=struct.unpack_from(s, contents, offset = 300)#was 300
        nsidc = np.array(z).reshape((332,316))
        nsidcdata = np.rot90(nsidc, 2)/2.5
    elif source=="bt": #bootstrap
        with open(filename, 'rb') as icefile:
            contents = np.fromfile(icefile, dtype='<i2')
        nsidc = np.array(contents).reshape((332,316))
        nsidcdata = np.rot90(nsidc, 2)/12.
    else:
        print("please cd into the directory of the file. files should start with either bt_ or nt_")
    print("Loaded data from date {}".format(date))
    return nsidcdata, date

def plot_map_nsidc(data, date):
    plt.clf()
    polar_crs = ccrs.SouthPolarStereo()
    plain_crs = ccrs.PlateCarree()
    polar_extent = [-180, 180, -90, -60]
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1, projection=polar_crs)
    ax.axis('off')
    ax.set_extent(polar_extent, crs=plain_crs)
    ax.coastlines(resolution='50m', color='k')
    ax.gridlines(linestyle='--', draw_labels=True)
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
    data[lats>-61] = np.nan
    t = ax.pcolormesh(lons, lats, data, transform = ccrs.PlateCarree(), cmap='Blues_r')
    cbar = plt.colorbar(t, orientation="horizontal", fraction=.03, aspect=46, pad=.07)
    cbar.set_label("% ice cover")
    plt.title("Sea ice concentration on {}".format(date))
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    return ax
if __name__ == '__main__':
    print("note that the plot_map_i function is specifically for the south pole")
    pass

# below is defunct as ASI moved their data around and I can't find where they store it anymore.
def get_data_asi(filename):
    datagroup = Dataset(filename, "r")
    time = datagroup["time"][:][0]
    zz = datagroup["sea_ice_area_fraction"][0]
    lands = datagroup["land"][:]
    date1901 = dt.datetime(1900,1,1,0,0,0)
    hours = dt.timedelta(0,0,0,0,0,time)
    realdate = (date1901+hours).date()
    return zz, realdate

# def plot_map_asi(ZZ, date, show=True, contours=True):
#   plt.clf()
#   polar_crs = ccrs.SouthPolarStereo()
#   plain_crs = ccrs.PlateCarree()
#   polar_extent = [-180, 180, -90, -60]

#   fig = plt.figure(figsize=(6, 6))
#   ax = fig.add_subplot(1, 1, 1, projection=polar_crs)
#   ax.set_extent(polar_extent, crs=plain_crs)
#   ax.coastlines(resolution='50m', color='k')
#   ax.gridlines(color='lightgrey', linestyle='-', draw_labels=True)

#   #test if the lat/lon grid is downloaded yet, if not then download:
#   if os.path.isfile("19911205_median5day.nc")==False:
#       commands.getoutput("wget ftp://ftp.icdc.zmaw.de/asi_ssmi_iceconc/ant/1991/19911205_median5day.nc") #get lat/lon grid from ASI

#   with Dataset("19911205_median5day.nc", "r") as datagroup:
#       lats = datagroup["latitude"][:]
#       lons = datagroup["longitude"][:]
#   mlons, mlats = m(lons, lats)
#   XX, YY = mlons, mlats
#   if contours:
#       plt.contour(XX,YY,ZZ,alpha=1, cmap = pylab.cm.jet)
#   plt.pcolormesh(XX,YY,ZZ, cmap = pylab.cm.YlGnBu, alpha=1)
#   plt.tight_layout( rect=[0,0.06,1,1])
#   cbar = plt.colorbar(orientation="horizontal", fraction=.03, aspect=46, pad=.07)
#   cbar.set_label("% ice cover")
#   plt.legend(loc=2)
#   if show:
#       plt.show()
#   else:
#       pass

#   return