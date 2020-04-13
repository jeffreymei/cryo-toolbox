# cryo-toolbox
Public repo for Python modules which may be useful for parsing/plotting sea ice data (e.g. from NSIDC, seismic traces)

Overview of modules:
- nsidc_sic is a way to visualize NSIDC sea ice concentration data for the south pole.
It assumes that the Basemap module has already been installed; if this is not the case, an easy way to install it is using the Anaconda distribution of Python. This is a bit outdated as basemap is no longer supported (instead, cartopy is now standard).
- parse_nsidc is a easy way to download files from via FTP from earthlink (NSIDC's offical data source). All you need to do is specify the URL where the data is, and the extension (* if you want all files, otherwise you may only want the jpg/h5 files and not the xml, for example)
- leadfinding demo is demonstrating how to obtain sea surface elevation estimates from the NSIDC L1 ATM data (which you can download with parse_nsidc...!)  The lidar data is available at https://nsidc.org/data/ILATM1B/versions/1# and the camera imagery at http://nsidc.org/data/docs/daac/icebridge/iodms1b/index.html.
- textural segmentation demo shows how to segment a (sea-surface-referenced) surface elevation scan, and using this information to extrapolate snow depth measurements. This work has been submitted to the J. Remote Sensing.
