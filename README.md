# cryo-toolbox
Public repo for Python modules which may be useful for parsing/plotting sea ice data (e.g. from NSIDC, seismic traces)

Overview of modules:
- nsidc_sic is a way to visualize NSIDC sea ice concentration data for the south pole. A [demo](sic_demo) is available.
- parse_nsidc is a easy way to download files from via FTP from earthlink ([NSIDC](www.nsidc.org)'s offical data source). All you need to do is specify the URL where the data is, and the extension (* if you want all files, otherwise you may only want the jpg/h5 files and not the xml, for example)
- Lead-finding [demo](leadfind_demo) is demonstrating how to obtain sea surface elevation estimates from the NSIDC L1 ATM data (which you can download with parse_nsidc...!)  The lidar data is available at https://nsidc.org/data/ILATM1B/versions/1# and the camera imagery at http://nsidc.org/data/docs/daac/icebridge/iodms1b/index.html. 
- Textural segmentation [demo](textureseg_demo) shows how to segment a (sea-surface-referenced) surface elevation scan, and using this information to extrapolate snow depth measurements. This work has been submitted to the J. Remote Sensing.
- Floe segmentation [demo](segment_demo.gif) - this uses OpenCV to make an interactive GUI - unfortunately this means I cannot make a Jupyter Notebook for this. However, you can click the demo link to see an animation of the GUI in action. You can in theory use any image, just run the script as "python fsd_code.py <path/to/file.jpg>". 
