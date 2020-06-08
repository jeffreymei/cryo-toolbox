This is a demonstration of the lead finding algorithm that is used to reference the L1 ATM elevation data from NSIDC (https://nsidc.org/data/ILATM1B/versions/1#) to the local sea surface height. The resulting data that uses this method is in an article [Mei & Maksym 2020](https://www.mdpi.com/2072-4292/12/9/1494) recently submitted to J. Remote Sensing. Check out the [notebook](Lead_Finding_Demo.ipynb) to get started. 

All the data that is needed for this is in the Data/ directory. They are taken from NSIDC, and somewhat truncated (as the files are very big).

All the modules should be available in a standard installation of Anaconda (https://www.anaconda.com/distribution/), except for OpenCV possibly. Please make sure you have OpenCV version 3+ (easily installed with pip; `pip install opencv-python`).
