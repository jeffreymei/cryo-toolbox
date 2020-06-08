This code is what is used to segment the snow surface (lidar scan) from the IceBridge lead-referenced L1-ATM surface elevation data (https://nsidc.org/data/ILATM1B/versions/1#). Check out the Jupyter [Notebook](Segmentor_Demo.ipynb) to get started.

The snow depth measurements were provided by Ron Kwok. The extrapolation of the snow depth allows for a more accurate mean snow depth over the lidar window (instead of taking just the mean of the snow points within that window, which may be biased). 

Many thanks to Adam Lefaivre for his implementation of textural segmentation with Gabor filters (https://github.com/lefaivre/textureSegmentation).
