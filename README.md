# Ego-Lane-fitting-Pointclouds

For personal project, I've decided to implement ego-lane polynomial fitting using raw binary files of pointcloud data earned from lidar. 
Along with, Python code, written and tested with Jupyter notebook, I added each part of code as detailed
as possible from data pre-processing to the self-made algorithm to extract polynomial fitting. 

The methods in this paper is largely divided into two parts: data pre-processing and
lane extraction. Individual point could be represented as 5 attributes: x, y, z coordinates,
intensity, and LiDAR beam number. Each raw point cloud binary file is consisted of roughly
in the order of 10,000 points, which could lead to a huge burden if we try to deal with all
those points. To avoid such catastrophe and for the sake of robust and efficient polynomial
fitting of ego lanes, removing unnecessary points while maximally trying to preserve essential
points in lane detection is a mandatory process. After going through cleaning procedure, we
should implement certain algorithms to select specific points that belong to ego lanes, such
that we could use those selected points to achieve 3-degree polynomial fitting. Flow chart
of my algorithm is shown as below figure:
