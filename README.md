# Ego-Lane-fitting-Pointclouds

## 1. Introduction
For a personal project, I've decided to implement ego-lane polynomial fitting using raw binary files of pointcloud data earned from lidar. (I obtained total 11 binary files, which I named them from scene 1 to scene 11 for the sake of convenience). Along with, Python code, written and tested with Jupyter notebook, I added each part of code as detailed
as possible from data pre-processing to the self-made algorithm to extract polynomial fitting. 

## 2. Workflow 
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

![workflow](https://user-images.githubusercontent.com/82307352/167679010-29f7d21b-90a2-4365-9e14-c2fad402ae21.jpg)

Along with source codes, following subsections will explain which filters I used for data
pre-processing, how and why I designed filters in such manner, and which algorithm did I
implement for ego lane extractions. 

There are several user-defined hyper parameters that are used throughout the
process of polynomial fitting process. These were tuned and managed by a single configuration file. Below is the source code including a short description of each hyper paramter,
constructing a configuration file, and parsing it to hyper parameters that will be used in
later codes.

```python
import configparser

def config_generator():
    '''
    %% Generate config files for 11 scenes
    Parameters
    ----------
    1. rot_flag: Decide whether rotate the coordinate system or not
            type: boolean
    2. width_flag: Decide whether implement crosswalk filtering or not
            type: boolean
    3. z_flag: Decide whether implement Z_filter or not
            type: boolean
    4. width_threshold: Threshold value of deciding whether it is crosswalk or not
            type: float
    5. z_threshold: Lower boundary of height values of points to be filtered
            type: int or float
    6. rot_angle: Amount of angle to be rotated
            type: int or float
    7. inten_up_bound: Upper boundary of intensity value for intensity filter
            type: int or float
    8. inten_low_bound: Lower boundary of intensity value for intensity filter
            type: int or float
    9. eps: Parameter for DBSCAN
            type: float
    10: min_samples: Parameter for DBSCAN
            type: int 
    ----------
    '''
    config = configparser.ConfigParser()

    config['scene1'] = {}
    config['scene1']['rot_flag'] = 'False'
    config['scene1']['width_flag'] = 'False'
    config['scene1']['z_flag'] = 'False'
    config['scene1']['width_threshold'] = '0'
    config['scene1']['z_threshold'] = '0'
    config['scene1']['rot_angle'] = '0'
    config['scene1']['inten_up_bound'] = '18'
    config['scene1']['inten_low_bound'] = '12'
    config['scene1']['eps'] = '0.08'
    config['scene1']['min_samples'] = '5'
    ...
    ...
    ...
    with open('lidar_config.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)
        
config_generator()

def get_params_from_config(config, scene_number):
    '''
    %% return tuple of parameters of specific scene
    '''
    scene_number = str(scene_number)
    
    rot_flag = eval(config.get('scene'+scene_number, 'rot_flag'))
    width_flag = eval(config.get('scene'+scene_number, 'width_flag'))
    z_flag = eval(config.get('scene'+scene_number, 'z_flag'))
    width_threshold = float(config.get('scene'+scene_number, 'width_threshold'))
    z_threshold = float(config.get('scene'+scene_number, 'z_threshold'))
    rot_angle = float(config.get('scene'+scene_number, 'rot_angle'))
    inten_up_bound = float(config.get('scene'+scene_number, 'inten_up_bound'))
    inten_low_bound = float(config.get('scene'+scene_number, 'inten_low_bound'))
    eps = float(config.get('scene'+scene_number, 'eps'))
    min_samples = int(config.get('scene'+scene_number, 'min_samples'))
    
    return (rot_flag, width_flag, z_flag, width_threshold, z_threshold, 
            rot_angle, inten_up_bound, inten_low_bound, eps, min_samples)

config = configparser.ConfigParser()
config.read('lidar_config.ini')
scene_number = 1 
rot_flag, width_flag, z_flag, width_threshold, z_threshold, rot_angle, inten_up_bound, inten_low_bound, eps, min_samples = get_params_from_config(config, scene_number)

```
The end part of the above process returns a tuple that carries hyper parameters that we
are going to use for data pre-processing, which will be shown in the upcoming section.

Below is the python library that I used throughout ego lane detection implementation:
```python
import os
import time
import itertools
import configparser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from scipy.optimize import fsolve
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
```

## Data Pre-processing
### 2.1.1 Coordinate Rotation
Among 11 scenes, scene 6 and 8 seemed to be required to go through a rotation of its
coordinate system in order to easily apply later filters. These scenes seem to be the moment
when the driver was trying to change the lane. Rotation seemed unavoidable due to the
following spatial filters since it selects the region of interest which relies on the axis of
driving (x-axis, red axis in the image).

**Below is the cropped-image of original pointcloud of scene 8 (left) and after rotation (right)**
![scene8_rotation](https://user-images.githubusercontent.com/82307352/167682700-940ac37b-dad0-4620-8cb9-84c19e3e393e.png)

```python
def rotation(lidar_data, rot_flag, rot_angle):
    '''
    %% rotate the coordinate system of the given lidar point-cloud
    1. Input:
        - lidar_data
            type: np.ndarray
            shape: (n, 5) where n denotes number of points in one lidar point-cloud data
        - rot_flag
            type: boolean
        - rot_angle
            type: int or float
    2. Return:
        - rotated_lidar_data
            type: np.ndarray
            shape: (n, 5) where n denotes number of points in one lidar point-cloud data
    '''
    rotated_lidar_data = lidar_data
    
    if rot_flag == True:
        rot_angle = rot_angle*np.pi/180
        cos = np.cos(rot_angle)
        sin = np.sin(rot_angle)
        rot_matrix = np.array([[cos, -sin, 0],
                              [sin, cos, 0],
                              [0, 0, 1]])
        for data in rotated_lidar_data:
            data[:3] = np.dot(rot_matrix, data[:3])
    
    return rotated_lidar_data
```
Above is the source code of the function of coordinate system rotation. With the input of
rotation angle and rotation flag, which decides whether a certain scene requires coordinate
rotation, rotation function derives rotation matrix around z-axis and consequently multiply
it with the original input LiDAR point cloud. For scene 6 and 8, rotation angle
of 10◦ and 44◦ was applied respectively.

### 2.1.2 Spatial Filter
After rotation, spatial filter only preserves points within the region of interest and disregard
the rest. For spatial filter, I chose the region of interests based on the following assumption.
For the autonomous driving purpose, we do not need information that is too far away from the
driver. Most important information is the points that are nearby, but how near? According
to the Korean road traffic law, the minimum safe distance is calculated by subtracting 15
from the speed. For example, if a driver is moving in 80km/h, than the minimum safe
distance is 80-15=65m. Also, by law, the interval between lanes is from 3m to 3.5m, and the
width of each lane is from 10cm to 15cm, depending on the road. From these information,
before applying spatial filter, I noticed that the dimensions of point cloud data are in metre
as I looked the given sample output. The y-intercept difference between two lines is 3.2,
which closely matches the interval between lanes from the road traffic law.

```python
def spatial_filter(lidar_data):
    '''
    %% Choose region of interest of the given lidar point-cloud
    1. Input: 
        - lidar_data
            type: np.ndarray
            shape: (n, 5) where n denotes number of points in one lidar point-cloud data
    2. Return: 
        - filtered_lidar_data
            type: np.ndarry
            shape: (m, 5) where m denotes number of remaining points after filter
    3. Parameters
    ----------
    xAmp: boundary along x-axis: keep points where -xAmp <= x_coordinate <= xAmp
            type: int or float
    yAmp: boundary along y-axis: keep points where -yAmp <= y_coordinate <= yAmp
            type: int or float
    ----------
    '''
    xAmp = 40
    yAmp = 3.9
    
    filtered_lidar_data = np.delete(lidar_data, np.where(
        (lidar_data[:, 0] >= xAmp) | (lidar_data[:, 0] <= -xAmp))[0], axis = 0)
    
    filtered_lidar_data = np.delete(filtered_lidar_data, np.where(
        (filtered_lidar_data[:, 1] >=yAmp) | (filtered_lidar_data[:, 1] <= -yAmp))[0], axis = 0)
   
    print("Number of xy-filtered data points = %d" %(len(lidar_data)-len(filtered_lidar_data)))
    print("Remaining data = %d" % (len(filtered_lidar_data)))
    return filtered_lidar_data
```
Assuming that the car is moving at speed of 55km/h, I set up the region of interest of
x-axis (driving axis) from -40 to 40 based on the calculation mentioned before.
Also for y axis, I chose to have points included from -3.9 to 3.9, which can includes
points of neighboring lanes. Then, I removed the points from the original data that are not
in the region of interests. All 11 scenes shared same x and y region of interest.
For z-axis, which could be seemed somewhat irrelevant to driving information, I also made
a filter that filters out points that are above the max height. Despite x and y coordinate
filters, there were unnecessary points which were located higher than the road (eg: speed
bumps, crash barriers, and central reservations).

**Below is the cropped-image of scene 6 pointcloud data. Red-dotted circle shows the area located higher than the lane.** 
![그림2](https://user-images.githubusercontent.com/82307352/167683315-b34900bb-2268-47a6-a994-da995a86ec82.jpg)

```python
def z_filter(lidar_data, z_flag, z_threshold):
    '''
    %% Delete points that are located higher than the z_threshold
    1. Input: 
        - lidar_data
            type: np.ndarray
            shape: (n, 5) where n denotes number of points in one lidar point-cloud data
        - z_flag
            type: boolean
        - z_threshold
            type: int or float
    2. Return: 
        - filtered_lidar_data
            type: np.ndarry
            shape: (m, 5) where m denotes number of remaining points after filter
    '''
    filtered_lidar_data = lidar_data
    
    if z_flag == True:
        filtered_lidar_data = np.delete(lidar_data, np.where(
        lidar_data[:, 2] >= z_threshold)[0], axis = 0)
    
    return filtered_lidar_data
```
Through z-filter function, I eliminated points above the given threshold for scene 6 and 7.
Threshold values were manually selected after multiple trials. 

### 2.1.3 Intensity Filter
Intensity of point clouds plays a significant role in distinguishing lane markings from unnecessary information such as asphalt, and the range of intensity is 0 to 255. Due to high
reflectivity of the road paint, lane marking points tend to have higher intensity compared to
that of other objects. However, there is no absolute standard of intensity values for each material: they might have a rough range but highly depend on other variables such as weather
condition or types of detectors.

Below is the intensity historgram of all 11 scenes. Most of intensity values are around 0 to 50.
![intensity histogram](https://user-images.githubusercontent.com/82307352/167683577-0642ff2b-5307-4774-9181-7f44f9db9345.jpg)
```python
def intensity_filter(lidar_data, inten_up_bound, inten_low_bound):
    '''
    %% Choose points within intensity boundary
    1. Input: 
        - lidar_data
            type: np.ndarray
            shape: (n, 5) where n denotes number of points in one lidar point-cloud data
        - inten_up_bound
            type: int or float
        - inten_low_bound
            type: int or float
    2. Return: 
        - filtered_lidar_data
            type: np.ndarry
            shape: (m, 5) where m denotes number of remaining points after filter
    '''
    filtered_lidar_data = np.delete(lidar_data, np.where(
        (lidar_data[:, 3] <= inten_low_bound) | (lidar_data[:, 3] >= inten_up_bound))[0], axis = 0)
    
    print("Number of filtered data points = %d" %(len(lidar_data)-len(filtered_lidar_data)))
    print("Remaining data = %d" % (len(filtered_lidar_data)))
    return filtered_lidar_data
```
Variable ’inten up bound’ and ’inten low bound’ decides the upper and lower boundary of
intensity values respectively. Then, I eliminated points that are off boundary. Based on the intensity histogram of each scene, I carefully chose ’inten up bound’ and ’inten low bound’ hyper parameters to maximally preserve points belonging to lane markings while eliminating others.

```python
data_folder = 'C:/Users/User/Desktop/Lidar_data/raw_pointclouds'
lidar_files = sorted(os.listdir(data_folder))
lidar_fpath = [os.path.join(data_folder, f) for f in lidar_files]
lidar_bin = lidar_fpath[0]
lidar_data = np.fromfile(lidar_bin, dtype=np.float32).reshape(-1, 5)

#Data filtering 
print("Original datapoint = %d points" %len(lidar_data))
lidar_data = rotation(lidar_data, rot_flag, rot_angle)
lidar_data = z_filter(lidar_data, z_flag, z_threshold)
lidar_data = spatial_filter(lidar_data)
lidar_data = intensity_filter(lidar_data, inten_up_bound, inten_low_bound)
print("After filtering = %d points remaining" %(len(lidar_data)))
```
Above filters seemed to be enough for pre-processing the point cloud at first glance, but
there was another issue that cannot be solved with above filters: a crosswalk. Crosswalk is
obviously not a lane mark, yet it is also drawn with white paint so that its intensity value is
similar to that of lane marks. Below figure shows the pre-processing results of two scenes,
one with crosswalk and one that does not.

**Below is the cropped-image of original pointcloud (right) and pointcloud after fitering processes (left) of scene 9. Red-dotted circle shows the crosswalk points that are not filtered.** 
![scene8_filtered_yet](https://user-images.githubusercontent.com/82307352/167684080-5e642fc5-89a0-470a-8554-ff520de35c46.png)

There must be other variables to be considered to get rid of crosswalk points, and according to Korean road traffic law, the width of the crosswalk lane is from 67.5cm to 75cm,
whereas that of lane mark is from 10cm to 15cm. Using this information, we could get rid of
groups of points that have a longer width along y-axis, suspecting that those would belong
to a crosswalk. In order to do this, I implemented a Density-based spatial clustering of
applications with noise (DBSCAN) algorithms, which would be explained in the following
section.

### 2.1.4 Crosswalk Filter
DBSCAN is one of data clustering algorithms that groups the points that are spatially close
and marking outlier points whose nearest neighbors are too far away. By using DBSCAN
algorithm, I could achieve both eliminating outliers that are not filtered previously and
earning cluster information that are needed to eliminate crosswalk points.

```python 
lidar_xyz = lidar_data[:, :3]
clustering = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(lidar_xyz)
print("Number of cluster = %d" %(max(clustering)+1))

outlier_index = []
cluster_info = []

for lidar_data_index, cluster_numbering in enumerate(clustering):
    if cluster_numbering == -1:
        outlier_index.append(lidar_data_index)
    else:
        cluster_info.append([cluster_numbering])        
print("Number of outlier = %d" %len(outlier_index))
filt_clust_data = np.delete(lidar_data, outlier_index, axis=0)
print("Remaining data = %d" %len(filt_clust_data))

#crosswalk-elimination
if width_flag == True:
    max_width_list = []
    lidar_data_withLabel = np.append(filt_clust_data, cluster_info, axis =1) 
    lidar_data_withLabel = lidar_data_withLabel[np.argsort(lidar_data_withLabel[:, 5])]
    for cluster_index in range(max(clustering)+1):
        lidar_inCluster = lidar_data_withLabel[lidar_data_withLabel[:, -1] == cluster_index]
        max_width = 0
        max_index = (0, 0)
        for index_one_point in range(len(lidar_inCluster) -1): 
            for index_theOther_point in range(1, len(lidar_inCluster)-1): 
                width = np.linalg.norm((lidar_inCluster[index_theOther_point]-lidar_inCluster[index_one_point])[:3])
                if width > max_width:
                    max_width = width
                    max_idx = (index_one_point, index_theOther_point)
        max_width_list.append(max_width)

    forbidden_cluster = []
    
    for index, width in enumerate(max_width_list):
        if width >= width_threshold:
            forbidden_cluster.append(index)
    
    for cluster_index in forbidden_cluster:
        lidar_data_withLabel = lidar_data_withLabel[lidar_data_withLabel[:, -1] != cluster_index]
    lidar_LabelRemoved = lidar_data_withLabel[:, :-1].astype(np.float32)
    filt_clust_data = lidar_LabelRemoved
    print("Remaining data = %d" %len(filt_clust_data))
    
fh = open("scene1.bin", "bw")
filt_clust_data.tofile(fh)
```

First, I extracted spatial information of points and implemented DBSCAN algorithm provided by scikit-learn packages. According to scikit-learn, DBSCAN algorithm
returns value -1 if the point is an outlier, and other int values based on which cluster the
point belongs to. I collected the indices of points that belong to outliers and remove them
from the original point clouds.

**Below is the cropped-image of scene9 before and after crosswalk elimination through DBSCAN clustering algorithm. Notice that points inside the red-dotted circle were removed.**
![scene9](https://user-images.githubusercontent.com/82307352/167685290-33b8de9c-5e88-452b-80c7-3c5d9a4a05ab.jpg)

After eliminating outliers and possessing cluster information of individual points, I measured the maximum distance in each cluster by measuring every single euclidean distance of
points within the same cluster. Then, I filtered so called ’forbidden clusters’
that has maximum distance longer than the width threshold (width threshold was set from
0.1 to 0.2 which is the approximate width of lane mark mentioned previously).
15
DBSCAN clustering was conducted to all 11 scenes whereas crosswalk filtering was only conducted to scene 4, 6, 7, and 9. Hyper parameters for DBSCAN (eps and min samples), and
width threshold were carefully tuned to achieve best results. Above image shows the results of
DBSCAN and crosswalk elimination of scene 9.

## 2.2 Lane Extraction
After filtering out seemingly unnecessary points from raw data, number of points reduced
from the order of few 10,000 to few 100. Also, since we need to conduct the polynomial
fitting in x-y plane, I decided to put value 0 to z coordinates of filtered points, assuming the
car is on the flatland with a negligible slope. Then, for lane detection, I decided to implement
RANdom SAmple Consensus (RANSAC) algorithm which is widely used for such tasks due
to its robustness and reliability against a high proportion of outliers. RANSAC algorithm
randomly chooses a number of data within the dataset, draw a hypothesis (model, or fitting)
out of those, count the number of data points that are positioned nearby the model with a
certain margin (inliers, or the consensus set), and repeat the above process N times to achieve
the hypothesis with the most inliers. RANSAC algorithm is well provieded in scikit-learn
package.
However, there is a problem. Despite going through numerous filters in previous sections,
filtered point cloud data still has a certain amount of points that are unrelated with the
lane marks. Also, there exists a possibility that even points of lane marks could be removed.
Above case happened to be a problem when I tried to run RANSAC algorithms immediately
after filtering process was done. If there are fewer lane mark points than unrelated points,
despite its random sampling process in fitting, RANSAC algorithm always returns the same
output that includes irrelevant points as inliers while excluding necessary points.
Of course I might be able to manually handle related hyper parameters of RANSAC
algorithm such as ’residual threshold’ (maximum residual for a data sample to be classified
as an inlier) to achieve the fitting that avoids such irrelevant points. However, unlike data
pre-processing where I individually fine-tuned every single hyper parameters, I did not want
to impose my subjective views in fitting process. Moreover, even if we try to iterate a process
to find optimized hyper parameters, it would be highly expensive to keep in track of effects
of all those hyper parameters.
To circumvent this while leaving the RANSAC algorithm hyper parameters by default, I
16
think of the following process: 1. partition the filtered data, analogous to using grid box
in YOLO object detection algorithm or US Electoral College system; 2. randomly choose
M samples from each grid cell and running RANSAC algorithms to extract two ego lanes;
3. iterate the 2nd process N times, and choose the ego lane pair with the lowest userdefined cost value. With this process, I thought I could both lighten the model complexity
by reducing the number of points to be tested by RANSAC algorithm and minimize the
chance of including wrong points into the fitting. Below is the simple figure for the better
understanding.

**Below is the simple schematic of the upcoming algorithm. Once I partitioned the filter point clouds into
K grid cells (K=20 in this figure), I randomly choose M samples from each grid cell (M=2 in
this figure for simplicity, red dots) and conduct the polynomial fitting (gray line). Repeat N
times and choose the best pair of ego lanes with minimal cost, which will be explained soon.
(-40, 3.9), (-40, -3.9), (40, 3.9) and (40, -3.9) are the Cartesian coordinates of 4 vertices
that defines the spatial boundary of the entire grid.**
![lane_extraction_그림](https://user-images.githubusercontent.com/82307352/167687028-d0f4c9b3-1127-4c4b-bff9-1f2b33eb0e56.jpg)

### 2.2.1 Grid cell
To do this, once we read the filtered LiDAR data, I first defined grid cells with
boundaries along x and y axis for each grid cell.

```python
data_folder = 'C:/Users/User/Desktop/Lidar_data/filtered_pointclouds'
lidar_files = sorted(os.listdir(data_folder))
lidar_fpath = [os.path.join(data_folder, f) for f in lidar_files]
lidar_bin = lidar_fpath[0]
lidar_data = np.fromfile(lidar_bin, dtype=np.float32).reshape(-1, 5)

grid_dict = {}
for y in (-1,1):
    grid_list = [] 

    for x in range(5, -5, -1):
        if y == -1:
            grid_x_up_bound = x*8
            grid_x_low_bound = (x-1)*8
            grid_y_up_bound = (y+1)*3.9
            grid_y_low_bound = y*3.9
            grid = [grid_x_up_bound, grid_x_low_bound, grid_y_up_bound, grid_y_low_bound]
            coord = (y, x)
            grid_dict[coord] = grid
        else:
            grid_x_up_bound = x*8
            grid_x_low_bound = (x-1)*8
            grid_y_up_bound = y*3.9
            grid_y_low_bound = (y-1)*3.9
            grid = [grid_x_up_bound, grid_x_low_bound, grid_y_up_bound, grid_y_low_bound]
            coord = (y, x)
            grid_dict[coord] = grid
```

Number of grid cells are same as the one described in the above figure. ’grid dict’ holds the
key as a simplified coordinate of a grid cell. Along the y axis, the data is partitioned into
two sections: points with negative y coordinates and those with positive y coordinates. For
simplicity, I denoted those two as -1 and 1 respectively. Similarly, for x axis, there are
10 partitions so that I put integers from -4 to 5 based on grid cell’s relative position. For
example, if we write ’grid dict[(-1, 5)]’, it will return a list of 4 elements that give upper and
lower boundaries of x and y coordinates within a rightmost, down grid cell.

After defining grid cells, we need to actually partition the point clouds. Below is the source
code.

```python
lidar_data[:, 2] = 0
data_in_grid = {}
for grid_cell_index, grid_cell_coord in enumerate(grid_dict):
    x_upper_bound, x_lower_bound, y_upper_bound, y_lower_bound = grid_dict[grid_cell_coord]
    inGrid_lidar_data = np.delete(lidar_data, np.where(
    (lidar_data[:, 0] > x_upper_bound) |
    (lidar_data[:, 0] <= x_lower_bound) |
    (lidar_data[:, 1] > y_upper_bound) |
    (lidar_data[:, 1] <= y_lower_bound))[0], axis = 0)
    
    data_in_grid[grid_cell_coord] = inGrid_lidar_data
```
’data in grid’ dictionary holds a key same as previously defined ’grid dict’ while the corresponding value is the data points inside that grid cell.

### 2.2.2 Cost Function
We also need to define the cost function to find the best pair of ego lanes. Running
RANSAC algorithms with selected points within each grid cell out of 20 returns the 3 degree
polynomial fitted coefficients of two lanes, one in upper, positive y region and the other in
lower, negative y region. If those two lanes are true ego lanes, than their distance should be
consistently around 3m.

**Below is the schematic of wrong distance (red-dotted line) and actual distance (blue-dotted line).**
![cost](https://user-images.githubusercontent.com/82307352/167687908-cae3bcdc-96cf-4daf-a443-afd19f633685.jpg)

Let’s say if I want to measure the distance between two lines at arbitrary x test, and
the polynomial functions of upper ego lane and lower ego lane are y up(x) and y down(x)
respectively. We cannot simply insert x test into these two functions and subtract each other
(i.e.y up(x test) − y down(x test)) to get the distance.

**Below is the schematic of obtaining a corrected distance.**
![cost_수정](https://user-images.githubusercontent.com/82307352/167688035-582fcc41-6500-423c-8ce9-8b8be7f61e26.jpg)

To correct such error, I decided to follow the procedure shown in above figure. First randomly
choose a testing point’s x coordinate - the lower ego lane will have a corresponding point with
same x coordinate and the slope of tangent line at that point (x test, y down(x test)) (red
line). Next, get the perpendicular line with respect to that tangent line using the relation
that the slopes of two perpendicular lines are negative reciprocals of each other (blue-dashed
line). Then, find the intersection point between the upper ego lane and the perpendicular
line from the lower ego lane. Finally, measure the euclidean distance between the intersection
point and the point from the lower ego lane. Below is the source code for the cost function.

```python
def cost(up_lane_coeffs, down_lane_coeffs):
    
    x_list = list(range(-40, 41, 1))
    interval_truth = list(itertools.repeat(3, 81)) 
    interval_measured = [] 

    for x_test in x_list:
        y_down = down_lane_coeffs[0]*x_test**3 + down_lane_coeffs[1]*x_test**2 + down_lane_coeffs[2]*x_test + down_lane_coeffs[3]
        deriv_y_down = 3*down_lane_coeffs[0]*x_test**2 + 2*down_lane_coeffs[1]*x_test+down_lane_coeffs[2]

        y_down_perp_line = lambda x: -1/deriv_y_down*(x-x_test) + y_down
        y_up = lambda x: up_lane_coeffs[0]*x**3 + up_lane_coeffs[1]*x**2 + up_lane_coeffs[2]*x + up_lane_coeffs[3]

        intersection_x = findIntersection(y_down_perp_line, y_up, x_test)[0]
        intersection_y = y_down_perp_line(intersection_x)

        dist = distance.euclidean([intersection_x, intersection_y], [x_test, y_down])
        interval_measured.append(dist)
    
    cost = mean_squared_error(interval_truth, interval_measured)
    return cost 

def findIntersection(fun1, fun2, x0):
    return fsolve(lambda x: fun1(x) - fun2(x), x0)
```
I decided to measure the interval between two lanes from x test=-40 to x = 40 with step
size 1 which will cover all parts of the pre-defined region of interest. I tried to
measure the mean squared error by comparing the measured interval and the ’ground truth’
interval, which I set it as 3m. For each point in x list, the measured interval is
appended in interval measured list and used in mean squared error.

### Fitting Best Ego Lane Pair
Below is the source code that explains the process of finding the best pair of ego lanes.

```python
min_sample_threshold = 5
data_repres_up = np.empty((0, 5)) 
data_repres_down = np.empty((0, 5)) 
iteration = 0
max_iter = 1000
prev_error = 1000

start = time.time()
while iteration <= max_iter:
    for y in (-1, 1):
        for x in range(5, -5, -1):
            if y == -1:
                if len(data_in_grid[y, x]) >= min_sample_threshold:
                    idx = np.random.randint(len(data_in_grid[y, x]), size = 3)
                    data_repres_up = np.append(data_repres_up, data_in_grid[y, x][idx, :], axis = 0)
                elif len(data_in_grid[y, x]) == 0:
                    pass
                else:
                    idx = np.random.randint(len(data_in_grid[y, x]), size = len(data_in_grid[y,x]))
                    data_repres_up = np.append(data_repres_up, data_in_grid[y, x][idx, :], axis = 0)

            elif y == 1:
                if len(data_in_grid[y, x]) >= min_sample_threshold:
                    idx = np.random.randint(len(data_in_grid[y, x]), size = 3)
                    data_repres_down = np.append(data_repres_down, data_in_grid[y, x][idx, :], axis = 0)
                elif len(data_in_grid[y, x]) == 0:
                    pass
                else:
                    idx = np.random.randint(len(data_in_grid[y, x]), size = len(data_in_grid[y,x]))
                    data_repres_up = np.append(data_repres_up, data_in_grid[y, x][idx, :], axis = 0)
                    
    up_grids_x = np.sort(data_repres_up, axis = 0)[:, 0] 
    up_grids_y = np.sort(data_repres_up, axis = 0)[:, 1]
    down_grids_x = np.sort(data_repres_down, axis = 0)[:, 0] 
    down_grids_y = np.sort(data_repres_down, axis = 0)[:, 1]

    ransac_up = RANSACRegressor(PolynomialRegression(degree=3),
                             min_samples = 7,    
                             max_trials = 10000,
                             random_state=0)
    ransac_up.fit(np.expand_dims(up_grids_x, axis=1), up_grids_y)
    up_grids_y_pred = ransac_up.predict(np.expand_dims(up_grids_x, axis=1))
    up_lane_coeffs= ransac_up.estimator_.get_params(deep = True)["coeffs"]

    ransac_down = RANSACRegressor(PolynomialRegression(degree=3), 
                             min_samples = 7,    
                             max_trials = 10000,
                             random_state=0)
    ransac_down.fit(np.expand_dims(down_grids_x, axis=1), down_grids_y)
    down_grids_y_pred = ransac_down.predict(np.expand_dims(down_grids_x, axis=1))
    down_lane_coeffs = ransac_down.estimator_.get_params(deep = True)["coeffs"]
    
    ego_lane_coeffs_pair = np.append(down_lane_coeffs, up_lane_coeffs, axis = 0)       
    curr_error = cost(up_lane_coeffs, down_lane_coeffs)
    
    if curr_error < prev_error:
        prev_error = curr_error 
        best_coeffs_pair = ego_lane_coeffs_pair

    iteration += 1
    end = time.time()
    print(end - start)
    print(iteration)

print("time spent: {:.2f} sec".format(time.time() - start)) 
print(best_coeffs_pair)
print(prev_error)
```
’min M samples’ is the minimum number of samples to be chosen in each grid cell.
For each grid cell, I randomly chose samples with randomly-picked indices. If the number of
samples within a grid cell is less than ’min M samples’, then I chose all of them. After that,
I appended those chosen points to ’data repres’, which denotes for the data representative. Then, I ran the RANSAC algorithm for both upper and lower parts of
data representatives respectively, get the pair of coefficients, calculate the cost, and update
coefficient pairs if the error in current loop is less than that of previous loop.
I set the initial previous loop error = 1000 as a starting point.
Additionally, one of the parameters for RANSACRegressor function is base estimator (in
the code, it is PolynomialRegression(degree=3)). Since scikit-learn package only provides
LinearRegression estimator, I made a new class ’PolynomialRegression(degree=3)’, which is
shown below:

```python
class PolynomialRegression(object):
    def __init__(self, degree=3, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))
```

# Results and Discussion
Below image shows the polynomial fittings of the left and right ego lanes,
earned with total 5000 iterations for each scene.

![final](https://user-images.githubusercontent.com/82307352/167688713-55578c34-924c-46ba-9434-13e9d0372b0e.jpg)

The algorithm of ego lane polynomial fitting was successful in most cases. However, there
seems to be a few drawbacks. The major drawback is that the algorithm is non-deterministic
since the underlying principle of the algorithm is to find out which one is the best amongst
large pairs of candidates. Of course we can find the best pair of ego lanes that are close to the
ground truth if we iterate as much as we can, theoretically. However, it would require a huge
amount of computation, which will not be suitable for pragmatic purposes such as real-time
related tasks. As an example, iterating 2000 times with the above algorithm approximately
took 150 seconds to complete.
Also, the algorithm seems to be vulnerable to the given point cloud data if the preprocessing has not done a good filtering although I implemented a random sampling with a
grid method to circumvent such issues. This also may cause a problem when there are few
points left after the filtering process. Since the algorithm divides those filtered data points
into several grids and chooses a certain number of representative points within each grid,
there is a possibility that only few points, or none at worst, would be involved in RANSAC
fitting algorithms if there exist few data points remaining after the pre-processing.
