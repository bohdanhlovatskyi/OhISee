## Visual Odometry

## Plan

- OpenCV
  - ORB feature detector
  - FLANN feature matcher
  - Epipolar Geometry and Essential matrix
  - Camera motion with epipolar constraint

- Do the whole thing manually in C++ (trying to optimize this as much as we can)

## Results
Current result: 
![](/results/matching.png)

## TODO:
- Consider using:
     - http://www.cvlibs.net/datasets/kitti/eval_odometry.php

     - tool to parse dataset: https://github.com/utiasSTARS/pykitti


- problem of pure rotation: if there is pure rotation, monocular SLAM could not be initialized 
- RANSAC