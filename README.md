## Visual Odometry

## Tests

Testing of the work was mainly focused at KITTY dataset. It is one of the most common datasets that provides ground truth for rotation and translation matrices.
Link: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

### Metrics:
- Absolute pose error, often used as absolute trajectory error. Corresponding poses are directly compared between estimate and reference given a pose relation. Then, statistics for the whole trajectory are calculated. This is useful to test the global consistency of a trajectory.

### Installation:
```shell
git clone https://github.com/MichaelGrupp/evo.git
cd evo
pip install --editable . --upgrade --no-binary evo
```

### To run the tests:
```shell
# compare the trajectories
evo_traj kitti result.txt --ref=rel/KITTI_00_gt.txt -p --plot_mode=xz

# find the metrics
mkdir test_res

evo_ape kitti rel/KITTI_00_gt.txt result.txt -va --plot --plot_mode xz --save_results test_res/result.zip

evo_res test_res/*.zip -p --save_table results/table.csv
```

## Plan
- [DONE] CV based implementation of Frontend of the SLAM
- [DONE] Visualization
- [DONE] Add some testing
  - [DONE] Find dataset that would be nice to use
  - [DONE] Find the way to compute some metrics
  - [] Read more on the metrics
  - [] Add timestamps so to make relative error estimation possible
- [] ORB feature detector
- [] BF feature matcher
- [] Epipolar Geometry and Essential matrix
- [] Camera motion with epipolar constraint
- [] Triangulation
- [] Graph pose optimization (probably via g2o)

- [] Fix report
- [] Fix README

