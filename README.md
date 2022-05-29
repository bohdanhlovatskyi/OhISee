## SLAM 
It is not the best SLAM, but it looks and feels as SLAM!

Visualization:

Presentation:

Report:

## Team
| **Bohdan Hlovatskyi** | **Mykhailo Pasichnyk** | **Stefan Malyk** |
| :--- | :---: | ---: |
| ![kakashi](readme/kakashi.jpg) | ![](readme/sasuke.png) | ![](readme/naruto.jpg) |
| [BohdanHlovatskyi](https://github.com/bohdanhlovatskyi) | [MykhailoPasychnyk](https://github.com/fox-flex) | [StefanMalyk](https://github.com/elem3ntary) |

## Requirements and Installation
Strongly depends on C++ library Pangoling, which is a convenient wrapper over OpenGL, for visualization.

For its building look at the Pangoling's github that is included as a submodule. Be careful to what python you link it!

After this:

```shell
# here use pip that corresponds to python you link the pangolin to!
pip3 install -r requirements.txt
```

## Results

Main metric - **relative pose error**. Read more in the report.

<img src="materials/error_mapped.png" width="256"/>


## Tests

Testing of the work was mainly focused at KITTY dataset. It is one of the most common datasets that provides ground truth for rotation and translation matrices.
Link: http://www.cvlibs.net/datasets/kitti/eval_odometry.php

### Metrics:
- Absolute pose error, often used as absolute trajectory error. Corresponding poses are directly compared between estimate and reference given a pose relation. Then, statistics for the whole trajectory are calculated. This is useful to test the global consistency of a trajectory.
- Instead of a direct comparison of absolute poses, the relative pose error compares motions ("pose deltas"). This metric gives insights about the local accuracy, i.e. the drift. For example, the translational or rotational drift per meter can be evaluated 
- https://github.com/MichaelGrupp/evo/wiki/Metrics

### Installation:
```shell
git clone https://github.com/MichaelGrupp/evo.git
cd evo
pip install --editable . --upgrade --no-binary evo
```

### Dataset

```shell
ffmpeg -framerate 10 -pattern_type glob -i '*.png' \
  -c:v libx264 ../../kitty01_1.mp4
```

### To run the tests:
```shell
mkdir test_res

# compare the trajectories
evo_traj kitti test_res/result.txt --ref=rel/poses.txt -p --plot_mode=xz        

# find the metrics

# to get absolute error
# you can also run without args : evo_ape kitti rel/KITTI_00_gt.txt result.txt
# so to get the cli output
evo_ape kitti rel/poses.txt test_res/result.txt --plot --plot_mode xz --save_results test_res/result_abs.zip

# to get relative error
evo_rpe kitti rel/poses.txt test_res/result.txt --plot --plot_mode xz --save_results test_res/result_rel.zip

evo_res test_res/*.zip -p --save_table test_res/table.csv
```

## License and Copyright

All the contributions are welcomed!

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

© 2022 Bohdan Hlovatskyi, Mykhailo Pasychnyk, Stefan-Yuriy Malyk

