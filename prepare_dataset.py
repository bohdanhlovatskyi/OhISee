import os
import numpy as np

from kitti_to_poses import get_pose

DATASET = "2011_09_26/2011_09_26_drive_0095_sync/oxts/data"
SAVE_TO = "poses.txt"

def traj_to_str(pose):
    np.set_printoptions(precision=9, suppress=True)
    res = np.array_str(pose[:3].flatten(), max_line_width=np.inf)
    res = res.strip("] [")
    res += "\n"

    return res

if __name__ == "__main__":
    poses = {}
    for filename in os.listdir(DATASET):
        pose = get_pose(os.path.join(DATASET, filename))
        filename = int(filename.split(".")[0])
        poses[filename] = pose

    initial = poses[0]
    initial_inv = np.linalg.inv(initial)
    
    poses = sorted(poses.items(), key=lambda x: x[0])
    
    with open(SAVE_TO, "w") as file:
        for idx, pose in poses:
            if idx == len(poses) - 1:
                break
            pose = initial_inv @ pose
            file.write(traj_to_str(pose))

    with open(SAVE_TO, "r") as file:
         data = file.readlines()

    for idx, line in enumerate(data):
        data[idx] = " ".join(line.split())

    with open(SAVE_TO, "w") as file:
        file.write("\n".join(data))
