import numpy as np

# https://www.mrt.kit.edu/z/publ/download/2013/GeigerAl2013IJRR.pdf
# data description: 2011_09_26/2011_09_26_drive_0095_sync/oxts/dataformat.txt

test_path = "2011_09_26/2011_09_26_drive_0095_sync/oxts/data/0000000000.txt"

def conv_to_t(lat,lon,scale):
    er = 6378137;
    mx = scale * lon * np.pi * er / 180;
    my = scale * er * np.log( np.tan((90+lat) * np.pi / 360) );

    return (mx, my)

def get_pose(path):
    with open(path) as file:
        data = list(map(float, file.readlines()[0].split()))

    lat, lon = data[0], data[1]

    t = np.zeros(3)
    t[1], t[2] = conv_to_t(lat, lon, lat_to_scale(lat))
    t[0] = data[2]

    t *= - 1


    rx, ry, rz = data[3], data[4], data[5]

    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

    R = Rz*Ry*Rx;
    
    # construct [R t]
    P = np.zeros(16).reshape((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t
    P[3, 3] = 1

    return P


def lat_to_scale(lat):
    return np.cos(lat * np.pi / 180.0);

if __name__ == "__main__":
    print(get_pose(test_path))