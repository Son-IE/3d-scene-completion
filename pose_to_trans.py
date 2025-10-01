import numpy as np
from scipy.spatial.transform import Rotation

for idx in range(0, 744 + 1):
    loc_pos = "poseData/pos/pos_data_frame{}.txt".format(idx)
    loc_orr = "poseData/orr/orr_data_frame{}.txt".format(idx)
    loc_pose = "poseData/finalData/pose_data_frame{}.txt".format(idx)
    pos = []
    orr = []

    with open(loc_pos, "r") as file:
        for line in file:
            pos.append(line)
    with open(loc_orr, "r") as file:
        for line in file:
            orr.append(line)

    rot_arr = Rotation.from_euler('xyz', orr)
    rot_mat = rot_arr.as_matrix()

    trans = np.identity(4)
    trans[:3, :3] = rot_mat
    trans[:3, 3] = pos
    trans[3, :4] = [0, 0, 0, 1]

    with open(loc_pose, "w") as file:
        string = np.array2string(trans).replace("[","")
        string = string.replace("]","")
        file.write(string)

    print(f"\rConverting: {(idx/276)*100}%", end="")

print("")