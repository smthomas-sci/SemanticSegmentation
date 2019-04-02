import os
import shutil


y_dir = "/home/simon/Documents/PhD/Data/Histo_Segmentation/MarginData/y/"



for i in range(1, 61):
    fname = os.path.join(y_dir, "SCC_" + str(i) + ".csv")

    fh = open(fname, "r")

    lines = fh.readlines()
    fh.close()

    keep = []
    for line in lines:
        if "X" in line:
            continue
        keep.append(line[2::])

    fh = open(fname, "w")
    for line in keep:
        fh.write(line)
    fh.close()




