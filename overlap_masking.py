"""
Creates masks for a given tile, indicating whether pixels
are inside a threshold of the edge of the tile.

DON'T NEED THIS BUT CONTAINS USEFUL CODE

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read in image and convert to RGB color space
histo = cv2.cvtColor(cv2.imread("/home/simon/Desktop/histo_demo.tif"), cv2.COLOR_BGR2RGB)

d = 100
threshold = 20  # must be an even number

tile = histo[0:d, 0:d, :]

mask = np.ones((d-threshold, d-threshold, 3))
pad = threshold // 2
# only pad axes 0 & 2
mask = np.pad(mask, ((pad, pad), (pad, pad), (0, 0)), mode='constant').astype(np.bool)
print("Mask shape:", mask.shape)
print("Tile shape:", tile.shape)















