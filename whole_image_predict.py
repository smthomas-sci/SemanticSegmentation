import matplotlib.pyplot as plt
import cv2
from keras.models import *

from seg_utils import *
from seg_models import ResNet_UNet


def calculate_tile_size(image_shape, lower=50, upper=150):
    """
    Calculates a tile size with optimal overlap

    Input:

        image - original histo image (large size)

        lower - lowerbound threshold for overlap

        upper - upper-bound threshold for overlap

    Output:

        dim - dimension of tile

        threshold - calculated overlap for tile and input image
    """
    dims = [x * 2 ** 5 for x in range(20, 80, 2)]
    w = image_shape[1]
    h = image_shape[0]
    thresholds = []
    for d in dims:
        w_steps = w // d
        w_overlap = (d - (w % d)) // w_steps
        h_steps = h // d
        h_overlap = (d - (h % d)) // h_steps
        # Threshold is the half the minimum overlap
        thresholds.append(min(w_overlap, h_overlap) // 2)
    # Loop through pairs and take first that satisfies
    for d, t in zip(dims, thresholds):
        if lower < t < upper:
            if d < 1560:
                return d, t  # dim, threshold
    # default
    return 1024, thresholds[dims.index(1024)]


num_classes = 12
# Import model
model = ResNet_UNet(num_classes=num_classes)
model.load_weights("/home/simon/Desktop/SegDemo/balanced_base_model.h5")

# Make Prediction model - no reshape at end
model_in = model.layers[0].get_input_at(0)
model_out = model.layers[-2].output
model = Model(inputs=[model_in], outputs=[model_out])

# Create color palette
colors = [
        [73, 0, 106],
        [108, 0, 115],
        [145, 1, 122],
        [181, 9, 130],
        [216, 47, 148],
        [236, 85, 157],
        [254, 246, 242],
        [248, 123, 168],
        [0, 0, 0],
        [127, 255, 255],
        [127, 255, 142],
        [255, 127, 127]
]
palette = Palette(colors)
cmap = getColorMap(colors)
image_num = 3
histo = load_image("/home/simon/Desktop/histo_demo_" + str(image_num) + ".tif")
canvas = np.zeros_like(histo)

# Tile info
w = histo.shape[1]
h = histo.shape[0]
dim, threshold = calculate_tile_size(histo.shape, lower=50, upper=100)
print("Tile size:", dim)
print("Tile threshold:", threshold)
d = dim
w_steps = w // d
w_overlap = (d - (w % d)) // w_steps
h_steps = h // d
h_overlap = (d - (h % d)) // h_steps
# starting positions
w_x, w_y = 0, d
h_x, h_y = 0, d


# Predict
step = 1
for i in range(h_steps+1):

    for j in range(w_steps+1):

        print("Processing tile", step, "of", (h_steps+1)*(w_steps+1),)
        step += 1

        # Grab a tile
        tile = histo[h_x:h_y, w_x:w_y, :][np.newaxis, ::]

        # Check and correct shape
        orig_shape = tile[0].shape
        if tile.shape != (dim, dim, 3):
            tile = cv2.resize(tile[0], dsize=(dim, dim))[np.newaxis, ::]

        # Predict
        probs = model.predict(tile)
        class_pred = np.argmax(probs[0], axis=-1)
        norm = plt.Normalize(vmin=class_pred.min(), vmax=class_pred.max())
        segmentation = np.copy(cmap(norm(class_pred))[:, :, 0:3])  # drop alpha

        # Add prediction to canvas
        canvas[h_x+threshold: h_y-threshold,
               w_x+threshold: w_y-threshold, :] = segmentation[threshold:-threshold,
                                                               threshold:-threshold, :]

        # Update column positions
        w_x += d - w_overlap
        w_y += d - w_overlap

    # Update row positions
    h_x += d - h_overlap
    h_y += d - h_overlap
    w_x, w_y = 0, d


# Show and save results
fname = "/home/simon/Desktop/" + str(image_num) + "_WSI_" + str(dim) + "px.png"
# plt.imshow(canvas)
# plt.axis("off")
# plt.imsave(fname, canvas)
#  # Make full screen
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.show()
print("saving...")
canvas *= 255.
canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
cv2.imwrite(fname, canvas)






