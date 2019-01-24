"""

Implementation of whole image prediction with variable overlap.

Important problem with Keras in Transfer Learning where
the Batch Norm layers causes problems. This can be
termporarily fixed using hte K.learning_phase(1) i.e. training phases
BN mean and variance values. Check out this blog.
http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/

Also consider this thread: https://github.com/keras-team/keras/issues/6977

This looks like it could be more of a hassle than its worth.
https://github.com/keras-team/keras/pull/9965

"""


import matplotlib.pyplot as plt
import cv2
from sys import stderr
from keras.models import *
import keras.backend as K

from seg_utils import *
from seg_models import ResNet_UNet


from numpy.random import seed
from tensorflow import set_random_seed
# Set seed
seed(1)
set_random_seed(2)


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
    dims = [x * 2 ** 5 for x in range(1, 80, 2)]
    w = image_shape[1]
    h = image_shape[0]
    thresholds = {}
    for d in dims:
        w_steps = w // d
        w_overlap = (d - (w % d)) // w_steps
        h_steps = h // d
        h_overlap = (d - (h % d)) // h_steps
        # Threshold is the half the minimum overlap
        thresholds[d] = min(w_overlap, h_overlap) // 2
    # Loop through pairs and take first that satisfies
    for d, t in sorted(thresholds.items(), key=lambda x: x[1]):
        if lower < t < upper:
            if d <= 1408:
                return d, t  # dim, threshold
    #default
    #return 512, thresholds[dims.index(512)]


num_classes = 12
# Import modelq
K.set_learning_phase(1)
model = ResNet_UNet(num_classes=num_classes)
gpus = 2
if gpus == 2:
    load_multigpu_checkpoint_weights(model, "./weights/Data_20_BS_24_PS_128_C_12_FT_False_E_200_LR_0.0001_checkpoint_0165.h5")
else:
    model.load_weights("./weights/Data_10_BS_3_PS_512_C_12_FT_False_E_100_LR_0.0001.h5")

# Make Prediction model - no reshape at end
model_in = model.layers[0].get_input_at(0)
model_out = model.layers[-2].output
model = Model(inputs=[model_in], outputs=[model_out])

# Create color palette
color_dict = {
    "EPI":  [73, 0, 106],
    "GLD":  [108, 0, 115],
    "INF":  [145, 1, 122],
    "RET":  [181, 9, 130],
    "FOL":  [216, 47, 148],
    "PAP":  [236, 85, 157],
    "HYP":  [254, 246, 242],
    "KER":  [248, 123, 168],
    "BKG":  [0, 0, 0],
    "BCC":  [127, 255, 255],
    "SCC":  [127, 255, 142],
    "IEC":  [255, 127, 127]
}

# Set up colors to match classes
colors = [color_dict[key] for key in color_dict.keys()]

palette = Palette(colors)
image_num = 1
images = [ #"/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H13362_2A_SCC_1.tif"
            "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H13362_2A_SCC_1.tif",
            "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H13362_2A_SCC_2.tif",
            "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H13688_1A_IEC.tif",
            "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H13692_2A_IEC_1.tif",
            "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H26657_1A_BCC_1.tif",
            "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H26657_1A_BCC_2.tif",
            "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H26658_1B_BCC_1.tif",
            "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H27203_1A_BCC.tif",
            "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H27251_1A_SCC_1.tif"
          ]

for file in images[4::]:
    name = file.split("/")[-1].split(".")[0]
    print("Processing image:", name, "Num:", image_num, "of", len(images))
    image_num += 1

    # Load in image and create canvas for segmentation
    try:
        histo = load_image(file, pre=True)

        canvas = np.zeros_like(histo)

        #prob_map = np.zeros((histo.shape[0], histo.shape[1], num_classes))

        # Tile info
        w = histo.shape[1]
        h = histo.shape[0]
        dim, threshold = calculate_tile_size(histo.shape, lower=50, upper=100)

    except Exception as e:
        print("Failed to process:", name, e, file=stderr)
        continue

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

            segmentation = apply_color_map(colors, class_pred)

            # Add prediction to canvas
            canvas[h_x+threshold: h_y-threshold,
                   w_x+threshold: w_y-threshold, :] = segmentation[threshold:-threshold,
                                                                   threshold:-threshold, :]
            # Add prediction map
            #prob_map[h_x + threshold: h_y - threshold,
            #         w_x + threshold: w_y - threshold, :] = probs[0][threshold:-threshold,
            #                                                          threshold:-threshold, :]

            # Update column positions
            w_x += d - w_overlap
            w_y += d - w_overlap

        # Update row positions
        h_x += d - h_overlap
        h_y += d - h_overlap
        w_x, w_y = 0, d

    # Compute ROC + AUC
    # true_map = create_prob_map_from_mask(mask_name, palette)
    # true_mask = np.argmax(true_map, axis=-1)
    # generate_ROC_AUC(true_map, pred_map, color_dict, colors)

    # Show and save results
    fname = "/home/simon/Desktop/" + name + "_WSI_" + str(dim) + "px.png"
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

    # Wipe canvas from memory
    del canvas

    # Next image...

print("Done.")










