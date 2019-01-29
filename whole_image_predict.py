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


num_classes = 12
gpus = 1

# Import model
model = ResNet_UNet(num_classes=num_classes)

# Load weights
if gpus == 2:
    load_multigpu_checkpoint_weights(model, "./weights/Data_20_BS_24_PS_128_C_12_FT_False_E_200_LR_0.0001_checkpoint_0165.h5")
else:
    model.load_weights("./weights/Data_20_BS_24_PS_128_C_12_FT_False_E_100_LR_0.0001_WM_N_o_n_e.h5")

# Create keras function instead of model - helps with Learning Phase errors
learning_phase = 1
model_in = model.layers[0].get_input_at(0)
model_out = model.layers[-2].output
model = K.function(inputs=[model_in, K.learning_phase()], outputs=[model_out])

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

# images = [
#             "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H13362_2A_SCC_1.tif",
#             "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H13362_2A_SCC_2.tif",
#             "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H13688_1A_IEC.tif",
#             "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H13692_2A_IEC_1.tif",
#             "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H26657_1A_BCC_1.tif",
#             "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H26657_1A_BCC_2.tif",
#             "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H26658_1B_BCC_1.tif",
#             "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H27203_1A_BCC.tif",
#             "/home/simon/Documents/PhD/Data/Histo_Segmentation/Images/17H27251_1A_SCC_1.tif"
#           ]

files = [
                "./WSI_test/images/histo_demo_1.tif",
                "./WSI_test/images/histo_demo_2.tif",
                "./WSI_test/images/histo_demo_3.tif"
            ]


output_directory = "./WSI_test/segmentations/"

whole_image_predict(files, model, output_directory, colors)







