"""

Implementation of whole image prediction with variable overlap.


"""

from seg_utils import *
from seg_models import ResNet_UNet, ResNet_UNet_ExtraConv

from numpy.random import seed
from tensorflow import set_random_seed
# Set seed
seed(1)
set_random_seed(2)


num_classes = 12
gpus = 1

# Import model
model = ResNet_UNet_ExtraConv(num_classes=num_classes)

# Load weights
if gpus == 2:
    load_multigpu_checkpoint_weights(model, "./weights/100_Images_BS_24_PS_512_C_12_FT_False_E_100_LR_0.0001_WM_F_Checkpoint_E_20.h5")
else:
    model.load_weights("./weights/2x_n_290_BS_3_PS_512_C_12_FT_False_E_30_LR_0.0001_WM_F_checkpoint_010.h5")


# Create Keras function instead of model - helps with Learning Phase errors
model_in = model.layers[0].get_input_at(0)
model_out = model.layers[-2].output
model = K.function(inputs=[model_in], outputs=[model_out])

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


base_dir = "/home/simon/Documents/PhD/Data/Histo_Segmentation/Datasets_n290/2x/Images/"
#fnames = os.listdir(base_dir)


with open("/home/simon/Documents/PhD/Data/Histo_Segmentation/Datasets_n290/2x/TrainingData/2x_n_290/test_files.txt", "r") as fh:
    fnames = [line.strip() + ".tif" for line in fh.readlines()]

files = [ base_dir + name for name in fnames]


output_directory = "/home/simon/Desktop/Outs/Masks/"

whole_image_predict(files, model, output_directory, colors, compare=False)







