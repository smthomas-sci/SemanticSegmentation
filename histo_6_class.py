"""

Experiment to test the segmentation ability of
the VGG_UNet model on a simple background segmentation
task on histology images.

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 26/10/18

"""


from seg_utils import *
from FCN_models import VGG_UNet
from keras.optimizers import *
import matplotlib.pyplot as plt

# Directory Setup
base_dir = "/home/simon/Documents/PhD/Data/Histo_Small/"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
X_dir = os.path.join(test_dir, "5DImg")
y_dir = os.path.join(test_dir, "5DMasks")

X_val_dir = os.path.join(test_dir, "Img")
y_val_dir = os.path.join(test_dir, "Masks")


# Create color palette
colors = [ [0,0,0],[255, 0, 0], [255, 106, 0],
          [255, 216, 0], [76, 255, 0], [0,38,255]]
palette = Palette(colors)

# Other parameters
batch_size = 4
dim = 512
num_classes = 6

# Create generators
train_gen = SegmentationGen(batch_size, X_dir, y_dir, palette,
                                x_dim=dim, y_dim=dim)
#val_gen = SegmentationGen(batch_size, X_val_dir, y_val_dir, palette,
#                                x_dim=dim, y_dim=dim)


# Import model
model = VGG_UNet(dim, num_classes)

#model.load_weights("fine_tune_model.h5")
# Unlock for fine tuning
#for layer in model.layers:
#    layer.trainable = True

# Compile model - include sampe_weights_mode="temporal"
#model.compile(optimizer=SGD(
#        lr=0.00001),
#        loss="categorical_crossentropy",
##        sample_weight_mode="temporal",
#        metrics=["accuracy"])

# Train
# history = model.fit_generator(
#                 epochs=100,
#                 generator = train_gen,
#                 steps_per_epoch = train_gen.n // batch_size)
# 
# 
# # Save weights:
# model.save_weights("final_model.h5")
# 
# 
# 
# # Predict
files = os.listdir(X_dir);
for file in files:
     im = io.imread(os.path.join(X_dir, file))
     im = im[:,:,0:3]
     im = resize(im, (dim, dim))
     preds, class_img = predictImage(model, im)

     plt.imsave(os.path.join(X_dir, "Out_" + file), class_img)
     plt.imshow(class_img)
     plt.show()




