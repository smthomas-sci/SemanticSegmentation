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
X_dir = os.path.join(train_dir, "Img")
y_dir = os.path.join(train_dir, "Masks")

X_val_dir = os.path.join(test_dir, "Img")
y_val_dir = os.path.join(test_dir, "Masks")


# Create color palette
colors = [[0],[255]]
palette = Palette(colors)

# Other parameters
batch_size = 5
dim = 512
num_classes = 2

# Create generators
train_gen = SegmentationGen(batch_size, X_dir, y_dir, palette,
                                x_dim=dim, y_dim=dim)
val_gen = SegmentationGen(batch_size, X_val_dir, y_val_dir, palette,
                                x_dim=dim, y_dim=dim)


# Import model
model = VGG_UNet(dim, num_classes)

# Compile model - include sampe_weights_mode="temporal"
model.compile(optimizer=SGD(
        lr=0.001),
        loss="categorical_crossentropy",
        sample_weight_mode="temporal",
        metrics=["accuracy"])

# Train
history = model.fit_generator(
                epochs=2,
                generator = train_gen,
                steps_per_epoch = train_gen.n // batch_size)

# Evaluate
loss, acc = model.evaluate_generator(
                generator = val_gen,
                steps = val_gen.n // batch_size)

print("Val Loss:", loss, ", Val Acc:", acc)

# Predict
im = io.imread(os.path.join(X_val_dir, "17H27258_1A_BCC_1.png"))
im = resize(im, (dim, dim))
preds, class_img = predictImage(model, im)

plt.imshow(class_img)
plt.show()




