"""

Experiment to test the segmentation ability of
the VGG_UNet model on a simple background segmentation
task on histology images.

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 26/10/18
Last Update: 31/10/18
"""


from seg_utils_old import *
from FCN_models import VGG_UNet, ResNet_UNet
from keras.optimizers import *
import matplotlib.pyplot as plt
from matplotlib import cm

# Directory Setup
base_dir = "/home/simon/Documents/PhD/Data/Histo_Small/"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
X_name = "6DImg"
y_name = "6DMasks"
out_dir = os.path.join(base_dir, "out")


# training
X_train_dir = os.path.join(train_dir, X_name)
y_train_dir = os.path.join(train_dir, y_name)
# valdiation
X_val_dir = os.path.join(test_dir, X_name)
y_val_dir = os.path.join(test_dir, y_name)


# Create color palette
colors = [ [0,0,0],[255, 0, 0], [255, 106, 0],
          [255, 216, 0], [76, 255, 0], [0,38,255]]
palette = Palette(colors)

# Other parameters
batch_size = 3
dim = 128
num_classes = 6

# Create generators
train_gen = SegmentationGen(batch_size, X_train_dir, y_train_dir, palette,
                            x_dim=dim, y_dim=dim, weight_mod = {5 : 1})
val_gen = SegmentationGen(batch_size, X_val_dir, y_val_dir, palette,
                                x_dim=dim, y_dim=dim)

# Import model
#model = VGG_UNet(dim, num_classes)
model = ResNet_UNet(input_shape=(dim, dim, 3), dim=dim, num_classes=num_classes)

model.load_weights("./weights/ResNet_UNet_base_model_model.h5")
# Unlock for fine tuning
for layer in model.layers:
    layer.trainable = True

# Compile model - include sampe_weights_mode="temporal"
model.compile(optimizer=SGD(
        lr=0.0001),
        loss="categorical_crossentropy",
        sample_weight_mode="temporal",
        metrics=["accuracy"])

# Train
epochs = 20
for i in range(epochs):
    print(i+1, "of", epochs)

    # Fit
    history = model.fit_generator(
                    epochs=1,
                    generator = train_gen,
                    steps_per_epoch = train_gen.n // batch_size
    )

    # Val
    loss, acc = model.evaluate_generator(
                    generator = val_gen,
                    steps = val_gen.n // batch_size
    )

    print("Validation - loss:", loss, ", acc:", acc)



#Save weights:
model.save_weights("./weights/ResNet_UNet_fine_tune_model.h5")

#
## Original colormap
cmap = getColorMap(colors)

# Predict
files = os.listdir(X_val_dir);
for file in files:
    im = io.imread(os.path.join(X_val_dir, file))
    im = im[:,:,0:3]
    im = resize(im, (dim, dim))
    preds, class_img = predictImage(model, im)

    # Normalise image and add color map
    norm = plt.Normalize(vmin=class_img.min(), vmax=class_img.max())
    image = cmap(norm(class_img))
    # Save it
    plt.imsave(os.path.join(out_dir, file), image)
    # Show it
    #plt.imshow(image)
    #plt.show()




