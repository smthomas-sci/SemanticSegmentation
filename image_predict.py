"""
Important problem with Keras in Transfer Learning where
the Batch Norm layers causes problems. This can be
termporarily fixed using hte K.learning_phase(1) i.e. training phases
BN mean and variance values. Check out this blog.
http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/

"""



import matplotlib.pyplot as plt
import cv2
from keras.models import *
from keras.optimizers import Adam
import argparse
import keras.backend as K

from seg_utils import *
from seg_models import ResNet_UNet



# Argparse setup
parser = argparse.ArgumentParser(description="Execute custom patch training regime")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training. Max of 12 on wiener")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
parser.add_argument("--weights", type=str, default=None, help="Path to pre-trained weights to load")
parser.add_argument("--dim", type=int, default=512, help="Patch size - Note: >512 may cause memory issues")
parser.add_argument("--num_classes", type=int, default=12, help="Number of classes to classify")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs available on machine")
parser.add_argument("--fine_tune", dest='fine_tune', action='store_true', help="Whether to fine-tune model")
parser.add_argument("--log_dir", type=str, default="logs", help="Path to tensorboard log directory")
parser.add_argument("--data_dir", type=str, default="./data/", help="Path to data directory")
parser.add_argument("--output_dir", type=str, default="./", help="Path to output directory")
parser.add_argument("--classes", type=str, nargs="+", default=None, help="Not yet implemented")
parser.set_defaults(fine_tune=False)

args = parser.parse_args()

# Assign to global names
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
weights = args.weights
dim = args.dim
num_classes = args.num_classes
gpus = args.gpus
fine_tune = args.fine_tune
log_dir = args.log_dir
data_dir = args.data_dir
output_dir = args.output_dir
classes = args.classes


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


data_dir = "./data/SingleImage/"

# training
X_train_dir = os.path.join(data_dir, "X_train")
y_train_dir = os.path.join(data_dir, "y_train")
train_n = get_number_of_images(X_train_dir)



num_classes = 12
# Import model
K.set_learning_phase(1)
model = ResNet_UNet(num_classes=num_classes)
path = "./weights/100_Images_BS_12_PS_512_C_12_FT_False_E_100_LR_0.0001_checkpoint.h5"
gpus = 1
if gpus == 2:
    load_multigpu_checkpoint_weights(model, "./weights/100_Images_BS_12_PS_512_C_12_FT_False_E_100_LR_0.0001_checkpoint.h5")
else:
    model.load_weights("./weights/Multi-Image_BS_1_PS_512_C_12_FT_False_E_30_LR_1e-05.h5")

# Make Prediction model - no reshape at end
#model_in = model.layers[0].get_input_at(0)
#model_out = model.layers[-2].output
#model = Model(inputs=[model_in], outputs=[model_out])

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

# Create generators
train_gen = segmentationGen(
                batch_size, X_train_dir,
                y_train_dir, palette,
                x_dim=dim, y_dim=dim
                )
model.compile(
            optimizer=Adam(lr=learning_rate),
            loss="categorical_crossentropy",
            sample_weight_mode="temporal",
            metrics=["accuracy"]
            )

step = 1
for i in range(28):
    print(i + 1, "of", 28)
    # Load image
    X_train, y_train, sample_weights = next(train_gen)

    # Evaluate
    #loss, acc = model.evaluate_generator(train_gen, steps=train_n)

    loss, acc = model.evaluate(X_train, y_train, 1, verbose=0)
    #print("loss:", loss, "acc:", acc)
    # Predict
    probs, class_pred = predict_image(model, X_train[0])

    #print(probs.shape)

    norm = plt.Normalize(vmin=0, vmax=11)
    segmentation = cmap(norm(class_pred))[:, :, 0:3]  # drop alpha

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(X_train[0])
    axes[0].set_title("Input")
    axes[1].imshow(class_pred)
    axes[1].set_title("Predict - Acc: " + str(round(acc, 5)))

    plt.show()
    #
    #
    # # # Show and save results
    # fname = "/home/simon/Desktop/" + str(i) + ".png"
    #
    # print("saving...")
    # segmentation *= 255.
    # cv2.imwrite(fname, segmentation)

    step += 1





