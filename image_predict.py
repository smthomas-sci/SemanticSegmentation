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
from pandas_ml import ConfusionMatrix


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


def grid_dimensions(n):
    x = 1
    while True:
        if n / x <= x and n % x == 0:
            break
        else:
            x += 1
    return x, n // x


def grid_position(row, col, pos):
    count = 0
    for i in range(row):
        for j in range(col):

            if count == pos:
                return i, j

            count += 1




data_dir = "/home/simon/Documents/PhD/Data/Histo_Segmentation/Datasets_n100/2x/TrainingData/Data_10/"
dim = 512

# training
X_train_dir = os.path.join(data_dir, "X_train")
y_train_dir = os.path.join(data_dir, "y_train")
train_n = get_number_of_images(X_train_dir)


num_classes = 12
# Import model
#K.set_learning_phase(1)
model = ResNet_UNet(num_classes=num_classes)
gpus = 1
if gpus == 2:
    load_multigpu_checkpoint_weights(model, "./weights/100_Images_BS_24_PS_512_C_12_FT_False_E_100_LR_0.0001_WM_F_Checkpoint_E_20.h5")
else:
    model.load_weights("./weights/Data_100_BS_12_PS_512_C_12_FT_True_E_30_LR_1e-06_WM_F.h5")

# Make Prediction model - no reshape at end
model_in = model.layers[0].get_input_at(0)
model_out = model.layers[-2].output
model = Model(inputs=[model_in], outputs=[model_out])


model.summary()



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

classes = np.asarray(list(color_dict.keys()))

# Create generators
train_gen = SegmentationGen(
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

# ------------------------------ #

# ----------------------------- #
#

stop = 30
step = 1
for i in range(stop):
    print(i + 1, "of", stop)
    # Load image
    X_train, y_train, sample_weights = next(train_gen)
#
#     # # Evaluate
#     # #loss, acc = model.evaluate_generator(train_gen, steps=train_n)
#     #
#     # loss, acc = model.evaluate(X_train, y_train, 1, verbose=0)
#     # #print("loss:", loss, "acc:", acc)
#     # # Predict
#
#
#     # RANDOM INPUT
    #X_train = np.random.randn(*X_train.shape)

    #K.set_learning_phase(1)
    preds = model.predict(X_train)

    #
    y_true = np.argmax(y_train[0], axis=-1).reshape(dim, dim)
    y_pred = np.argmax(preds[0], axis=-1)
#
#
#     # Compute Acc.
#     true_counts = np.ndarray.flatten(y_true)
#     pred_counts = np.ndarray.flatten(y_pred)
#
#     T_unique, T_counts = np.unique(true_counts, return_counts=True)
#     P_unique, P_counts = np.unique(pred_counts, return_counts=True)
#
#     T_dict = dict(zip(T_unique, T_counts))
#     P_dict = dict(zip(P_unique, P_counts))
#
#     for i in range(num_classes):
#
#         if T_dict.get(i, None):
#             acc = min(P_dict.get(i, 0) / T_dict.get(i), 1)
#             print(classes[i], acc)
#     print()
#     print("True:", T_dict)
#     print("Pred:", P_dict)

    # n = int(model.output.shape[-1])
    # row, col = grid_dimensions(n)
    # print(row, col)
    # fig, axes = plt.subplots(row, col, figsize=(12, 12))
    # for j in range(n):
    #     row_pos, col_pos = grid_position(row, col, j)
    #     channel = preds[0, :, :, j]
    #     norm = plt.Normalize(vmin=0, vmax=1)
    #     axes[row_pos, col_pos].imshow(norm(channel), cmap="jet")
    #     #axes[row_pos, col_pos].set_title(classes[j])
    # plt.show()

#     #confusion_matrix = ConfusionMatrix(y_true, y_pred)
#
#     # # Change names
#     # indices = confusion_matrix.classes
#     # if len(indices) == 2:
#     #     indices = [np.max(y_true), np.min(y_true)]
#     #     print("indices", indices)
#     #
#     # cm = confusion_matrix.to_array()
#     # metrics = {}
#     # pres_classes = classes[indices]
#     # for pos in range(len(pres_classes)):
#     #
#     #     total = np.sum(cm[pos, :])
#     #     if total == 0:
#     #         continue
#     #
#     #     metrics[pres_classes[pos]] = {}
#     #     for j in range(len(pres_classes)):
#     #         pro = cm[pos, j] / total
#     #         metrics[pres_classes[pos]][pres_classes[j]] = round(pro, 5)
#     #
#     # # show results
#     # # for clss in metrics.keys():
#     # #     print(clss, metrics[clss])
#     #
#     #
#     # print()
#     # print()
#     #
    ground_truth = apply_color_map(colors, y_true)
    segmentation = apply_color_map(colors, y_pred)

    image = X_train[0]
    #
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 8))
    axes2[0].imshow(image)
    axes2[0].set_title("Input")
    axes2[1].imshow(ground_truth)
    axes2[1].set_title("Ground Truth")
    axes2[2].imshow(segmentation)
    axes2[2].set_title("Prediction")
    # #plt.savefig("/home/simon/Desktop/Out/image_" + str(step) + ".png")
    plt.show()
#     #
#     #
#     # # # Show and save results
#     # fname = "/home/simon/Desktop/" + str(i) + ".png"
#     #
#     # print("saving...")
#     # segmentation *= 255.
#     # cv2.imwrite(fname, segmentation)

    step += 1
