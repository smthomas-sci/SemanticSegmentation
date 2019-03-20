"""

A quick script to evaluate a trained model.

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 11/03/19
Last Update: 11/03/19

"""

import argparse


from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model


from seg_utils import *
from seg_models import ResNet_UNet, ResNet_UNet_ExtraConv

from numpy.random import seed as set_np_seed
from tensorflow import set_random_seed as set_tf_seed

# Set seed
seed = 1
set_np_seed(seed)
set_tf_seed(seed)


# Argparse setup
parser = argparse.ArgumentParser(description="Execute custom patch training regime")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training. Max of 12 on wiener")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
parser.add_argument("--weights", type=str, default=None, help="Path to pre-trained weights to load")
parser.add_argument("--dim", type=int, default=512, help="Patch size - Note: >512 may cause memory issues")
parser.add_argument("--num_classes", type=int, default=12, help="Number of classes to classify")
parser.add_argument("--data", type=str, default="./data/", help="Path to data directory")
args = parser.parse_args()

# Assign to global names
batch_size = args.batch_size
learning_rate = args.learning_rate
weights = args.weights
dim = args.dim
num_classes = args.num_classes
data_dir = args.data

print("[INFO] - EVALUATION RUN")
print("[INF0] - random seed -", seed)
print("[INFO] hyper-parameter details ...")
print("Batch Size:", batch_size)
print("Learning Rate:", learning_rate)
print("Weights:", weights)
print("Patch Dim:", dim)
print("Num Classes:", num_classes)
print("Data:", data_dir)


# Path & Directory Setup
X_eval_dir = os.path.join(data_dir, "X_train")
y_eval_dir = os.path.join(data_dir, "y_train")


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

# Create color and palette for generators
classes = list(color_dict.keys())
colors = [color_dict[key] for key in color_dict.keys()]
palette = Palette(colors)

# Create generator
eval_gen = SegmentationGen(
                batch_size, X_eval_dir,
                y_eval_dir, palette,
                x_dim=dim, y_dim=dim,
                )

# Import model for single GPU
model = ResNet_UNet(dim=dim, num_classes=num_classes)

# Load pre-trained weights
if weights:
    model.load_weights(weights)

# Lock weights for evaluation
model.trainable = False

# Compile model for training
model.compile(
            optimizer=Adam(lr=learning_rate),
            loss="categorical_crossentropy",
            sample_weight_mode="temporal",
            metrics=["accuracy"],
            weighted_metrics=["accuracy"]
            )

# Evaluate
print("[INFO] evaluating...")
results = model.evaluate_generator(generator=eval_gen,
                                   steps=eval_gen.n // eval_gen.batch_size,
                                   verbose=1)
print("Loss:", results[0], "Acc:", results[1], "Weighted Acc:", results[-1])

# Confusion Matrix
epoch_cm = np.zeros((len(classes), len(classes)))

# Loop through validation set
for n in range(eval_gen.n // eval_gen.batch_size):

    print("Step", n+1, "of", eval_gen.n // eval_gen.batch_size)

    # Grab next batch
    X, y_true, _ = next(eval_gen)

    # Make prediction with model
    y_pred = model.predict(X)

    # Find highest classes prediction
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    # Flatten batch into single array
    y_true = np.ndarray.flatten(y_true)
    y_pred = np.ndarray.flatten(y_pred)

    # Create batch CM
    batch_cm = ConfusionMatrix(y_true, y_pred)

    # Get all classes in batch
    all_classes = list(batch_cm.classes)

    batch_cm = batch_cm.to_array()

    # Update epoch CM
    for i in all_classes:
        for j in all_classes:
            epoch_cm[i, j] += batch_cm[all_classes.index(i), all_classes.index(j)]


# Create Colorful CM
# Compute row sums for Recall
row_sums = epoch_cm.sum(axis=1)
matrix = np.round(epoch_cm / row_sums[:, np.newaxis], 3)

# Set up colors
color = [255, 118, 25]
orange = [c / 255. for c in color]
white_orange = LinearSegmentedColormap.from_list("", ["white", orange])

fig = plt.figure(figsize=(12, 14))
ax = fig.add_subplot(111)
cax = ax.matshow(matrix, interpolation='nearest', cmap=white_orange)
fig.colorbar(cax)

ax.set_xticklabels([''] + classes, fontsize=8)
ax.set_yticklabels([''] + classes, fontsize=8)

# Get ticks to show properly
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))

ax.set_title("Recall - " + X_eval_dir)
ax.set_ylabel("Ground Truth")
ax.set_xlabel("Predicted")

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j - 0.1, i, str(matrix[i, j]), fontsize=8)

plt.savefig("/clusterdata/s4200058/train_CM.png", format="png")

plt.close(fig)

# Save CM
# fname = "/home/simon/Desktop/CMs/CM.np"
# print("Saving CM to", fname)
# np.save(fname, epoch_cm)

print("Finished.")


# ------------ #
# ROC ANALYSIS
# ------------ #

# Create ROC curves for all tissue types
#     ROC = {}
#     for tissue_class in color_dict.keys():
#         # Get class index
#         class_idx = colors.index(color_dict[tissue_class])
#
#         true = np.ravel(true_map[:, :, class_idx])
#         pred = np.ravel(prob_map[:, :, class_idx])
#
          # Pickle true and pred for later...

#         # Get FPR and TPR
#         fpr, tpr, thresholds = roc_curve(true, pred)
#         roc_auc = auc(fpr, tpr)
#         if np.isnan(roc_auc):
#             # class not present
#             continue
#         # Update values
#         ROC[tissue_class] = {"AUC": roc_auc, "TPR": tpr, "FPR": fpr, "raw_data": (true, pred)}
#
#     return ROC



