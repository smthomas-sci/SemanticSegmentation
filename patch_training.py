"""

Training script to train ResNet-UNet architecture
for segmentation on n-classes.

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 09/01/19
Last Update: 09/01/19

    Data:
        The data directory should include the following subdirectories and each should
        contain the appropriate images: X_train, y_train, X_val, y_val, X_test, y_test

    Usage:
        python patch_training.py --batch_size 1 --epochs 10 --learning_rate 0.001 \
        --dim 512 --num_classes 12 --gpus 1  --log_dir ./logs/ --data data/ \
        --fine_tune --weights ./weights/BS_1_PS_512_C_12_FT_True_E_10_LR_0.001.h5

"""

import argparse

import tensorflow as tf
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import TensorBoard

from seg_utils import *
from seg_models import ResNet_UNet

from numpy.random import seed
from tensorflow import set_random_seed
# Set seed
seed(1)
set_random_seed(2)


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

# Create unique run name
run_name = "BS_" + str(batch_size) + "_PS_" + str(dim) + \
           "_C_" + str(num_classes) + "_FT_" + str(fine_tune) + \
           "_E_" + str(epochs) + "_LR_" + str(learning_rate)

print("Run:", run_name)

# Path & Directory Setup
os.system("mkdir -p " + log_dir)
os.system("mkdir -p " + log_dir + "/" + run_name)
os.system("mkdir -p weights")

# training
X_train_dir = os.path.join(data_dir, "X_train")
y_train_dir = os.path.join(data_dir, "y_train")
train_n = get_number_of_images(X_train_dir)
# validation
X_val_dir = os.path.join(data_dir, "X_val")
y_val_dir = os.path.join(data_dir, "y_val")
val_n = get_number_of_images(X_val_dir)
# test
X_test_dir = os.path.join(data_dir, "X_test")
y_test_dir = os.path.join(data_dir, "y_test")
test_n = get_number_of_images(X_test_dir)

# Create color palette
colors = [
        [73, 0, 106],       # EPI
        [108, 0, 115],      # GLD
        [145, 1, 122],      # INF
        [181, 9, 130],      # RET
        [216, 47, 148],     # FOL
        [236, 85, 157],     # PAP
        [254, 246, 242],    # HYP
        [248, 123, 168],    # KER
        [0, 0, 0],          # BKG
        [127, 255, 255],    # BCC
        [127, 255, 142],    # SCC
        [255, 127, 127]     # IEC
]
palette = Palette(colors)

# Create generators
train_gen = segmentationGen(
                batch_size, X_train_dir,
                y_train_dir, palette,
                x_dim=dim, y_dim=dim
                )

val_gen = segmentationGen(
                batch_size, X_val_dir,
                y_val_dir, palette,
                x_dim=dim, y_dim=dim
                )

test_gen = segmentationGen(
                batch_size, X_test_dir,
                y_test_dir, palette,
                x_dim=dim, y_dim=dim
                )


if gpus > 1:
    print("[INFO] training with {} GPUs...".format(gpus))

    # Store a copy of the model on *every* GPU, and then combine
    # then combine the results from the gradient updates from the CPU
    with tf.device("/cpu:0"):
        # Import model
        orig_model = ResNet_UNet(dim=dim, num_classes=num_classes)

        # Load pre-trained weights
        if weights:
            orig_model.load_weights(weights)

        # Lock / unlock weights for training
        set_weights_for_training(orig_model, fine_tune)

    # Create multi-GPU version
    model = multi_gpu_model(orig_model, gpus=gpus)

else:
    print("[INFO] training with 1 GPU...")

    # Import model for single GPU
    model = ResNet_UNet(dim=dim, num_classes=num_classes)

    # Load pre-trained weights
    if weights:
        model.load_weights(weights)

    # Lock / unlock weights for training
    set_weights_for_training(model, fine_tune)


# Compile model for training
model.compile(
            optimizer=Adam(lr=learning_rate),
            loss="categorical_crossentropy",
            sample_weight_mode="temporal",
            metrics=["accuracy"]
            )

# Create Tensorboard Callbacks
callback_list = [TensorBoard(log_dir=log_dir + "/" + run_name)]

# Train
history = model.fit_generator(
                        epochs=epochs,
                        generator=train_gen,
                        steps_per_epoch=train_n // batch_size,
                        validation_data=val_gen,
                        validation_steps=val_n // batch_size,
                        callbacks=callback_list
                        )

# Save weights
weight_path = "./weights/" + run_name + ".h5"
print("Saving weights as:", weight_path)
if gpus > 1:
    orig_model.save_weights(weight_path)
else:
    model.save_weights(weight_path)

# Evaluate
loss, acc = model.evaluate_generator(
                    generator=test_gen,
                    steps=test_n // batch_size
                    )
print("Test set evaluation - Loss:", loss, "Acc:", acc)






