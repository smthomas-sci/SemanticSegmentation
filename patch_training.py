"""

Training script to train ResNet-UNet architecture
for segmentation on n-classes.

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 09/01/19
Last Update: 11/01/19

    Python v3.6:
        For less than 12 classes the python interpreter must be >=3.6. This is purely
        to ensure that dictionary.keys() returns the items in the original order
        as per update described https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-compactdict

    Data:
        The data directory should include the following subdirectories and each should
        contain the appropriate images: X_train, y_train, X_val, y_val, X_test, y_test

    Usage:
        python patch_training.py --batch_size 1 --epochs 10 --learning_rate 0.001 \
        --dim 512 --num_classes 12 --gpus 1  --log_dir ./logs/ --data data/ \
        --fine_tune --weights ./weights/BS_1_PS_512_C_12_FT_True_E_10_LR_0.001.h5


    Known Errors:
        BatchNoralisation layers, of which ResNet has many, apply different
        normalisations between training and predict/evaluation time. Need
        to consider this when evaluating as loss/acc can under-perform
        just because the data is not passed through the network in the same
        way. K.learning_phase(1) sets BN layers to use training time mean and
        stdev during predict/evaluate calls. Temporary fix.

"""

import argparse

import tensorflow as tf
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K

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
classes = args.classes

# Create unique run name
run_name = str(data_dir.split("/")[-2]) + "_BS_" + str(batch_size) + \
           "_PS_" + str(dim) + "_C_" + str(num_classes) + \
           "_FT_" + str(fine_tune) + "_E_" + str(epochs) + \
           "_LR_" + str(learning_rate)

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
#X_test_dir = os.path.join(data_dir, "X_test")
#y_test_dir = os.path.join(data_dir, "y_test")
#test_n = get_number_of_images(X_test_dir)

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
if not classes:
    colors = [color_dict[key] for key in color_dict.keys()]

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

#test_gen = segmentationGen(
#                batch_size, X_test_dir,
#                y_test_dir, palette,
#                x_dim=dim, y_dim=dim
#                )


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
weight_path = "./weights/" + run_name + "_checkpoint_{epoch:04d}-{val_acc:.2f}.h5"

callback_list = [
                    TensorBoard(log_dir=log_dir + "/" + run_name),
        
                    ModelCheckpoint(
                                    weight_path,
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=False,
                                    mode='auto',
                                    save_weights_only=True,
                                    period=5 # save every 10 epochs
                                    )
                ]

# Train
history = model.fit_generator(
                        epochs=epochs,
                        generator=train_gen,
                        steps_per_epoch=train_n // batch_size,
                        validation_data=val_gen,
                        validation_steps=(val_n / 4) // batch_size,     # 1/4 of data
                        callbacks=callback_list
                        )

# Save final weights
weight_path = "./weights/" + run_name + "_final.h5"
print("Saving weights as:", weight_path)
if gpus > 1:
    orig_model.save_weights(weight_path)
else:
    model.save_weights(weight_path)

# Final Evaluation - Switch to training mode
K.training_phase(1)
loss, acc = model.evaluate_generator(
                    generator=val_gen,
                    steps=val_n // batch_size
                    )
print("Validation set evaluation - Loss:", loss, "Acc:", acc)






