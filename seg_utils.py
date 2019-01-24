"""

A collection of classes and functions for training fully-convolutional
CNNs for semantic segmentation. Includes a custom generator class for
performing model.fit_genertor() method in Keras. This is specifically used
for segementation ground truth labels which requires sample weights
to be used. (See https://github.com/keras-team/keras/issues/3653)

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 24/10/18
Last Update: 24/01/19

"""

import numpy as np
from skimage import io
from sys import stderr
import h5py
import os
import io as IO


from cv2 import resize

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as cols
from matplotlib.pyplot import Normalize

from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from pandas_ml import ConfusionMatrix




class Palette(object):
    """
    A color pallete which is essentially a channel
    colour LUT.
    """
    def __init__(self, ordered_list):
        """
        Takes and order list of colours and stores in dictionary
        format.

        Input:

            ordered_list - list of rgb tuples in class order

        Output:

            self[index] - rgb tuple associated with index/class
        """

        self.colors = dict((i, color) for (i, color) in enumerate(ordered_list))

    def __getitem__(self, arg):
        """
        Returns item with input key, i.e. channel
        """
        return self.colors[arg]


    def __str__(self):
        """
        Print representation
        """
        return "Channel Color Palette:\n" + str(self.colors)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.colors.keys())


def segmentationGen(
                    batch_size, X_dir, y_dir,
                    palette, x_dim, y_dim,
                    suffix=".png", weight_mod=None,
                    sparse=False):
    """
    A generator that returns X, y & sampe_weight data in designated batch sizes,
    specifically for segmentation problems. It converts y images 2D arrays
    to work with sample_weights which are calculated on the fly for each batch.

    The output predictions are in shape (batch_size, dim*dim*, num_classes)
    and therefore need to be reshaped inorder to be interpreted / visualised.

    Validation needs to be done separately due to implementation differences
    with keras, but remains fairly straight forward.

    Training is very sensitive to batchsize and traininging rate.

    See again:  https://github.com/keras-team/keras/issues/3653

    Example:

        >>> colours = [(0,0,255),(0,255,0),(255,0,0)]
        >>> palette = Palette(colours)
        >>> batch_size = 10
        >>> dim = 128
        >>> train_gen = SegmentationGen(batch_size, X_dir, y_dir, palette,
        ...                        x_dim=dim, y_dim=dim)
        >>> # Compile mode - include sampe_weights_mode="temporal"
        ... model.compile(  optimizer=SGD(lr=0.001),
        ...                 loss="categorical_crossentropy",
        ...                 sample_weight_mode="temporal",
        ...                 metrics=["accuracy"])
        >>> # Train
        ... history = model.fit_generator(
        ...                 generator = train_gen,
        ...                 steps_per_epoch = train_n // batch_size
        ...                 validation_data = val_gen,
        ...                 validation_steps = val_n // batch_size )
        >>> # Evaluate
        ... loss, acc = model.evaluate_generator(
        ...                 generator = test_gen,
        ...                 steps = 2)


    Input:

        batch_size - number of images in a batch

        X_dir - full path directory of training images

        y_dir - full path directory of training labels

        palette - color palette object where each index (range(n-classes))
                is the class colour from the segmented ground truth. Get be
                obtained from the LUT of come standard segmentaiton datasets.

        dim - batches require images to be stacked so for
                batch_size > 1 image_size is required.

        suffix - the image type in the raw images. Default is ".png"

        weight_mod -  a dictionary to modify certain weights by index
                      i.e. weight_mod = {0 : 1.02} increases weight 0 by 2%.
                      Default is None.

        sparse -    *NOT IMPLEMENTED* bool indicating whether ground-truth is
                    sparse-encoded compared to one-hot-encoded.


    Output:
        using the global next() function or internal next() function the class
        returns X_train, y_train numpy arrays:
            X_train.shape = (batch_size, image_size, dim, 3)
            y_train.shape = (batch_size, image_size, dim, num_classes)
    """
    # Helper functions
    def _getClassMask(rgb, im):
        """
        Takes an rgb tuple and returns a binary mask of size
        im.shape[0] x im.shape[1] indicated where each color
        is present.

        Input:
            rgb - tuple of (r, g, b)

            im - segmentation ground truth image

        Output:
            mask - binary mask
        """
        # Colour mask
        if len(rgb) == 3:
            r, g, b = rgb
            r_mask = im[:,:, 0] == r
            g_mask = im[:,:, 1] == g
            b_mask = im[:,:, 2] == b
            mask = r_mask & g_mask & b_mask
            return mask
        # 8-bit mask
        return im[:,:] == rgb



    def _calculateWeights(y_train):
        """
        Calculates the balanced weights of all the classes
        in the batch.

        Input:
            y_train - (dim, dim,num_classes) ground truth

        Ouput:
            weights - a list of the weights for each class
        """
        class_counts = []
        # loop through each class
        for i in range(num_classes):
            batch_count = 0
            # Sum up each class count in each batch image
            for b in range(y_train.shape[0]):
                batch_count += np.sum(y_train[b][:,:,i])
            class_counts.append(batch_count)

        # create Counts
        y = []
        present_classes = []
        absent_classes = []
        for i in range(num_classes):
            # Adjusts for absence
            if class_counts[i] == 0:
                absent_classes.append(i)
                continue
            else:
                present_classes.append(i)
                y.extend([i]*int(class_counts[i]))
        # Calculate weights
        weights = compute_class_weight("balanced", present_classes, y)
        for c in absent_classes:
            weights = np.insert(weights, c, 0)

        # Modify weight for a particular class
        if weight_mod:
            for key in weight_mod.keys():
                weights[key] *= weight_mod[key]

        return weights

    def _createBatches(positions):
        """
        Creates X_train and y_train batches from the given
        positions i.e. files in the directory

        Input:
            positions - list of integers representing files

        Output:
            X_train, y_train - batches
        """
        # Store images in batch
        X_batch = []
        y_batch = []

        # Loop through current batch
        for pos in positions:
            # Get image name
            fname = files[pos][:-4]

            # load X-image
            im = io.imread(os.path.join(X_dir, fname + suffix))[:,:,0:3]    # drop alpha
            im = resize(im, (x_dim, x_dim))
            X_batch.append(im)

            # Load y-image
            im = io.imread(os.path.join(y_dir, fname + ".png"))[:,:,0:3]    # drop alpha
            im = resize(im, (y_dim, y_dim))
            # Convert to 3D ground truth
            y = np.zeros((im.shape[0], im.shape[1], num_classes), dtype=np.float32)
            # Loop through colors in palette and assign to new array
            for i in range(num_classes):
                rgb = palette[i]
                mask = _getClassMask(rgb, im)
                y[mask, i] = 1.

            y_batch.append(y)

        # Combine images into batches and normalise
        X_train = np.stack(X_batch, axis=0).astype(np.float32)
        y_train = np.stack(y_batch, axis=0)

        # Preprocess X_train
        X_train /= 255.
        X_train -= 0.5
        X_train *= 2.

        # Calculate sample weights
        weights = _calculateWeights(y_train)
        # # Modify weights
        # if weight_mod:
        #     for i in weight_mod:
        #         weights[i] = weight_mod[i] #*= weight_mod[i]

        #print("Weights:", weights)
        # Take weight for each correct position
        sample_weights = np.take(weights, np.argmax(y_train, axis=-1))

        # Reshape to suit keras
        sample_weights = sample_weights.reshape(y_train.shape[0], y_dim*y_dim)
        y_train = y_train.reshape(y_train.shape[0],
                                  y_dim*y_dim,
                                  num_classes)

        return X_train, y_train, sample_weights

    # -----------------------------------------------------------------------------#
    #                                   RUN
    # -----------------------------------------------------------------------------#

    files = os.listdir(X_dir)
    num_classes = len(palette)
    n = len(files)
    cur = 0
    order = list(range(n))
    np.random.shuffle(order)

    while True:
        # Reset
        if cur == n:
            np.random.shuffle(order)
            cur = 0

        # Most batches will be equal to batch_size
        if cur < (n - batch_size):
            # Get positions of files in batch
            positions = order[cur:cur + batch_size]

            cur += batch_size

            # create Batches
            X_train, y_train, sample_weights = _createBatches(positions)

            yield (X_train, y_train, sample_weights)

        # Final batch is smaller than batch_size
        else:
            positions = order[cur::]

            # Step is maximum
            cur = n

            # Create Batches
            X_train, y_train, sample_weights = _createBatches(positions)

            yield (X_train, y_train, sample_weights)


def predict_image(model, image):
    """
    Simplifies image prediction for segmentation models. Automatically
    reshapes output so it can be visualised.

    Input:

        model - ResNet training model where model.layers[-1] is a reshape
                layer.

        image - rgb image of shape (dim, dim, 3) where dim == model.input_shape
                image should already be pre-processed using load_image() function.

    Output:

        preds - probability heatmap of shape (dim, dim, num_classes)

        class_img - argmax of preds of shape (dim, dim, 1)
    """
    # Add new axis to conform to model input
    x = image[np.newaxis, ::]

    # Prediction
    preds = model.predict(x)[0].reshape(
                                    image.shape[0],
                                    image.shape[0],
                                    model.layers[-1].output_shape[-1])
    # class_img
    class_img = np.argmax(preds, axis=-1)

    return preds, class_img


def getColorMap(colors):
    """
    Returns a matplotlib color map of the list of RGB values

    Input:

        colors - a list of RGB colors

    Output:

        cmap -  a matplotlib color map object
    """
    # Normalise RGBs
    norm_colors = []
    for color in colors:
        norm_colors.append([val / 255. for val in color])
    # create color map
    cmap = cols.ListedColormap(norm_colors)

    return cmap


def apply_color_map(colors, image):
    """
    Applies the color specified by colors to the input image.

    Input:

        colors - list of colors in color map

        image - image to apply color map to

    Output:

        color_image

    """
    cmap = getColorMap(colors)
    norm = Normalize(vmin=0, vmax=len(colors))
    color_image = cmap(norm(image))[:, :, 0:3]  # drop alpha
    return color_image



def load_image(fname, pre=True):
    """
    Loads an image, with optional resize and pre-processing
    for ResNet50.

    Input:

        fname - path + name of file to load

        pre - whether to pre-process image

    Output:

        im - image as numpy array
    """
    im = io.imread(fname).astype("float32")
    if pre:
            im /= 255.
            im -= 0.5
            im *= 2.
    return im


def set_weights_for_training(model, fine_tune, layer_num=[81, 174]):
    """
    Takes a model and a training state i.e. fine_tune = True
    and sets weights accordingly. Fine-tuning unlocks
    from layer 81 - res4a_branch2a


    Input:

        model - ResNet_UNet model by default, can be any model

        fine_tune - bool to signify training state

        layer_num - layer to lock/unlock from. default is
                    173 add_16, where 174 is up_sampling2d_1

    Output:

        None
    """
    if not fine_tune:
        print("[INFO] base model...")
        # ResNet layers
        for layer in model.layers[0:layer_num[1]]:
            layer.trainable = False
        # UNet layers
        for layer in model.layers[layer_num[1]::]:
            layer.trainable = True
    else:
        print("[INFO] fine tuning model...")
        # ResNet layers
        for layer in model.layers[layer_num[0]:layer_num[1]]:
            layer.trainable = True
        # UNet layers
        for layer in model.layers[layer_num[1]::]:
            layer.trainable = True


def get_number_of_images(dir):
    """
    Returns number of files in given directory

    Input:

        dir - full path of directory

    Output:

        number of files in directory
    """
    return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])


def load_multigpu_checkpoint_weights(model, h5py_file):
    """
    Loads the weights of a weight checkpoint from a multi-gpu
    keras model.

    Input:

        model - keras model to load weights into

        h5py_file - path to the h5py weights file

    Output:
        None
    """

    print("Setting weights...")
    with h5py.File(h5py_file, "r") as file:

        # Get model subset in file - other layers are empty
        weight_file = file["model_1"]

        for layer in model.layers:

            try:
                layer_weights = weight_file[layer.name]

            except:
                # No weights saved for layer
                continue

            try:
                weights = []

                # Extract weights
                for term in layer_weights:
                    if isinstance(layer_weights[term], h5py.Dataset):
                        # Convert weights to numpy array and prepend to list
                        weights.insert(0, np.array(layer_weights[term]))

                # Load weights to model
                layer.set_weights(weights)

            except Exception as e:
                print("Error: Could not load weights for layer:", layer.name, file=stderr)


def create_prob_map_from_mask(filename, palette):
    """

    Creates a probability map with the input mask

    Input:

        filename - path to mask file

        pallette - color palette of mask

    Output:
        prob_map - numpy array of size image.h x image.w x num_classes
    """

    # Helper functions
    def _getClassMask(rgb, im):
        """
        Takes an rgb tuple and returns a binary mask of size
        im.shape[0] x im.shape[1] indicated where each color
        is present.

        Input:
            rgb - tuple of (r, g, b)

            im - segmentation ground truth image

        Output:
            mask - binary mask
        """
        # Colour mask
        if len(rgb) == 3:
            r, g, b = rgb
            r_mask = im[:, :, 0] == r
            g_mask = im[:, :, 1] == g
            b_mask = im[:, :, 2] == b
            mask = r_mask & g_mask & b_mask
            return mask
        # 8-bit mask
        return im[:, :] == rgb

    num_classes = len(palette)
    # Load y-image
    im = io.imread(filename)
    # Convert to 3D ground truth
    prob_map = np.zeros((im.shape[0], im.shape[1], num_classes), dtype=np.float32)
    # Loop through colors in palette and assign to new array
    for i in range(num_classes):
        rgb = palette[i]
        mask = _getClassMask(rgb, im)
        prob_map[mask, i] = 1.
    return prob_map


# def generate_ROC_AUC(true_map, prob_map, color_dict, colors):
#     """
#     Generates ROC curves and AUC values for all class in image, as well
#     as keeps raw data for later use.
#
#     Input:
#         true_map - map of true values, generated from mask using
#                     create_prob_map_from_mask()
#         prob map - 3 dimensional prob_map created from model.predict()
#
#         color_dict - color dictionary containing names and colors
#
#         colors - list of colors
#     Output:
#
#         ROC - dictionary:
#                 "AUC" - scalar AUC value
#                 "TPR" - array of trp for different thresholds
#                 "FPR" - array of fpr for different thresholds
#                 "raw_data" - type of (true, pred) where each are arrays
#
#     ! NEED TO INCLUDE SAMPLE WEIGHTS !
#
#     """
#     # Create ROC curves for all tissue types
#     ROC = {}
#     for tissue_class in color_dict.keys():
#         # Get class index
#         class_idx = colors.index(color_dict[tissue_class])
#
#         true = np.ravel(true_map[:, :, class_idx])
#         pred = np.ravel(prob_map[:, :, class_idx])
#
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


def write_confusion_matrix(matrix, classes):
    """
     Writes a confusion matrix to the tensorboard session

     Input:
        matrix - numpy confusion matrix

        classes - ordered list of classes

    Output:
        None
    """
    # Compute row sums for Recall
    row_sums = matrix.sum(axis=1)
    matrix = np.round(matrix / row_sums[:, np.newaxis], 3)

    # Import colors
    color = [255, 118, 25]
    orange = [ c / 255. for c in color]
    white_orange = LinearSegmentedColormap.from_list("", ["white", orange])

    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest', cmap=white_orange)
    fig.colorbar(cax)

    ax.set_xticklabels(['']+classes, fontsize=8)
    ax.set_yticklabels(['']+classes, fontsize=8)

    # Get ticks to show properly
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.set_title("Recall")
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j-0.1, i, str(matrix[i, j]), fontsize=8)

    buffer = IO.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    #plt.show()
    return buffer


class Validation(Callback):
    """
    A custom callback to perform validation at the
    end of each epoch. Also writes useful class
    metrics to tensorboard logs.
    """

    def __init__(self, generator, steps, classes, run_name, color_list):
        """

        Initialises the callback

        Input:

            generator -  validation generator of type segmentationGen()

            steps - number of steps in validation e.g. n // batch_size

            classes - an ordered list of classes ie. [ "EPI", "GLD" etc ]
        """
        self.validation_data = generator
        self.validation_steps = steps
        self.classes = np.asarray(classes)
        self.cms = []
        self.run_name = run_name
        self.color_list = color_list

    def write_predict_plot(self, mask, prediction, name, epoch):
        """
        Write mask and prediction to Tensorboard
        """
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(apply_color_map(self.color_list, mask))
        axes[0].set_title("Ground Truth")
        axes[1].imshow(apply_color_map(self.color_list, prediction))
        axes[1].set_title("Predict")
        plt.axis('off')

        # save to buffer
        plot_buffer = IO.BytesIO()
        plt.savefig(plot_buffer, format="png")
        plot_buffer.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(plot_buffer.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        # Add image summary
        summary_op = tf.summary.image(name,
                                      image,
                                      max_outputs=1,
                                      family="Predictions")

        with tf.Session() as sess:
            summary = sess.run(summary_op)
        # Write summary
        writer = tf.summary.FileWriter(logdir="./logs/" + self.run_name)
        writer.add_summary(summary, epoch)
        writer.close()

    def write_current_plot(self, epoch):
        """
        Write confusion matrix to Tensorboard
        """
        # Get the matrix
        matrix = self.cms[-1]

        # Prepare the plot
        plot_buffer = write_confusion_matrix(matrix, list(self.classes))

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(plot_buffer.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        # Add image summary
        summary_op = tf.summary.image("confusion_matrix_epoch_" + str(epoch),
                                      image,
                                      max_outputs=24,
                                      family="Confusion Matrix")

        with tf.Session() as sess:
            summary = sess.run(summary_op)
        # Write summary
        writer = tf.summary.FileWriter(logdir="./logs/" + self.run_name)
        writer.add_summary(summary, epoch)
        writer.close()


    def on_train_end(self, logs={}):
        # 1. Print final confusion matrix
        print()
        print("Confusion Matrix (final epoch):")
        for i in range(12):
            for j in range(12):
                print(self.cms[-1][i, j], end=", ")
            print()

    def on_epoch_end(self, epoch, logs={}):

        # Create new validation model
        val_model = K.function(inputs=[self.model.input, K.learning_phase()], outputs=[self.model.output], )

        # Confusion Matrix
        epoch_cm = np.zeros((len(self.classes), len(self.classes)))

        # Loop through validation set
        for n in range(self.validation_steps):

            # Grab next batch
            X, y_true, _ = next(self.validation_data)

            # Make prediction with model
            learning_phase = 1
            y_pred = val_model([X, learning_phase])

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

        # End of epoch - compute stats
        print("Validation")
        for i in range(12):
            # Recall - TP / (TP + FN)
            try:
                precision = np.round(epoch_cm[i, i] / np.sum(epoch_cm[i, :]), 5)
            except ZeroDivisionError:
                precision = 0
            # Update Logs
            name = self.classes[i] + "_P"
            print(self.classes[i], "P:", precision, end=" ")
            logs[name] = precision

            # Precision - TP / (TP + FP)
            try:
                recall = np.round(epoch_cm[i, i] / np.sum(epoch_cm[:, i]), 5)
            except ZeroDivisionError:
                recall = 0
            # Update Logs
            name = self.classes[i] + "_R"
            print("R:", recall, end=" ")
            logs[name] = recall

            print()

        # PREDICT A SAMPLE OF IMAGES

        # Grab next batch
        X, y_true, _ = next(self.validation_data)

        # Make prediction with model
        learning_phase = 1
        y_pred = val_model([X, learning_phase])[0]

        y_pred = y_pred.reshape(
                            y_pred.shape[0],    # batch_size
                            X[0].shape[0],  # dim
                            X[0].shape[1],  # dim
                            y_pred.shape[-1]    # number of classes
                            )
        y_true = y_true.reshape(
                            y_pred.shape[0],  # batch_size
                            X[0].shape[0],  # dim
                            X[0].shape[1],  # dim
                            y_pred.shape[-1]
                            )

        # Loop through each image in batch                )
        for i in range(y_pred.shape[0]):
            mask, pred = np.argmax(y_true[i], axis=-1), np.argmax(y_pred[i], axis=-1)

            self.write_predict_plot(mask, pred, "Pred_E", epoch)

        # Clear memory of val model
        del val_model
        self.cms.append(epoch_cm)

        # Write confusion matrix to tensorboard
        self.write_current_plot(epoch)

        return


class Test(Callback):
    """
    A custom callback to perform testing at the
    end of the whole training run. Also writes useful class
    metrics to tensorboard logs.
    """

    def __init__(self, generator, steps, classes):
        """

        Initialises the callback

        Input:

            generator -  validation generator of type segmentationGen()

            steps - number of steps in validation e.g. n // batch_size
        """
        self.test_data = generator
        self.test_steps = steps
        self.classes = classes

    def on_train_end(self, epoch, logs={}):

        return


# ------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------ #


if __name__ == "__main__":

    # -------------------------- #
    # Build a demo model to test #
    # -------------------------- #
    def demoModel(dim, num_classes):
        """
        Builds a simple encoder decoder network - don't be too impresssed!
        """
        import numpy as np
        from keras.models import Sequential, Model
        from keras.layers import Input
        from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Conv2DTranspose, Cropping2D
        from keras.layers import concatenate, UpSampling2D, Reshape
        import keras.backend as K

        # Build model
        input_image = Input(shape=(dim, dim, 3))

        conv = Conv2D(24, (3, 3), activation='relu', padding='same')(input_image)

        pool = MaxPooling2D((2, 2), strides=(2, 2), name="pool")(conv)

        conv1x1 = Conv2D(24, (1, 1), padding='same', activation='relu')(pool)

        up = UpSampling2D(size=(2, 2))(conv1x1)
        up_conv = Conv2D(24, 2, activation='relu', padding='same')(up)
        merge = concatenate([conv, up_conv], axis=3)

        conv = Conv2D(12, 3, activation='relu', padding='same')(merge)

        activation = Conv2D(num_classes, (1, 1), activation="softmax")(conv)

        # need to reshape for training
        output = Reshape((dim * dim, 3))(activation)

        model = Model(inputs=[input_image], outputs=output)

        model.summary()

        return model


    # SET RANDOM SEED ---- |
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)
    # -------------------- |

    from keras.optimizers import SGD
    import matplotlib.pyplot as plt

    # Directory Setup
    base_dir = "./Demo_Images/"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    X_dir = os.path.join(train_dir, "img")
    y_dir = os.path.join(train_dir, "tag")
    X_val_dir = os.path.join(test_dir, "img")
    y_val_dir = os.path.join(test_dir, "tag")

    # Batch Size
    batch_size = 5

    # Color Palette
    colours = [(17, 16, 16),(0, 255, 0), (255, 0, 0)]
    palette = Palette(colours)

    # Model parameters
    num_classes = len(palette)
    dim = 128

    # Create Generators 
    train_gen = segmentationGen(batch_size, X_dir, y_dir, palette,
                                x_dim=dim, y_dim=dim, suffix=".jpg")
    val_gen = segmentationGen(batch_size, X_val_dir, y_val_dir,
                              palette,x_dim=dim, y_dim=dim, suffix=".jpg")

    # build demo model
    model = demoModel(dim, num_classes)

    # Compile model - include sampe_weights_mode="temporal"
    model.compile(optimizer=SGD(
            lr=0.001),
            loss="categorical_crossentropy",
            sample_weight_mode="temporal",
            metrics=["accuracy"])

    # Train
    history = model.fit_generator(
        epochs=5,
        generator=train_gen,
        steps_per_epoch=20 // batch_size,
        validation_data=val_gen,
        validation_steps=2)

    # Predict raw image
    im = io.imread(os.path.join(X_dir, "1.jpg"))
    im = resize(im, (dim, dim))
    preds, class_img = predict_image(model, im)

    plt.imshow(class_img)
    plt.show()





