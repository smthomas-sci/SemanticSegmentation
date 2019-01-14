"""

A collection of classes and functions for training fully-convolutional
CNNs for semantic segmentation. Includes a custom generator class for
performing model.fit_genertor() method in Keras. This is specifically used
for segementation ground truth labels which requires sample weights
to be used. (See https://github.com/keras-team/keras/issues/3653)

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 24/10/18
Last Update: 07/01/19

"""

import numpy as np
from skimage import io
from sys import stderr
import h5py
import os
from cv2 import resize
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.colors as cols
from keras.layers import BatchNormalization


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
                    sparse=False,
                    custom_class=[None, None]):
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

        custom_class - a list pair of [classes, color_dictionary] to convert label
                        images to simplier problems. Undesired classes are subsumed
                        into background or other relevant classes.

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
            im = io.imread(os.path.join(X_dir, fname + suffix))
            im = resize(im, (x_dim, x_dim))
            X_batch.append(im)

            # Load y-image
            im = io.imread(os.path.join(y_dir, fname + ".png"))
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
        X_train = np.stack(X_batch, axis=0).astype(np.float32) / 255.
        y_train = np.stack(y_batch, axis=0)

        # Calculate sample weights
        weights = _calculateWeights(y_train)
        # Modify weights
        if weight_mod:
            for i in weight_mod:
                weights[i] *= weight_mod[i]

        # Take weight for each correct position
        sample_weights = np.take(weights, np.argmax(y_train, axis=-1))

        # Reshape to suit keras
        sample_weights = sample_weights.reshape(y_train.shape[0], y_dim*y_dim)
        y_train = y_train.reshape(y_train.shape[0], y_dim*y_dim,
                                  num_classes)

        return X_train, y_train, sample_weights

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
        for i in range(num_classes):
            # Adjusts for absence
            if class_counts[i] == 0:
                class_counts[i] = 1
            y.extend([i]*int(class_counts[i]))
        # Calculate weights
        weights = compute_class_weight("balanced", list(range(num_classes)), y)

        return weights

    # TO IMPLEMENT - Custom
    classes, color_dict = custom_class
    if not classes and not color_dict:
        pass

    # RUN --------------------------------------------------------------------------
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
        im = im / 255.
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
                print("Setting weights for layer:", layer.name)
                layer.set_weights(weights)

            except Exception as e:
                print("Error: Could not load weights for layer:", layer.name, file=stderr)



        # print("Number of layers:", len(file.keys()))
        # weight_file = file["model_1"]
        # for layer_name in weight_file:
        #
        #     print(layer_name, weight_file[layer_name])
        #
        #     layer_weights = weight_file[layer_name]
        #
        #     weights = []
        #     for term in layer_weights:
        #
        #         if isinstance(layer_weights[term], h5py.Dataset):
        #             # Convert weights to numpy array
        #             weights.append(np.array(layer_weights[term]))
        #
        #     # Set weights to layer

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

    up = UpSampling2D(size=(2,2))(conv1x1)
    up_conv =  Conv2D(24, 2, activation = 'relu', padding = 'same')(up)
    merge = concatenate([conv,up_conv], axis = 3)

    conv = Conv2D(12, 3, activation = 'relu', padding = 'same')(merge)

    activation = Conv2D(num_classes, (1, 1), activation = "softmax")(conv)

    # need to reshape for training
    output = Reshape((dim*dim, 3))(activation)

    model = Model(inputs=[input_image], outputs=output)

    model.summary()

    return model


# ---- #
# DEMO #
# ---- #
if __name__ == "__main__":

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
    colours = [(17,16,16),(0,255,0),(255,0,0)]
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





