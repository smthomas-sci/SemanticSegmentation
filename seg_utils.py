"""

A custom generator class (+ associated classes) for performing
model.fit_genertor() method in Keras. This is specifically used
for segementation ground truth labels. It requires sample weights
to be used.

See: https://github.com/keras-team/keras/issues/3653

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 24/10/18
Last Update: 25/10/18

"""

import numpy as np
from skimage import io
import os
from cv2 import resize
from sklearn.utils.class_weight import compute_class_weight


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
            orderd_list - list of rgb tuples in class order

        Output:
            self[index] - rgb tuple associated with index/class
        """

        self.colors = dict((i, color) for (i, color) in enumerate(ordered_list))

    def __getitem__(self, arg):
        """
        Returns item with input key, i.e. channel
        """
        return colours[arg]


    def __str__(self):
        """
        Print representation
        """
        return "Channel Color Palette:\n" + str(self.colors)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.colors.keys())



class SegmentationGen(object):
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
        ...                 steps_per_epoch = train_gen.n // batch_size)
        >>> # Evaluate
        ... loss, acc = model.evaluate_generator(
        ...                 generator = val_gen,
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

    Output:
        using the global next() function or internal next() function the class
        returns X_train, y_train numpy arrays:
            X_train.shape = (batch_size, image_size, dim, 3)
            y_train.shape = (batch_size, image_size, dim, num_classes)
    """
    def __init__(self, batch_size, X_dir, y_dir, palette, x_dim, y_dim):
        self.batch_size = batch_size
        self.X_dir = X_dir
        self.y_dir = y_dir
        self.files = os.listdir(X_dir)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_classes = len(palette)
        self.n = len(self.files)
        self.cur = 0
        self.order = list(range(self.n))
        np.random.shuffle(self.order)


    def getClassMask(self, rgb, im):
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
        r, g, b = rgb
        r_mask = im[:,:, 0] == r
        g_mask = im[:,:, 1] == g
        b_mask = im[:,:, 2] == b
        mask = r_mask & g_mask & b_mask
        return mask

    def createBatches(self, positions):
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
            fname = self.files[pos][:-4]

            # load X-image
            im = io.imread(os.path.join(self.X_dir, fname + ".jpg"))
            im = resize(im, (self.x_dim, self.x_dim))
            X_batch.append(im)

            # Load y-image ----------------- ||
            im = io.imread(os.path.join(self.y_dir, fname + ".png"))
            im = resize(im, (self.y_dim, self.y_dim))
            # Convert to 3D ground truth
            y = np.zeros((im.shape[0], im.shape[1], self.num_classes),
                                    dtype=np.float32)
            # Loop through colors in palette and assign to new array
            for i in range(self.num_classes):
                rgb = palette[i]
                mask = self.getClassMask(rgb, im)
                y[mask, i] = 1.

            y_batch.append(y)

        # Combine images into batches
        X_train = np.stack(X_batch, axis=0).astype(np.float32) / 255;
        y_train = np.stack(y_batch, axis=0)

        # Calculate sample weights
        weights = self._calculateWeights(y_train)


        # Take weight for each correct position
        sample_weights = np.take(weights, np.argmax(y_train, axis=-1))

        # Reshape to suit keras
        sample_weights = sample_weights.reshape(y_train.shape[0], self.y_dim*self.y_dim)
        y_train = y_train.reshape(y_train.shape[0], self.y_dim*self.y_dim,
                                  self.num_classes)

        return (X_train, y_train, sample_weights)

    def _calculateWeights(self, y_train):
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
        for i in range(self.num_classes):
            batch_count = 0
            # Sum up each class count in each batch image
            for b in range(y_train.shape[0]):
                batch_count += np.sum(y_train[b][:,:,i])
            class_counts.append(batch_count)

        # create Counts
        y = []
        for i in range(self.num_classes):
            y.extend([i]*int(class_counts[i]))
        # Calcualte weights
        weights = compute_class_weight("balanced", list(range(self.num_classes)), y)

        return weights

    def __iter__(self):
        return self


    def __next__(self):
        return self.next()

    def next(self):
        """
        Returns the next X_train and y_train arrays.
        """
        # Most batches will be equal to batch_size
        if self.cur < (self.n - self.batch_size):
            # Get positions of files in batch
            positions = self.order[self.cur:self.cur + self.batch_size]

            self.cur += self.batch_size

            # create Batches
            X_train, y_train, sample_weights = self.createBatches(positions)

            return X_train, y_train, sample_weights

        # Final batch is smaller than batch_size
        if self.cur < self.n:
            positions = self.order[self.cur::]

            # Step is maximum - next will return None
            self.cur = self.n

            # Create Batches
            X_train, y_train, sample_weights = self.createBatches(positions)

            return X_train, y_train, sample_weights

        else:
            # reshuffle order for next batch
            np.random.shuffle(self.order)

            # Reset cur
            self.cur = 0

            # Signal end of epoch
            return None


def predictImage(model, image):
    """
    Simplifies image prediction for segmentation models. Automatically
    reshapes output so it can be visualised.

    Input:
        model - CNN keras mode
        image - rgb image of shape (dim, dim, 3) where dim == model.input_shape

    Output:
        preds - probability heatmap of shape (dim, dim, num_classes)
        class_img - argmax of preds of shape (dim, dim, 1)
    """
    # Reshape
    x = image[np.newaxis, ::]

    # Standardise range
    x = x.astype(np.float32) / 255.

    # Prediction
    preds = model.predict(x)[0].reshape(image.shape[0],
                                           image.shape[0],
                                           model.layers[-1].output_shape[-1])
    # class_img
    class_img = np.argmax(preds, axis=-1)

    return (preds, class_img)



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

    x = Conv2D(num_classes, (1, 1), activation = "softmax")(conv)

    # need to reshape for training
    output = Reshape((dim*dim, 3))(x)

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
    import tensorflow as tf
    import matplotlib.pyplot as plt


    #   Directory Setup
    base_dir = "../."
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    X_dir = os.path.join(train_dir, "img")
    y_dir = os.path.join(train_dir, "tag")
    X_val_dir = os.path.join(test_dir, "img")
    y_val_dir = os.path.join(test_dir, "tag")

    # Batch Size
    batch_size = 10

    # Color Palette
    colours = [(17,16,16),(0,255,0),(255,0,0)]
    palette = Palette(colours)

    # Model parameters
    num_classes = len(palette)
    dim = 128

    # Create Generators 
    train_gen = SegmentationGen(batch_size, X_dir, y_dir, palette,
                                x_dim=dim, y_dim=dim)
    val_gen = SegmentationGen(batch_size, X_val_dir, y_val_dir,
                              palette,x_dim=dim, y_dim=dim)

    # build demo model
    model = demoModel(dim, num_classes)

    # Load best model from before
    model.load_weights("best_model.h5")


    # Compile mode - include sampe_weights_mode="temporal"
    model.compile(optimizer=SGD(
            lr=0.001),
            loss="categorical_crossentropy",
            sample_weight_mode="temporal",
            metrics=["accuracy"])

    # Train
    history = model.fit_generator(
        generator = train_gen,
        steps_per_epoch = train_gen.n // batch_size)

    # Evaluate
    loss, acc = model.evaluate_generator(
        generator = val_gen,
        steps = 2)

    print("Val Loss:", loss, ", Val Acc:", acc)


    # Predict raw image
    im = io.imread(os.path.join(X_dir, "1.jpg"))
    im = resize(im, (dim, dim))
    preds, class_img = predictImage(model, im)

    plt.imshow(class_img)
    plt.show()





