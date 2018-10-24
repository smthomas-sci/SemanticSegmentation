"""

Preipares data for training


Date: 24/10/18
Author: Simon Thomas
Email: simon.thomas@uq.edu.au

"""

import numpy as np
from skimage import io
import os
from cv2 import resize

# Colours

class Palette(object):
    """
    A color pallete which is essentially a channel
    colour LUT.
    """
    def __init__(self, ordered_list):
        """
        Takes and order list of colours and stores in dictionary
        format.
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
    A generator that returns X and y data in designated batch sizes,
    specifically for segmentation problems. It converts y images into
    3D arrays where n-dim = number of classes.

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
    def __init__(self, batch_size, X_dir, y_dir, palette, dim):
        self.batch_size = batch_size
        self.X_dir = X_dir
        self.y_dir = y_dir
        self.files = os.listdir(X_dir)
        self.dim = dim
        self.num_classes = len(palette)
        self.n = len(self.files)
        self.cur = 0
        self.order = list(range(self.n))
        # Randomise the order in which the files are loaded
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
            im = io.imread(os.path.join(X_dir, fname + ".jpg"))
            X_batch.append(im)

            # Load y-image ----------------- ||
            im = io.imread(os.path.join(y_dir, fname + ".png"))
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

        return (X_train, y_train)


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
            X_train, y_train = self.createBatches(positions)

            return X_train, y_train

        # Final batch is smaller than batch_size
        if self.cur < self.n:
            positions = self.order[self.cur::]

            # Step is maximum - next will return None
            self.cur = self.n

            # Create Batches
            X_train, y_train = self.createBatches(positions)

            return X_train, y_train

        else:
            # reshuffle order for next batch
            np.random.shuffle(self.order)

            # Reset cur
            self.cur = 0

            # Signal end of epoch
            raise StopIteration


# BUILD MODEL
def buildModel():
    """
    Builds a simple encoder decoder network
    """
    import numpy as np
    from keras.models import Sequential, Model
    from keras.layers import Input, Dropout, Permute
    from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Conv2DTranspose, Cropping2D
    from keras.layers import concatenate, UpSampling2D
    import keras.backend as K

    from keras.optimizers import SGD
    from keras import backend as keras

    # Build model
    input_image = Input(shape=(512, 512, 3))

    block1_conv1 = Conv2D(12, (3, 3), input_shape=(512, 512, 3),
                          activation='relu',
                          padding='same',name='block1_conv1')(input_image)

    block1_conv2 = Conv2D(12, (3, 3), activation='relu', padding='same',
                          name='block1_conv2')(block1_conv1)

    block1_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block1_pool")(block1_conv2)


    fc7 = Conv2D(200, (1, 1), padding='same', activation='relu',
                 name='fc7')(block1_pool)
    # Score layer
    #score = Conv2D(4, (1, 1), padding='same', activation='relu', name='score')(fc7)

    # Upsampling - Unet Decoder starts here...
    up1 = UpSampling2D(size=(2,2))(fc7)
    up1_conv =  Conv2D(12, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up1)
    merge1 = concatenate([block1_conv2,up1_conv], axis = 3)

    conv1 = Conv2D(24, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1)

    conv2 = Conv2D(12, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv1)

    output = Conv2D(3, (1, 1), activation = "softmax")(conv2)

    model = Model(inputs=[input_image], outputs=output)

    model.compile(optimizer=SGD(lr=0.01), loss="categorical_crossentropy",
                  metrics = ["accuracy"])

    model.summary()

    return model


# ---------------------------------------------------------------------------------# 

# Test
if __name__ == "__main__":

    #model = buildModel()


    # ------------------------------------------------------------------ #

    #   Directories
    base_dir = "/home/simon/Documents/PhD/Data/EarPen"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    X_dir = os.path.join(train_dir, "img")
    y_dir = os.path.join(train_dir, "tag")


    # Batch Size
    batch_size = 5


    # Test Palette
    colours = [(0,0,0),(0,255,0),(255,0,0)]

    palette = Palette(colours)

    gen = SegmentationGen(batch_size, X_dir, y_dir, palette, dim = 512)

    # Test loop
    X_train, y_train = next(gen)
    print(X_train.shape, y_train.shape)



    print("end.")


#
#history = model.fit_generator(
#         train_generator,
#         steps_per_epoch = T_spe,
#         epochs = 50,
#         validation_data = validation_generator,
#         validation_steps = V_spe,
#         callbacks = callback_list
#
# X, y = test_generator.next()

#model.fit_generator(generator=training_generator,
#                validation_data=validation_generator,
#                use_multiprocessing=True,
#                workers=6)


