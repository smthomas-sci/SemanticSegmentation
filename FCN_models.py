"""

A collection of Encoder-Decoder networks, namely U-net and
U-net like decoders combined with regular CNNs e.g. VGG, ResNEt etc.)
The model architectures are suitbale for training Semantic Segmentation only.
You will need to save the trained model and rebuilt so it can take any input
size. 

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 26/10/18
Last Update: 26/10/18

"""


def VGG_UNet(dim, num_classes, channels=3):
    """
    Returns a VGG16 Nework with a U-Net
    like upsampling stage. Inlcudes 3 skip connections
    from previous VGG layers.

    Input:
        dim - the size of the input image. Note that is should be
              a square of 2 so that downsampling and upsampling
              always match. ie. 128 -> 64 -> 32 -> 64 -> 128

        num_classes - the number of classes in the whole problem. Used to
                      determine the dimension of output map. i.e. model.predict()
                      returns array that can be reshaped to (dim, dim,
                      num_classes).

        channels - number of channels in input image. Defaut of 3 for RGB

    Output:
        model - an uncompied keras model. Check output shape before use.

    """

    import keras.backend as K
    from keras.models import Sequential, Model
    from keras.layers import Input, Dropout, Permute
    from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Conv2DTranspose, Cropping2D
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.applications.vgg16 import VGG16

    # Import a headless VGG16 - extract weighs and then delete
    vgg16 = VGG16(include_top=False)

    weights = []
    for layer in vgg16.layers[1::]:
        weights.append(layer.get_weights())

    del vgg16
    K.clear_session()

    # Build VGG-Unet using functional API

    input_image = Input(shape=(dim, dim, channels))

    # Conv Block 1
    block1_conv1 = Conv2D(64, (3, 3), activation='relu',
                          padding='same',name='block1_conv1')(input_image)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                          name='block1_conv2')(block1_conv1)
    block1_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block1_pool")(block1_conv2)

    # Conv Block 2
    block2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same',
                          name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',
                          name='block2_conv2')(block2_conv1)
    block2_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block2_pool")(block2_conv2)

    # Conv Block 3
    block3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(256, (3, 3),activation='relu',padding='same',
                          name='block3_conv3')(block3_conv2)
    block3_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block3_pool")(block3_conv3)

    # Conv Block 4
    block4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block4_conv3')(block4_conv2)
    block4_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block4_pool")(block4_conv3)


    # Conv Block 5
    block5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block5_conv1')(block4_pool)
    block5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block5_conv2')(block5_conv1)
    block5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block5_conv3')(block5_conv2)
    block5_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block5_pool")(block5_conv3)


    # Upsampling 1
    up1 = UpSampling2D(size=(2,2))(block5_pool)
    up1_conv = Conv2D(512, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up1)
    merge1 = concatenate([block5_conv3,up1_conv], axis = 3)
    merge1_conv1 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1)
    merge1_conv2 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1_conv1)

    # Upsampling 2
    up2 = UpSampling2D(size = (2,2))(merge1_conv2)
    up2_conv = Conv2D(256, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up2)
    merge2 = concatenate([block4_conv3,up2_conv], axis = 3)
    merge2_conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2)
    merge2_conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2_conv1)

    # Upsampling 3
    up3 = UpSampling2D(size = (2,2))(merge2_conv2)
    up3_conv = Conv2D(128, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3)
    merge3 = concatenate([block3_conv3,up3_conv], axis = 3)
    merge3_conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3)
    merge3_conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3_conv1)

    # Upsampling 4
    up4 = UpSampling2D(size=(2,2))(merge3_conv2)
    up4_conv = Conv2D(64, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up4)
    merge4 = concatenate([block2_conv2,up4_conv], axis = 3)
    merge4_conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge4)
    merge4_conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge4_conv1)

    # Upsamplig 5
    up5 = UpSampling2D(size = (2,2))(merge4_conv2)
    up5_conv = Conv2D(64, 2, activation = 'relu', padding = 'same',
                  kernel_initializer = 'he_normal')(up5)
    merge5 = concatenate([block1_conv2,up5_conv], axis = 3)
    merge5_conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(merge5)
    merge5_conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(merge5_conv1)

    # Activation and reshape for training
    activation = Conv2D(num_classes, 1, activation = "softmax")(merge5_conv2)
    output = Reshape((dim*dim, num_classes))(activation)

    # Link model
    model = Model(inputs=[input_image], outputs=output)

    # Set VGG weights and lock from training
    for layer, weight in zip(model.layers[1:19], weights):
        # Set
        layer.set_weights(weight)
        # Lock
        layer.trainable = False

    return model

def ResNet_UNet(dim, num_classes):
    """
    Returns a ResNet50 Nework with a U-Net
    like upsampling stage. Inlcudes 3 skip connections
    from previous VGG layers.

    Input:
        dim - the size of the input image. Note that is should be
              a square of 2 so that downsampling and upsampling
              always match. ie. 128 -> 64 -> 32 -> 64 -> 128

        num_classes - the number of classes in the whole problem. Used to
                      determine the dimension of output map. i.e. model.predict()
                      returns array that can be reshaped to (dim, dim,
                      num_classes).

        channels - number of channels in input image. Defaut of 3 for RGB

    Output:
        model - an uncompiled keras model. Check output shape before use.


    """
    import keras.backend as K
    from keras.models import Sequential, Model
    from keras.layers import Input, Dropout, Permute
    from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Conv2DTranspose, Cropping2D
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.applications.resnet50 import ResNet50

    # Import a headless VGG16
    resnet = ResNet50(input_shape = (dim, dim, 3), include_top=False)

    # Attached U-net from second last layer - activation_49
    res_out = resnet.layers[-2].output

    # Upsampling 1
    up1 = UpSampling2D(size=(2,2))(res_out)
    up1_conv = Conv2D(512, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up1)

    prev_layer = resnet.get_layer("activation_40").output
    merge1 = concatenate([prev_layer,up1_conv], axis = 3)
    merge1_conv1 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1)
    merge1_conv2 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1_conv1)

    # Upsampling 2
    up2 = UpSampling2D(size = (2,2))(merge1_conv2)
    up2_conv = Conv2D(256, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up2)
    prev_layer = resnet.get_layer("activation_22").output
    merge2 = concatenate([prev_layer,up2_conv], axis = 3)
    merge2_conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2)
    merge2_conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2_conv1)

    # Upsampling 3 & 4
    up3 = UpSampling2D(size = (2,2))(merge2_conv2)
    up3_conv1 = Conv2D(128, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3)
    up3_conv2 = Conv2D(128, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3_conv1)
    up4 = UpSampling2D(size = (2,2))(up3_conv2)
    up4_conv = Conv2D(128, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up4)
    prev_layer = resnet.get_layer("activation_1").output
    merge3 = concatenate([prev_layer,up4_conv], axis = 3)
    merge3_conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3)
    merge3_conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3_conv1)

    # Upsample 5
    up5 = UpSampling2D(size = (2,2))(merge3_conv2)
    up5_conv = Conv2D(64, 2, activation = 'relu', padding = 'same',
                  kernel_initializer = 'he_normal')(up5)
    merge5_conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(up5_conv)
    merge5_conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(merge5_conv1)

    # Activation and reshape for training
    activation = Conv2D(num_classes, 1, activation = "softmax")(merge5_conv2)
    output = Reshape((dim*dim, num_classes))(activation)


    # Build model
    model = Model(inputs=[resnet.input], outputs=[output])

    return model




if __name__ == "__main__":

    dim = 512
    num_classes = 20
    #model = VGG_UNet(dim, num_classes)
    #model.summary()
    model = ResNet_UNet(dim, num_classes)
    model.summary()





