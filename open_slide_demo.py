import matplotlib.pyplot as plt
import numpy as np
from openslide import open_slide
from PIL import Image
import os

# IMPORTANT
Image.MAX_IMAGE_PIXELS = None

slide_dir = "./WSI/"
slides = os.listdir(slide_dir)


for fname in slides[-1::]:

    print(fname)
    # Grab size
    size = 256

    # Open ground truth in open slide
    slide = open_slide(os.path.join(slide_dir, fname))

    image = slide.get_thumbnail((size, size))

    print(slide.dimensions)

    # Grab subsection
    #image = np.array(slide.read_region((0, 0), 0, (size, size)))[..., 0:3]


    w = h = 1
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)


    ax.imshow(image)
    ax.set_title(fname)
    plt.show()

    # Close slide
    slide.close()

