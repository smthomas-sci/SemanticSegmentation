import cv2

from seg_utils import *
from seg_models import ResNet_UNet

image = cv2.cvtColor(
    cv2.imread("/home/simon/Documents/PhD/Data/Histo_Segmentation/Datasets_n290/2x/Images/BCC_85.tif"), cv2.COLOR_BGR2RGB)



# Tile properties
max_row = image.shape[0]
max_col = image.shape[1]
dim = 1408

# Model properties
num_classes = 12
model = ResNet_UNet(num_classes=num_classes)
model.load_weights("./weights/Data_100_BS_12_PS_512_C_12_FT_True_E_30_LR_1e-06_WM_F.h5")

# Make Prediction model - no reshape at end
model_in = model.layers[0].get_input_at(0)
model_out = model.layers[-2].output
model = Model(inputs=[model_in], outputs=[model_out])


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
# Contour color
green = tuple([ val / 255. for val in [85, 255, 51] ])
red = tuple([ val / 255. for val in [255, 0, 0] ])
orange = tuple([ val / 255. for val in [255,69,0] ])

# Loop through image
count = 0


print(len(cols), len(rows))

for i in range(0,max_col-dim,50):

    # Grab tile
    tile = image[max_row-dim:max_row, i:i+dim]

    # Input - Pre-process title
    X = np.copy(tile).astype("float32")
    X /= 255.
    X -= 0.5
    X *= 2.

    # Predict segmentation
    pred = model.predict(np.expand_dims(X, 0))[0]

    # Scale prediction map
    h = w = 20
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(tile)
    # Sum across all cancer channels
    map = np.sum(pred[:,:,-3:-1], axis=-1) > 0.5

    #plt.contour(map, levels=[0.9, 0.95], linewidths=3, colors=[orange, red], linestyles="dashed")
    ax.contour(map, linewidths=10, colors=[green], linestyles="dashed")
    #plt.axis("off")

    fname = "/home/simon/Desktop/Contours/out/{:03d}.png".format(count)
    print("Saving...", fname)
    plt.savefig(fname)

    #plt.pause(0.01)

    #plt.cla()
    plt.close()

    count += 1
