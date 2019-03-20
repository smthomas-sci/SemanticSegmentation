import cv2
import os
import matplotlib.pyplot as plt



out_dir = "/home/simon/Desktop/Outs/Masks/Test_Comps/"
pred_dir = "/home/simon/Desktop/Outs/Masks/2x_290_test/"
mask_dir = "/home/simon/Documents/PhD/Data/Histo_Segmentation/Datasets_n290/2x/Masks/"
image_dir = "/home/simon/Documents/PhD/Data/Histo_Segmentation/Datasets_n290/2x/Images/"


files = os.listdir(pred_dir)

step = 1
for file in files:
    print("Processing", step, "of", len(files))
    fname = "_".join(file.split("_")[0:-2])

    # Load images
    image = cv2.cvtColor(cv2.imread(os.path.join(image_dir, fname + ".tif")), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, fname + ".png")), cv2.COLOR_BGR2RGB)
    pred = cv2.cvtColor(cv2.imread(os.path.join(pred_dir, file)), cv2.COLOR_BGR2RGB)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), frameon=False)
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(mask)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.savefig(out_dir + fname + ".png")

    plt.close()

    step += 1
