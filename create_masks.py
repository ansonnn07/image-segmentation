import json
import matplotlib
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import cv2
import os
import sys
from skimage import io
import matplotlib.pyplot as plt
from IPython.display import display

# add this line to use plt.show() on non-Jupyter environment
matplotlib.use("tkagg")


# def get_mask(row, shape):
## https://github.com/sam-watts/futoshiki-solver/blob/v2.0/puzzle_segmentation/semantic_seg.ipynb
#     row = json.loads(row)
#     coords = np.array(
#         [[x, y] for x, y in zip(row["all_points_x"], row["all_points_y"])]
#     )
#     mask = np.zeros((*shape, 1))
#     cv2.fillPoly(mask, [coords], 255)
#     return mask


def getClassName(classID):
    if classID <= len(CLASSES) - 1:
        return CLASSES[classID]
    raise ValueError(f"The given class ID '{classID}' is not found.")


# Set to True to check several images with masks overlayed
# Set to False to save the mask images
IS_CHECKING = True
SHOW_COCO_MASKS = False
FIGSIZE = (10, 5)
MASK_DIR = os.path.join("dataset", "masks")
if not os.path.exists(MASK_DIR):
    os.makedirs(MASK_DIR)

json_path = r"dataset\result.json"
json_file = json.loads(open(json_path).read())
# print(json_file)
img_dict_list = json_file["images"]
annotations = json_file["annotations"]

images_df = pd.json_normalize(json_file, record_path=["images"])
print("Images")
display(images_df.head())
categories_df = pd.json_normalize(json_file, record_path=["categories"])
print("Categories")
display(categories_df.head())
annotations_df = pd.json_normalize(json_file, record_path=["annotations"])
print("Annotations")
display(annotations_df.head())

# https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047
coco = COCO(json_path)
catIDs = coco.getCatIds()
categories = coco.loadCats(catIDs)
print(f"\n{categories = }")
CLASSES = [i["name"] for i in categories]
# add a background class at index 0
CLASSES = ["background"] + CLASSES
print(f"{CLASSES = }")

# randomly assign colors for different classes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3), dtype=np.uint8)
# add [0, 0, 0] on top of it for background color
COLORS = np.vstack([[0, 0, 0], COLORS]).astype(np.uint8)

# initialize the legend visualization
legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype=np.uint8)
# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
    # draw the class name + color on the legend
    color = [int(c) for c in color]
    cv2.putText(
        legend,
        className,
        (5, (i * 25) + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color), -1)

for i, img_dict in enumerate(img_dict_list):
    img_path = os.path.join("dataset", img_dict["file_name"])
    filename = os.path.basename(img_path)

    annIds = coco.getAnnIds(imgIds=img_dict["id"], catIds=catIDs, iscrowd=None)
    anns = coco.loadAnns(annIds)
    if SHOW_COCO_MASKS:
        print("[INFO] Showing masks using coco.showAnns ...")
        img = io.imread(img_path)
        plt.imshow(img)
        coco.showAnns(anns)
        plt.show()
        continue

    mask = np.zeros((img_dict["height"], img_dict["width"]))
    classes_found = []
    for annot in anns:
        # considering 'background' class at index 0
        className = getClassName(annot["category_id"] + 1)
        classes_found.append(className)
        pixel_value = CLASSES.index(className)
        # print(pixel_value)
        # the final mask contains the pixel values for each class
        mask = np.maximum(coco.annToMask(annot) * pixel_value, mask)
    # print(cv2.resize(mask, (16, 16)))

    if not IS_CHECKING:
        # save the mask images
        mask_path = os.path.join(MASK_DIR, filename)
        cv2.imwrite(mask_path, mask)
    else:
        if i == 5:
            # stop after checking at most 5 images
            break
        print(
            f"{filename} | "
            f"Unique pixel values = {np.unique(mask)} | "
            f"Classes found = {classes_found}"
        )
        ## check one hot mask
        # print(cv2.resize(tf.one_hot(mask, len(CLASSES)).numpy(), (16, 16)).shape)

        ## plot using matplotlib with colorbar (optional)
        # img = io.imread(img_path)
        # plt.figure(figsize=FIGSIZE)
        # plt.title(filename)
        # plt.imshow(img)
        # fig = plt.figure(figsize=FIGSIZE)
        # plt.title("Semantic mask")
        # print(np.max(mask))
        # ax = plt.imshow(mask, cmap="jet", vmin=0, vmax=len(CLASSES) - 1)
        # fig.colorbar(ax, ticks=np.arange(len(CLASSES)))
        # plt.show()

        ## show original and grayscale mask side by side
        # mask = np.expand_dims(mask, axis=-1)
        # mask = np.concatenate([mask, mask, mask], axis=-1)
        # img = img.astype(np.uint8)
        # mask = mask.astype(np.uint8)
        # img = cv2.imread(img_path)
        # cv2.imshow(f"Original | {filename}", img)
        # cv2.imshow(f"Mask | {filename}", mask * 255.)
        # key = cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if key == 27:
        #     break

        # resize the mask such that its dimensions match the
        # original size of the input image
        # https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/
        img = cv2.imread(img_path)
        # given the class ID map obtained from the mask, we can map each of
        # the class IDs to its corresponding color
        colored_mask = COLORS[mask.astype(np.uint8)]
        colored_mask = cv2.resize(
            colored_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        # perform a weighted combination of the input image with the colored_mask to
        # form an output visualization with different colors for each class
        output = ((0.4 * img) + (0.6 * colored_mask)).astype("uint8")
        # show the input and output images
        cv2.imshow("Legend", legend)
        cv2.imshow("Input", img)
        cv2.imshow("mask", mask)
        cv2.imshow("Output", output)

        key = cv2.waitKey(0)
        if key == 27:
            # press 'ESC' to exit
            break
cv2.destroyAllWindows()
