from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
import numpy as np


def draw_bounding_box(img, bbox, labels):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    color = tuple(np.random.choice(range(100, 256), size=3))

    draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline=color)

    for i, k in enumerate(labels.keys()):
        w, h = font.getsize(labels[k])
        # draw.rectangle((bbox[0], bbox[1] + i*h, bbox[0] + w, bbox[1] + (i+2)*h), fill=color)
        draw.text((bbox[0], bbox[1] + i*h), "{0}:{1:.3} ".format(k, labels[k]), fill=color)

    return img


def draw_all_boxes(img, bboxes, categories):
    for bbox, c in zip(bboxes, categories):
        img = draw_bounding_box(img, bbox, c)

    img.show()


def visualize_bboxes(image, bboxes):
    """

    :param image: PIL image
    :param bboxes:
    :return:
    """
    print("Number of GT bboxes", bboxes.shape[0])
    for idx, bbox in enumerate(bboxes):
        bbox = np.round(np.array(bbox))
        # print(bbox)
        image = draw_bounding_box(image, bbox, {"name": "{0}".format(idx)})

    image.show(title="BBoxes")

def render_and_save_bboxes(image, image_id, bboxes, scores, scales, directory="qualitative"):
    """
    Render the bboxes on the image and save the image
    :param image: PIL image
    :param image_id:
    :param bboxes:
    :param scores:
    :param scales:
    :param directory:
    :return:
    """
    for idx, bbox in enumerate(bboxes):
        bbox = np.round(np.array(bbox))
        image = draw_bounding_box(image, bbox, {'score': scores[idx], 'scale': scales[idx]})

    image.save("{0}/{1}.jpg".format(directory, image_id))

def get_image_file_list(img_file):
    import imghdr
    import os
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists