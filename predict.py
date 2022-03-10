import numpy as np
from paddle.vision import transforms
from models.model import DetectionModel
import paddle
import trainer
import json
import time
import cv2
import os
from argparse import ArgumentParser
from utils.visualize import get_image_file_list


num_templates = 25
nms_thresh = 0.2
prob_thresh = 0.9
templates = json.load(open('./datasets/templates.json'))
templates = np.round_(np.array(templates), decimals=8)

dets = np.empty((0, 5))  # store bbox (x1, y1, x2, y2), score
rf = {'size': [859, 859],
      'stride': [8, 8],
      'offset': [-1, -1]}
val_transforms = transforms.Compose([transforms.Transpose(),
                                     transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                          std=[255, 255, 255]),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

def main(args):
    img_list = get_image_file_list(args.image_path)
    

    model = DetectionModel(num_objects=1, num_templates=num_templates)
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = paddle.load(args.checkpoint)
            model.set_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))
    os.makedirs(args.save_path,exist_ok=True)
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        start = time.time()
        img_raw = cv2.imread(img_path)
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB).astype('float32')

        input = paddle.to_tensor(val_transforms(img)).unsqueeze(0)
        dets = trainer.get_detections(model, input, templates, rf,
                                    val_transforms, prob_thresh,
                                    nms_thresh,scales=[0])
        end = time.time()
        for idx, bbox in enumerate(dets):
            bbox = np.round(bbox)
            cv2.rectangle(img_raw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        print("Inference Speed:", end-start)
        cv2.imwrite(os.path.join(args.save_path,img_name), img_raw)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, help="path to test image")
    parser.add_argument('--checkpoint', type=str,
                        default="",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_path', default="inference_models", help="inference model save path")
    args = parser.parse_args()
    main(args)

