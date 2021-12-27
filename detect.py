import numpy as np
from paddle.vision import transforms
from models.model import DetectionModel
import paddle
import trainer
import json
import time
import cv2


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

def main():
    img = cv2.cvtColor(cv2.imread('./assets/test.jpg'), cv2.COLOR_BGR2RGB).astype('float32')
    img_raw = cv2.imread('./assets/test.jpg')

    input = paddle.to_tensor(val_transforms(img)).unsqueeze(0)

    model = DetectionModel(num_objects=1, num_templates=num_templates)
    model.set_state_dict(paddle.load('./weights/checkpoint_80.pdparams')["model"])
    start = time.time()
    dets = trainer.get_detections(model, input, templates, rf,
                                val_transforms, prob_thresh,
                                nms_thresh)
    end = time.time()
    for idx, bbox in enumerate(dets):
        bbox = np.round(bbox)
        cv2.rectangle(img_raw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    print("Inference Speed:", end-start)
    print("Saved result.jpg")
    cv2.imwrite('result.jpg', img_raw)


if __name__ == '__main__':
    main()

