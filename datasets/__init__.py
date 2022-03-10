import numpy as np
import os.path as osp
import json
from utils.cluster import compute_kmedoids
from .wider_face import WIDERFace
from paddle.io import DataLoader, DistributedBatchSampler


def get_dataloader(args, label_path, img_root, num_templates=25,
                   template_file="templates.json", img_transforms=None,
                   train=True):
    template_file = osp.join("datasets", template_file)

    if osp.exists(template_file):
        templates = json.load(open(template_file))

    else:
        # Cluster the bounding boxes to get the templates
        dataset = WIDERFace(label_path, img_root, [],train=train)
        clustering = compute_kmedoids(dataset.get_all_bboxes(), 1, indices=num_templates,
                                      option='pyclustering', max_clusters=num_templates)

        print("Canonical bounding boxes computed")
        templates = clustering[num_templates]['medoids'].tolist()

        # record templates
        json.dump(templates, open(template_file, "w"))

    templates = np.round_(np.array(templates), decimals=8)

    dataset = WIDERFace(label_path, img_root, templates,
                        img_transforms=img_transforms,
                        debug=args.debug,
                        train=train)
    batch_sampler=DistributedBatchSampler(dataset, batch_size=args.batch_size, shuffle=train)
    data_loader = DataLoader(dataset, batch_sampler = batch_sampler, num_workers = args.workers)
    return data_loader, templates
