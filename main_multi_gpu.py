import argparse
import os
import os.path as osp

import paddle
from paddle.optimizer import Momentum, Adam
import paddle.optimizer.lr as lr_scheduler
from logger import Logger
from paddle.vision import transforms
import paddle.distributed as dist
import time

import trainer
from datasets import get_dataloader
from models.loss import DetectionCriterion
from models.model import DetectionModel


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_img_root", default="")
    parser.add_argument("--val_img_root", default="")
    parser.add_argument("--train_label_path", default="")
    parser.add_argument("--val_label_path", default="")
    parser.add_argument("--dataset", default="WIDERFace")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save-every", default=10, type=int)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save_dir", default="weights")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main():
    dist.init_parallel_env()
    args = arguments()
    os.makedirs(args.save_dir, exist_ok=True)

    log = Logger(os.path.join(args.save_dir, 'training.log'), level='info')
    num_templates = 25  # aka the number of clusters

    img_transforms = transforms.Compose([transforms.Transpose(),
                                        transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                             std=[255, 255, 255]),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    train_loader, _ = get_dataloader(args, args.train_label_path, args.train_img_root, num_templates,
                                     train=True, img_transforms=img_transforms)

    model = DetectionModel(num_objects=1, num_templates=num_templates)
    model = paddle.DataParallel(model)
    loss_fn = DetectionCriterion(num_templates)

    # directory where we'll store model weights

    scheduler = lr_scheduler.StepDecay(learning_rate=args.lr,
                                       step_size=20,
                                       last_epoch=args.start_epoch-1)

    optimizer = Momentum(parameters=model.parameters(),
                         learning_rate=scheduler,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)

    # optimizer = optim.Adam(parameters=model.parameters(), learning_rate==args.lr, weight_decay=args.weight_decay)

    if args.checkpoint:
        checkpoint = paddle.load(args.checkpoint)
        model.set_state_dict(checkpoint['model'])
        optimizer.set_state_dict(checkpoint['optimizer'])
        # Set the start epoch if it has not been
        if not args.start_epoch:
            args.start_epoch = checkpoint['epoch']

    # train and evalute for `epochs`
    for epoch in range(args.start_epoch, args.epochs):
        trainer.train(model, loss_fn, optimizer, train_loader, epoch, log)
        scheduler.step()

        if (epoch+1) % args.save_every == 0:
            trainer.save_checkpoint({
                'epoch': epoch + 1,
                'batch_size': train_loader.batch_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename="checkpoint_{0}.pdparams".format(epoch+1), save_path=args.save_dir)
            trainer.save_checkpoint({
                'epoch': epoch + 1,
                'batch_size': train_loader.batch_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename="latest.pdparams", save_path=args.save_dir)


if __name__ == '__main__':
    # dist.spawn(main)
    main()