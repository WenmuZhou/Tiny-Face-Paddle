# Tiny-Face-Paddle
This is a PaddlePaddle implementation of Peiyun Hu's awesome tiny face detector.
下载WIDERFACE数据集，配置路径如下
```
/tmp
|--WIDER_test
|--WIDER_train
|--WIDER_val
|--wider_face_split
```
本仓库利用四卡1080ti耗时三天训练100个epoch，所有文件在此处下载
训练时，下载initial.pdparams权重，运行run.sh
```
bash run.sh
```

测试是，配置完成widerface_evaluate，见里面的readme文件
```shell
python evaluate.py /tmp/WIDERFACE/wider_face_split/wider_face_val_bbx_gt.txt --dataset-root /tmp/WIDERFACE/ --checkpoint weights/checkpoint_80.pdparams --split val
cd widerface_evaluate
python evaluation.py
```
