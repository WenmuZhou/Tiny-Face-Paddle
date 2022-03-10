# Tiny-Face-Paddle
This is a PaddlePaddle implementation of Peiyun Hu's awesome tiny face detector. PaddlePaddle version is 2.2.1-gpu

## 环境配置
除了常用的图像视觉库，你可能需要安装如下依赖
```shell
pip install pyclust treelib pyclustering
```

## 数据准备
下载WIDERFACE数据集，配置路径如下
```
/tmp
|--WIDER_test
|--WIDER_train
|--WIDER_val
|--wider_face_split
```
本仓库利用四卡1080ti耗时三天训练100个epoch，所有文件在[此处下载](https://pan.baidu.com/s/1grltos3o03ybsRwNdy8-DA)，code:ser1，AIStudio项目仓库[点击此处](https://aistudio.baidu.com/aistudio/projectdetail/3208437?contributionType=1)

**注意到：`main_multi_gpu.py`使用`spawn`机制，完美兼容单卡训练**


## 训练
训练时，运行run.sh
```
bash run.sh
```
训练保存日志如下所示：
```
2021-12-27 23:13:18,619 - Tiny-Face-Paddle/trainer.py[line:27] - INFO: Epoch: [0][0/1074]	avg_reader_cost: 25.66170 sec	avg_batch_cost: 29.31611 sec	avg_samples: 12.0 samples	avg_ips 0.40933 images/sec	loss_cls: 101.686981	loss_reg: 1.730689
2021-12-27 23:13:22,903 - Tiny-Face-Paddle/trainer.py[line:27] - INFO: Epoch: [0][1/1074]	avg_reader_cost: 28.01650 sec	avg_batch_cost: 31.45801 sec	avg_samples: 12.0 samples	avg_ips 0.38146 images/sec	loss_cls: 101.402120	loss_reg: 1.744150
2021-12-27 23:13:26,579 - Tiny-Face-Paddle/trainer.py[line:27] - INFO: Epoch: [0][2/1074]	avg_reader_cost: 30.59909 sec	avg_batch_cost: 33.39737 sec	avg_samples: 12.0 samples	avg_ips 0.35931 images/sec	loss_cls: 99.606205	loss_reg: 1.575557
```

## 测试

### widerface_evaluate配置
测试时，通过如下命令配置完成`widerface_evaluate`
```shell
cd widerface_evaluate/
python setup.py build_ext --inplace
```

### 测试文件生成
通过evaluat.py生成测试文件
```shell
python evaluate.py --val_img_root x --val_label_path x --checkpoint x
```

### 精度评估
进入`widerface_evaluate`文件夹，通过`evaluation.py`测试精度
```shell
cd widerface_evaluate
python evaluation.py
```

项目日志通过Logger保存在Experiments文件夹下
原始精度0.902 0.892 0.797
实现精度0.906 0.895 0.789(checkpoint_80.pdparams)

验证时，运行test.sh即可，**注意到要修改路径
每个权重的测试结果在results.txt**

## 推理
运行`detect.py`自动获取示例图像的推断结果，保存为`result.jpg`
```shell
python predict.py --image_path --checkpoint
```
如下所示

<img src="https://user-images.githubusercontent.com/49911294/147483964-896a7991-cfc7-416a-b5d7-3093a798db8f.jpg" width="400"/>   <img src="https://user-images.githubusercontent.com/49911294/147483984-3e887c1b-d6c4-4972-bccd-a34a32888507.jpg" width="400"/>

## 模型导出
运行`export_model.py`进行动转静
```shell
python export_model.py --checkpoint
```

## 推理模型预测
运行`infer.py`进行推理
```shell
python export_model.py --checkpoint
```

# **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| CSDN主页        | [Deep Hao的CSDN主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
| GitHub主页        | [Deep Hao的GitHub主页](https://github.com/GuoQuanhao) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
