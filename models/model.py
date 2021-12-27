import numpy as np
import math
import paddle
from paddle import nn
from paddle.vision.models import resnet50, resnet101
from paddle import ParamAttr
from .initializer import kaiming_uniform_, _calculate_fan_in_and_fan_out
import sys


class DetectionModel(nn.Layer):
    """
    Hybrid Model from Tiny Faces paper
    """

    def __init__(self, base_model=resnet101, num_templates=1, num_objects=1):
        super().__init__()
        # 4 is for the bounding box offsets
        output = (num_objects + 4)*num_templates
        self.model = base_model(pretrained=True)

        # delete unneeded layer
        del self.model.layer4
        self.score_res3 = nn.Conv2D(in_channels=512,
                                    out_channels=output,
                                    kernel_size=1,
                                    padding=0,
                                    weight_attr=ParamAttr(learning_rate=0.1),
                                    bias_attr=ParamAttr(learning_rate=0.1))

        self.score_res3.weight.set_value(kaiming_uniform_(paddle.empty(self.score_res3.weight.shape, dtype=paddle.float32), a=math.sqrt(5)))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.score_res3.weight)
        bound = 1 / math.sqrt(fan_in)
        self.score_res3.bias.set_value(paddle.uniform(shape=self.score_res3.bias.shape, dtype='float32', min=-bound, max=bound))
        
        self.score_res4 = nn.Conv2D(in_channels=1024, out_channels=output,
                                    kernel_size=1, padding=0)
        self.score_res4.weight.set_value(kaiming_uniform_(paddle.empty(self.score_res4.weight.shape, dtype=paddle.float32), a=math.sqrt(5)))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.score_res4.weight)
        bound = 1 / math.sqrt(fan_in)
        self.score_res4.bias.set_value(paddle.uniform(shape=self.score_res4.bias.shape, dtype='float32', min=-bound, max=bound))

        self.score4_upsample = nn.Conv2DTranspose(in_channels=output,
                                                  out_channels=output,
                                                  kernel_size=4,
                                                  stride=2,
                                                  padding=1,
                                                  weight_attr=ParamAttr(learning_rate=0.0),
                                                  bias_attr=False)
        self._init_bilinear()

    def _init_weights(self):
        pass

    def _init_bilinear(self):
        """
        Initialize the ConvTranspose2d layer with a bilinear interpolation mapping
        :return:
        """
        k = self.score4_upsample._kernel_size[0]
        factor = np.floor((k+1)/2)
        if k % 2 == 1:
            center = factor
        else:
            center = factor + 0.5
        C = np.arange(1, 5)

        f = np.zeros((self.score4_upsample._in_channels,
                      self.score4_upsample._out_channels, k, k))

        for i in range(self.score4_upsample._out_channels):
            f[i, i, :, :] = (np.ones((1, k)) - (np.abs(C-center)/factor)).T @ \
                            (np.ones((1, k)) - (np.abs(C-center)/factor))

        self.score4_upsample.weight.set_value(paddle.to_tensor(f, dtype=paddle.float32))


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        # res2 = x

        x = self.model.layer2(x)
        res3 = x

        x = self.model.layer3(x)
        res4 = x

        score_res3 = self.score_res3(res3)

        score_res4 = self.score_res4(res4)
        score4 = self.score4_upsample(score_res4)

        # We need to do some fancy cropping to accomodate the difference in image sizes in eval
        if not self.training:
            # from vl_feats DagNN Crop
            cropv = score4.shape[2] - score_res3.shape[2]
            cropu = score4.shape[3] - score_res3.shape[3]
            # if the crop is 0 (both the input sizes are the same)
            # we do some arithmetic to allow python to index correctly
            if cropv == 0:
                cropv = -score4.shape[2]
            if cropu == 0:
                cropu = -score4.shape[3]

            score4 = score4[:, :, 0:-cropv, 0:-cropu]
        else:
            # match the dimensions arbitrarily
            score4 = score4[:, :, 0:score_res3.shape[2], 0:score_res3.shape[3]]

        score = score_res3 + score4

        return score

