import paddle
from paddle import nn

from .utils import balance_sampling


class AvgMeter:
    def __init__(self):
        self.average = 0
        self.num_averaged = 0

    def update(self, loss, size):
        n = self.num_averaged
        m = n + size
        self.average = ((n * self.average) + float(loss)) / m
        self.num_averaged = m

    def reset(self):
        self.average = 0
        self.num_averaged = 0


def soft_margin_loss(input, label, reduction):
    assert reduction in ['none', 'mean', 'sum']
    if reduction == 'none':
        return paddle.log(1+paddle.exp(-input.multiply(label.astype(paddle.float32))))
    if reduction == 'mean':
        return paddle.sum(paddle.log(1+paddle.exp(-input.multiply(label.astype(paddle.float32)))) / input.numel())
    if reduction == 'sum':
        return paddle.sum(paddle.log(1+paddle.exp(-input.multiply(label.astype(paddle.float32)))))


class SoftMarginLoss(nn.Layer):
    """
    SoftMarginLoss Implement
    """
    def __init__(self, reduction='none'):
        super(SoftMarginLoss, self).__init__()
        self.reduction = reduction
        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, input, label):
        return soft_margin_loss(input, label, self.reduction)


class DetectionCriterion(nn.Layer):
    """
    The loss for the Tiny Faces detector
    """

    def __init__(self, n_templates=25, reg_weight=1, pos_fraction=0.5):
        super().__init__()

        # We don't want per element averaging.
        # We want to normalize over the batch or positive samples.
        self.regression_criterion = nn.SmoothL1Loss(reduction='none')
        self.classification_criterion = SoftMarginLoss(reduction='none')
        self.n_templates = n_templates
        self.reg_weight = reg_weight
        self.pos_fraction = pos_fraction

        self.class_average = AvgMeter()
        self.reg_average = AvgMeter()

        self.masked_class_loss = None
        self.masked_reg_loss = None
        self.total_loss = None

    def balance_sample(self, class_map):
        label_class_np = class_map.cpu().numpy()
        # iterate through batch
        for idx in range(label_class_np.shape[0]):
            label_class_np[idx, ...] = balance_sampling(label_class_np[idx, ...],
                                                        pos_fraction=self.pos_fraction)

        class_map = paddle.to_tensor(label_class_np)

        return class_map

    def hard_negative_mining(self, classification, class_map):
        loss_class_map = soft_margin_loss(classification.detach(), class_map,
                                                        reduction='none')
        class_map[loss_class_map < 0.03] = 0
        return class_map

    def forward(self, output, class_map, regression_map):
        classification = output[:, 0:self.n_templates, :, :]
        regression = output[:, self.n_templates:, :, :]

        # online hard negative mining
        class_map = self.hard_negative_mining(classification, class_map)
        # balance sampling
        class_map = self.balance_sample(class_map)

        class_loss = self.classification_criterion(classification, class_map)

       # weights used to mask out invalid regions i.e. where the label is 0
        class_mask = (class_map != 0).astype(output.dtype)
        # Mask the classification loss
        self.masked_class_loss = class_mask * class_loss

        reg_loss = self.regression_criterion(regression, regression_map)
        # make same size as reg_map
        reg_mask = (class_map > 0).tile([1, 4, 1, 1]).astype(output.dtype)

        self.masked_reg_loss = reg_mask * reg_loss  # / reg_loss.shape[0]

        self.total_loss = self.masked_class_loss.sum() + \
            self.reg_weight * self.masked_reg_loss.sum()

        self.class_average.update(self.masked_class_loss.sum(), output.shape[0])
        self.reg_average.update(self.masked_reg_loss.sum(), output.shape[0])

        return self.total_loss

    def reset(self):
        self.class_average.reset()
        self.reg_average.reset()
