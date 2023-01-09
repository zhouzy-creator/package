#!/usr/bin/env python
# coding: utf-8


import mindspore
import   mindspore.dataset  as   ds
import mindspore.dataset.vision.c_transforms  as  transforms
class_labels={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22, '23': 23, '24': 24, '25': 25, '26': 26, '27': 27, '28': 28, '29': 29, '30': 30, '31': 31, '32': 32, '33': 33, '34': 34, '35': 35, '36': 36, '37': 37, '38': 38, '39': 39, '40': 40, '41': 41, '42': 42, '43': 43, '44': 44, '45': 45, '46': 46, '47': 47, '48': 48, '49': 49, '50': 50, '51': 51, '52': 52, '53': 53}

#{'工艺品/仿唐三彩': 0, '工艺品/仿宋木叶盏': 1, '工艺品/布贴绣': 2, '工艺品/景泰蓝': 3, '工艺品/木马勺脸谱': 4, '工艺品/柳编': 5, '工艺品/葡萄花鸟纹银香囊': 6, '工艺品/西安剪纸': 7, '工艺品/陕历博唐妞系列': 8, '景点/关中书院': 9, '景点/兵马俑': 10, '景点/南五台': 11, '景点/大兴善寺': 12, '景点/大观楼': 13, '景点/大雁塔': 14, '景点/小雁塔': 15, '景点/未央宫城墙遗址': 16, '景点/水陆庵壁塑': 17, '景点/汉长安城遗址': 18, '景点/西安城墙': 19, '景点/钟楼': 20, '景点/长安华严寺': 21, '景点/阿房宫遗址': 22, '民俗/唢呐': 23, '民俗/皮影': 24, '特产/临潼火晶柿子': 25, '特产/山茱萸': 26, '特产/玉器': 27, '特产/阎良甜瓜': 28, '特产/陕北红小豆': 29, '特产/高陵冬枣': 30, '美食/八宝玫瑰镜糕': 31, '美食/凉皮': 32, '美食/凉鱼': 33, '美食/德懋恭水晶饼': 34, '美食/搅团': 35, '美食/枸杞炖银耳': 36, '美食/柿子饼': 37, '美食/浆水面': 38, '美食/灌汤包': 39, '美食/烧肘子': 40, '美食/石子饼': 41, '美食/神仙粉': 42, '美食/粉汤羊血': 43, '美食/羊肉泡馍': 44, '美食/肉夹馍': 45, '美食/荞面饸饹': 46, '美食/菠菜面': 47, '美食/蜂蜜凉粽子': 48, '美食/蜜饯张口酥饺': 49, '美食/西安油茶': 50, '美食/贵妃鸡翅': 51, '美食/醪糟': 52, '美食/金线油塔': 53}
image_size=380
mindspore.set_seed(666777)

#mindspore.context.set_context(device_target="Ascend")

def create_dataset(path, batch_size=32, train=True, image_size=image_size):
    dataset = ds.ImageFolderDataset(path, num_parallel_workers=8, class_indexing=class_labels)

    # 图像增强操作
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if train:
        trans = [
            transforms.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(prob=0.5),
            transforms.Normalize(mean=mean, std=std),
            transforms.HWC2CHW()
        ]
    else:
        trans = [
            transforms.Decode(),
            transforms.Resize([image_size,image_size]),
            transforms.Normalize(mean=mean, std=std),
            transforms.HWC2CHW()
        ]

    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=8)
    # 设置batch_size的大小，若最后一次抓取的样本数小于batch_size，则丢弃
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset



# 加载训练数据集
train_path = "/data/deep/mindcon/food_dataset/train"
dataset_train = create_dataset(train_path, train=True)

# 加载验证数据集
val_path = "/data/deep/mindcon/food_dataset/val"
dataset_val = create_dataset(val_path, train=False)



from mindspore import Tensor
from typing import Any, Type, Union, List,Optional
from mindspore import nn
from mindspore.ops import operations as P
class ConvNormActivation(nn.Cell):
    """
    Convolution/Depthwise fused with normalization and activation blocks definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
        layer. Default: nn.BatchNorm2d.
        activation (nn.Cell, optional): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.ReLU.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> conv = ConvNormActivation(16, 256, kernel_size=1, stride=1, groups=1)
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm: Optional[nn.Cell] = nn.BatchNorm2d,
                 activation: Optional[nn.Cell] = nn.ReLU
                 ) -> None:
        super(ConvNormActivation, self).__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                pad_mode='pad',
                padding=padding,
                group=groups
            )
        ]

        if norm:
            layers.append(norm(out_planes))
        if activation:
            layers.append(activation())

        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output

class GlobalAvgPooling(nn.Cell):
    """
    Global avg pooling definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GlobalAvgPooling()
    """

    def __init__(self,
                 keep_dims: bool = False
                 ) -> None:
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x
    
class DenseHead(nn.Cell):
    """
    LinearClsHead architecture.

    Args:
        input_channel (int): The number of input channel.
        num_classes (int): Number of classes.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive]): activate function applied to the output. Eg. `ReLU`. Default: None.
        keep_prob (float): Dropout keeping rate, between [0, 1]. E.g. rate=0.9, means dropping out 10% of input.
            Default: 1.0.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self,
                 input_channel: int,
                 num_classes: int,
                 has_bias: bool = True,
                 activation: Optional[Union[str, nn.Cell]] = None,
                 keep_prob: float = 1.0
                 ) -> None:
        super(DenseHead, self).__init__()

        self.dropout = nn.Dropout(keep_prob)
        self.dense = nn.Dense(input_channel, num_classes, has_bias=has_bias, activation=activation)

    def construct(self, x):
        if self.training:
            x = self.dropout(x)
        x = self.dense(x)
        return x

class BaseClassifier(nn.Cell):
    """
    generate classifier
    """

    def __init__(self, backbone, neck=None, head=None):
        super(BaseClassifier, self).__init__()
        self.backbone = build_backbone(backbone) if isinstance(backbone, dict) else backbone
        if neck:
            self.neck = build_neck(neck) if isinstance(neck, dict) else neck
            self.with_neck = True
        else:
            self.with_neck = False
        if head:
            self.head = build_head(head) if isinstance(head, dict) else head
            self.with_head = True
        else:
            self.with_head = False

    def construct(self, x):
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        if self.with_head:
            x = self.head(x)
        return x
    

class ResidualBlockBase(nn.Cell):
    """
    ResNet residual block base definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        group (int): Group convolutions. Default: 1.
        base_with (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlockBase(3, 256, stride=2)
    """

    expansion: int = 1

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 stride: int = 1,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None
                 ) -> None:
        super(ResidualBlockBase, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d
        assert group != 1 or base_width == 64, "ResidualBlockBase only supports groups=1 and base_width=64"
        self.conv1 = ConvNormActivation(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            norm=norm)
        self.conv2 = ConvNormActivation(
            out_channel,
            out_channel,
            kernel_size=3,
            norm=norm,
            activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlockBase construct."""
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)

        return out


class ResidualBlock(nn.Cell):
    """
    ResNet residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the second convolutional layer. Default: 1.
        group (int): Group convolutions. Default: 1.
        base_with (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.
        down_sample (nn.Cell, optional): Downsample structure. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """

    expansion: int = 4

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 stride: int = 1,
                 group: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None
                 ) -> None:
        super(ResidualBlock, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d
        out_channel = int(out_channel * (base_width / 64.0)) * group

        self.conv1 = ConvNormActivation(
            in_channel, out_channel, kernel_size=1, norm=norm)
        self.conv2 = ConvNormActivation(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            groups=group,
            norm=norm)
        self.conv3 = ConvNormActivation(
            out_channel,
            out_channel *
            self.expansion,
            kernel_size=1,
            norm=norm,
            activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlock construct."""
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.down_sample:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Type[Union[ResidualBlockBase, ResidualBlock]]): Block for network.
        layer_nums (List[int]): Numbers of block in different layers.
        group (int): Group convolutions. Default: 1.
        base_width (int): Width of per group. Default: 64.
        norm (nn.Cell, optional): Module specifying the normalization layer to use. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet(ResidualBlock, [3, 4, 6, 3])
    """

    def __init__(self,
                 block: Type[Union[ResidualBlockBase, ResidualBlock]],
                 layer_nums: List[int],
                 group: int = 1,
                 base_with: int = 64,
                 norm: Optional[nn.Cell] = None
                 ) -> None:
        super(ResNet, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d
        self.norm = norm
        self.in_channels = 64
        self.group = group
        self.base_with = base_with
        self.conv1 = ConvNormActivation(
            3, self.in_channels, kernel_size=7, stride=2, norm=norm)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layer_nums[0])
        self.layer2 = self._make_layer(block, 128, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_nums[3], stride=2)

    def _make_layer(self,
                    block: Type[Union[ResidualBlockBase, ResidualBlock]],
                    channel: int,
                    block_nums: int,
                    stride: int = 1
                    ):
        """Block layers."""
        down_sample = None

        if stride != 1 or self.in_channels != self.in_channels * block.expansion:
            down_sample = ConvNormActivation(
                self.in_channels,
                channel * block.expansion,
                kernel_size=1,
                stride=stride,
                norm=self.norm,
                activation=None)
        layers = []
        layers.append(
            block(
                self.in_channels,
                channel,
                stride=stride,
                down_sample=down_sample,
                group=self.group,
                base_width=self.base_with,
                norm=self.norm
            )
        )
        self.in_channels = channel * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.in_channels,
                    channel,
                    group=self.group,
                    base_width=self.base_with,
                    norm=self.norm
                )
            )

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def _resnet(arch: str,
            block: Type[Union[ResidualBlockBase, ResidualBlock]],
            layers: List[int],
            num_classes: int,
            pretrained: bool,
            input_channel: int,
            **kwargs: Any
            ) -> ResNet:
    """ResNet architecture."""
    backbone = ResNet(block, layers, **kwargs)
    neck = GlobalAvgPooling()
    head = DenseHead(input_channel=input_channel, num_classes=num_classes)
    model = BaseClassifier(backbone, neck, head)

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load
        # checkpoint file.
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model




def resnet18(
        num_classes: int = 1000,
        pretrained: bool = False,
        **kwargs: Any) -> ResNet:
    """
    ResNet18 architecture.

    Args:
        num_classes (int): Number of classification. Default: 1000.
        pretrained (bool): Download and load the pre-trained model. Default: False.

    Returns:
        ResNet

    Examples:
        >>> resnet18(num_classes=10, pretrained=True, **kwargs)
    """
    return _resnet(
        "resnet18", ResidualBlockBase, [
            2, 2, 2, 2], num_classes, pretrained, 512, **kwargs)



def resnet50(
        num_classes: int = 1000,
        pretrained: bool = False,
        **kwargs: Any) -> ResNet:
    """
    ResNet50 architecture.

    Args:
        num_classes (int): Number of classification. Default: 1000.
        pretrained (bool): Download and load the pre-trained model. Default: False.

    Returns:
        ResNet

    Examples:
        >>> resnet50(num_classes=10, pretrained=True, **kwargs)
    """
    return _resnet(
        "resnet50", ResidualBlock, [
            3, 4, 6, 3], num_classes, pretrained, 2048, **kwargs)



import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.loss import LossBase
import mindspore.ops as ops

class CrossEntropySmooth(LossBase):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = mindspore.Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = ms.Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss




from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net
from mindspore import nn
LR = 1e-3  

network = resnet50(10)
param_dict = load_checkpoint("/data/deep/mindcon/resnet50_224.ckpt")
param_filter = [x.name for x in network.head.get_parameters()]

def filter_ckpt_parameter(origin_dict, param_filter):
    """删除origin_dict中包含param_filter参数名的元素"""
    for key in list(origin_dict.keys()): # 获取模型的所有参数名
        for name in param_filter: # 遍历模型中待删除的参数名
            if name in key:
                print("Delete parameter from checkpoint:", key)
                del origin_dict[key]
                break

# # 删除全连接层
filter_ckpt_parameter(param_dict, param_filter)

load_param_into_net(network, param_dict)


# 定义优化器
network_opt = nn.Momentum(params=network.trainable_params(), learning_rate=0.001, momentum=0.9)
#network_opt = nn.Adam(params=network.trainable_params(),learning_rate=LR,beta1=0.9,beta2=0.999)
# 定义损失函数
network_loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, num_classes=10)
#network_loss = nn.SoftmaxCrossEntropyWithLogits()
#network_opt = nn.SGD(network.trainable_params(), learning_rate=LR)
# 定义评价指标
metrics = {"Accuracy": nn.Accuracy()}

# 初始化模型
model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)



import os,stat
from mindvision.check_param import Rel, Validator as validator
from mindspore.train.callback import Callback
from mindspore import save_checkpoint
class ValAccSaveMonitor(Callback):
    """
    Train loss and validation accuracy monitor, after each epoch save the
    best checkpoint file with highest validation accuracy.

    Usage:
        >>> monitor = TrainLossAndValAccMonitor(model, dataset_val, num_epochs=10)
    """

    def __init__(self,
                 model,
                 dataset_val,
                 num_epochs,
                 interval=1,
                 eval_start_epoch=1,
                 save_best_ckpt=True,
                 ckpt_directory="./",
                 best_ckpt_name="best.ckpt",
                 metric_name="Accuracy",
                 dataset_sink_mode=True):
        super(ValAccSaveMonitor, self).__init__()
        self.model = model
        self.dataset_val = dataset_val
        self.num_epochs = num_epochs
        self.eval_start_epoch = eval_start_epoch
        self.save_best_ckpt = save_best_ckpt
        self.metric_name = metric_name
        self.interval = validator.check_int(interval, 1, Rel.GE, "interval")
        self.best_res = 0
        self.dataset_sink_mode = dataset_sink_mode
        self.ckpt_directory=ckpt_directory
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, best_ckpt_name)

    def apply_eval(self):
        """Model evaluation, return validation accuracy."""
        return self.model.eval(self.dataset_val, dataset_sink_mode=self.dataset_sink_mode)[self.metric_name]

    def epoch_end(self, run_context):
        """
        After epoch, print train loss and val accuracy,
        save the best ckpt file with highest validation accuracy.
        """
        callback_params = run_context.original_args()
        cur_epoch = callback_params.cur_epoch_num

        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            # Validation result
            res = self.apply_eval()

            print("-" * 20)
            print(f"Epoch: [{cur_epoch: 3d} / {self.num_epochs: 3d}], "
                  f"Train Loss: [{callback_params.net_outputs.asnumpy() :5.3f}], "
                  f"{self.metric_name}: {res: 5.3f}")

            def remove_ckpt_file(file_name):
                os.chmod(file_name, stat.S_IWRITE)
                os.remove(file_name)

            # Save the best ckpt file
            if res >= self.best_res:
                self.best_res = res
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        remove_ckpt_file(self.best_ckpt_path)
                    save_checkpoint(callback_params.train_network, self.best_ckpt_path)
            if(cur_epoch%10==0):
                save_path = os.path.join(self.ckpt_directory, "foodsave_{}_{}.ckpt".format(cur_epoch,res))
                save_checkpoint(callback_params.train_network, save_path)                
    # pylint: disable=unused-argument
    def end(self, run_context):
        print("=" * 80)
        print(f"End of validation the best {self.metric_name} is: {self.best_res: 5.3f}, "
              f"save the best ckpt file in {self.best_ckpt_path}", flush=True)



from mindspore.train.callback import TimeMonitor

num_epochs = 100

# 模型训练与验证，训练完成后保存验证精度最高的ckpt文件（best.ckpt）到当前目录下
model.train(num_epochs, 
            dataset_train,
            callbacks=[ValAccSaveMonitor(model, dataset_val, num_epochs,ckpt_directory="/data/deep/mindcon/result/",
                 best_ckpt_name="resnet_foodbest_2.ckpt",), TimeMonitor()])






