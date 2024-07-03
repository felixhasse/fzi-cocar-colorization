import torchvision
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)

        self.net.classifier = DeepLabHead(2048, num_classes)

    def forward(self, inputs):
        return self.net(inputs)


class DeepLabV3MobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=None)

        self.net.classifier = DeepLabHead(960, num_classes)

    def forward(self, inputs):
        return self.net(inputs)
