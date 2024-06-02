import torch.nn as nn
from torchvision import models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        
        # Load the pre-trained ResNet18 model
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)
        
        # Modify the first convolutional layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=self.resnet.conv1.out_channels, 
            kernel_size=self.resnet.conv1.kernel_size, 
            stride=self.resnet.conv1.stride, 
            padding=self.resnet.conv1.padding,
            bias=False
        )
        
        # Replace the final fully connected layer with a custom layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class CustomResNet101(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet101, self).__init__()
        
        # Load the pre-trained ResNet18 model
        weights = models.ResNet101_Weights.DEFAULT
        self.resnet = models.resnet101(weights=weights)
        
        # Modify the first convolutional layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=self.resnet.conv1.out_channels, 
            kernel_size=self.resnet.conv1.kernel_size, 
            stride=self.resnet.conv1.stride, 
            padding=self.resnet.conv1.padding,
            bias=False
        )
        
        # Replace the final fully connected layer with a custom layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# test the module independently
if __name__ == "__main__":
    model = CustomResNet101(num_classes=10)
    print(model)
