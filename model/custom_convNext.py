import torch
import torch.nn as nn
import torchvision.models as models

class CustomConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNeXt, self).__init__()
        self.convnext = models.convnext_base(pretrained=True)
        # Replace the classifier head
        self.convnext.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.convnext.classifier[2].in_features, num_classes)
        )

    def forward(self, x):
        return self.convnext(x)

# Optional: If you want to test the module independently
if __name__ == "__main__":
    model = CustomConvNeXt(num_classes=10)
    input_image = torch.randn(1, 3, 32, 32)
    output = model(input_image)
    print(output.size())
