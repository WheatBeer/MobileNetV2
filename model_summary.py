from utils.mobilenetv2 import MobileNetV2 
from torchsummary import summary

model = MobileNetV2()
model.eval()

summary(model, (3, 224, 224))
