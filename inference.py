import json
import argparse
import torch
from PIL import Image
from torchvision import transforms
from utils.mobilenetv2 import MobileNetV2

parser = argparse.ArgumentParser(description='Training MobileNetV2')

parser.add_argument('-w', '--weights', metavar='PATH', 
                    default='./data/mobilenet_v2-b0353104.pth',
                    help='Pretrained parameters PATH | Default: ./data/mobilenet_v2-b0353104.pth')

parser.add_argument('-i', '--input', metavar='PATH', 
                    default='./data/dog.jpg',
                    help='Input image PATH | Default: ./data/dog.jpg')

def main():
    args = parser.parse_args()
    
    # Define model
    model = MobileNetV2()
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('===> Test on', device)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print('===> Using', torch.cuda.device_count(), 'GPUs!')
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(Image.open(args.input))
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    print('===> Image preprocessing is done!')
    
    idx2label = []
    cls2label = {}
    with open("./data/imagenet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        
    with torch.no_grad():
        output = model(input_batch)
    output_pro = torch.nn.functional.softmax(output[0], dim=0)
    values, index = torch.max(output_pro, 0)
    print('===> Probability:', '%3.1f' %(float(values)*100),'%')
    print('===> Index:', idx2label[index])

if __name__ == '__main__':
    main()
