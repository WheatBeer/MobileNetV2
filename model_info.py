import argparse
import torch
import torch.nn as nn
from utils.mobilenetv2 import MobileNetV2
from torchsummary import summary

parser = argparse.ArgumentParser(description='Model Information')

parser.add_argument('-w', '--weights', metavar='PATH',
                    default='./data/mobilenet_v2-b0353104.pth',
                    help='Pretrained parameters PATH | Default: ./data/mobilenet_v2-b0353104.pth')

parser.add_argument('-i', '--info_num', type=int, default=0,
                    metavar='#', help='Info you want | 0: summary 1: arcitecture 2: paramters value | Default: 0')

def main():
	args = parser.parse_args()
	
	model = MobileNetV2()
	
	# model summary
	if args.info_num == 0:
		summary(model, (3, 224, 224))
	# model architecture
	elif args.info_num == 1:
		model.load_state_dict(torch.load(args.weights))
		print(model.eval())
	# model parameters
	elif args.info_num == 2:
		model.load_state_dict(torch.load(args.weights))
		print(model.state_dict())
	else:
		print('You must put the information number(-i / --info_num)')
		
if __name__ == '__main__':
    main()
