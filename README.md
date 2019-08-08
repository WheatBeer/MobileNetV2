# MobileNetV2_8bits
MobileNetV2 8-bits Precision Validation & Retraining

## Dependencies
- [Pytorch][pytorch] (ver.1.1.0)
- torchvision
- tqdm (for progress bar)
- You can check other dependencies on top of *.py files

## MobileNetV2(Model + Pretrained weights & bias)
- Paper: ["Inverted Residuals and Linear Bottlenecks"][paper] 
- [Code][code]
- [Imagenet Dataset][imagenet]
- Model Architecture: [model_summary.txt][model_summuary.txt] or 
```python3 model_summary.py```
  ### Top 1 & Top 5
  - Using pretrained model(./data/mobilenet_v2-b0353104.pth)
  - Top 1 Accuracy: 71.88 % 	 
  - Top 5 Accuracy: 90.29 %
 
## Description
### inference.py

~~~
python3 train.py -e 10 -l 0.001 -n 5 -d 0
~~~

[pytorch]: https://pytorch.org/
[paper]: https://arxiv.org/abs/1801.04381
[code]: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
[model_summuary.txt]: https://github.com/WheatBeer/MobileNetV2_8bits/blob/master/model_summary.txt
[imagenet]: http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
