import torch
import torchvision.transforms as transforms
import cv2
from timeit import default_timer as timer
import time
import sys

from lib.utils.augmentations import letterbox_for_img

# load model
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
model.eval()
model.to('cuda')

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

image = cv2.imread('test.jpg', cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

def inference():
    img, _, _ = letterbox_for_img(image)
    img = transform(img)
    img = img.unsqueeze(0).to('cuda')
    det_out, da_seg_out, ll_seg_out = model(img)

# warmup
for i in range(0, 5):
    inference()

total_time = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
times = []
iterrations = 100

# benchmark
for i in range(0, iterrations):
    start.record()
    inference()
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

total_time = sum(times)
avg_time = total_time / float(iterrations)
print("Avg time: {:}ms".format(round(avg_time,5)))