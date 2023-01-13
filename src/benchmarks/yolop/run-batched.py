import torch
import torchvision.transforms as transforms
import cv2
from timeit import default_timer as timer
import time
import sys

from lib.utils.augmentations import letterbox_for_img

# load model
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
before = timer()
model.eval()
model.to('cuda')
after = timer()

print('model eval time:')
print(after - before)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

image = cv2.imread('test.jpg', cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def inference():
    img, _, _ = letterbox_for_img(image)
    img = transform(img)
    img = img.unsqueeze(0).to('cuda')
    det_out, da_seg_out, ll_seg_out = model(img)
    # print(det_out[0], file=sys.stderr)
    # print(da_seg_out[0], file=sys.stderr)
    # print(ll_seg_out[0], file=sys.stderr)


total_time = 0
times = []

# warmup
for i in range(0, 5):
    inference()

iterations = 10
# benchmark
for i in range(0, iterations):
    start = timer()
    inference()
    end = timer()
    torch.cuda.synchronize()
    times.append(end - start)

total_time = sum(times) * 1000
avg_time = total_time / float(iterations)
print("Avg time: {:}ms".format(round(avg_time, 5)))
