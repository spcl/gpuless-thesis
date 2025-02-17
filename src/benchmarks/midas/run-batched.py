import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt

from timeit import default_timer as timer
import time

model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

before = timer()
midas.eval()

device = torch.device("cuda")
midas.to(device)
after = timer()

print('model eval time:')
print(after - before)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread('dog.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def inference():
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    # plt.imshow(output)
    # plt.show()
    # plt.savefig('out.pdf')


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
