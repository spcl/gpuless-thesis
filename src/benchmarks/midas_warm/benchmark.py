import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt

from timeit import default_timer as timer
import time

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)


midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

device = torch.device("cuda")
midas.to(device)

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
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
times = []

# warmup
for i in range(0, 5):
    inference()

iterrations = 10
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
