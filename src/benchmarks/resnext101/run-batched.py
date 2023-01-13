import torch
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer
import time
import sys

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)

before = timer()
model.eval()
model.to('cuda')
after = timer()

print('model eval time:')
print(after - before)

input_image = Image.open('dog.jpg')


def inference(model, image):
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    input_batch = input_batch.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    top_prob.to("cpu")
    top_catid.to("cpu")
    # print(top_catid[0].item(), file=sys.stderr)


total_time = 0
times = []

# warmup
for i in range(0, 5):
    inference(model, input_image)

iterations = 100
# benchmark
for i in range(0, iterations):
    start = timer()
    inference(model, input_image)
    end = timer()
    torch.cuda.synchronize()
    times.append(end - start)

total_time = sum(times) * 1000
avg_time = total_time / float(iterations)
print(*times, sep="\n")
