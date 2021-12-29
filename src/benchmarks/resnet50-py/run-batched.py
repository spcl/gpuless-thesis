import torch
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer
import time
import sys

def inference(model):
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    print(top_catid[0].item(), file=sys.stderr)

start = timer()

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.eval()

input_image = Image.open('dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

input_batch = input_batch.to('cuda')
model.to('cuda')

# warmup
for i in range(0, 5):
    inference(model)

# benchmark
for i in range(0, 100):
    start = timer()
    inference(model)
    end = timer()
    print(end - start)
    time.sleep(0.5)

