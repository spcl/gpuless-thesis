import json
import bottle
from bottle import post, request, response
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64

initialized = False
categories = []
model = None
preprocess = None

bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024

def init():
    global initialized, categories, model, preprocess
    if initialized:
        return

    with open("./imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
    model.eval()
    model.to('cuda')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    initialized = True

@post('/invoke')
def invoke():
    global initialized, categories, model, preprocess
    init()

    event_body = request.json['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(event_body)))

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    top_catid = top_catid[0].item()
    top_prob = top_prob.item()
    label = categories[top_catid]

    response.headers['Content-Type'] = 'application/json'
    return json.dumps({'label': label})

if __name__ == '__main__':
    bottle.run(host='0.0.0.0', port = 9000)
