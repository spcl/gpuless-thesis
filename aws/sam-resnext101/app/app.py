import torch
import torchvision
from torchvision import transforms
import base64
import json

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from io import BytesIO

model = torch.load( '/opt/ml/model')
model.eval()

categories = []
with open("/opt/ml/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

def lambda_handler(event, context):
    api_gateway_body = event['body'].encode('utf-8')
    event_body = (json.loads(api_gateway_body))['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(event_body)))

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_catid = torch.topk(probabilities, 1)
    label = categories[top_catid[0].item()]

    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": label,
            }
        )
    }
