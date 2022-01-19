import bottle
from bottle import post, request, response
import torch
import torchvision
from torchvision import transforms
import base64
import json
import cv2
import numpy
import matplotlib.pyplot as plt
import os
import signal
import sys

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from io import BytesIO

initialized = False
model = None
preprocess = None

bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024

def init():
    global initialized, model, transform
    if initialized:
        return

    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()

    device = torch.device("cuda")
    model.to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform


    initialized = True

@post('/invoke')
def invoke():
    global initialized, model, transform
    init()

    event_body = request.json['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(event_body)))

    cv2_img = cv2.cvtColor(numpy.array(image), cv2.COLOR_BGR2RGB)
    input_batch = transform(cv2_img).to('cuda')

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=cv2_img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    plt.imshow(output)
    plt.savefig('out.jpg')

    response.headers['Content-Type'] = 'application/json'
    return json.dumps({})

@get('/exit')
def exit():
    sys.stderr.close()

if __name__ == '__main__':
    bottle.run(host='0.0.0.0', port = 9000)
