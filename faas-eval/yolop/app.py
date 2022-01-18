import json
import bottle
from bottle import post, request, response
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
import sys

import cv2
import numpy

from lib.utils.utils import select_device
from lib.config import cfg
from lib.models import get_net
from lib.utils.augmentations import letterbox_for_img
from pathlib import Path

initialized = False
categories = []
model = None
transform = None

bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024

def init():
    global initialized, categories, model, transform
    if initialized:
        return

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

    initialized = True

@post('/invoke')
def invoke():
    global initialized, categories, model, transform
    init()

    event_body = request.json['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(event_body)))

    img = cv2.cvtColor(numpy.array(image), cv2.COLOR_BGR2RGB)
    img, _, _ = letterbox_for_img(img)
    img = transform(img).to('cuda')
    img = img.unsqueeze(0)

    det_out, da_seg_out, ll_seg_out = model(img)
    print(det_out[0], file=sys.stderr)
    print(da_seg_out[0], file=sys.stderr)
    print(ll_seg_out[0], file=sys.stderr)

    response.headers['Content-Type'] = 'application/json'
    return json.dumps({})

if __name__ == '__main__':
    bottle.run(host='0.0.0.0', port = 9000)
