import torch
import torchvision.transforms as transforms
import base64
import json
import cv2
import numpy

from lib.utils.utils import select_device
from lib.config import cfg
from lib.models import get_net
from lib.utils.augmentations import letterbox_for_img
from pathlib import Path

from PIL import Image
from io import BytesIO

model = get_net(cfg)
checkpoint = torch.load('/opt/ml/model', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

def lambda_handler(event, context):
    api_gateway_body = event['body'].encode('utf-8')
    event_body = (json.loads(api_gateway_body))['body'].encode('utf-8')
    # event_body = event['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(event_body)))

    img = cv2.cvtColor(numpy.array(image), cv2.COLOR_BGR2RGB)
    img, _, _ = letterbox_for_img(img)
    img = transform(img)
    img = img.unsqueeze(0)

    det_out, da_seg_out, ll_seg_out = model(img)

    return {
        'statusCode': 200,
        'body': '',
    }
