import torch
import torchvision
from torchvision import transforms
import base64
import json
import cv2
import numpy

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from io import BytesIO

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from midas import transforms

midas = DPTDepthModel(
        path=None,
        backbone="vitl16_384",
        non_negative=True,
        )
midas.load_state_dict(torch.load('/opt/ml/model', map_location=torch.device('cpu')))

transform = torchvision.transforms.Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

def lambda_handler(event, context):
    api_gateway_body = event['body'].encode('utf-8')
    event_body = (json.loads(api_gateway_body))['body'].encode('utf-8')
    # event_body = event['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(event_body)))

    cv2_img = cv2.cvtColor(numpy.array(image), cv2.COLOR_BGR2RGB)
    input_batch = transform(cv2_img)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=cv2_img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    return {
        'statusCode': 200,
        'body': '',
    }

