import json
import bottle
from bottle import post, get, request, response
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
import os
import signal
import array
import torchvision
import pickle
import numpy as np
import sys

import inference_utils as infu
from global_vars import *

initialized = False
categories = []
model = None
preprocess = None
input_data = None

with open('./case_00000.pkl', 'rb') as f:
    input_data = f.read()

bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024

device = torch.device("cuda")

def to_tensor(arr):
    return torch.from_numpy(arr).float().to(device)

def from_tensor(tensor):
    return tensor.cpu().numpy().astype(float)

def do_infer(input_tensor):
    with torch.no_grad():
        return model(input_tensor)

def infer_single_query(query):
    image = query[np.newaxis, ...]
    result, norm_map, norm_patch = infu.prepare_arrays(image, ROI_SHAPE)
    t_image = to_tensor(image)
    t_result = to_tensor(result)
    t_norm_map = to_tensor(norm_map)
    t_norm_patch = to_tensor(norm_patch)

    subvol_cnt = 0
    for i, j, k in infu.get_slice_for_sliding_window(t_image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
        subvol_cnt += 1
        result_slice = t_result[
            ...,
            i:(ROI_SHAPE[0] + i),
            j:(ROI_SHAPE[1] + j),
            k:(ROI_SHAPE[2] + k)]

        input_slice = t_image[
            ...,
            i:(ROI_SHAPE[0] + i),
            j:(ROI_SHAPE[1] + j),
            k:(ROI_SHAPE[2] + k)]

        norm_map_slice = t_norm_map[
            ...,
            i:(ROI_SHAPE[0] + i),
            j:(ROI_SHAPE[1] + j),
            k:(ROI_SHAPE[2] + k)]

        result_slice += do_infer(input_slice) * t_norm_patch
        norm_map_slice += t_norm_patch

    result = from_tensor(t_result)
    norm_map = from_tensor(t_norm_map)
    final_result = infu.finalize(result, norm_map)
    return final_result

def init():
    global initialized, categories, model, preprocess, input_data
    if initialized:
        return

    device = torch.device("cuda")
    model = torch.jit.load('./3dunet_kits19_pytorch.ptc', map_location='cuda')
    model.eval()

    initialized = True

@post('/invoke')
def invoke():
    global initialized, categories, model, preprocess, input_data
    init()

    # event_body = request.json['body'].encode('utf-8')
    # image = Image.open(BytesIO(base64.b64decode(event_body)))

    query = pickle.loads(input_data)[0]
    result = infer_single_query(query)
    response_array = array.array("B", result.tobytes())
    bi = response_array.buffer_info()

    return json.dumps({})

@get('/exit')
def exit():
    sys.stderr.close()

if __name__ == '__main__':
    bottle.run(host='0.0.0.0', port = 9000)
