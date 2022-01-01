import torch
import torchvision
from torchvision import transforms
import base64
import json
import pickle
import numpy as np
import array

import inference_utils as infu
from global_vars import *

device = torch.device("cpu")
model = torch.jit.load('/opt/ml/model', map_location='cpu')
model.eval()

with open('/opt/ml/case_00000.pkl', 'rb') as f:
    input_data = f.read()

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

def lambda_handler(event, context):
    # api_gateway_body = event['body'].encode('utf-8')
    # event_body = (json.loads(api_gateway_body))['body'].encode('utf-8')

    query = pickle.loads(input_data)[0]
    result = infer_single_query(query)
    response_array = array.array("B", result.tobytes())
    bi = response_array.buffer_info()

    return {
        'statusCode': 200,
        'body': '',
    }

