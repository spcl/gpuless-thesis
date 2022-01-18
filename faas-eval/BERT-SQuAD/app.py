import json
import bottle
from bottle import post, request, response
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
from bert import QA
import os
import signal

initialized = False
model = None

bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024

def init():
    global initialized, model
    if initialized:
        return

    model = QA('model')
    initialized = True

@post('/invoke')
def invoke():
    global initialized, model
    init()

    doc = request.json['doc']
    question = request.json['question']
    answer = model.predict(doc, question)
    return json.dumps({
        'statusCode': 200,
        'body': json.dumps(answer),
    })

@get('/exit')
def exit():
    os.kill(os.getpid(), signal.SIGTERM)

if __name__ == '__main__':
    bottle.run(host='0.0.0.0', port = 9000)
