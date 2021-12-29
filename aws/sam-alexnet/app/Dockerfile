FROM public.ecr.aws/lambda/python:3.9

COPY app.py requirements.txt ./
COPY model /opt/ml/model
COPY imagenet_classes.txt /opt/ml/imagenet_classes.txt

RUN python3.9 -m pip install -r requirements.txt -t .

CMD ["app.lambda_handler"]
