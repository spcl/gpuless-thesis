FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./
COPY model /opt/ml/model
RUN python3.9 -m pip install -r requirements.txt -t .
COPY app.py ./
COPY case_00000.pkl /opt/ml/case_00000.pkl
COPY global_vars.py inference_utils.py ./
COPY meta ./meta

CMD ["app.lambda_handler"]
