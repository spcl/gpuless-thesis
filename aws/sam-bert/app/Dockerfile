FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./
COPY model ./model
RUN python3.9 -m pip install -r requirements.txt -t .

COPY bert.py utils.py app.py ./

CMD ["app.lambda_handler"]
