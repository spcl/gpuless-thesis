from bert import QA
from timeit import default_timer as timer
import json

model = QA('model')

def lambda_handler(event, context):
    api_gateway_body = event['body'].encode('utf-8')
    event_json = json.loads(api_gateway_body)
    doc = event_json['doc']
    question = event_json['question']

    answer = model.predict(doc, question)

    return {
        'statusCode': 200,
        'body': json.dumps(answer),
    }
