from timeit import default_timer as timer
import time
import requests

api_endpoints = {
        'alexnet': 'https://kiqvi099ek.execute-api.eu-central-1.amazonaws.com/Prod',
        'resnet50': 'https://1f4mkdykmk.execute-api.eu-central-1.amazonaws.com/Prod',
        'resnext50': 'https://0eqknn6wa9.execute-api.eu-central-1.amazonaws.com/Prod',
        'resnext101': 'https://lg0rsjbxr2.execute-api.eu-central-1.amazonaws.com/Prod',
        'vgg19': 'https://9td0jyy6y0.execute-api.eu-central-1.amazonaws.com/Prod'
        }

with open('dog.json') as f:
    test_json = f.read()

def run_benchmark(name, api_endpoint):
    # warmup
    url = api_endpoint + '/classify_digit'
    headers = {
            "Content-Type": "application/json",
            }

    for i in range(0, 5):
        time.sleep(1.0)
        r = requests.post(url, data=test_json, headers=headers)
        # print(r.status_code)
        # print(r.content)

    # benchmark
    for i in range(0, 100):
        start = timer()
        r = requests.post(url, data=test_json, headers=headers)
        assert(r.status_code==200)
        end = timer()
        print(end - start)

for name, endpoint in api_endpoints.items():
    print(f'benchmarking: {name} ({endpoint})')
    run_benchmark(name, endpoint)

