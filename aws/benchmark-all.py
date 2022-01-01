from timeit import default_timer as timer
import time
import requests

api_endpoints = {
        'alexnet': 'https://kiqvi099ek.execute-api.eu-central-1.amazonaws.com/Prod',
        'resnet50': 'https://1f4mkdykmk.execute-api.eu-central-1.amazonaws.com/Prod',
        'resnext50': 'https://0eqknn6wa9.execute-api.eu-central-1.amazonaws.com/Prod',
        'resnext101': 'https://lg0rsjbxr2.execute-api.eu-central-1.amazonaws.com/Prod',
        'vgg19': 'https://9td0jyy6y0.execute-api.eu-central-1.amazonaws.com/Prod'
        'midas': 'https://vx84twe3ed.execute-api.eu-central-1.amazonaws.com/Prod',
        'yolop': 'https://s2mopwrhn8.execute-api.eu-central-1.amazonaws.com/Prod',
        'bert': 'https://38gmvj5c1l.execute-api.eu-central-1.amazonaws.com/Prod',
        # '3d-unet-kits19': 'https://hrja1m1zvd.execute-api.eu-central-1.amazonaws.com/Prod',
        }

def run_benchmark(name, api_endpoint):
    # warmup
    url = api_endpoint + '/classify_digit'
    headers = {
            "Content-Type": "application/json",
            }

    with open(f'sam-{name}/test.json') as f:
        test_json = f.read()

    # warmup
    for i in range(0, 5):
        # start = timer()
        r = requests.post(url, data=test_json, headers=headers)
        # end = timer()
        # print(end - start)
        # print(r.status_code)
        # print(r.content)
        assert(r.status_code==200)
        time.sleep(0.3)

    # benchmark
    for i in range(0, 100):
        start = timer()
        r = requests.post(url, data=test_json, headers=headers)
        end = timer()
        assert(r.status_code==200)
        print(end - start)
        time.sleep(0.3)

for name, endpoint in api_endpoints.items():
    print(f'benchmarking: {name} ({endpoint})')
    run_benchmark(name, endpoint)

