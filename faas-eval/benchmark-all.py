from timeit import default_timer as timer
import time
import requests

api_endpoints = {
        'alexnet': 'http://0.0.0.0:9005',
        # 'resnet50': 'http://0.0.0.0:9000',
        # 'resnext50': 'http://0.0.0.0:9000',
        # 'resnext101': 'http://0.0.0.0:9000',
        # 'vgg19': 'http://0.0.0.0:9000'
        # 'yolop': 'http://0.0.0.0:9000',
        # 'midas': 'http://0.0.0.0:9000',
        # '3d-unet-kits19': 'http://0.0.0.0:9000',
        }

def run_benchmark_cold(name, api_endpoint):
    # warmup
    url = api_endpoint
    headers = { "Content-Type": "application/json" }

    with open(f'{name}/test.json') as f:
        test_json = f.read()

    for i in range(0, 30):
        start = timer()
        r = requests.post(url + '/invoke', data=test_json, headers=headers)
        end = timer()
        print(end - start)

        time.sleep(0.5)
        # os.system('./kill-server-runner.sh')
        r = requests.get(url + '/exit')
        time.sleep(3.0)

    # # warmup
    # for i in range(0, 5):
    #     # start = timer()
    #     r = requests.post(url, data=test_json, headers=headers)
    #     # end = timer()
    #     # print(end - start)
    #     # print(r.status_code)
    #     # print(r.content)
    #     assert(r.status_code == 200)
    #     time.sleep(0.3)

    # # benchmark
    # for i in range(0, 100):
    #     start = timer()
    #     r = requests.post(url, data=test_json, headers=headers)
    #     end = timer()
    #     assert(r.status_code==200)
    #     print(end - start)
    #     time.sleep(0.3)

for name, endpoint in api_endpoints.items():
    print(f'benchmarking: {name} ({endpoint})')
    run_benchmark_cold(name, endpoint)

