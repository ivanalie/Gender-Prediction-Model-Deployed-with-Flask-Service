import requests

url = 'http://localhost:5000/predict_json'
name_list = ['barack', 'donald', 'biden', 'hillary', 'diana', 'elizabeth']

for name in name_list:
    r = requests.post(url,json={'name': name})
    print(r.json())


