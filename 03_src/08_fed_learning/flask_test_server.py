import requests
import json

answer = requests.post("http://localhost:5000/get_params", json.dumps({"test": True}))

print(answer.json())