import json

def read_json(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return json_data
