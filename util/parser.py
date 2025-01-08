import json

def read_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)

if __name__ == '__main__':
    read_json("../config/test.json")