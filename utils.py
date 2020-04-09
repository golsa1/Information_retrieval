import json
def save_file(fileName, file):
    with open(fileName, 'w') as outfile:
        json.dump(file, outfile)

def open_json(fileName):
    try:
        with open(fileName,encoding='utf8') as json_data:
            d = json.load(json_data)
    except Exception as s:
        d=s
        print(d)
    return d