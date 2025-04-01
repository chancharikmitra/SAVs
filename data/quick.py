import json
import os

data_list = []
with open('camerabench_trainset/has_zoom_out.jsonl', 'r') as readfile:
    for line in readfile:
        item = json.loads(line)
        label = item.pop('question')
        item['question'] = 'Does the camera zoom out?'
        data_list.append(item)

with open('camerabench_trainset/has_zoom_out.jsonl', 'w') as writefile:
    for json_obj in data_list:
        writefile.write(json.dumps(json_obj) + '\n')