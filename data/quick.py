import json
import os

label_list = []
with open('vlguard_train_full.json', 'r') as readfile:
    data = json.load(readfile)
    for item in data:
        label = item.pop('answer')
        item['label'] = label

with open('vlguard_full_train.json', 'w') as writefile:
    json.dump(data, writefile, indent=4)