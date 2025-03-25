from datasets import load_dataset
from random import sample
import json

dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

data = []
for item in dataset['validation']:
    data.append(item)

examples = []
for item in dataset['train']:
    examples.append(item)

sample100 = sample(data, 100)
data = []
for item in sample100:
    data.append({
        'radiology_report': item['radiology_report'],
        'layman_report': item['layman_report']
    })

with open('sample100.json', 'w') as f:
    json.dump(data, fp=f, indent=2)


examples_3shot = sample(examples, 1000)
data = []
for item in examples_3shot:
    data.append({
        'radiology_report': item['radiology_report'],
        'layman_report': item['layman_report']
    })

with open('examples_3shot.json', 'w') as f:
    json.dump(data, fp=f, indent=2)