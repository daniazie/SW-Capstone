from openai import OpenAI
from random import sample
import os
import json

with open('sample100.json', 'r') as f:
    dataset = json.load(fp=f)

with open('examples_3shot.json', 'r') as f:
    examples = json.load(fp=f)

client = OpenAI(api_key=os.environ['OPENAI_KEY'])

model = "gpt-4o"

data = []
idx = 1
for item in dataset:
    ex_3shot = sample(examples, 3)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are translating professional radiology reports into layman's terms. Do not include any medical jargon. Write concisely. When rewriting the radiology reports, follow these examples:\n"
                                            f"Radiology report: {ex_3shot[0]['radiology_report']}\n Layman's report: {ex_3shot[0]['layman_report']}\n"
                                            f"Radiology report: {ex_3shot[1]['radiology_report']}\n Layman's report: {ex_3shot[1]['layman_report']}\n"
                                            f"Radiology report: {ex_3shot[2]['radiology_report']}\n Layman's report: {ex_3shot[2]['layman_report']}\n"},
            {"role": "user", "content": f"Radiology report: {item['radiology_report']}\n Layman's report: "}
        ],
        temperature=0.3,
    )
    data.append({
        'radiology_report': item['radiology_report'],
        'prediction': completion.choices[0].message.content,
        'target_summary': item['layman_report'],
    })
    print(f"Processing item {idx}/{len(dataset)}")
    idx += 1

with open('sample100_results.json', 'w') as f:
    json.dump(data, fp=f, indent=2)