from openai import OpenAI
import argparse
import json
from tqdm import tqdm
import os

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--output_file', type=str)
    
    return parser

def feedback_prompt(radiology_report, lay_report):

    prompt = f"""### You are an expert medical language reviewer. You are given a radiology report and the full output generated by a language model in response to it. Evaluate the quality of the **entire model output** (not just the lay report section) based on the following 3 criteria.

    For each, provide a **concise explanation (1-2 sentences max)** and a **score in the format x/10**. At the end, provide the total score as the **sum of all three criteria**, formatted as **n/30**.
    1. **Factuality (x/10)**: How factually consistent is the output with the original radiology report? Highlight factually incorrect or inconsistent phrases and penalize accordingly.
    2. **Completeness (x/10)**: Does the output include all important information from the radiology report? Penalize omissions.
    3. **Format (x/10)**: Penalize any commentary or non-report language, such as “Here is your revised report,” “Translation:”, or any explanation of changes. Full marks only if the output **only** contains the lay summary, without extra headers or commentary.
    4. **Total Score (n/30)**: Sum of the seven individual scores."""

    query = f"""### Original radiology report:
    {radiology_report}

    ### Lay report:
    {lay_report}

    ### Feedback:"""

    return [{"role": "system", "content": prompt},
            {"role": "user", "content": query}]

parser = init_parser()
args = parser.parse_args()

with open(args.input_file, 'r') as file:
    data = json.load(file)

client = OpenAI(api_key=os.environ['OPENAI_KEY'])

examples = []

for i, item in enumerate(tqdm(data, desc="Generating")):
    try:
        radiology_report = item['document']
        lay_report = item['generated_caption']
    except:
        radiology_report = item['radiology_report']
        try:
            lay_report = item['prediction']
        except:
            lay_report = item['predicted_summary']

    completion = client.chat.completions.create(
        model=args.model,
        messages=feedback_prompt(radiology_report, lay_report),
        temperature=0.5
    )

    feedback = completion.choices[0].message.content

    examples.append({
        'radiology_report': radiology_report,
        'lay_report': lay_report,
        'feedback': feedback
    })

if os.path.exists(args.output_file):
    with open(args.output_file, 'r') as file:
        feedbacks = json.load(file)
else:
    feedbacks = []

feedbacks.extend(examples)
with open(args.output_file, 'w') as file:
    json.dump(feedbacks, fp=file, indent=2)