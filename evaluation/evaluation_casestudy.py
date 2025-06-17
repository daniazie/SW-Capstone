import os, sys, json
sys.path.append("./AlignScore/src")
sys.path.append("./summac")
import torch
import nltk 
import textstat
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
from alignscore import AlignScore
from lens import download_model, LENS
from summac.model_summac import SummaCConv
import argparse
from f1chexbert import F1CheXbert
from radgraph import RadGraph, F1RadGraph
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import evaluate


nltk.download('punkt')

def calc_rouge(preds, refs):
  # Get ROUGE F1 scores
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], \
                                    use_stemmer=True, split_summaries=True)
  scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
  del scorer
  return (np.mean([s['rouge1'].fmeasure for s in scores])+ \
         np.mean([s['rouge2'].fmeasure for s in scores])+ \
         np.mean([s['rougeLsum'].fmeasure for s in scores]))/3.0

def cal_bleu(preds, refs):
  bleu = evaluate.load('sacrebleu')
  scores = bleu.compute(predictions=preds, references=refs)["score"]
  del bleu
  return scores

def cal_meteor(preds, refs):
  meteor = evaluate.load('meteor')
  scores = meteor.compute(predictions=preds, references=refs)["meteor"]
  del meteor
  return scores

def calc_bertscore(preds, refs):
  # Get BERTScore F1 scores
  P, R, F1 = score(preds, refs, lang="en", verbose=True, device='cuda')#:0
  return np.mean(F1.tolist())

def calc_readability(preds):
  fkgl_scores = []
  cli_scores = []
  dcrs_scores = []
  for pred in preds:
    fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
    cli_scores.append(textstat.coleman_liau_index(pred))
    dcrs_scores.append(textstat.dale_chall_readability_score(pred))
  return np.mean(fkgl_scores), np.mean(cli_scores), np.mean(dcrs_scores)

def calc_lens(preds, refs, docs):
  lens_path = download_model("davidheineman/lens")
  metric = LENS(lens_path, rescale=True)
  abstracts = [d.split("\n")[0] for d in docs]
  refs = [[x] for x in refs]

  scores = metric.score(abstracts, preds, refs, batch_size=8)#, gpus=1
  del metric
  return np.mean(scores)

def calc_alignscore(preds, docs):
  model_path = hf_hub_download(repo_id="yzha/AlignScore", filename="AlignScore-base.ckpt")
  alignscorer = AlignScore(model='roberta-base', batch_size=16, device='cuda', \
                           ckpt_path=model_path, evaluation_mode='nli_sp')
  return np.mean(alignscorer.score(contexts=docs, claims=preds))

def cal_summac(preds, docs):
  model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
  return np.mean(model_conv.score(docs, preds)['scores'])

def cal_f1bert(preds, refs):
  f1chexbert = F1CheXbert()
  f1chexbert_score = f1chexbert(hyps=preds, refs=refs) 
  f1chexbert_score, _, _, _ = f1chexbert(hyps=preds, refs=refs) 
  del f1chexbert
  return f1chexbert_score

def cal_radgraph(preds, refs):
  f1radgraph = F1RadGraph(reward_level="all")
  f1radgraph_score,  _, _, _ = f1radgraph(hyps=preds, refs=refs)
  del f1radgraph
  return f1radgraph_score[-1]

def read_file_lines(path):
  with open(path, 'r') as f:
    lines = json.load(f)[:10]
  return lines

def cal_similarity(preds, refs):
  model = SentenceTransformer("all-MiniLM-L6-v2")
  scores = [np.array(model.similarity(model.encode(p), model.encode(r))) for p,r in zip(preds,refs)]
  del model
  return np.mean(scores)


def evaluate_all(pred_path, gold_path,task_name):
  # Load data from files
  refs_dicts = read_file_lines(gold_path)
  preds = read_file_lines(pred_path)
  # print(preds[0].keys)

  assert len(refs_dicts)==len(preds)

  refs = [d['reference'] for d in refs_dicts]
  docs = [d['document'] for d in refs_dicts]
  preds = [d['generated_caption'] for d in preds]

  # preds = [d for d in preds_list]
  # print(type(refs[0]),refs[0])
  scores = []
  for pred, ref, doc in zip(preds, refs, docs):
    score_dict = {}

    # Relevance scores
    score_dict['ROUGE'] = calc_rouge([pred], [ref])
    score_dict['BLEU'] = cal_bleu([pred], [ref])
    score_dict['METEOR'] = cal_meteor([pred], [ref])  
    score_dict['BERTScore'] = calc_bertscore([pred], [ref])

    # # Readability scores
    fkgl_score, cli_score, dcrs_score = calc_readability([pred])
    score_dict['FKGL'] = fkgl_score
    score_dict['DCRS'] = dcrs_score
    score_dict['CLI'] = cli_score
    

    # Factuality scores
    if task_name == "lay_summ":
      score_dict['AlignScore'] = calc_alignscore([pred], [doc])
      score_dict['LENS'] = calc_lens([pred], [ref], [doc])
      score_dict['SummaC'] = cal_summac([pred], [doc])
    else:
      score_dict['similarity'] = cal_similarity([pred], [doc])
      score_dict["f1chexbert"] = cal_f1bert([pred], [ref])
      score_dict["radgraph"] = cal_radgraph([pred], [ref])

    print(score_dict)
    scores.append(score_dict)

  return scores

def write_scores(score_dict, output_filepath):
  # Write scores to file
  with open(output_filepath, 'w') as f:
    for key, value in score_dict.items():
      f.write(f"{key}: {value}\n")

def final_score(task_name):
  submit_dir = "./input_data"# ("elife.txt" and "plos.txt") with predictions
  truth_dir = "./reference_data"#directory with jsonl files containing references and articles 
  output_dir = "./output"

  if task_name == "lay_summ":
    # Calculate eLife scores
    elife_scores = evaluate_all(
      os.path.join(submit_dir, 'elife.txt'), 
      os.path.join(truth_dir, 'eLife_test.jsonl'),
      )
    torch.cuda.empty_cache()

    # Calculate PLOS scores
    plos_scores = evaluate_all(
      os.path.join(submit_dir, 'plos.txt'), 
      os.path.join(truth_dir, 'PLOS_test.jsonl'),
      )

    # Calculate overall scores
    final_scores = {key: np.mean([elife_scores[key], plos_scores[key]]) for key in elife_scores.keys()}
  else:
    # Calculate RRG scores
    final_scores = evaluate_all(
      os.path.join(submit_dir, 'rrg.txt'), 
      os.path.join(truth_dir, 'RRG_test.jsonl'),
      )
    
  # Write overall score
  write_scores(final_scores, os.path.join(output_dir, 'scores.txt'))

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj



if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Evaluate medical text generation outputs.")
    parser.add_argument('--prediction_file', type=str, required=True,default= 'BioLaySumm2025-eLife_result.json', help='Path to the predictions JSON file.')
    parser.add_argument('--groundtruth_file', type=str, default= 'BioLaySumm2025-eLife_result.json',required=True, help='Path to the ground truth JSON file.')
    parser.add_argument('--task_name',  type=str,  default= 'lay_summ', required=True, help='The name of the task.') #"rrg" 
    args = parser.parse_args()

    scores = evaluate_all(args.prediction_file, args.groundtruth_file, args.task_name)
    scores = convert_numpy(scores)

    file_name = args.prediction_file.split('/')[-1]

    if 'v0' in file_name:
        with open(f'evaluation_results/{file_name.split("_result")[0]}_v0_eval.json', 'w') as file:
            json.dump(scores, fp=file, indent=2)
    else:
        with open(f'evaluation_results/{file_name.split("_result")[0]}_eval.json', 'w') as file:
            json.dump(scores, fp=file, indent=2)