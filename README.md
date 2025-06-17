# README

## 소프트웨어융합캡스톤디자인 프로젝트

### 실행 방법
실행하도록 conda 가상환경을 생성해야 합니다.
```conda create -n bls python=3.10```

만약 CUDA를를 사용하려면 아래 명령어를 사용하시시면 됩니다.
```conda create -n bls python=3.10 cudatoolkit=CUDA_VERSION```
* CUDA_VERSION은 본인 CUDA 버전으로 대입하시면 됩니다.

그리고 requirements.txt를 설치해야 합니다.
```pip install -r requirements.txt```

그리고 모델을 평가하도록 BioLaySumm2025 레포를 clone하셔야 합니다.
```git clone https://github.com/gowitheflow-1998/BioLaySumm2025.git```

CheXbert-F1과 RadGraph-F1으로 제대로 평가하도록 CXR-Report-Metric 레포도 clone하셔야 하고 파이썬 3.7인 conda 가상환경을 따로 만들어야 합니다.
```git clone https://github.com/rajpurkarlab/CXR-Report-Metric.git```
```conda create -n clinical_eval python=3.7```

evaluation 디렉터리에 있는 run_clinical_eval.sh와 clinical_metric.py를 CXR-Report-Metric 디렉터리에 이동하시면 됩니다. 비슷하게도, evaluation 디렉터리에 있는 run_bls_eval.sh와 evaluation_casestudy.py를 BioLaySumm2025 디렉터리에 이동하시면 됩니다.

파인튜닝 파일들은 sft_sh에 들어가 있습니다. 

gen_sft.py로 생성하시려면 명령어에서 놓은은 아웃풋 이름에 무조건 프롬프트 세팅을 포함해야 합니다.
* 예시: qwen3-4b-sft_3shot_sample_results.json


### 코드 설명

* **sft.py**: 지도 파인튜닝을 실행하는 파일
* **gen_sft.py**: 지도 파인튜닝된 모델로 환자 친화적 의료 영상 판독문 생성하는 파일
* **feedback_gen.py**: Refinement에 few-shot 피드백을 위한 피드백 예시를 생성하는 파일
* **gen_refine.py**: Refinement 프레임워크로 환자 친화적 의료 영상 판독문 생성하는 파일
* **refinement.py**: 존재하는 환자 친화적 의료 영상 판독문을 기반으로 refinement를 실행하는 파일 (SFT 모델로 따로 생성하지 않음)
* **evaluation_casestudy.py**: Refinement 프레임워크를 분석하도록 case study를 위한 평가 파일
---

### Guidelines
Create a conda environment as below:
```conda create -n ble python=3.10```

If you intend to use CUDA, use the following command to create a conda environment:
```conda create -n bls python=3.10 cudatoolkit=CUDA_VERSION```
* Substitute CUDA_VERSION with your own CUDA version

Install the necessary packages using the requirements.txt file:
```pip install -r requirements.txt```

Clone the BioLaySumm2025 repository for evaluation:
```git clone https://github.com/gowitheflow-1998/BioLaySumm2025.git```

Clone the CXR-Report-Metric repository for CheXbert-F1 and RadGraph-F1:
```git clone https://github.com/rajpurkarlab/CXR-Report-Metric.git```

When using CXR-Report-Metric, you need to use a separate conda environment as it requires Python 3.7:
```conda create -n clinical_eval python=3.7```

You can then move the clinical_metrics.py and run_clinical_eval.sh files from the evaluation folder into the CXR-Report-Metric directory. Similarly, you can move the evaluation_casestudy.py and run_bls_eval.sh files from the evaluation directory into the BioLaySumm2025 directory.

The sh files for finetuning are in the sft_sh directory.

If you intend to generate samples using gen_sft.py, ensure that you include the prompt setting (3shot or 0shot) in the name of the output file as below:
* Ex: qwen3-4b-sft_3shot_sample_results.json

### Codes
* **sft.py**: Executes supervised fine-tuning
* **gen_sft.py**: Generates lay radiology reports using the SFT model
* **feedback_gen.py**: Generates example feedbacks for few-shot feedback generation (for refinement)
* **gen_refine.py**: Uses the refinement framework starting from the SFT model
* **refinement.py**: Executes refinement given existing lay reports (No first generation from SFT model)
* **evaluation_casestudy.py**: Evaluates the case study results for refinement (if the case_study setting was used).