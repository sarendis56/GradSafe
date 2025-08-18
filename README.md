# GradSafe
Official Code for ACL 2024 paper "GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis"
https://arxiv.org/abs/2402.13494

## Overview
Large Language Models (LLMs) face threats from unsafe prompts.
Existing methods for detecting unsafe prompts for LLMs are primarily online moderation APIs or finetuned LLMs. These strategies, however, often require extensive and resource-intensive data collection and training processes.
In this study, we propose GradSafe, which effectively detects unsafe prompts by scrutinizing the gradients of safety-critical parameters in LLMs. 
Our methodology is grounded in a pivotal observation: when LLMs are trained on unsafe prompts paired with compliance responses, the resulting gradients on certain safety-critical parameters exhibit consistent patterns. In contrast, safe prompts lead to markedly different gradient patterns.
Building on this observation, GradSafe analyzes the gradients from prompts (paired with compliance responses) to accurately detect unsafe prompts. 
We show that GradSafe, applied to Llama-2 without further training, outperforms Llama Guard—despite its extensive finetuning with a large collected dataset—in detecting unsafe content. 
This superior performance is consistent across both zero-shot and adaptation scenarios, as evidenced by our evaluations on the ToxicChat and XSTest.


## Dataset

The ToxicChat dataset is available at https://huggingface.co/datasets/lmsys/toxic-chat, we use toxicchat1123 in our evaluation.

The XSTest dataset is available at https://huggingface.co/datasets/natolambert/xstest-v2-copy

Please download the dataset and save at the ./data

## Base model

Please download Llama-2 7b from https://huggingface.co/meta-llama/Llama-2-7b-chat-hf, and save at ./model

## Multi-Modal GradSafe Evaluation (LLaVA)

This repository has been extended to support multi-modal language models like LLaVA with vision-language capabilities.

### Prerequisites

1. **Environment Setup**:
   - Ensure you have the `llava` conda environment activated:
     ```bash
     conda activate llava
     ```

2. **Model**:
   - Download LLaVA model (e.g., `llava-v1.6-vicuna-7b`) and save it in the `./model` directory

### Running Multi-Modal Evaluation

#### Quick Test (200 samples from XSTest):
```bash
python run_gradsafe_evaluation.py --quick_test
```

#### Full Benchmark (1800 samples):
```bash
python run_gradsafe_evaluation.py [--model_path MODEL_PATH] [--use_training_data] [--output_file OUTPUT_FILE]
```

#### Command-line Options:
- `--model_path`: Path to LLaVA model directory (default: `model/llava-v1.6-vicuna-7b`)
- `--use_training_data`: Use training data to find critical parameters (default: use built-in examples)
- `--output_file`: Output file for results (default: `gradsafe_llava_results.json`)
- `--skip_env_check`: Skip conda environment check
- `--quick_test`: Run quick test on 100+100 XSTest samples
- `--disable_cache`: Disable caching of gradients and scores
- `--cooling_interval`: Samples before cooling break (default: 10)
- `--cooling_time`: Cooling break duration in seconds (default: 60)

### Benchmark Datasets

The multi-modal evaluation uses a comprehensive vision-language benchmark:

**Training Set (2,000 examples, 1:1 ratio)**:
- Benign (1,000): Alpaca (500) + MM-Vet (218) + OpenAssistant (282)
- Malicious (1,000): AdvBench (300) + JailbreakV-28K (550) + DAN variants (150)

**Test Set (1,800 examples, 1:1 ratio)**:
- Safe (900): XSTest safe (250) + FigTxt safe (300) + VQAv2 (350)
- Unsafe (900): XSTest unsafe (200) + FigTxt unsafe (350) + VAE (200) + JailbreakV-28K (150)

## Citation

If you find this work useful for your research, please cite the original paper (you don't need to cite this repository):
```
@inproceedings{xie2024gradsafe,
  title={GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis},
  author={Xie, Yueqi and Fang, Minghong and Pi, Renjie and Gong, Neil},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={507--518},
  year={2024}
}
```
