# MermaidSeqBench-Eval

Evaluate your LLM on the [MermaidSeqBench](https://huggingface.co/datasets/ibm-research/MermaidSeqBench) dataset using LLMaJ and a RESTful OpenAI-compatible API.

## Getting started

### (optional) Create & activate the Conda environment
```
conda create -n merm python=3.10
conda activate merm
```

### Install Python dependencies
```
pip install -r requirements.txt
```

### Set up env vars
Make sure you have set an env variable with your OpenAI-compatible key to use in your OpenAI Python client: `OPENAI_API_KEY`

### Run the evaluation script
#### Usage
```
usage: python eval.py [-h] --model MODEL --model_api_endpoint MODEL_API_ENDPOINT --judge JUDGE --judge_api_endpoint JUDGE_API_ENDPOINT [--output OUTPUT] [--crit_file CRIT_FILE]

Evaluate an LLM on the MermaidSeqBench dataset using LLMaJ with the RESTful OpenAI client.

options:
  -h, --help            show this help message and exit
  --model MODEL         Name or identifier of the model to evaluate.
  --model_api_endpoint MODEL_API_ENDPOINT
                        REST API endpoint for the model (e.g., OpenAI-compatible endpoint URL).
  --judge JUDGE         Name or identifier of the judging model.
  --judge_api_endpoint JUDGE_API_ENDPOINT
                        REST API endpoint for the judging model (LLMaJ).
  --output OUTPUT       Output folder for the evaluation results and log file (default: current directory).
  --crit_file CRIT_FILE
                        Path to the YAML file defining judgment criteria (default: judge-criteria.yaml).
```
A `csv` file named `results.csv` (timestamped variant) will be saved in the specified output folder.

#### Example
```
python eval.py \
  --model llama-3-70b \
  --model_api_endpoint https://api.fireworks.ai/inference/v1 \
  --judge gpt-4-turbo \
  --judge_api_endpoint https://api.openai.com/v1 \
  --output ./results \
  --crit_file judge-criteria.yaml
```

### Cite this work
If you would like to cite this work in a paper or a presentation, the following is recommended (BibTeX entry):
```
@misc{shbita2025mermaidseqbenchevaluationbenchmarkllmtomermaid,
      title={MermaidSeqBench: An Evaluation Benchmark for LLM-to-Mermaid Sequence Diagram Generation},
      author={Basel Shbita and Farhan Ahmed and Chad DeLuca},
      year={2025},
      eprint={2511.14967},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2511.14967},
}
```