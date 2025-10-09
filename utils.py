import os
import re
import socket
import random
import logging
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import warnings


warnings.filterwarnings("ignore")
load_dotenv()
tqdm.pandas()
SEED = 42
random.seed(SEED)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_PROMPT_COL = "input_prompt"
LLM_OUTPUT_COL = "llm_response"
EXPECTED_OUTPUT_COL = "expected_output"
DEFAULT_MAX_NEW_TOKENS_MODEL = 1024
DEFAULT_MAX_NEW_TOKENS_JUDGE = 1024
JUDGE_SYS_PROMPT = "You are an objective evaluator responsible for assessing MermaidJS sequence diagrams based on a given natural language specification."
JUDGE_SCORING_GUIDELINE = '''
---
Provide a numerical score (0.000 to 1.000) and a concise explanation.
Format the output as: <score>; <explanation>

Scoring scale:
- 0.000 to 0.200: Very poor;
- 0.201 to 0.400: Poor;
- 0.401 to 0.600: Fair;
- 0.601 to 0.800: Good;
- 0.801 to 0.999: Very good;
- 1.000: Perfect;
'''


# ----------- logging -----------


def init_logger(output_folder):
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y_%m_%d.%H_%M_%S")
    log_path = os.path.join(output_folder, f'log_{hostname}__{timestamp}.log')
    
    logging.basicConfig(filename=log_path,
                        filemode='w',
                        level=logging.DEBUG, # .INFO
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def get_logger(name):
    return logging.getLogger(name)


# ----------- helpers -----------


def extract_float_from_string(s):
    if s:
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)  # match float or integer
        return float(match.group()) if match else None
    return None


def render_prompt(template, context):
    full_template = template.strip() + "\n" + JUDGE_SCORING_GUIDELINE
    return full_template.format(
        agent_prompt=context.get("agent_prompt", ""),
        agent_response=context.get("agent_response", ""),
        expected_agent_response=context.get("expected_agent_response", "")
    )