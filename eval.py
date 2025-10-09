import pandas as pd
import argparse
import yaml
from datasets import load_dataset, Dataset
from openai import OpenAI
from utils import *


log = get_logger(__name__)


# -------- LLM helper --------


def evaluate(judge_client, judge_model, prompt_template, agent_prompt, agent_response, expected_agent_response):
    global log
    
    context = {
        "agent_prompt": agent_prompt,
        "agent_response": agent_response,
        "expected_agent_response": expected_agent_response
    }
    rendered_prompt = render_prompt(prompt_template, context)
    log.debug(f'[evaluate] rendered_prompt: {rendered_prompt}')
    response = judge_client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": str(JUDGE_SYS_PROMPT)},
            {"role": "user", "content": str(rendered_prompt)}
        ],
        max_completion_tokens=DEFAULT_MAX_NEW_TOKENS_JUDGE,
        temperature=0
    )
    chat_response = response.choices[0].message.content
    log.debug(f"[evaluate] Response from API: {chat_response}")
    return chat_response


# -------- main --------------


def main(args):
    global log
    
    os.makedirs(args.output, exist_ok=True)
    init_logger(args.output)

    mermseqbench_ds = load_dataset("ibm-research/MermaidSeqBench")
    df = mermseqbench_ds["train"].to_pandas()    
    log.info(f'Loaded data:\n"{df}"...')

    with open(args.crit_file, 'r') as file:
        crit_config = yaml.safe_load(file)
    log.info(f"Loaded LLMaJ criteria ({len(crit_config['evaluation_criteria'])} in total)")

    # LLM under test
    model = args.model
    endpoint = args.model_api_endpoint
    client = OpenAI(
        base_url=endpoint,
        api_key=OPENAI_API_KEY,
    )
    log.info(f'Running LLM: {model}')
    for i, row in tqdm(df.iterrows(), total=len(df), desc="model chat completions"):
        user_prompt = row.get(LLM_PROMPT_COL)
        log.debug(f"[model] user_prompt: {user_prompt}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": str(user_prompt)}],
                max_completion_tokens=DEFAULT_MAX_NEW_TOKENS_MODEL,
                temperature=0,
            )
            chat_response = response.choices[0].message.content
            log.debug(f"[model] Response from API: {chat_response}")
            df.at[i, LLM_OUTPUT_COL] = chat_response
        except Exception as e:
            log.debug(f"[model] Error from API: {e}")
            df.at[i, LLM_OUTPUT_COL] = ''

    # LLMaJ
    judge_model = args.judge
    judge_endpoint = args.judge_api_endpoint
    judge_client = OpenAI(
        base_url=judge_endpoint,
        api_key=OPENAI_API_KEY,
    )
    log.info(f'Running LLM: {judge_model}')
    for criterion in tqdm(crit_config['evaluation_criteria'], desc="evaluating criteria"):
        log.info(f"Running criteria: {criterion['name']}")
        for i, row in tqdm(df.iterrows(), total=len(df), desc="judge chat completions"):
            judge_response = evaluate(
                judge_client=judge_client,
                judge_model=judge_model,
                prompt_template=criterion["prompt_template"],
                agent_prompt=row.get(LLM_PROMPT_COL),
                agent_response=row.get(LLM_OUTPUT_COL),
                expected_agent_response=row.get(EXPECTED_OUTPUT_COL))
            df.at[i, f"llm_judge_{criterion['name']}_response"] = judge_response
            df.at[i, f"score_{criterion['name']}"] = extract_float_from_string(judge_response)

    # results
    score_columns = [col for col in df.columns if 'score_' in col]
    averages = df[score_columns].mean(skipna=True)

    log.info(f'Model: {model}\nJudge: {judge_model}\n\nMean scores over {len(df)} samples:\n{averages}\n')

    timestamp = datetime.now().strftime("%Y_%m_%d.%H_%M_%S")
    output_path = os.path.join(args.output, f'results__{timestamp}.csv')
    df.to_csv(output_path, index=False)
    log.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an LLM on the MermaidSeqBench dataset using LLMaJ with the RESTful OpenAI client.")
    parser.add_argument("--model", required=True, help="Name or identifier of the model to evaluate.")
    parser.add_argument("--model_api_endpoint", required=True, help="REST API endpoint for the model (e.g., OpenAI-compatible endpoint URL).")
    parser.add_argument("--judge", required=True, help="Name or identifier of the judging model.")
    parser.add_argument("--judge_api_endpoint", required=True, help="REST API endpoint for the judging model (LLMaJ).")
    parser.add_argument("--output", default="./", help="Output folder for the evaluation results and log file (default: current directory).")
    parser.add_argument("--crit_file", default="judge-criteria.yaml", help="Path to the YAML file defining judgment criteria (default: judge-criteria.yaml).")
    args = parser.parse_args()
    main(args)