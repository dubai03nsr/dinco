"""
Judge candidate answers for TriviaQA
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pickle as pkl
import argparse

JUDGE_PROMPT = """
Below is a question along with the ground-truth answer and a candidate answer. Your task is to determine whether the candidate answer is semantically consistent with the ground-truth answer in the context of the question. Only output "Yes" or "No".

Question: {question}
Ground-truth answer: {gold_ans}
Candidate answer: {cand_ans}
""".strip()

def run(batch_msgs, option_tok_ids):
    input_ids = tokenizer.apply_chat_template(
        batch_msgs,
        enable_thinking=False,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        max_new_tokens=1,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        return_dict_in_generate=True,
        output_scores=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    return torch.softmax(outputs.scores[-1], dim=-1)[:, option_tok_ids]

if __name__ == "__main__":
    # get model name
    parser = argparse.ArgumentParser()
    model_names = [
        'Qwen/Qwen3-8B',
        'Qwen/Qwen3-1.7B',
        'meta-llama/Llama-3.2-3B-Instruct',
        'google/gemma-3-4b-it',
    ]
    parser.add_argument('--model_name', type=str, default=model_names[1], choices=model_names, help='huggingface model name')
    args = parser.parse_args()
    model_name = args.model_name
    half_model_name = model_name.split('/')[1]
    results_fstem = f'results/triviaqa/{half_model_name}'

    # load judge
    judge_name = 'meta-llama/Llama-3.1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(judge_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(judge_name, device_map='auto', dtype=torch.float16)
    tokenizer.pad_token = tokenizer.eos_token

    # load dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    n_qst = 1000
    ds = ds['validation'].shuffle(seed=17).select(range(n_qst))

    # load candidate answers
    with open(f'{results_fstem}/beam_strs_cleaned.pkl', 'rb') as f:
        kept_ansss = pkl.load(f)
    cand_anss = [anss[0] for anss in kept_ansss]

    option_tok_ids = torch.tensor(tokenizer.convert_tokens_to_ids(['Yes', 'No']))
    judge_probs = -torch.ones(n_qst, len(option_tok_ids), dtype=torch.float16)

    prev_ds_logprobs_i, ds_logprobs_i = 0, 0
    batch_size = 100
    batch_msgs = []

    # judge
    for ex_i, ex in enumerate(ds):
        # make msg
        prompt = JUDGE_PROMPT.format(question=ex['question'], gold_ans=ex['answer']['value'], cand_ans=cand_anss[ex_i])
        msg = [
            {'role': 'user', 'content': prompt}
        ]

        if len(batch_msgs) >= batch_size:
            batch_judge_probs = run(batch_msgs, option_tok_ids)
            judge_probs[prev_ds_logprobs_i:ds_logprobs_i] = batch_judge_probs
            prev_ds_logprobs_i = ds_logprobs_i
            batch_msgs = []

        # add to batch
        if ds_logprobs_i >= prev_ds_logprobs_i:
            batch_msgs.append(msg)
        ds_logprobs_i += 1
    # final run
    if len(batch_msgs) > 0:
        batch_judge_probs = run(batch_msgs, option_tok_ids)
        judge_probs[prev_ds_logprobs_i:ds_logprobs_i] = batch_judge_probs

    torch.save(judge_probs, f'{results_fstem}/judge-probs.pt')
