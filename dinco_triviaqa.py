"""
Run DINCO on TriviaQA

(Results will not exactly match the paper due to slight code modifications, but differences in results are minimal.)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import transformers
from datasets import load_dataset
import pickle as pkl
import re
import os
import argparse

PROMPT = """
Here are 2 sets of example prompt and answer.

Example Prompt: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
Example Answer: Sinclair Lewis

Example Prompt: Where in England was Dame Judi Dench born?
Example Answer: York

---

Now, here is a new prompt to answer. Answer with a concise phrase, as in the examples.

Prompt: {question}
Answer:
""".strip()

PTRUE_PROMPT = """
Below is a question and a candidate answer. Your task is to determine whether the answer is correct or not. Only output \"Yes\" (correct) or \"No\" (incorrect).

Question: {question}
Candidate answer: {candidate_answer}
""".strip()

def beam_search(ds, model, tokenizer, num_beams=5, length_penalty=0.0, max_new_tokens=100):
    beam_strs = []
    beam_lls = torch.zeros((len(ds), num_beams))

    for ex_i, ex in enumerate(ds):
        msg = [
            {'role': 'user', 'content': PROMPT.format(question=ex['question'])}
        ]
        input_ids = tokenizer.apply_chat_template(
            [msg],
            enable_thinking=False,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        beam_strs.append(tokenizer.batch_decode(outputs.sequences[:, input_ids.shape[1]:], skip_special_tokens=True))
        beam_lls[ex_i] = outputs.sequences_scores.cpu()

    return beam_strs, beam_lls

def clean_str(s):
    s = s.split('\n')[0]
    s = s.replace('Answer:', '')
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def lexical_cleaning(beam_strs, beam_lls):
    filtered_beam_lls = -float('inf') * torch.ones_like(beam_lls)

    for ex_i in range(len(beam_strs)):
        # argsort beam_lls[ex_i] in descending order
        top_ll_is = torch.topk(beam_lls[ex_i], k=len(beam_lls[ex_i])).indices

        strs = []
        norm_strs = set()
        for seq_i in top_ll_is:
            s = clean_str(beam_strs[ex_i][seq_i])

            # skip if empty
            if len(s) == 0:
                continue

            # get normalized version
            norm_s = s.lower()
            norm_s = norm_s.replace('.', '')

            # add if first occurrence
            if norm_s not in norm_strs:
                norm_strs.add(norm_s)
                strs.append(s)
                filtered_beam_lls[ex_i, len(strs) - 1] = beam_lls[ex_i, seq_i]

        # replace with filtered strings
        beam_strs[ex_i] = strs

    return beam_strs, filtered_beam_lls

def get_ptrue(ds, model, tokenizer, beam_strs):
    ptrues = -torch.ones(n_qst, max([len(strs) for strs in beam_strs]))

    option_tok_ids = tokenizer.convert_tokens_to_ids(['Yes', 'No'])

    for ex_i, ex in enumerate(ds):
        msgs = []
        for ans_i, ans in enumerate(beam_strs[ex_i]):
            msgs.append([
                {'role': 'user', 'content': PTRUE_PROMPT.format(question=ex['question'], candidate_answer=ans)},
            ])

        seqs = tokenizer.apply_chat_template(
            msgs,
            enable_thinking=False,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                seqs,
                attention_mask=seqs.ne(tokenizer.pad_token_id),
                logits_to_keep=1,
            )
        ptrues[ex_i, :len(msgs)] = torch.softmax(outputs.logits[:, -1, option_tok_ids], dim=-1)[:, 0]

    return ptrues

def run_nli(ds, beam_strs):
    nli_model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name, device_map='auto')

    beam_width = max([len(strs) for strs in beam_strs])
    nlis = -torch.ones(len(ds), beam_width, beam_width, 3)

    for ex_i, ex in enumerate(ds):
        assert(len(beam_strs[ex_i]) >= 1)
        if len(beam_strs[ex_i]) == 1: # no pairs
            continue

        premises, hypotheses = [], []
        nlis_mask = torch.zeros(beam_width, beam_width, dtype=torch.bool)
        for i in range(len(beam_strs[ex_i])):
            premises.extend([f"Question: {ex['question']}\nAnswer: {beam_strs[ex_i][i]}"] * (len(beam_strs[ex_i]) - 1))
            hypotheses.extend([f"Answer: {beam_strs[ex_i][j]}" for j in range(len(beam_strs[ex_i])) if j != i])
            nlis_mask[i, :len(beam_strs[ex_i])] = 1
            nlis_mask[i, i] = 0

        inputs = nli_tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
        ).to(nli_model.device)
        with torch.no_grad():
            nlis[ex_i][torch.nonzero(nlis_mask, as_tuple=True)] = torch.softmax(nli_model(**inputs).logits, dim=-1).cpu()

    return nlis

def get_normalized_verbalized_confidence(ptrues, nlis):
    entail_i = 0
    contra_i = 2

    sym_nlis = (nlis + nlis.swapdims(-3, -2)) / 2
    contra_weights = sym_nlis[:, :, :, contra_i]
    sims = nlis[:, :, :, entail_i]
    degrees = torch.sum(torch.max(torch.tensor(0.), sims), dim=-2) + 1

    n_qst = ptrues.shape[0]
    nvcs = torch.empty(n_qst)

    for ex_i in range(n_qst):
        main_ans_i = 0 # set main answer to highest probability generation
        numerator = ptrues[ex_i, main_ans_i]
        denominator = numerator.clone()

        for ans_i in range(ptrues.shape[1]):
            if ans_i == main_ans_i:
                continue
            if ptrues[ex_i, ans_i] < 0:
                break

            denominator += ptrues[ex_i, ans_i] * contra_weights[ex_i, main_ans_i, ans_i] / (degrees[ex_i, ans_i] - sims[ex_i, main_ans_i, ans_i])

        if denominator > 1:
            nvcs[ex_i] = numerator / denominator
        else:
            nvcs[ex_i] = numerator

    return nvcs

def sample_generations(ds, model, tokenizer, n_sample=5, max_new_tokens=100):
    sampled_strs = []

    for ex_i, ex in enumerate(ds):
        msg = [
            {'role': 'user', 'content': PROMPT.format(question=ex['question'])}
        ]
        input_ids = tokenizer.apply_chat_template(
            [msg],
            enable_thinking=False,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids.repeat(n_sample, 1),
            attention_mask=input_ids.ne(tokenizer.pad_token_id).repeat(n_sample, 1),
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        sampled_strs.append(tokenizer.batch_decode(outputs.sequences[:, input_ids.shape[1]:], skip_special_tokens=True))

    return sampled_strs

def run_sc_nli(main_strs, sampled_strs):
    nli_model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name, device_map='auto')
    entail_i = 0

    n_qst = len(main_strs)
    n_sample = max([len(strs) for strs in sampled_strs])
    nlis = torch.zeros(n_qst, n_sample)

    for ex_i, ex in enumerate(ds):
        main_str = clean_str(main_strs[ex_i])
        premises, hypotheses = [], []

        assert(len(sampled_strs[ex_i]) == n_sample)
        nli_mask = torch.ones(n_sample, dtype=torch.bool)

        for sample_i, sampled_str in enumerate(sampled_strs[ex_i]):
            sampled_str = clean_str(sampled_str)
            if main_str == sampled_str: # exact match
                nli_mask[sample_i] = 0
            else:
                ans_pair = [main_str, sampled_str]
                for perm in [(0, 1), (1, 0)]:
                    premises.append(f"Question: {ex['question']}\nAnswer: {ans_pair[perm[0]]}")
                    hypotheses.append(f"Answer: {ans_pair[perm[1]]}")

        nlis[ex_i, ~nli_mask] = 1.0 # exact matches

        if len(premises) > 0:
            inputs = nli_tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
            ).to(nli_model.device)
            with torch.no_grad():
                outputs = torch.softmax(nli_model(**inputs).logits, dim=-1)[:, entail_i].cpu()
            nlis[ex_i, nli_mask] = (outputs[0::2] + outputs[1::2]) / 2

    return nlis

if __name__ == "__main__":
    transformers.set_seed(17)

    # load dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    n_qst = 1000
    ds = ds['validation'].shuffle(seed=17).select(range(n_qst))

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
    print(model_name)
    half_model_name = model_name.split('/')[1]
    results_fstem = f'results/triviaqa/{half_model_name}'

    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if 'Qwen' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', dtype=torch.float16)
    elif 'Llama' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', dtype=torch.float16)
        tokenizer.pad_token = tokenizer.eos_token
    elif 'gemma' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', dtype=torch.bfloat16)
        tokenizer.eos_token_id = 106 # <end_of_turn>

    # beam search
    beam_strs, beam_lls = beam_search(ds, model, tokenizer)
    # save
    if not os.path.exists(results_fstem):
        os.makedirs(results_fstem)
    with open(f'{results_fstem}/beam_strs.pkl', 'wb') as f:
        pkl.dump(beam_strs, f)
    torch.save(beam_lls, f'{results_fstem}/beam_lls.pth')
    print('Saved beam search results', flush=True)

    # perform lexical cleaning
    beam_strs, beam_lls = lexical_cleaning(beam_strs, beam_lls)
    # save
    with open(f'{results_fstem}/beam_strs_cleaned.pkl', 'wb') as f:
        pkl.dump(beam_strs, f)
    torch.save(beam_lls, f'{results_fstem}/beam_lls_cleaned.pth')
    print('Saved cleaned beam search results', flush=True)

    # collect verbalized confidence
    ptrues = get_ptrue(ds, model, tokenizer, beam_strs)
    # save
    torch.save(ptrues, f'{results_fstem}/ptrues.pth')
    print('Saved P(true) results', flush=True)

    # get NLI predictions
    nlis = run_nli(ds, beam_strs)
    # save
    torch.save(nlis, f'{results_fstem}/nlis.pth')
    print('Saved NLI results', flush=True)

    # compute normalized verbalized confidence
    nvcs = get_normalized_verbalized_confidence(ptrues, nlis)
    # save
    torch.save(nvcs, f'{results_fstem}/nvcs.pth')
    print('Saved NVC results', flush=True)

    # run self-consistency
    sampled_strs = sample_generations(ds, model, tokenizer)
    # save
    with open(f'{results_fstem}/sampled_strs.pkl', 'wb') as f:
        pkl.dump(sampled_strs, f)
    print('Saved sampled generations', flush=True)

    # compute self-consistency matches
    main_strs = [strs[0] for strs in beam_strs] # highest probability generation
    sc_nlis = run_sc_nli(main_strs, sampled_strs)
    # save
    torch.save(sc_nlis, f'{results_fstem}/sc_nlis.pth')
    print('Saved self-consistency NLI results', flush=True)

    # compute SC confidences
    # append 1 to dim=-1 for main answer
    sc_nlis = torch.cat((sc_nlis, torch.ones(n_qst, 1)), dim=-1)
    sc_match_threshold = 0.9
    sc_confs = torch.mean((sc_nlis > sc_match_threshold).float(), dim=-1)

    # compute NVC and SC
    dinco_confs = (nvcs + sc_confs) / 2
    # save
    torch.save(dinco_confs, f'{results_fstem}/dinco_confs.pth')
    print('Saved DINCO results', flush=True)
