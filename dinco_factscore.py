"""
Run DINCO on TriviaQA
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import transformers
from datasets import load_dataset
import pickle as pkl
import re
import os
import json
import argparse

PROMPT = "Write me a paragraph biography on {entity}."

DISTRACTOR_PROMPT = """
You will be given a fact about a person. Assuming the fact is accurate, your task is to generate a plausible but inaccurate statement of a similar nature. The distractor statement should form a minimal pair with the original statement, i.e. the distractor should be as similar to the original as possible while ensuring that the distractor is not factual. The distractor should be crafted so that someone with only superficial knowledge about the topic is likely to be fooled.

Let's see some examples before the real task.

Topic: Barack Obama
Fact: Barack Obama was born in Hawaii.
Distractor: Barack Obama was born in Kenya.

Topic: Wright brothers
Fact: Wright airplanes were involved in fatal crashes.
Distractor: Wright airplanes were praised for their safety.

Topic: John Clempert
Fact: John Clempert was inspired by Houdini when developing acts.
Distractor: John Clempert was inspired by Penn and Teller when developing acts.

Now for the real task. Only output a distractor as in the examples.

Topic: {entity}
Fact: {claim}
Distractor:
""".strip()

PTRUE_PROMPT = """
Your task is to determine whether the following claim related to {entity} is correct. Only output "Yes" (correct) or "No" (incorrect).

Claim: {claim}

Yes or No:
""".strip()

NLI_PROMPT = """
You will be given a passage and a claim. Your task is to determine whether the passage supports, refutes, or does not mention the claim. Output only "Support", "Refute", or "No Mention".

Let's see some examples before the real task.

Passage: Barack Obama was the 44th President of the United States, serving from 2009 to 2017. Born on August 4, 1961, in Honolulu, Hawaii, he was the first African American to hold the office. Before his presidency, Obama served as a state senator in Illinois and later as the 47th Governor of Illinois. A former constitutional law professor, he was known for his eloquence, bipartisan approach, and focus on issues such as healthcare reform, climate change, and foreign policy. His presidency was marked by significant legislative achievements, including the Affordable Care Act, and a commitment to diplomacy and international cooperation. After leaving office, he authored memoirs and remained active in public life, advocating for social justice and community engagement.
Claim: Barack Obama was born in Hawaii.
Relationship: Support

Passage: Tiger Woods is one of the most iconic and accomplished golfers in history, known for his extraordinary talent, dominance on the course, and global influence on the sport. Born on December 30, 1975, in Cypress, Florida, Woods rose to fame in the mid-1990s and quickly became a household name, winning his first major championship at the 1997 Masters at just 21 years old. Over his career, he has claimed 15 major titles, the most in PGA Tour history, and has consistently ranked among the world's top golfers for over two decades. His aggressive playing style, precision, and mental toughness set him apart, making him a symbol of excellence in golf. Despite personal challenges and setbacks, Woods has remained a dominant force in the sport, inspiring millions of fans around the world.
Claim: Tiger Woods won a major championship at 19 years old.
Relationship: Refute

Passage: Albert Einstein was a theoretical physicist renowned for developing the theory of relativity, which revolutionized the understanding of space, time, and gravity. Born in 1879 in Ulm, Germany, he later moved to Switzerland and eventually to the United States, where he became a naturalized citizen. Einstein's work, including the famous equation E=mc², laid the foundation for modern physics and contributed to the development of nuclear energy. Despite his scientific achievements, he was also a passionate advocate for peace, civil rights, and education. His legacy endures as one of the most influential scientists in history.
Claim: Albert Einstein became a US citizen.
Relationship: No Mention

Now for the real task.

Passage: {bio}
Claim: {claim}
Relationship:
""".strip()

def beam_search(decomps, model, tokenizer, num_beams=5, length_penalty=0.0, max_new_tokens=100):
    beam_strs = []
    max_n_claim = max([len(decomp['decisions']) for decomp in decomps])
    beam_lls = torch.zeros((len(decomps), max_n_claim, num_beams))

    for decomp_i, decomp in enumerate(decomps):
        beam_strs.append([])

        for claim_i, claim in enumerate(decomp['decisions']):
            msg = [
                {'role': 'user', 'content': DISTRACTOR_PROMPT.format(entity=decomp['entity'], claim=claim['atom'])}
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

            beam_strs[decomp_i].append(tokenizer.batch_decode(outputs.sequences[:, input_ids.shape[1]:], skip_special_tokens=True))
            beam_lls[decomp_i, claim_i] = outputs.sequences_scores.cpu()

    return beam_strs, beam_lls

def clean_str(s):
    s = s.split('\n')[0]
    s = s.replace('Distractor:', '')
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def lexical_cleaning(beam_strs, beam_lls):
    filtered_beam_lls = -float('inf') * torch.ones_like(beam_lls)

    for entity_i in range(len(beam_strs)):
        for claim_i in range(len(beam_strs[entity_i])):
            # argsort beam_lls[entity_i] in descending order
            top_ll_is = torch.topk(beam_lls[entity_i][claim_i], k=len(beam_lls[entity_i][claim_i])).indices

            strs = []
            norm_strs = set()
            for seq_i in top_ll_is:
                s = clean_str(beam_strs[entity_i][claim_i][seq_i])

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
                    filtered_beam_lls[entity_i, len(strs) - 1] = beam_lls[entity_i, seq_i]

            # replace with filtered strings
            beam_strs[entity_i][claim_i] = strs

    return beam_strs, filtered_beam_lls

def get_ptrue(decomps, model, tokenizer, beam_strs):
    max_n_claim = max([len(decomp_strs) for decomp_strs in beam_strs])
    max_n_distractor = max([len(claim_strs) for decomp_strs in beam_strs for claim_strs in decomp_strs])
    ptrues = -torch.ones(len(decomps), max_n_claim, max_n_distractor)

    option_tok_ids = tokenizer.convert_tokens_to_ids(['Yes', 'No'])

    for decomp_i, decomp in enumerate(decomps):
        for claim_i in range(len(decomp['decisions'])):
            msgs = []
            for distractor_i, distractor in enumerate(beam_strs[decomp_i][claim_i]):
                msgs.append([
                    {'role': 'user', 'content': PTRUE_PROMPT.format(entity=decomp['entity'], claim=distractor)},
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
            ptrues[decomp_i, claim_i, :len(msgs)] = torch.softmax(outputs.logits[:, -1, option_tok_ids], dim=-1)[:, 0]

    return ptrues

def run_nli(beam_strs):
    nli_model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name, device_map='auto')

    max_n_claim = max([len(decomp_strs) for decomp_strs in beam_strs])
    max_n_distractor = max([len(claim_strs) for decomp_strs in beam_strs for claim_strs in decomp_strs])
    nlis = -torch.ones(len(beam_strs), max_n_claim, max_n_distractor, max_n_distractor, 3)

    for entity_i in range(len(beam_strs)):
        for claim_i in range(len(beam_strs[entity_i])):
            n_distractor = len(beam_strs[entity_i][claim_i])
            assert(n_distractor >= 1)
            if n_distractor == 1: # no pairs
                continue

            premises, hypotheses = [], []
            nlis_mask = torch.zeros(max_n_distractor, max_n_distractor, dtype=torch.bool)
            for distractor_i in range(n_distractor):
                premises.extend([beam_strs[entity_i][claim_i][distractor_i]] * (n_distractor - 1))
                hypotheses.extend([beam_strs[entity_i][claim_i][distractor_j] for distractor_j in range(n_distractor) if distractor_j != distractor_i])
                nlis_mask[distractor_i, :n_distractor] = 1
                nlis_mask[distractor_i, distractor_i] = 0

            inputs = nli_tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
            ).to(nli_model.device)
            with torch.no_grad():
                nlis[entity_i, claim_i][torch.nonzero(nlis_mask, as_tuple=True)] = torch.softmax(nli_model(**inputs).logits, dim=-1).cpu()

    return nlis

def get_normalized_verbalized_confidence(ptrues, nlis):
    entail_i = 0
    contra_i = 2

    sym_nlis = (nlis + nlis.swapdims(-3, -2)) / 2
    contra_weights = sym_nlis[..., contra_i]
    sims = nlis[..., entail_i]
    degrees = torch.sum(torch.max(torch.tensor(0.), sims), dim=-2) + 1

    n_entity = ptrues.shape[0]
    nvcs = -torch.ones(n_entity, ptrues.shape[1])

    for entity_i in range(n_entity):
        n_claim = torch.sum(ptrues[entity_i, :, 0] > -1).item()
        for claim_i in range(n_claim):
            main_ans_i = 0
            numerator = ptrues[entity_i, claim_i, main_ans_i]
            denominator = numerator.clone()

            for distractor_i in range(ptrues.shape[2]):
                if distractor_i == main_ans_i:
                    continue
                if ptrues[entity_i, claim_i, distractor_i] < 0:
                    break

                denominator += ptrues[entity_i, claim_i, distractor_i] * contra_weights[entity_i, claim_i, main_ans_i, distractor_i] / (degrees[entity_i, claim_i, distractor_i] - sims[entity_i, claim_i, main_ans_i, distractor_i])

            if denominator > 1:
                nvcs[entity_i, claim_i] = numerator / denominator
            else:
                nvcs[entity_i, claim_i] = numerator

    return nvcs

def sample_generations(decomps, model, tokenizer, n_sample=5, max_new_tokens=500):
    sampled_strs = []

    for decomp_i, decomp in enumerate(decomps):
        msg = [
            {'role': 'user', 'content': PROMPT.format(entity=decomp['entity'])}
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

def run_sc_nli(decomps, sampled_strs):
    def get_options():
        text_options = ['Support', 'Refute', 'No Mention']
        token_options = torch.zeros(len(text_options), dtype=torch.long)
        for i in range(len(text_options)):
            token_options[i] = tokenizer(text_options[i], add_special_tokens=False).input_ids[0]
        return token_options

    option_tok_ids = get_options()

    max_n_claim = max([len(decomp['decisions']) for decomp in decomps])
    n_sample = max([len(strs) for strs in sampled_strs])
    nlis = -torch.ones(len(decomps), max_n_claim, n_sample)

    for entity_i, entity in enumerate(decomps):
        for claim_i, claim in enumerate(entity['decisions']):
            assert(len(sampled_strs[entity_i]) == n_sample)
            msgs = []
            for sample_i in range(n_sample):
                msgs.append([
                    {'role': 'user', 'content': NLI_PROMPT.format(bio=sampled_strs[entity_i][sample_i], claim=claim['atom'])},
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
            nlis[entity_i, claim_i] = torch.softmax(outputs.logits[:, -1, option_tok_ids], dim=-1)[:, 0]

    return nlis

if __name__ == "__main__":
    transformers.set_seed(17)

    # get model name
    parser = argparse.ArgumentParser()
    model_names = [
        'Qwen/Qwen3-8B',
        'google/gemma-3-4b-it',
    ]
    parser.add_argument('--model_name', type=str, default=model_names[1], choices=model_names, help='huggingface model name')
    args = parser.parse_args()
    model_name = args.model_name
    print(model_name)
    half_model_name = model_name.split('/')[1]
    results_fstem = f'results/factscore/{half_model_name}'

    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if 'Qwen' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', dtype=torch.float16)
    elif 'gemma' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', dtype=torch.bfloat16)
        tokenizer.eos_token_id = 106 # <end_of_turn>

    # load claim decompositions
    with open(f'factscore_claims/{half_model_name}.jsonl') as f:
        decomps = [json.loads(line) for line in f.readlines() if line.strip()]

    # beam search
    beam_strs, beam_lls = beam_search(decomps, model, tokenizer)
    # save
    if not os.path.exists(f'{results_fstem}'):
        os.makedirs(f'{results_fstem}')
    with open(f'{results_fstem}/beam_strs.pkl', 'wb') as f:
        pkl.dump(beam_strs, f)
    torch.save(beam_lls, f'{results_fstem}/beam_lls.pth')
    print('Saved beam search results', flush=True)

    # prepend original claim to distractors
    for decomp_i, decomp in enumerate(decomps):
        for claim_i, claim in enumerate(decomp['decisions']):
            beam_strs[decomp_i][claim_i].insert(0, claim['atom'])
    beam_lls = torch.cat([torch.zeros_like(beam_lls[..., :1]), beam_lls], dim=-1)

    # perform lexical cleaning
    beam_strs, beam_lls = lexical_cleaning(beam_strs, beam_lls)
    # save
    with open(f'{results_fstem}/beam_strs_cleaned.pkl', 'wb') as f:
        pkl.dump(beam_strs, f)
    torch.save(beam_lls, f'{results_fstem}/beam_lls_cleaned.pth')
    print('Saved cleaned beam search results', flush=True)
    # load beam_strs, beam_lls
    with open(f'{results_fstem}/beam_strs_cleaned.pkl', 'rb') as f:
        beam_strs = pkl.load(f)
    beam_lls = torch.load(f'{results_fstem}/beam_lls_cleaned.pth')

    # collect verbalized confidence
    ptrues = get_ptrue(decomps, model, tokenizer, beam_strs)
    # save
    torch.save(ptrues, f'{results_fstem}/ptrues.pth')
    print('Saved P(true) results', flush=True)

    # get NLI predictions
    nlis = run_nli(beam_strs)
    # save
    torch.save(nlis, f'{results_fstem}/nlis.pth')
    print('Saved NLI results', flush=True)

    # compute normalized verbalized confidence
    nvcs = get_normalized_verbalized_confidence(ptrues, nlis)
    # save
    torch.save(nvcs, f'{results_fstem}/nvcs.pth')
    print('Saved NVC results', flush=True)

    # run self-consistency
    sampled_strs = sample_generations(decomps, model, tokenizer)
    # save
    with open(f'{results_fstem}/sampled_strs.pkl', 'wb') as f:
        pkl.dump(sampled_strs, f)
    print('Saved sampled generations', flush=True)

    # compute self-consistency matches
    sc_nlis = run_sc_nli(decomps, sampled_strs)
    # save
    torch.save(sc_nlis, f'{results_fstem}/sc_nlis.pth')
    print('Saved self-consistency NLI results', flush=True)

    # mean over samples
    sc_confs = torch.mean(sc_nlis, dim=-1)

    # compute NVC and SC
    dinco_confs = (nvcs + sc_confs) / 2
    # save
    torch.save(dinco_confs, f'{results_fstem}/dinco_confs.pth')
    print('Saved DINCO results', flush=True)
