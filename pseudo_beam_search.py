"""
Pseudo-beam search for the OpenAI API
"""

import numpy as np
import json

PROMPT = """
You will be given a prompt along with a prefix to begin your answer with. Your answer should start with the given prefix. If the prefix itself is your final answer, you can simply output just the prefix.

Let's look at 2 examples before the real task.

Example Prompt: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
Example Answer Prefix: Sin
Example Answer: Sinclair Lewis

Example Prompt: Where in England was Dame Judi Dench born?
Example Answer Prefix: York
Example Answer: York

---

Now, here is a new prompt to answer. Answer with a concise phrase starting with the given prefix, as in the examples.

Prompt: {question}
Prefix: {prefix}
Answer:
""".strip()

def pseudo_beam_search():
    """
    Use output from first inference call (which got the main prediction) to make input to complete high-likelihood prefixes.
    """

    # load dataset
    with open('simple_qa_test_set.jsonl') as f:
        ds = [json.loads(line) for line in f.readlines() if line.strip()]

    # load gpt output from first inference call
    model_name = "gpt-4.1-2025-04-14"
    with open(f'gpt_output/{model_name}.jsonl') as f:
        jsons = [json.loads(line) for line in f.readlines() if line.strip()]

    n_top_logprob = 11
    ll_thresh = -20
    n_distractor = 10

    with open(f'gpt_input/{model_name}_beams.jsonl', 'w') as f:
        for json_line in jsons:
            content_json = json_line['response']['body']['choices'][0]['logprobs']['content']
            n_token = len(content_json)
            prefix_str = ''
            prefix_ll = 0.0
            strs = []
            lls = []

            for tokpos_i in range(n_token):
                top_logprobs_json = content_json[tokpos_i]['top_logprobs']

                for rank_i in range(1, n_top_logprob): # skip greedy option used in main answer
                    token = top_logprobs_json[rank_i]['token']
                    if token.startswith('<|') and token.endswith('|>'): # skip special tokens
                        continue
                    strs.append(prefix_str + token)

                    lls.append(prefix_ll + top_logprobs_json[rank_i]['logprob'])

                prefix_str += content_json[tokpos_i]['token']
                prefix_ll += content_json[tokpos_i]['logprob']

            # get top n_distractor prefixes with ll > ll_thresh
            order = np.argsort(lls)[::-1]
            order = [i for i in order if lls[i] > ll_thresh][:n_distractor]
            strs = [strs[i] for i in order]
            lls = [lls[i] for i in order]

            ex_i = int(json_line['custom_id'].split('-')[-1])
            qst = ds[ex_i]['problem']
            for distractor_i, distractor in enumerate(strs):
                user_prompt = PROMPT.format(question=qst, prefix=distractor)

                json.dump({
                    'custom_id': f"qid-{ex_i}_beam-{distractor_i}",
                    'method': 'POST',
                    'url': '/v1/chat/completions',
                    'body': {
                        'model': model_name,
                        'messages': [
                            {
                                'role': 'user',
                                'content': user_prompt
                            }
                        ],
                        'temperature': 0,
                        'max_completion_tokens': 50,
                    }
                }, f)
                f.write('\n')