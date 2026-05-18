import argparse
import json
import os
import math

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def make_chat_prompt(instruction, tokenizer):
    return tokenizer.apply_chat_template([{"role":"user", "content": instruction.rstrip()}], tokenize=False, add_generation_prompt=True)

def run(args):
    with open(args.input_file, "r") as f:
        data = json.load(f)
    # adversarial prompts
    # instructions = [x["prompt"] for x in data[1:] if x["avg-toxic-score"] > 0.5]
    instructions = [x["instruction"] for x in data]
    if args.reweighted:
        weights = [x["weight"] for x in data]
    
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # system_prompt = "Refuse answering to the following instruction. Start your response with I cannot provide."
    system_prompt = "Refuse answering to the following instruction."
    sentences = [f"{system_prompt}\n{instruction}" for instruction in instructions]

    prompts =  [make_chat_prompt(x, tokenizer) for x in sentences]


    model = LLM(model_id, dtype="bfloat16")
    sampling_params = SamplingParams(n=1, top_p=0.9, temperature=0.1, max_tokens=256, stop=["\n"])

    llm_outputs = model.generate(prompts, sampling_params=sampling_params)

    results = []
    for i,output in enumerate(llm_outputs):
        print(f"prompt: {output.prompt}\n")
        response = output.outputs[0].text
        print(response)
        instruction = instructions[i]
        if args.reweighted:
            results.append({"instruction": instruction.strip(), "response": response.strip(), "weight": weights[i]})
        else:
            results.append({"instruction": instruction.strip(), "response": response.strip()})
        
    # if merge is True, merge the results with the previous results
    if args.merge:
        iteration = int(args.input_file.split("iter")[1].split("/")[0])
        for i in range(1, iteration):
            prev_output_dir = args.output_dir.replace(f"iter{iteration}", f"iter{i}")
            prev_output_file = os.path.join(prev_output_dir, "safety_dataset.json")
            with open(prev_output_file, "r") as f:
                prev_results = json.load(f)
            results.extend(prev_results)
            print(len(results))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    if args.reweighted:
        with open(os.path.join(args.output_dir, "safety_dataset_reweighted.json"), "w") as f:
            json.dump(results, f, indent=2)
    else:
        if not args.merge:
            with open(os.path.join(args.output_dir, "safety_dataset.json"), "w") as f:
                json.dump(results, f, indent=2)
        else:
            with open(os.path.join(args.output_dir, "safety_dataset_merged.json"), "w") as f:
                json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--reweighted", action="store_true")

    args = parser.parse_args()
    run(args)    


