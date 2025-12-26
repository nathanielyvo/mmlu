# -*- encoding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from categories import categories, subcategories

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

from torch import nn
import torch
from tqdm import tqdm

def reset_model(model):
    for l in tqdm(range(len(model.model.layers))):
        for n in range(len(model.model.layers[l].mlp.experts)):
            proto = model.model.layers[l].mlp.gate.weight[n]
            gate = model.model.layers[l].mlp.experts[n].gate_proj.weight
            cos_sim = nn.functional.cosine_similarity(proto, gate, dim=-1)
            topk = torch.topk(cos_sim, k=len(cos_sim)//3)
            topk_indices = topk[1]
            
            # 为了方便操作，获取当前要修改的 expert 对象
            # 注意：你的循环变量是 l，但代码里写的是 layers[0]，这里我沿用 layers[0] 以匹配你的片段
            # 如果目的是遍历所有层，请记得将 layers[0] 改为 layers[l]
            current_expert = model.model.layers[l].mlp.experts[n]
            
            # 获取新的中间层大小
            new_inter_size = len(topk_indices)
            
            # --- 1. 处理 gate_proj (筛选行) ---
            # 提取旧权重，形状为 [intermediate, hidden]
            old_gate_weight = current_expert.gate_proj.weight.data
            # 创建新层 (保持 bias=False 设置)
            new_gate_layer = nn.Linear(current_expert.hidden_size, new_inter_size, bias=False)
            # 复制筛选后的权重到新层 (注意保持 device 和 dtype)
            new_gate_layer.weight.data = old_gate_weight[topk_indices, :].clone()
            # 替换模型中的层
            model.model.layers[l].mlp.experts[n].gate_proj = new_gate_layer

            # --- 2. 处理 up_proj (筛选行) ---
            # 提取旧权重，形状为 [intermediate, hidden]
            old_up_weight = current_expert.up_proj.weight.data
            new_up_layer = nn.Linear(current_expert.hidden_size, new_inter_size, bias=False)
            # 同样根据 topk_indice 筛选行
            new_up_layer.weight.data = old_up_weight[topk_indices, :].clone()
            model.model.layers[l].mlp.experts[n].up_proj = new_up_layer

            # --- 3. 处理 down_proj (筛选列) ---
            # 提取旧权重，形状为 [hidden, intermediate]
            old_down_weight = current_expert.down_proj.weight.data
            # 注意：down_proj 的输入维度变了，输出维度(hidden_size)保持不变
            new_down_layer = nn.Linear(new_inter_size, current_expert.hidden_size, bias=False)
            # 对于 down_proj，intermediate 维度在列上 (dim=1)，所以筛选列
            new_down_layer.weight.data = old_down_weight[:, topk_indices].clone()
            model.model.layers[l].mlp.experts[n].down_proj = new_down_layer

            # --- 4. 更新 expert 的属性 ---
            # 更新 intermediate_size，防止后续 forward 中用到该属性导致报错
            current_expert.intermediate_size = new_inter_size

            model.model.layers[0].mlp.experts[n] = current_expert
            
    return model



def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model.split("/")[-1]))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model.split("/")[-1])))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    start_time = time.time()
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model.split("/")[-1]), "{}.csv".format(subject)
            ),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    results_file = os.path.join(
        args.save_dir, "accuracies_{}.json".format(args.model.replace("/", "_"))
    )
    end_time = time.time()
    results["cost_time"] = end_time - start_time
    with open(results_file, "w") as f:
        f.write(json.dumps(results, indent=4))

    model = reset_model(model)

    model.eval()

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model.split("/")[-1]))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model.split("/")[-1])))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    start_time = time.time()
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model.split("/")[-1]), "{}.csv".format(subject)
            ),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    results_file = os.path.join(
        args.save_dir, "accuracies_{}.json".format(args.model.replace("/", "_"))
    )
    end_time = time.time()
    results["cost_time"] = end_time - start_time
    with open(results_file, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-30B-A3B")
    args = parser.parse_args()
    main(args)
    