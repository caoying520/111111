
import numpy as np
import pandas as pd
import torch
import pickle, math
import os, sys
from tqdm import tqdm
from esm.pretrained import esm2_t33_650M_UR50D
import json
import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
current_dir = os.path.dirname(sys.argv[0])

# ESM2最多输入1024个token，其中包含BOS和EOS
MAX_SEQ_LENGTH = 1022


def construct_seq_dict(fasta_path):
    seq_dict = {}
    seq_name = None
    seq_parts = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if seq_name is not None:
                    seq_dict[seq_name] = "".join(seq_parts)

                seq_name = line[1:].strip()
                seq_parts = []

            else:
                seq_parts.append(line)

        if seq_name is not None:
            seq_dict[seq_name] = "".join(seq_parts)

    return seq_dict


def cal_esm2_to_json(fasta_path, save_esm2_path):

    model, alphabet = esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    res_dict = {}
    res_dict_json = {}

    seq_dict = construct_seq_dict(fasta_path)
    seq_dict_len = len(seq_dict)

    for index, seq_name in enumerate(seq_dict):

        start_dt = datetime.datetime.now()

        print('======>', index + 1, '/', seq_dict_len)

        sequence = seq_dict[seq_name]

        if len(sequence) == 0:
            print("Warning: empty sequence:", seq_name)
            continue

        # 长度不超过1022，保持原来的整条序列提取方式
        if len(sequence) <= MAX_SEQ_LENGTH:
            seq_segments = [sequence]

        # 超过1022 aa，进行连续分段
        else:
            seq_segments = [
                sequence[i:i + MAX_SEQ_LENGTH]
                for i in range(0, len(sequence), MAX_SEQ_LENGTH)
            ]

            print(
                "Long sequence detected:",
                seq_name,
                "length:",
                len(sequence),
                "segments:",
                len(seq_segments)
            )

        esm2_vector = []

        total_length = 0

        # 每个片段独立提取ESM2特征
        for segment in seq_segments:

            batch_labels, batch_strs, batch_tokens = batch_converter(
                ((seq_name, segment),)
            )

            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():

                results = model(
                    batch_tokens,
                    repr_layers=[33],
                    return_contacts=False
                )

                token_representations = results["representations"][33]

                tokens_len = batch_lens[0].item()

                # 去除BOS和EOS，只保留氨基酸残基
                segment_vector = (
                    token_representations[0, 1:tokens_len - 1]
                    .mean(0)
                    .cpu()
                    .detach()
                    .numpy()
                )

            segment_length = len(segment)

            # 按照片段长度加权累积
            esm2_vector.append(segment_vector * segment_length)

            total_length += segment_length

            del batch_tokens, results, token_representations

        # 加权平均，最终仍然是1280维
        esm2_vector = np.sum(esm2_vector, axis=0) / total_length

        res_dict[seq_name] = esm2_vector
        res_dict_json[seq_name] = esm2_vector.tolist()

        end_dt = datetime.datetime.now()

        time_cost = (end_dt - start_dt).seconds

        print(
            "time cost:",
            time_cost,
            "s ------> remain time:",
            int((seq_dict_len - index - 1) * time_cost / 60),
            "min"
        )

    # 保持原来的JSON输出格式
    with open(save_esm2_path, "w") as fp:
        json.dump(res_dict_json, fp)

    print("\nESM2 feature extraction completed!")
    print("Total sequences:", seq_dict_len)
    print("Successfully extracted:", len(res_dict_json))
    print("Feature dimension:", len(next(iter(res_dict_json.values()))))
    print("Saved to:", save_esm2_path)

    return res_dict


if __name__ == '__main__':

    fasta_path = r'/datasets/test_dataset.fasta'

    save_esm2_path = r'/datasets/test_dataset_esm2.json'

    cal_esm2_to_json(fasta_path, save_esm2_path)
