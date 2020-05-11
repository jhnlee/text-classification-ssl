from transformers import BertTokenizer, BertConfig
import numpy as np
import logging
import os
import csv
import pickle

import torch
from torch.utils.data import DataLoader, Dataset


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len=128):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, "cached_" + filename)
        self.max_len = max_len

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.sentence, self.label = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            data = []
            with open(file_path, encoding="utf-8") as tsvreader:
                for line in csv.reader(tsvreader, delimiter="\t"):
                    data.append(line)
            text, label = list(zip(*data[1:]))
            tokenized_text = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t)) for t in text]
            tokenized_text = [tokenizer.build_inputs_with_special_tokens(t) for t in tokenized_text]
            label = [float(l) for l in label]

            if max(label) == len(set(label)):
                label = [l - 1.0 for l in label]

            # adjust order
            label_unique, counts = np.unique(label, return_counts=True)
            counts_min = counts.min()
            sentences = []
            labels = []
            for l in label_unique:
                idx = np.array([t == l for t in label])
                sentences.append(np.array(tokenized_text)[idx][:counts_min])
                labels.append(np.array(label)[idx][:counts_min])

            sentence = np.concatenate(np.array(sentences).T).tolist()
            label = np.concatenate(np.array(labels).T).tolist()

            self.sentence = sentence
            self.label = label

            logger.info("Saving features into cached file %s", cached_features_file)

            with open(cached_features_file, "wb") as handle:
                pickle.dump((self.sentence, self.label), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, item):
        if len(self.sentence[item]) > self.max_len:
            sentence = torch.tensor(
                [self.sentence[item][0]]
                + self.sentence[item][-(self.max_len - 1) : -1]
                + [self.sentence[item][-1]],
                dtype=torch.long,
            )
        else:
            sentence = torch.tensor(self.sentence[item], dtype=torch.long)
        label = torch.tensor(self.label[item], dtype=torch.long)
        return sentence, label


class AugmentedDataset(TextDataset):
    def __init__(
        self,
        tokenizer,
        file_path,
        num_labeled,
        num_aug,
        mlm_prob,
        num_mlm_repeat,
        max_len=128,
        use_bt=False,
    ):
        super(AugmentedDataset, self).__init__(tokenizer, file_path)
        self.max_len = max_len

        directory, filename = os.path.split(file_path)
        directory = os.path.join(directory, str(num_labeled))
        if use_bt:
            aug_prefix = "bt_"
            directory, _ = os.path.split(directory)
            directory, dataset = os.path.split(directory)
            directory = os.path.join(directory, "bt", dataset)
        else:
            aug_prefix = (
                "augmented" + str(num_aug) + "_" + str(mlm_prob) + "_" + str(num_mlm_repeat) + "_"
            )
        aug_features_file = os.path.join(
            directory, aug_prefix + filename if aug_prefix not in filename else filename
        )
        assert os.path.isfile(aug_features_file)
        logger.info("Loading features from augmented file : %s", aug_features_file)
        with open(aug_features_file, "rb") as handle:
            self.aug_sentence, self.teacher_ori_logit = pickle.load(handle)

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, item):
        if len(self.sentence[item]) > self.max_len:
            sentence = torch.tensor(
                [self.sentence[item][0]]
                + self.sentence[item][-(self.max_len - 1) : -1]
                + [self.sentence[item][-1]],
                dtype=torch.long,
            )
        else:
            sentence = torch.tensor(self.sentence[item], dtype=torch.long)
        if len(self.aug_sentence[item]) > self.max_len:
            aug_sentence = torch.tensor(
                [self.aug_sentence[item][0]]
                + self.aug_sentence[item][-(self.max_len - 1) : -1]
                + [self.aug_sentence[item][-1]],
                dtype=torch.long,
            )
        else:
            aug_sentence = torch.tensor(self.aug_sentence[item], dtype=torch.long)
        label = torch.tensor(self.label[item], dtype=torch.long)
        sentence_logit = torch.tensor(self.teacher_ori_logit[item], dtype=torch.float32)

        return (
            sentence,
            aug_sentence,
            label,
            sentence_logit,
        )
