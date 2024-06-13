import os
import json
import numpy as np
import pandas as pd
import einops

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import sentencepiece as spm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class NLMCXR(data.Dataset):
    def __init__(
        self,
        directory,
        input_size=(256, 256),
        random_transform=False,
        view_pos=["AP", "PA", "LATERAL"],
        max_views=2,
        sources=["image", "history"],
        targets=["label"],
        max_len=1000,
        vocab_file="nlmcxr_unigram_1000.model",
    ):

        self.source_sections = ["INDICATION", "COMPARISON"]
        self.target_sections = ["FINDINGS"]
        self.dir = directory
        self.input_size = input_size
        self.random_transform = True
        self.view_pos = view_pos
        self.max_views = max_views
        self.sources = sources
        self.targets = targets
        self.max_len = max_len
        self.vocab_file = vocab_file
        self.vocab = spm.SentencePieceProcessor(model_file=directory + vocab_file)
        self.__input_data(binary_mode=True)

        if random_transform:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(0.1, 0.1, 0.1),
                            transforms.RandomRotation(15, expand=True),
                        ]
                    ),
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(input_size), transforms.ToTensor()]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        sources, targets = [], []
        temp_rep = self.captions[self.file_report[file_name]["image"][0] + ".png"]

        if "image" in self.sources:
            imgs, vpos = [], []
            images = self.file_report[file_name]["image"]

            new_orders = np.random.permutation(len(images))
            img_files = np.array(images)[new_orders].tolist()

            for i in range(min(self.max_views, len(img_files))):
                img = Image.open(self.dir + "images/" + img_files[i] + ".png").convert(
                    "RGB"
                )
                imgs.append(self.transform(img).unsqueeze(0))  # [1,C,W,H]
                vpos.append(1)

            cur_len = len(vpos)
            for i in range(cur_len, self.max_views):
                imgs.append(torch.zeros_like(imgs[0]))
                vpos.append(-1)

            imgs = torch.cat(imgs, dim=0)
            vpos = np.array(vpos, dtype=np.int64)

        info = self.file_report[file_name]["report"]

        source_info = []
        for section, content in info.items():
            if section in self.source_sections:
                source_info.append(content)
        source_info = " ".join(
            source_info
        )  # 将 source_info 列表中的所有元素用 " " 连接

        encoded_source_info = (
            [self.vocab.bos_id()]
            + self.vocab.encode(source_info)
            + [self.vocab.eos_id()]
        )  # encode source_info
        source_info = (
            np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        )  # 全padding 长度为maxlen
        source_info[: min(len(encoded_source_info), self.max_len)] = (
            encoded_source_info[: min(len(encoded_source_info), self.max_len)]
        )  # 有值的部分赋值 其余部分为padding

        target_info = []
        for section, content in info.items():
            if section in self.target_sections:
                target_info.append(content)
        #target_info = " ".join(target_info)

        target_info = temp_rep

        np_labels = np.zeros(len(self.top_np), dtype=float)
        for i in range(len(self.top_np)):
            if self.top_np[i] in target_info:
                np_labels[i] = 1

        encoded_target_info = (
            [self.vocab.bos_id()]
            + self.vocab.encode(target_info)
            + [self.vocab.eos_id()]
        )
        target_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        target_info[: min(len(encoded_target_info), self.max_len)] = (
            encoded_target_info[: min(len(encoded_target_info), self.max_len)]
        )

        for i in range(len(self.sources)):
            if self.sources[i] == "image":
                sources.append((imgs, vpos))
            elif self.sources[i] == "history":
                sources.append(source_info)
            elif self.sources[i] == "label":
                sources.append(
                    np.concatenate([np.array(self.file_labels[file_name]), np_labels])
                )
            elif self.sources[i] == "caption":
                sources.append(target_info)
            elif self.sources[i] == "caption_length":
                sources.append(min(len(encoded_target_info), self.max_len))

        for i in range(len(self.targets)):
            if self.targets[i] == "label":
                targets.append(
                    np.concatenate([np.array(self.file_labels[file_name]), np_labels])
                )
            elif self.targets[i] == "caption":
                targets.append(target_info)
            elif self.targets[i] == "caption_length":
                targets.append(min(len(encoded_target_info), self.max_len))

        return sources if len(sources) > 1 else sources[0], (
            targets if len(targets) > 1 else targets[0]
        )

    def __get_nounphrase(self, top_k=100, file_name="count_nounphrase.json"):
        with open(self.dir + file_name, "r") as f:
            count_nounphrase = json.load(f)
        count_nounphrase = dict(
            sorted(count_nounphrase.items(), key=lambda x: x[1], reverse=True)
        )
        top_np = list(count_nounphrase.keys())[:top_k]
        return top_np

    def __input_data(self, binary_mode=True):
        self.__input_caption()
        self.__input_report()
        self.__input_label()
        self.__filter_inputs()
        self.top_np = self.__get_nounphrase()

    def __input_caption(self):
        with open(self.dir + "captions.json", "r") as f:
            captions = json.load(f)
        self.captions = captions

    def __input_report(self):
        with open(self.dir + "reports_ori.json", "r") as f:
            reports = json.load(f)
        self.file_list = list(reports.keys())
        self.file_report = reports

    def __input_label(self):
        with open(self.dir + "file2label.json", "r") as f:
            labels = json.load(f)
        self.file_labels = labels

    def __filter_inputs(self):
        filtered_files_report = {}
        for file_name, report in self.file_report.items():
            if (len(report["image"]) > 0) and (
                ("FINDINGS" in report["report"])
                and (report["report"]["FINDINGS"] != "")
            ):
                filtered_files_report[file_name] = report
        self.file_report = filtered_files_report
        self.file_list = list(self.file_report.keys())

    def get_subsets(self, train_size=0.7, val_size=0.1, test_size=0.2, seed=0):
        np.random.seed(seed)
        indices = np.random.permutation(len(self.file_list))
        train_pvt = int(train_size * len(self.file_list))
        val_pvt = int((train_size + val_size) * len(self.file_list))
        train, val, test = (
            indices[:train_pvt],
            indices[train_pvt:val_pvt],
            indices[val_pvt:],
        )

        master_file_list = np.array(self.file_list)
        train_dataset = NLMCXR(
            self.dir,
            self.input_size,
            self.random_transform,
            self.view_pos,
            self.max_views,
            self.sources,
            self.targets,
            self.max_len,
            self.vocab_file,
        )
        train_dataset.file_list = master_file_list[train].tolist()

        val_dataset = NLMCXR(
            self.dir,
            self.input_size,
            self.random_transform,
            self.view_pos,
            self.max_views,
            self.sources,
            self.targets,
            self.max_len,
            self.vocab_file,
        )
        val_dataset.file_list = master_file_list[val].tolist()

        test_dataset = NLMCXR(
            self.dir,
            self.input_size,
            self.random_transform,
            self.view_pos,
            self.max_views,
            self.sources,
            self.targets,
            self.max_len,
            self.vocab_file,
        )
        test_dataset.file_list = master_file_list[test].tolist()

        return train_dataset, val_dataset, test_dataset


class NLMCXR_ONE(data.Dataset):

    def __init__(
        self,
        directory,
        input_size=(256, 256),
        random_transform=False,
        view_pos=["AP", "PA", "LATERAL"],
        max_views=2,
        sources=["caption"],
        targets=["label"],
        max_len=1000,
        vocab_file="nlmcxr_unigram_1000.model",
    ):

        self.source_sections = ["INDICATION", "COMPARISON"]
        self.target_sections = ["FINDINGS"]
        self.dir = directory
        self.input_size = input_size
        self.random_transform = True
        self.view_pos = view_pos
        self.max_views = max_views
        self.sources = sources
        self.targets = targets
        self.max_len = max_len
        self.vocab_file = vocab_file
        self.vocab = spm.SentencePieceProcessor(model_file=directory + vocab_file)
        self.__input_data(binary_mode=True)

        if random_transform:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(0.1, 0.1, 0.1),
                            transforms.RandomRotation(15, expand=True),
                        ]
                    ),
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(input_size), transforms.ToTensor()]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        sources, targets = [], []
        temp_rep = self.captions[self.file_report[file_name]["image"][0] + ".png"]

        if "image" in self.sources:
            imgs, vpos = [], []
            images = self.file_report[file_name]["image"]

            new_orders = np.random.permutation(len(images))
            img_files = np.array(images)[new_orders].tolist()

            for i in range(min(self.max_views, len(img_files))):
                img = Image.open(self.dir + "images/" + img_files[i] + ".png").convert(
                    "RGB"
                )
                imgs.append(self.transform(img).unsqueeze(0))  # [1,C,W,H]
                vpos.append(1)

            cur_len = len(vpos)
            for i in range(cur_len, self.max_views):
                imgs.append(torch.zeros_like(imgs[0]))
                vpos.append(-1)

            imgs = torch.cat(imgs, dim=0)
            vpos = np.array(vpos, dtype=np.int64)

        info = self.file_report[file_name]["report"]

        source_info = []
        for section, content in info.items():
            if section in self.source_sections:
                source_info.append(content)
        source_info = " ".join(
            source_info
        )  # 将 source_info 列表中的所有元素用 " " 连接

        encoded_source_info = (
            [self.vocab.bos_id()]
            + self.vocab.encode(source_info)
            + [self.vocab.eos_id()]
        )  # encode source_info
        source_info = (
            np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        )  # 全padding 长度为maxlen
        source_info[: min(len(encoded_source_info), self.max_len)] = (
            encoded_source_info[: min(len(encoded_source_info), self.max_len)]
        )  # 有值的部分赋值 其余部分为padding

        target_info = []
        for section, content in info.items():
            if section in self.target_sections:
                target_info.append(content)
        # target_info = " ".join(target_info)

        target_info = temp_rep

        np_labels = np.zeros(len(self.top_np), dtype=float)
        for i in range(len(self.top_np)):
            if self.top_np[i] in target_info:
                np_labels[i] = 1

        encoded_target_info = (
            [self.vocab.bos_id()]
            + self.vocab.encode(target_info)
            + [self.vocab.eos_id()]
        )
        target_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        target_info[: min(len(encoded_target_info), self.max_len)] = (
            encoded_target_info[: min(len(encoded_target_info), self.max_len)]
        )

        for i in range(len(self.sources)):
            if self.sources[i] == "image":
                sources.append((imgs, vpos))
            elif self.sources[i] == "history":
                sources.append(source_info)
            elif self.sources[i] == "label":
                sources.append(
                    np.concatenate([np.array(self.file_labels[file_name]), np_labels])
                )
            elif self.sources[i] == "caption":
                sources.append(target_info)
            elif self.sources[i] == "caption_length":
                sources.append(min(len(encoded_target_info), self.max_len))

        for i in range(len(self.targets)):
            if self.targets[i] == "label":
                targets.append(
                    np.concatenate([np.array(self.file_labels[file_name]), np_labels])
                )
            elif self.targets[i] == "caption":
                targets.append(target_info)
            elif self.targets[i] == "caption_length":
                targets.append(min(len(encoded_target_info), self.max_len))

        return sources if len(sources) > 1 else sources[0], (
            targets if len(targets) > 1 else targets[0]
        )

    def __get_nounphrase(self, top_k=100, file_name="count_nounphrase.json"):
        with open(self.dir + file_name, "r") as f:
            count_nounphrase = json.load(f)
        count_nounphrase = dict(
            sorted(count_nounphrase.items(), key=lambda x: x[1], reverse=True)
        )
        top_np = list(count_nounphrase.keys())[:top_k]
        return top_np

    def __input_data(self, binary_mode=True):
        self.__input_caption()
        self.__input_report()
        self.__input_label()
        self.top_np = self.__get_nounphrase()

    def __input_caption(self):
        with open(self.dir + "testcaption.json", "r") as f:
            captions = json.load(f)
        self.captions = captions

    def __input_report(self):
        with open(self.dir + "testone.json", "r") as f:
            reports = json.load(f)
        self.file_list = list(reports.keys())
        self.file_report = reports

    def __input_label(self):
        with open(self.dir + "file2label.json", "r") as f:
            labels = json.load(f)
        self.file_labels = labels
