from argparse import ArgumentParser
import datasets
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold, train_test_split


class DataModule(pl.LightningDataModule):

    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "imdb": ["text"],
        "trec": ["text"],
        "twitter": ["text"],
        # "health_fact": ["main_text"],
    }

    task_label_field_map = {
        "cola": ["label"],
        "sst2": ["label"],
        "imdb": ["label"],
        "trec": ["label-coarse"],
        "twitter": ["label"],
        # "health_fact": ["label"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "imdb": 2,
        "trec": 6,
        "twitter": 3,
        # "health_fact": 5,
    }

    metrics_names = {
        "cola": "f1",
        "sst2": "f1",
        "imdb": "f1",
        "trec": "accuracy",
        "twitter": "accuracy",
        # "health_fact": "accuracy",
    }

    dataset_names = {
        "cola": ["glue", "cola"],
        "sst2": ["glue", "sst2"],
        "imdb": ["imdb"],
        "trec": ["trec"],
        "twitter": ["tweet_eval", "sentiment"],
        # "health_fact": ["health_fact"],
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        cache_dataset: bool = False,
        cached_dataset_filepath: str = "",
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.cached_train = None
        self.cached_vals = None
        self.cache_dataset = cache_dataset
        if self.cache_dataset:
            if not cached_dataset_filepath:
                cached_dataset_filepath = f"{self.task_name}.cached.dataset.pt"
            self.load_cache_dataset(cached_dataset_filepath)

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True
        )
        # self.max_seq_length = self.tokenizer.model_max_length

    def setup(self, stage):
        if not self.cache_dataset:
            self.dataset = datasets.load_dataset(*self.dataset_names[self.task_name])

            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    remove_columns=[self.task_label_field_map[self.task_name][0]],
                )
                self.columns = [
                    c
                    for c in self.dataset[split].column_names
                    if c in self.loader_columns
                ]
                self.dataset[split].set_format(type="torch", columns=self.columns)
        else:
            if self.task_name in ["cola", "sst2"]:
                self.dataset["test"] = self.dataset["validation"]
            split_dict = self.dataset["train"].train_test_split(test_size=0.1, seed=42)
            self.dataset["train"] = split_dict["train"]
            self.dataset["validation"] = split_dict["test"]

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        if not self.cache_dataset:
            datasets.load_dataset(*self.dataset_names[self.task_name])
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        AutoModel.from_pretrained(self.model_name_or_path)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if len(self.eval_splits) <= 1:
            return DataLoader(
                self.dataset["validation"],
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def train_dataset(self):
        return self.dataset["train"]

    @property
    def val_dataset(self):
        if len(self.eval_splits) == 1:
            return self.dataset["validation"]
        elif len(self.eval_splits) > 1:
            return [self.dataset[x] for x in self.eval_splits]

    @property
    def metric(self):
        return datasets.load_metric(self.metrics_names[self.task_name])

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(
                    example_batch[self.text_fields[0]],
                    example_batch[self.text_fields[1]],
                )
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch[self.task_label_field_map[self.task_name][0]]

        return features

    def kfold(self, k_folds=10, seed=420):
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.train_dataset)):
            train_ids = train_ids.tolist()
            val_ids = val_ids.tolist()

            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)
            yield fold, DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                sampler=train_subsampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ), DataLoader(
                self.train_dataset,
                batch_size=self.eval_batch_size,
                sampler=val_subsampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            ),

    def load_cache_dataset(self, cached_dataset_filepath):
        print(f"Load cached dataset {cached_dataset_filepath}")
        self.dataset = torch.load(cached_dataset_filepath)

    @staticmethod
    def add_cache_arguments(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--cache-dataset",
            action="store_true",
            help="If use cached dataset",
        )
        parser.add_argument(
            "--cache-dataset-filepath", type=str, default="", help="Cached dataset path"
        )
        parser.add_argument("--k-folds", type=int, default=10)

        return parser
