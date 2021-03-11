#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm



# Downloaded by Philip Huebner March 11, 2021, and simplified for single-GPU pre-training of Roberta.
"""


import logging

from datasets import Dataset, DatasetDict

from transformers.models.roberta import RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling, Trainer, set_seed, TrainingArguments

from babybert.io import load_sentences_from_file
from babybert.utils import make_sequences
from babybert import configs
from babybert.params import param2default, Params


LINE_BY_LINE = True  # TODO interesting to evaluate model performance when this is False
num_sentences_per_input = 1
SEED = 0
MAX_SEQ_LENGTH = 256
preprocessing_num_workers = 4


params = Params.from_param2val(param2default)

training_args = TrainingArguments(
    output_dir='output',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    do_predict=False,
    per_device_train_batch_size=params.batch_size,
    learning_rate=params.lr,
    evaluation_strategy='no',
    max_steps=160_000,
    warmup_steps=10_000,
    seed=SEED,
)

logger = logging.getLogger(__name__)


def main():

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    logger.info("Loading data")

    # Set seed before initializing model.
    set_seed(SEED)

    # load data - inserted by PH
    data_path = configs.Dirs.corpora / f'{params.corpus_name}.txt'
    sentences = load_sentences_from_file(data_path,
                                         training_order=params.training_order,
                                         include_punctuation=params.include_punctuation,
                                         allow_discard=True)
    data_in_dict = {'text': make_sequences(sentences, num_sentences_per_input)}
    datasets = DatasetDict({'train': Dataset.from_dict(data_in_dict)})
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    config = RobertaConfig.from_pretrained('config.json')
    tokenizer = RobertaTokenizerFast.from_pretrained(configs.Dirs.tokenizers / params.bbpe)

    logger.info("Initialising Roberta from scratch")
    model = RobertaForMaskedLM(config)
    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = datasets["train"].column_names
    text_column_name = "text"

    if MAX_SEQ_LENGTH > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({MAX_SEQ_LENGTH}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(MAX_SEQ_LENGTH, tokenizer.model_max_length)

    if LINE_BY_LINE:
        # When using line_by_line, we just tokenize each nonempty line.

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=False,
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=False,
        )

    if "train" not in tokenized_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = tokenized_datasets["train"]

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)


if __name__ == "__main__":
    main()