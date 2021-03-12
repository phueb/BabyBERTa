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

from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling, Trainer, set_seed, TrainingArguments

from babybert.io import load_sentences_from_file
from babybert.utils import make_sequences
from babybert import configs
from babybert.params import param2default, Params
from babybert.utils import load_tokenizer


SEED = 1
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
    max_steps=160_000,
    warmup_steps=10_000,
    seed=SEED,
    save_steps=40_000,
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
    data_in_dict = {'text': make_sequences(sentences, params.num_sentences_per_input)}
    datasets = DatasetDict({'train': Dataset.from_dict(data_in_dict)})
    print(datasets['train'])
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # vocab_fn = 'vocab.json'
    # merges_fn = 'merges.txt'
    # tokenizer = RobertaTokenizerFast(vocab_file=str(configs.Dirs.tokenizers / params.bbpe / vocab_fn),
    #                                  merges_file=str(configs.Dirs.tokenizers / params.bbpe / merges_fn),
    #                                  add_prefix_space=params.add_prefix_space)

    tokenizer = load_tokenizer(params, configs.Dirs.root)

    print(tokenizer.encode('the dog on the <mask> .', add_special_tokens=False).tokens)
    raise SystemExit

    config = RobertaConfig(vocab_size=tokenizer.vocab_size,
                           hidden_size=params.hidden_size,
                           num_hidden_layers=params.num_layers,
                           num_attention_heads=params.num_attention_heads,
                           intermediate_size=params.intermediate_size,
                           initializer_range=params.initializer_range,
                           )

    logger.info("Initialising Roberta from scratch")
    model = RobertaForMaskedLM(config)
    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = datasets["train"].column_names
    text_column_name = "text"

    if params.max_num_tokens_in_sequence > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({params.max_num_tokens_in_sequence}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]

        # TODO testing
        return tokenizer.encode(
            examples["text"],
            add_special_tokens=True,
        )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=False,
    )

    train_dataset = tokenized_datasets["train"]
    print(f'Length of train data={len(train_dataset)}')

    # Data collator will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm_probability=params.mask_probability)

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


if __name__ == "__main__":
    main()