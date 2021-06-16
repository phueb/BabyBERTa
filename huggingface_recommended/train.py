"""
Train a Roberta model using code provided by library maintainers
"""

import logging

from datasets import Dataset, DatasetDict

from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling, Trainer, set_seed, TrainingArguments

from babyberta.io import load_sentences_from_file
from babyberta.utils import make_sequences
from babyberta import configs
from babyberta.params import param2default, Params


def main():

    params = Params.from_param2val(param2default)

    # get new rep
    rep = 0
    path_out = configs.Dirs.root / 'official_implementation' / str(rep)
    while path_out.exists():
        rep += 1
        path_out = configs.Dirs.root / 'official_implementation' / str(rep)

    print(f'replication={rep}')

    training_args = TrainingArguments(
        output_dir=str(path_out),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        do_predict=False,
        per_device_train_batch_size=params.batch_size,
        learning_rate=params.lr,
        max_steps=160_000,
        warmup_steps=params.num_warmup_steps,
        seed=rep,
        save_steps=40_000,
    )

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    set_seed(rep)

    logger.info("Loading data")
    data_path = configs.Dirs.corpora / 'aonewsela.txt'  # we use aonewsela for reference implementation
    sentences = load_sentences_from_file(data_path,
                                         include_punctuation=params.include_punctuation,
                                         allow_discard=True)
    data_in_dict = {'text': make_sequences(sentences, params.num_sentences_per_input)}
    datasets = DatasetDict({'train': Dataset.from_dict(data_in_dict)})
    print(datasets['train'])
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    logger.info("Loading tokenizer")
    tokenizer = RobertaTokenizerFast(vocab_file=None,
                                     merges_file=None,
                                     tokenizer_file=str(configs.Dirs.tokenizers / params.tokenizer / 'tokenizer.json'),
                                     )
    logger.info("Initialising Roberta from scratch")
    config = RobertaConfig(vocab_size=tokenizer.vocab_size,
                           hidden_size=params.hidden_size,
                           num_hidden_layers=params.num_layers,
                           num_attention_heads=params.num_attention_heads,
                           intermediate_size=params.intermediate_size,
                           initializer_range=params.initializer_range,
                           )
    model = RobertaForMaskedLM(config)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    text_column_name = "text"

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=params.max_input_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=[text_column_name],
        load_from_cache_file=True,
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
    trainer.train()
    trainer.save_model()  # Saves the tokenizer too


if __name__ == "__main__":
    main()
