from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from transformers import DataCollatorForTokenClassification, AutoTokenizer


class CoNLL2002DataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
            self,
            language: str = 'es',
            text_column_name: str = 'tokens',
            label_column_name: str = 'ner_tags',
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            tokenizer: str = 'distilroberta-base',
            max_seq_length: int = 128,
            padding: bool = True,
            num_classes: int = -1
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.collate_fn = None

        self.data_train_dataset: Optional[Dataset] = None
        self.data_val_dataset: Optional[Dataset] = None
        self.data_test_dataset: Optional[Dataset] = None

        self.tokenizer = None
        self.number_of_classes = None
        self.num_classes_passed = num_classes
        self.label_to_idx = None

    def num_classes(self):
        return self.number_of_classes

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        load_dataset('conll2002', self.hparams.language)

    def process_dataset(self, dataset):
        return dataset.map(self.tokenize_and_align_labels,
                           batched=True, num_proc=1).remove_columns(['id', 'tokens', 'pos_tags', 'ner_tags'])

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = load_dataset('conll2002', self.hparams.language, split='train')
            self.data_val = load_dataset('conll2002', self.hparams.language, split='validation')
            self.data_test = load_dataset('conll2002', self.hparams.language, split='test')

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, add_prefix_space=True)
        self.collate_fn = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )

        self.number_of_classes = self.data_train.features[self.hparams.label_column_name].feature.num_classes

        print('########### passed', self.num_classes_passed)
        print('########### dataset classes', self.number_of_classes)
        assert self.num_classes_passed == self.number_of_classes, 'Number of passed classes should be the same as in ' \
                                                                  'dataset'

        self.data_train_dataset = self.process_dataset(self.data_train)
        self.data_val_dataset = self.process_dataset(self.data_val)
        self.data_test_dataset = self.process_dataset(self.data_test)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.hparams.collate_fn,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples[self.hparams.text_column_name], truncation=True, is_split_into_words=True,
            # padding=self.hparams.padding, max_length=self.hparams.max_seq_length,
        )
        all_labels = examples[self.hparams.label_column_name]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)

            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    # def tokenize_and_align_labels(self, examples):
    #     tokenized_inputs = self.tokenizer(
    #         examples[self.hparams.text_column_name],
    #         padding=self.hparams.padding,
    #         truncation=True,
    #         max_length=self.hparams.max_seq_length,
    #         # We use this argument because the texts in our dataset are lists of words (with a label for each word).
    #         is_split_into_words=True,
    #     )
    #     labels = []
    #     for i, label in enumerate(examples[self.hparams.label_column_name]):
    #         word_ids = tokenized_inputs.word_ids(batch_index=i)
    #         previous_word_idx = None
    #         label_ids = []
    #         for word_idx in word_ids:
    #             # Special tokens have a word id that is None. We set the label to -100 so they are automatically
    #             # ignored in the loss function.
    #             if word_idx is None:
    #                 label_ids.append(-100)
    #             # We set the label for the first token of each word.
    #             elif word_idx != previous_word_idx:
    #                 label_ids.append(label[word_idx])
    #             # For the other tokens in a word, we set the label to either the current label or -100, depending on
    #             # the label_all_tokens flag.
    #             else:
    #                 label_ids.append(-100)
    #             previous_word_idx = word_idx
    #
    #         labels.append(label_ids)
    #     tokenized_inputs["labels"] = labels
    #     return tokenized_inputs


if __name__ == "__main__":
    datamodule = CoNLL2002DataModule()
    datamodule.prepare_data()
    datamodule.setup()
    print(datamodule.num_classes())
    # print(f'Train size: {len(datamodule.train_dataloader())}')
    # print(f'Validation size: {len(datamodule.val_dataloader())}')
    # print(f'Test size: {len(datamodule.test_dataloader())}')
    # batch: DataCollatorForTokenClassification = next(iter(datamodule.train_dataloader()))
    # print(type(batch))
    # print(batch)
