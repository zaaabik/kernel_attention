import torch.nn
from torch import nn

from transformers import AutoModelForTokenClassification, AutoConfig


class PreTrainedLLM(nn.Module):
    def __init__(
            self,
            model_name: str,
            num_classes: int = 10,
    ):
        super().__init__()

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_classes,
            finetuning_task='ner'
        )

    def forward(self, x):
        return self.model(**x)


if __name__ == "__main__":
    _ = PreTrainedLLM('xlm-roberta-base')
