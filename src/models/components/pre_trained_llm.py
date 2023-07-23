from torch import nn

from transformers import AutoModelForTokenClassification


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


class PreTrainedLLMOnlyLastLayer(nn.Module):
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

        self.model.roberta.requires_grad_(False)
        self.model.roberta.encoder.layer[-1].requires_grad_(True)
        self.model.classifier.requires_grad_(True)

    def forward(self, x):
        return self.model(**x)


if __name__ == "__main__":
    _ = PreTrainedLLM('xlm-roberta-base')
