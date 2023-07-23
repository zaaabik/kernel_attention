import torch.nn
from torch import nn
import gpytorch

from transformers import AutoModelForTokenClassification

from src.models.components.layers.kernel_attention import KernelAttention


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


class PreTrainedLLMKernelAttention(nn.Module):
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

        self.kernel_attention = KernelAttention(
            768, 1,
            gpytorch.kernels.RBFKernel(), num_classes,
        )

        self.model.classifier = self.kernel_attention

    def forward(self, x):
        return self.model(**x)


if __name__ == "__main__":
    _ = PreTrainedLLM('xlm-roberta-base')
