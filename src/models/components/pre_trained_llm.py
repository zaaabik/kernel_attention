import torch.nn
from torch import nn

from transformers import AutoModelForTokenClassification

from src.models.components.layers.kernel_attention import KernelAttention, LinearAttention, LinearTransformerLayer, \
    EncoderBlock


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
            kernel_attention_num_heads: int = 1,
            kernel_function=None,
            kernel_regularization: float = 0.01,
            inverse_function=None
    ):
        super().__init__()

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_classes,
            finetuning_task='ner'
        )

        self.model.roberta.requires_grad_(False)

        self.kernel_attention = KernelAttention(
            768,
            n_heads=kernel_attention_num_heads,
            kernel_class=kernel_function,
            num_classes=num_classes,
            lmbda=kernel_regularization,
            inverse_function=inverse_function
        )

        self.model.classifier = self.kernel_attention

    def forward(self, x):
        return self.model(**x)


class PreTrainedLLMAttentionLayerCLS(nn.Module):
    def __init__(
            self,
            model_name: str,
            num_classes: int = 10,
            kernel_attention_num_heads: int = 1,
            ff=False,
            attention=True
    ):
        super().__init__()

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_classes,
            finetuning_task='ner'
        )

        self.model.roberta.requires_grad_(False)

        self.kernel_attention = torch.nn.Sequential(
            EncoderBlock(input_dim=768, dim_feedforward=768 * 4,
                         num_heads=kernel_attention_num_heads, ff=ff,
                         attention=attention),
            torch.nn.Linear(768, num_classes)
        )

        self.model.classifier = self.kernel_attention

    def forward(self, x):
        return self.model(**x)


if __name__ == "__main__":
    _ = PreTrainedLLM('xlm-roberta-base')
