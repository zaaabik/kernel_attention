import torch.nn
from torch import nn

from transformers import AutoModelForTokenClassification

from src.data.conll2002_datamodule import CoNLL2002DataModule

class PreTrainedLLM(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
    ):
        super().__init__()

        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.num_labels = num_classes
        self.model.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, x):
        return self.model(**x)


if __name__ == "__main__":
    _ = PreTrainedLLM('xlm-roberta-base')

    # datamodule = CoNLL2002DataModule(batch_size=1)
    #
    # datamodule.prepare_data()
    # datamodule.setup()
    # print(f'Train size: {len(datamodule.train_dataloader())}')
    # print(f'Validation size: {len(datamodule.val_dataloader())}')
    # print(f'Test size: {len(datamodule.test_dataloader())}')
    # batch = next(iter(datamodule.train_dataloader()))
    # for k,v in batch.items():
    #     print(k, v.shape)
    # print(type(batch))
    # print(batch)
    #
    # model = PreTrainedLLM('xlm-roberta-base', datamodule.num_classes())
    # out = model(batch)
    # print(out.loss)
    # print(out.logits.argmax(axis=-1))
