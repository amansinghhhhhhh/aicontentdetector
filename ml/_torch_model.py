"""Torch-only pieces of the detector. Imported lazily by ml.model so the ONNX
backend can run without torch installed."""

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)


class AIDetectionModel(PreTrainedModel):
    """Custom head used by wasitaigenerated/tropa-mini (see model card)."""

    config_class = AutoConfig
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        n = int(
            getattr(config, "detector_num_labels", None)
            or getattr(config, "num_labels", None)
            or 2
        )
        self.classifier = nn.Linear(config.hidden_size, n)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out[0]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


def load_torch_model(source: str, config):
    """Returns (model, is_tropa) with weights loaded from `source`."""
    if getattr(config, "detector_num_labels", None):
        model = AIDetectionModel.from_pretrained(source, config=config)
        return model, True
    model = AutoModelForSequenceClassification.from_pretrained(source, config=config)
    return model, False
