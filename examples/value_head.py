"""
https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_openai.html#OpenAIGPTDoubleHeadsModel
"""

import torch.nn as nn
import torch
from transformers import PretrainedConfig
from typing import Callable, Optional
from transformers.modeling_utils import SequenceSummary

from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm



class ModelWithValueHead(nn.Module):
    def __init__(self, pretrained, config_file):

        super().__init__()
        
        self.transformer = pretrained
        config = PretrainedConfig.from_pretrained(config_file)
        config.num_labels = 1
        self.head = SequenceSummary(config)
        self.config = pretrained.config
        # self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        # labels=None,
        # mc_labels=None,
        # output_attentions=None,
        # output_hidden_states=None,
        **kwargs
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs[0]
        output = self.head(hidden_states, mc_token_ids).squeeze(-1)
        return output

    def save_head(self, path):
        torch.save(self.head.state_dict(), path)