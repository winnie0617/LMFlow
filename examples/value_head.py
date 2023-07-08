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

torch.set_default_dtype(torch.bfloat16)

# Define the class for single layer NN  
class one_layer_net(nn.Module):    
    # Constructor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(one_layer_net, self).__init__()
        # hidden layer 
        self.linear_one = nn.Linear(input_size, hidden_neurons)
        self.linear_two = nn.Linear(hidden_neurons, output_size) 

    # prediction function
    def forward(self, x):
        self.act = torch.tanh(self.linear_one(x))
        y_pred = self.linear_two(self.act)
        return y_pred

class ModelWithValueHead(nn.Module):
    def __init__(self, pretrained, config_file):

        super().__init__()
        
        self.transformer = pretrained
        config = PretrainedConfig.from_pretrained(config_file)
        config.num_labels = 1
        # self.head = SequenceSummary(config)
        self.head = one_layer_net(2560, 256, 1)
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
        cls_index = torch.full_like(
            hidden_states[..., :1, :],
            hidden_states.shape[-2] - 1,
            dtype=torch.long,
        )
        # cls at last index
        cls = hidden_states.gather(-2, cls_index).squeeze(-2)
        output = self.head(cls)
        return output

    def save_head(self, path):
        torch.save(self.head.state_dict(), path)