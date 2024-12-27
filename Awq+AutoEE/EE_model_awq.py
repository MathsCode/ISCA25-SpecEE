import copy
import json
import time
import os
import torch
import torch.nn as nn
from transformers import AutoConfig
from model_llama_ee import LlamaForCausalLM as LlamaForCausalLMEE
from transformers import AutoTokenizer
from configs import EConfig
from cnets import Model
from awq import AutoAWQForCausalLM
class EEModel(nn.Module):
    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias)

        low_memory=False
        # breakpoint()
        device = base_model.model.model.blocks[-1].attn.qkv_proj.qweight.device
        if device!=base_model.model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.model.lm_head.qweight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
            
        self.ea_layer.to(self.base_model.model.dtype).to(device)
        
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer
    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            is_offload = False,
            skip_model = None,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        base_model = AutoAWQForCausalLM.from_quantized(
                base_model_path, **kwargs
            )
        # breakpoint()
        configpath=os.path.join(ea_model_path,"config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath,
        )
        load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path,
                                         map_location=base_model.model.device)
        model.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)

        return model
    
    def forward(
            self,
            input_ids=None,
            max_new_tokens=10, 
            exit_layer_id_list = None,
    ):
        
        self.ea_layer.reset_kv()
        with torch.inference_mode():
            input_len = input_ids.shape[1]
            outputs,token = self.base_model.model.model(input_ids=input_ids, is_causal = True,lm_head = self.base_model.model.lm_head,exit_layer_id_list=exit_layer_id_list)
            hidden_states = outputs[0].clone()
            past_key_values = outputs[1]
            token = token.to(input_ids.device)
            input_ids = torch.cat((input_ids, token), dim=1)
            topk_index, topk_prob, top_head_weight = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.model.lm_head)
            for _ in range(max_new_tokens - 1):
                outputs,token = self.base_model.model.model(input_ids=token,is_causal = True,init=False,draft_lm_head_weight = top_head_weight,draft_token_index = topk_index,lm_head = self.base_model.model.lm_head,exit_layer_id_list = exit_layer_id_list)
                hidden_states = outputs[0].clone()
                past_key_values = outputs[1]
                input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

                topk_index, topk_prob, top_head_weight = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.model.lm_head)
                if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                    return input_ids
            return input_ids
                
                
                
            
            
            