import torch
import torch.nn as nn
from typing import List
from awq.utils import fused_utils
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
)
from awq.modules.fused.block import (
    MPTBlock,
    FalconDecoderLayer,
    LlamaLikeBlock,
    MixtralBlock,
    Phi3Block,
    CohereBlock,
)


class MixtralModel(nn.Module):
    def __init__(self, vocab_size, blocks, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[MixtralBlock] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        mask = fused_utils.prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h,
        )

        for layer in self.blocks:
            h, mask = fused_utils.prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, past_key_value = layer(
                h, None, attention_mask=mask, is_causal=is_causal
            )

        h = self.norm(h)

        return MoeModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=past_key_value,
            hidden_states=(),
            attentions=(),
            router_logits=(),
        )

import torch.nn.functional as F
class LlamaLikeModel(nn.Module):
    """
    LlamaLikeModel is intended to be reused across models that have
    an architecture that closely resembles Llama, e.g. Mistral and Aquila.
    """

    def __init__(self, vocab_size, blocks, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[LlamaLikeBlock] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0
        self.predictors = [torch.load('/home/xujiaming/xujiaming/train_machine/PJY/ASPLOS-24-1/classifiers-7B-awq-4/model'+str(layer_idx)+'.pth').to(torch.float16) for layer_idx in range(40)]
        self.last_token = None
        # self.lm_head = None
        # print(11111111111111)
        # self.exit_layer_id_list = []
    @property
    def embed_tokens(self):
        return self.embedding

    @property
    def layers(self):
        return self.blocks

    # @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
        
        init = True,
        draft_lm_head_weight = None,
        draft_token_index = None,
        lm_head = None,
        exit_layer_id_list = None,
        *args,
        **kwargs,
    ):
        # self.lm_head = lm_head
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        mask = fused_utils.prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h,
        )
        # last_prob = None
        overall_layers = set([15,16,17,19,21,24,26,28])
        dynamic_layers = set()
        if len(exit_layer_id_list) >= 5:
            for x in exit_layer_id_list[-5:]:
                for y in [-2,0,2]:
                    dynamic_layers.add(x + y - 1)
        layer_selected = overall_layers | dynamic_layers    
        for idx in range(len(self.blocks)):
            layer = self.blocks[idx]
            h, mask = fused_utils.prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, _ = layer(h, None, attention_mask=mask, is_causal=is_causal)
            if not init:
                if idx not in layer_selected:
                    continue
                tmp_h = self.norm(h)
                draft_logits = F.linear(tmp_h,draft_lm_head_weight)
                draft_prob = F.softmax(draft_logits)
                # if last_prob is None:
                #     prob_gap = draft_prob
                # else:
                #     prob_gap = draft_prob - last_prob
                # last_prob = draft_prob
                feature = torch.cat([tmp_h,draft_logits,draft_prob],dim=-1).squeeze(0)         
                pred = self.predictors[idx](feature)  
                if pred > 0.95:
                   logits = lm_head(tmp_h)
                   token =  torch.argmax(logits[:, -1])
                   token = token[None,None]
                #    if token != self.last_token:
                #    if token in draft_token_index:
                   exit_layer_id_list.append(idx+1)
                #    self.last_token = token
                   return BaseModelOutputWithPast(
                            last_hidden_state=tmp_h,
                            past_key_values=None,
                            hidden_states=(),
                            attentions=(),
                        ),token

        h = self.norm(h)
        token = torch.argmax(lm_head(h[:,-1]))
        token = token[None,None]
        exit_layer_id_list.append(32)
        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        ),token


class CohereModel(nn.Module):
    def __init__(self, vocab_size, blocks, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[CohereBlock] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0

    @property
    def embed_tokens(self):
        return self.embedding

    @property
    def layers(self):
        return self.blocks

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        mask = fused_utils.prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h,
        )

        for layer in self.blocks:
            h, mask = fused_utils.prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, _ = layer(h, None, attention_mask=mask, is_causal=is_causal)
        h = self.norm(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        )


class MPTModel(nn.Module):
    def __init__(self, vocab_size, blocks, wte, norm_f):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = wte
        self.blocks: List[MPTBlock] = nn.ModuleList(blocks)
        self.norm_f = norm_f
        self.attn_uses_sequence_id = False
        self.prefix_lm = False
        self.last_forward_num_tokens = 0

    @torch.inference_mode()
    def forward(
        self,
        input_ids,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.wte(input_ids)

        mask = fused_utils.prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h,
        )

        for layer in self.blocks:
            h, mask = fused_utils.prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, past_key_value = layer(
                h, None, attention_mask=mask, is_causal=is_causal
            )
        h = self.norm_f(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=past_key_value,
            hidden_states=(),
            attentions=(),
        )


class FalconModel(nn.Module):
    def __init__(self, vocab_size, blocks, word_embeddings, ln_f):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embeddings = word_embeddings
        self.blocks: List[FalconDecoderLayer] = nn.ModuleList(blocks)
        self.ln_f = ln_f
        self.attn_uses_sequence_id = False
        self.prefix_lm = False
        self.last_forward_num_tokens = 0

    @torch.inference_mode()
    def forward(
        self,
        input_ids,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.word_embeddings(input_ids)

        mask = fused_utils.prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h,
        )

        for layer in self.blocks:
            h, mask = fused_utils.prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, past_key_value = layer(
                h, None, attention_mask=mask, is_causal=is_causal
            )
        h = self.ln_f(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=past_key_value,
            hidden_states=(),
            attentions=(),
        )

class Phi3Model(nn.Module):
    """
    Phi3LikeModel is intended to be reused across models that have
    an architecture that closely resembles Phi-3.
    """

    def __init__(self, vocab_size, blocks, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[Phi3Block] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0

    @property
    def embed_tokens(self):
        return self.embedding

    @property
    def layers(self):
        return self.blocks

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        mask = fused_utils.prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h,
        )

        for layer in self.blocks:
            h, mask = fused_utils.prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, _ = layer(
                h, None, attention_mask=mask, is_causal=is_causal
            )
        h = self.norm(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        )
