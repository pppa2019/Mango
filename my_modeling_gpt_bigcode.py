import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, NLLLoss
from torch.nn.functional import softmax

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.models.gpt_bigcode import GPTBigCodeForCausalLM, GPTBigCodeConfig, GPTBigCodeModel


def token_ids2seg_ids(token_ids, seg_id=None, code_seg_id=None):
    seg_flag = (token_ids==seg_id)
    ins_seg_ids = torch.cumsum(seg_flag, dim=-1)
    ins_pos = torch.sum(ins_seg_ids==0)
    output_pos = torch.sum(ins_seg_ids<=1)
    code_flag = (token_ids==code_seg_id)
    code_flag[:, :output_pos] = torch.zeros((code_flag.shape[0], output_pos), dtype=torch.bool)
    code_seg_ids = torch.cumsum(code_flag, dim=-1)
    begin_code_pos = torch.sum(code_seg_ids==0)
    end_code_pos = torch.sum(code_seg_ids<=1)
    return ins_pos, output_pos, begin_code_pos, end_code_pos


class My_GPTBigCodeForCausalLM(GPTBigCodeForCausalLM):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTBigCodeModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs
    

    def calc_reward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            pure_code_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        ins_pos, output_pos, begin_code_pos, end_code_pos = token_ids2seg_ids(input_ids, seg_id=self.config.ins_token_id, code_seg_id=self.config.code_token_id)
        if self.config.use_emb_output:
            output_hidden_states = True
        
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        loss_vector = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_vector = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = torch.mean(loss_vector)

       
        if pure_code_ids is not None:
            if "margin_contrastive" in self.config.train_target:
                ins_ids = input_ids[:, :output_pos]
                negative_ids = torch.cat([ins_ids,pure_code_ids],dim=-1)
                attention_mask = torch.ones(negative_ids.shape).to(self.device)
                neg_labels = negative_ids
                neg_outputs = self.transformer(
                    negative_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                neg_lm_logits = self.lm_head(neg_outputs[0])
                loss_fct = CrossEntropyLoss()
                neg_shift_logits = neg_lm_logits[..., output_pos:-1, :].contiguous()
                pos_shift_logits = shift_logits[..., output_pos:, :].contiguous()

                neg_loss = loss_fct(neg_shift_logits.view(-1, neg_shift_logits.size(-1)), neg_labels[..., output_pos+1:].contiguous().view(-1))
                pos_loss = loss_fct(pos_shift_logits.view(-1, pos_shift_logits.size(-1)), shift_labels[..., output_pos:].view(-1))

                margin = self.config.margin

                self.contrastive_loss = max(0, margin + pos_loss - neg_loss)

                
                if neg_shift_logits.shape[1]>10:
                    loss = loss + self.contrastive_loss
                del neg_shift_logits
                del pos_shift_logits
        return loss, transformer_outputs, lm_logits, loss_vector

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pure_code_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_seq_loss: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        sample_ids = None
        if pure_code_ids is not None:
            loss, transformer_outputs, lm_logits, loss_vector = self.calc_reward(
                input_ids,
                sample_ids,
                labels,
                pure_code_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        
        else:
            loss, transformer_outputs, lm_logits, loss_vector = self.calc_reward(
                input_ids,
                sample_ids,
                labels,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
        
            
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            if return_seq_loss:
                output = (output,loss_vector)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(layer_past.index_select(0, beam_idx.to(layer_past.device)) for layer_past in past_key_values)