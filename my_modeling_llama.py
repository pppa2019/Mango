from transformers.models.llama import LlamaForCausalLM, LlamaModel, LlamaConfig
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn, softmax
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings


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
    

class My_LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def calc_reward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            sample_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            comment_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pure_code_ids: Optional[torch.Tensor] = None,
    ):
        ins_pos, output_pos, begin_code_pos, end_code_pos = token_ids2seg_ids(input_ids, seg_id=self.config.ins_token_id, code_seg_id=self.config.code_token_id)
        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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
                neg_labels = negative_ids.contiguous().to(self.device)
                neg_outputs = self.model(
                    negative_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                neg_lm_logits = self.lm_head(neg_outputs[0])
                loss_fct = CrossEntropyLoss()
                neg_shift_logits = neg_lm_logits[..., output_pos:-1, :].contiguous()
                pos_shift_logits = shift_logits[..., output_pos:, :].contiguous()

                pos_loss = loss_fct(pos_shift_logits.view(-1, pos_shift_logits.size(-1)), shift_labels[..., output_pos:].view(-1))
                neg_loss = loss_fct(neg_shift_logits.view(-1, neg_shift_logits.size(-1)), neg_labels[..., output_pos+1:].view(-1))
                
                margin = self.config.margin
                self.contrastive_loss = max(0, margin + pos_loss - neg_loss)                
                if neg_shift_logits.shape[1]>10:
                    loss = loss + self.contrastive_loss
                del neg_shift_logits
                del pos_shift_logits
        else:
            print('No pure_code_ids')
            exit(-1)
        

        return loss, transformer_outputs, lm_logits, loss_vector

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pure_code_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sample_ids = None
        if pure_code_ids is not None:
            loss, outputs, logits, loss_vector = self.calc_reward(
                input_ids,
                sample_ids,
                labels,
                pure_code_ids=pure_code_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
        
            loss, outputs, logits, loss_vector = self.calc_reward(
                input_ids,
                sample_ids,
                labels,
                past_key_values=past_key_values,
                attention_mask=attention_mask,

                position_ids=position_ids,

                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
                        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past