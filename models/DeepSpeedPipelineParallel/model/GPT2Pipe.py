import deepspeed
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.distributed as dist

# OSLO는.. sequential한 function 때문에 받은 모든 input을 주는데다가,
# output_shape처럼 - 처음에 계산해야 나중에 postblock에서 써먹을 수 있는 값들도 다 보낸다.
# 그런데 정작 원래 GPT2에서는 깔끔하게 output 몇 종류만 보낸다. 예를 들어, 굳이 input_ids를 쭉쭉 output까지 보내지 않는다.
# 너무 복잡하니까 이번에는 정말 GPT2 또는 GPT2LM 또는 GPT2 Sequence Classification에서 내보내는 output만 내보내 보겠다.
# 그리고 GPT2에서는 output을 tuple로 보낼 때도 있다. tuple이 그렇게 근본 없는 data carry 방법은 아닌 것 같다.

GPT2_FLOW_KEYS = (  # 처음에 받은 값이거나 중간에 생겨나는/고쳐지는 값이지만 끝까지 전달이 되어야 함
    'input_ids',
    'inputs_embeds_shape',
    'output_shape',  # torch.Size이려나? list이려나?
    'output_hidden_states',  # bool
    'return_dict',  # bool
    'all_self_attentions',  # 이것도 presents랑 똑같아
    'all_cross_attentions',  # 이것도 presents랑 똑같아
    'all_hidden_states',  # 이것도 presents랑 똑같아
    'attention_mask',  # Optional[torch.FloatTensor] = None,
    'presents',
)


GPT2_BLOCK_KEYS = (
    'hidden_states',  # Optional[Tuple[torch.FloatTensor]],
    'past_key_values',
    'head_mask',  # Optional[torch.FloatTensor] = None,
    'encoder_hidden_states',  # Optional[torch.Tensor] = None,
    'encoder_attention_mask',  # Optional[torch.FloatTensor] = None,
    'use_cache',  # Optional[bool] = False,
    'output_attentions',  # Optional[bool] = False,
)

GPT2_OUTPUT_KEYS = (
    'last_hidden_state',  # hidden_states, # torch.FloatTensor = None
    'hidden_states',
    'attentions',  # Optional[Tuple[torch.FloatTensor]] = None
    'cross_attentions',  # Optional[Tuple[torch.FloatTensor]] = None
    'past_key_values',
)

GPT2FORSEQCLS_OUTPUT_KEYS = (
    'loss',  # Optional[torch.FloatTensor] = None
    'logits',  # torch.FloatTensor = None
    'hidden_states',  # Optional[Tuple[torch.FloatTensor]] = None
    'attentions',  # Optional[Tuple[torch.FloatTensor]] = None
    # Optional[Tuple[Tuple[torch.FloatTensor]]] = None # 문제! # XXX tensor 한 개로 stack 했다
    'past_key_values',
)


def make_outputs(**outputs):
    out = dict()
    out.update(outputs)
    return out


def make_tuple_outputs(args, keys, outputs):
    """
    add fixed sequence of values to a tuple
    args: Tuple
    key: Tuple[str]
    outputs: Tuple
    """
    out = list(args)
    for key in keys:
        temp = outputs.get(key, torch.tensor(
            [], device='cuda', dtype=torch.int8))
        # None
        if temp is None:
            temp = torch.tensor([], device='cuda', dtype=torch.int8)
        # bool type
        elif isinstance(temp, bool):
            temp = torch.tensor(temp, device='cuda')
        # [None, None, None, ....], Maximum: 1D So just...manually
        elif isinstance(temp, list) or isinstance(temp, tuple):
            if len(temp) > 0 and (isinstance(temp[0], list) or isinstance(temp[0], tuple)):
                temp = torch.stack(
                    [torch.tensor([], device='cuda') if item is None else torch.stack(item, dim=0) for item in temp], dim=0)
                if temp.size(-1) == 0:
                    temp = temp.to(dtype=torch.int8)
            else:
                temp = torch.tensor(
                    [[] if item is None else item for item in temp], device='cuda')
                if temp.size(-1) == 0:
                    temp = temp.to(dtype=torch.int8)
            temp = temp.clone().detach()
        out.append(temp)
    return tuple(out)


def convert_from_tuple(inputs):
    """
    This convert [] -> None, Bool Tensor -> Bool, [[],[],[],...] -> [None, None, ....]
    """
    inputs = list(inputs)
    for i, input in enumerate(inputs):
        #  Manage boolean
        if isinstance(input, tuple):
            continue
        if input.dtype == torch.bool and len(input.size()) == 0:
            inputs[i] = bool(input)
        # None
        elif input.size(-1) == 0 and len(input.size()) == 1:
            inputs[i] = None
        # [None, None, None, None, .....]
        elif input.size(-1) == 0 and len(input.size()) != 1:
            inputs[i] = [None for j in range(input.size(0))]
    return inputs


class GPT2ModelPipe(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

    class PreblockClass(nn.Module):
        def __init__(self, config, wte, wpe, drop, dtype):
            super().__init__()
            self.config = config
            self.wte = wte
            self.wpe = wpe
            self.drop = drop
            self.dtype = dtype

        def get_head_mask(
            self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
        ) -> Tensor:
            """
            Prepare the head mask if needed.
            Args:
                head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                    The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
                num_hidden_layers (`int`):
                    The number of hidden layers in the model.
                is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                    Whether or not the attentions scores are computed by chunks or not.
            Returns:
                `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
                `[None]` for each layer.
            """
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(
                    head_mask, num_hidden_layers)
                if is_attention_chunked is True:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers

            return head_mask

        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            assert head_mask.dim(
            ) == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
            # switch to float if need + fp16 compatibility
            head_mask = head_mask.to(dtype=self.dtype)
            return head_mask

        def forward(
            self,
            inputs
        ):
            inputs = list(inputs)
            inputs = convert_from_tuple(inputs)
            # input 길이 맞춰주기. default를 None으로 상정.
            if len(inputs) < 13:
                inputs = inputs + [None] * (13 - len(inputs))

            # add non-existing keys
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions, output_hidden_states, return_dict, past_key_values = inputs
            # print("DEBUG-DEEPSPEED: input_ids: ", input_ids, input_ids.shape)
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            # TOGGLE POINT
            use_cache = False  # DEBUG: if use_cache = True, DeepSpeed source code must be modified
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time"
                )
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                batch_size = input_ids.shape[0]
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                batch_size = inputs_embeds.shape[0]
            else:
                raise ValueError(
                    "You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, input_shape[-1])
            if position_ids is not None:
                position_ids = position_ids.view(-1, input_shape[-1])
            if past_key_values is None:
                past_length = 0
                past_key_values = tuple([None] * self.config.n_layer)
            else:
                past_length = past_key_values[0][0].size(-2)
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    input_shape[-1] + past_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(
                    0).view(-1, input_shape[-1])

            # GPT2Attention mask.
            if attention_mask is not None:
                if batch_size <= 0:
                    raise ValueError("batch_size has to be defined and > 0")
                attention_mask = attention_mask.view(batch_size, -1)
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and -10000.0 for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(
                    dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * -10000.0

            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.add_cross_attention and encoder_hidden_states is not None:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
                encoder_hidden_shape = (
                    encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(
                        encoder_hidden_shape, device=device)
                encoder_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask)
            else:
                encoder_attention_mask = None

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # head_mask has shape n_layer x batch x n_heads x N x N
            head_mask = self.get_head_mask(head_mask, self.config.n_layer)

            if inputs_embeds is None:
                inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

            if token_type_ids is not None:
                token_type_embeds = self.wte(token_type_ids)
                hidden_states = hidden_states + token_type_embeds
            # print("DEBUG-DEEPSPEED: preblock hidden_states: ", hidden_states, hidden_states.shape)
            # print("DEBUG-DEEPSPEED: self.training", self.training)
            hidden_states = self.drop(hidden_states)
            output_shape = input_shape + (hidden_states.size(-1),)

            presents = () if use_cache else None
            all_self_attentions = () if output_attentions else None
            all_cross_attentions = (
                () if output_attentions and self.config.add_cross_attention else None
            )
            all_hidden_states = () if output_hidden_states else None

            inputs_embeds_shape = torch.tensor(
                inputs_embeds.shape, device='cuda')
            attention_mask = torch.where(attention_mask > -5000, True, False)
            # print("DEBUG-DEEPSPEED: preblock hidden_states 2: ", hidden_states, hidden_states.shape)
            outputs = make_outputs(
                hidden_states=hidden_states,
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds_shape=inputs_embeds_shape,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                output_shape=output_shape,
                presents=presents,
                all_self_attentions=all_self_attentions,
                all_cross_attentions=all_cross_attentions,
                all_hidden_states=all_hidden_states,
            )
            out = tuple()
            out = make_tuple_outputs(out, GPT2_BLOCK_KEYS, outputs)
            out = make_tuple_outputs(out, GPT2_FLOW_KEYS, outputs)
            return out

    class BlockClass(nn.Module):
        def __init__(self, config, layer, layer_id, gradient_checkpointing):
            super().__init__()
            self.config = config
            self.layer = layer  # 이렇게 해도 메모리를 아낄 수가 있나? 확인을 해봐야지!
            self.layer_id = layer_id
            self.gradient_checkpointing = gradient_checkpointing

        def forward(
            self,
            inputs,
        ):
            inputs = convert_from_tuple(inputs)

            hidden_states, past_key_values, head_mask, encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions, input_ids, inputs_embeds_shape, output_shape, output_hidden_states, return_dict, all_self_attentions, all_cross_attentions, all_hidden_states, attention_mask,  presents = inputs
            # 빈 튜플이 맞는 것만 따로 처리해주자
            attention_mask = torch.where(attention_mask, -0.0, -10000.0)

            presents = presents if presents is not None else ()
            all_self_attentions = all_self_attentions if all_self_attentions is not None else ()
            all_cross_attentions = all_cross_attentions if all_cross_attentions is not None else ()
            all_hidden_states = all_hidden_states if all_hidden_states is not None else ()
            # block 설정
            block = self.layer
            # should be past_key_values[layer_id], but in this case I didn't
            layer_past = past_key_values[self.layer_id]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    print(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[self.layer_id],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[self.layer_id],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]

            if use_cache is True:
                presents = tuple([[pre for pre in present]
                                 for present in presents]) if len(presents) != 0 else ()
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = tuple([[attn for attn in attention]
                                             for attention in all_self_attentions]) if len(all_self_attentions) != 0 else ()
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = tuple([[attn for attn in attention]
                                                  for attention in all_cross_attentions]) if len(all_cross_attentions) != 0 else ()
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )
            # print("DEBUG-DEEPSPEED: layer ", self.layer_id, "\nhidden_states.shape: ", hidden_states.shape, "\nhidden_states: ", hidden_states)
            attention_mask = torch.where(attention_mask > -5000, True, False)
            outputs = make_outputs(
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                input_ids=input_ids,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                output_shape=output_shape,
                presents=presents,
                all_self_attentions=all_self_attentions,
                all_cross_attentions=all_cross_attentions,
                all_hidden_states=all_hidden_states,
                # inputs_embeds=inputs_embeds,
                inputs_embeds_shape=inputs_embeds_shape
            )
            # print("Layer ID : ", self.layer_id, "OUTPUTS : ", outputs)
            out = tuple()
            out = make_tuple_outputs(out, GPT2_BLOCK_KEYS, outputs)
            out = make_tuple_outputs(out, GPT2_FLOW_KEYS, outputs)
            # print("Layer ID : ", self.layer_id, "OUTPUTS : ", out)
            return out

    class PostblockClass(nn.Module):
        def __init__(self, config, ln_f):
            super().__init__()
            self.config = config
            self.ln_f = ln_f

        def forward(
            self,
            inputs
        ):
            inputs = convert_from_tuple(inputs)

            hidden_states, past_key_values, head_mask, encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions, input_ids, inputs_embeds_shape, output_shape, output_hidden_states, return_dict, all_self_attentions, all_cross_attentions, all_hidden_states, attention_mask, presents = inputs
            # 빈 튜플이 맞는 것만 따로 처리해주자
            presents = presents if presents is not None else ()
            all_self_attentions = all_self_attentions if all_self_attentions is not None else ()
            all_cross_attentions = all_cross_attentions if all_cross_attentions is not None else ()
            all_hidden_states = all_hidden_states if all_hidden_states is not None else ()

            attention_mask = torch.where(attention_mask, -0.0, -10000.0)

            hidden_states = self.ln_f(hidden_states)
            hidden_states = hidden_states.view(*output_shape)
            # Add last hidden state
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            attention_mask = torch.where(attention_mask > -5000, True, False)
            outputs = make_outputs(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                output_shape=output_shape,
                presents=presents,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
                hidden_states=all_hidden_states,
                input_ids=input_ids,
                # input_embeds=input_embeds,
                inputs_embeds_shape=inputs_embeds_shape
            )
            # return self.organize_fn(**outputs)
            out = tuple()
            out = make_tuple_outputs(
                out, ("input_ids", "inputs_embeds_shape",) + GPT2_OUTPUT_KEYS, outputs)
            return out

    def to_layers(self):
        preblock = self.PreblockClass(
            self.config, self.wte, self.wpe, self.drop, self.dtype)
        blocks = nn.ModuleList(
            [self.BlockClass(self.config, self.h[i], i, self.gradient_checkpointing) for i in range(len(self.h))])
        postblock = self.PostblockClass(self.config, self.ln_f)
        return [
            preblock,
            *blocks,
            postblock
        ]


class GPT2ForSequenceClassificationPipe(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2ModelPipe(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    class HeadClass(nn.Module):
        def __init__(self, config, score):
            super().__init__()
            self.config = config
            self.score = score

        def forward(self, inputs):
            input_ids, inputs_embeds_shape, last_hidden_state, hidden_states, attentions, cross_attentions, past_key_values = inputs
            logits = self.score(last_hidden_state)
            if input_ids is not None:
                batch_size, sequence_length = input_ids.shape[:2]
            else:
                # batch_size, sequence_length = inputs_embeds.shape[:2]
                batch_size, sequence_length = torch.Size(
                    inputs_embeds_shape)[:2]

            assert (
                self.config.pad_token_id is not None or batch_size == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (
                        torch.ne(
                            input_ids, self.config.pad_token_id).sum(-1) - 1
                    )
                else:
                    sequence_lengths = -1
                    logger.warning(
                        f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                        f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                    )

            pooled_logits = logits[range(batch_size), sequence_lengths]
            logits = pooled_logits
            return logits

    def to_layers(self):
        head = self.HeadClass(self.config, self.score)
        return [
            *self.transformer.to_layers(),
            head
        ]

    def loss_fn(self, logits, labels):
        if labels is None:
            return None

        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (
                labels.dtype == torch.long or labels.dtype == torch.int
            ):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        else:
            loss = None
        torch.set_printoptions(precision=10)
        print("DEBUG: loss_ds : ", loss)
        return loss
