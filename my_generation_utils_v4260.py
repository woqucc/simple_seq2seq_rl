import copy
import itertools
import logging
from typing import Optional, Iterable, Callable, Union, List

import marisa_trie
import torch

from transformers import T5ForConditionalGeneration, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, ConstrainedBeamSearchScorer, PhrasalConstraint, DisjunctiveConstraint, GenerationConfig, StoppingCriteriaList, Constraint
from transformers.generation import validate_stopping_criteria
from transformers.generation.utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput, GenerateOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput

import inspect
import warnings
import torch.distributed as dist

logger = logging.getLogger(__name__)


def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = False,
        **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](./generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which had the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complement the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        kwargs:
            Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchDecoderOnlyOutput`],
                - [`~generation.SampleDecoderOnlyOutput`],
                - [`~generation.BeamSearchDecoderOnlyOutput`],
                - [`~generation.BeamSampleDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchEncoderDecoderOutput`],
                - [`~generation.SampleEncoderDecoderOutput`],
                - [`~generation.BeamSearchEncoderDecoderOutput`],
                - [`~generation.BeamSampleEncoderDecoderOutput`]
    """
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()

    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    if generation_config is None:
        # legacy: users may modify the model configuration to control generation -- update the generation config
        # model attribute accordingly, if it was created from the model config
        if self.generation_config._from_model_config:
            new_generation_config = GenerationConfig.from_model_config(self.config)
            if new_generation_config != self.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use a generation configuration file (see"
                    " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                )
                self.generation_config = new_generation_config
        generation_config = self.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    sampling_with_log_probs = model_kwargs.pop("sampling_with_log_probs", False)

    self._validate_model_kwargs(model_kwargs.copy())
    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
        if (
                generation_config.pad_token_id is not None
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids = self._prepare_decoder_input_ids_for_generation(
            batch_size,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            model_kwargs=model_kwargs,
            device=inputs_tensor.device,
        )
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_seq_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to"
            f" {generation_config.max_length} (`generation_config.max_length`). Controlling `max_length` via the"
            " config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif has_default_max_length and generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
    elif not has_default_max_length and generation_config.max_new_tokens is not None:
        raise ValueError(
            "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
            " limit to the generated output length. Remove one of those arguments. Please refer to the"
            " documentation for more information. "
            "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
        )

    if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
        raise ValueError(
            f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
            f" the maximum length ({generation_config.max_length})"
        )
    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 7. determine generation mode
    is_constraint_gen_mode = (
            generation_config.constraints is not None or generation_config.force_words_ids is not None
    )

    is_contrastive_search_gen_mode = (
            generation_config.top_k is not None
            and generation_config.top_k > 1
            and generation_config.do_sample is False
            and generation_config.penalty_alpha is not None
            and generation_config.penalty_alpha > 0
    )

    is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
    )
    is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
    )
    is_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
    )
    is_beam_sample_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
    )
    is_group_beam_gen_mode = (
            (generation_config.num_beams > 1)
            and (generation_config.num_beam_groups > 1)
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
    )

    if generation_config.num_beam_groups > generation_config.num_beams:
        raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and generation_config.do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    # 9. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    # 10. go into different generation modes
    if is_greedy_gen_mode:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing"
                " greedy search."
            )

        # 11. run greedy search
        return self.greedy_search(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_contrastive_search_gen_mode:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing"
                " contrastive search."
            )

        return self.contrastive_search(
            input_ids,
            top_k=generation_config.top_k,
            penalty_alpha=generation_config.penalty_alpha,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_sample_gen_mode:
        # 11. prepare logits warper
        logits_warper = self._get_logits_warper(generation_config)

        # 12. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run sample
        # return sample(
        #     self,
        #     input_ids=input_ids,
        #     logits_processor=logits_processor,
        #     logits_warper=logits_warper,
        #     max_length=generation_config.max_length,
        #     # stopping_criteria=stopping_criteria,
        #     pad_token_id=generation_config.pad_token_id,
        #     eos_token_id=generation_config.eos_token_id,
        #     output_scores=generation_config.output_scores,
        #     return_dict_in_generate=generation_config.return_dict_in_generate,
        #     # synced_gpus=synced_gpus,
        #     sampling_with_log_probs=sampling_with_log_probs,
        #     **model_kwargs,
        # )
        return sample(self,
                      input_ids,
                      logits_processor=logits_processor,
                      logits_warper=logits_warper,
                      stopping_criteria=stopping_criteria,
                      pad_token_id=generation_config.pad_token_id,
                      eos_token_id=generation_config.eos_token_id,
                      output_scores=generation_config.output_scores,
                      return_dict_in_generate=generation_config.return_dict_in_generate,
                      synced_gpus=synced_gpus,
                      sampling_with_log_probs=sampling_with_log_probs,
                      **model_kwargs,
                      )

    elif is_beam_gen_mode:
        if generation_config.num_return_sequences > generation_config.num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        return self.beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_beam_sample_gen_mode:
        # 11. prepare logits warper
        logits_warper = self._get_logits_warper(generation_config)

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")
        # 12. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size * generation_config.num_return_sequences,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
        )

        # 13. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams * generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 14. run beam sample
        return self.beam_sample(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_group_beam_gen_mode:
        if generation_config.num_return_sequences > generation_config.num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if generation_config.num_beams % generation_config.num_beam_groups != 0:
            raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        has_default_typical_p = kwargs.get("typical_p") is None and generation_config.typical_p == 1.0
        if not has_default_typical_p:
            raise ValueError("Decoder argument `typical_p` is not supported with beam groups.")

        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            max_length=stopping_criteria.max_length,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            num_beam_groups=generation_config.num_beam_groups,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        return self.group_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_constraint_gen_mode:
        if generation_config.num_return_sequences > generation_config.num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        if generation_config.num_beams <= 1:
            raise ValueError("`num_beams` needs to be greater than 1 for constrained generation.")

        if generation_config.do_sample:
            raise ValueError("`do_sample` needs to be false for constrained generation.")

        if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
            raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

        final_constraints = []
        if generation_config.constraints is not None:
            final_constraints = generation_config.constraints

        if generation_config.force_words_ids is not None:

            def typeerror():
                raise ValueError(
                    "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                    f"of positive integers, but is {generation_config.force_words_ids}."
                )

            if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
            ):
                typeerror()

            for word_ids in generation_config.force_words_ids:
                if isinstance(word_ids[0], list):
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any(not isinstance(token_ids, list) for token_ids in word_ids):
                        typeerror()
                    if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                    ):
                        typeerror()

                    constraint = DisjunctiveConstraint(word_ids)
                else:
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                        typeerror()

                    constraint = PhrasalConstraint(word_ids)
                final_constraints.append(constraint)

        # 11. prepare beam search scorer
        constrained_beam_scorer = ConstrainedBeamSearchScorer(
            constraints=final_constraints,
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        return self.constrained_beam_search(
            input_ids,
            constrained_beam_scorer=constrained_beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )


def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        sampling_with_log_probs: Optional[bool] = None,
        **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
    For an overview of generation strategies and code examples, check the [following
    guide](./generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     TopKLogitsWarper,
    ...     TemperatureLogitsWarper,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> model.generation_config.pad_token_id = model.config.eos_token_id

    >>> input_prompt = "Today is a beautiful day, and"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> # instantiate logits processors
    >>> logits_warper = LogitsProcessorList(
    ...     [
    ...         TopKLogitsWarper(50),
    ...         TemperatureLogitsWarper(0.7),
    ...     ]
    ... )

    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
    >>> outputs = model.sample(
    ...     input_ids,
    ...     logits_processor=logits_processor,
    ...     logits_warper=logits_warper,
    ...     stopping_criteria=stopping_criteria,
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Today is a beautiful day, and a wonderful day.\n\nI was lucky enough to meet the']
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    this_peer_finished = False  # used by synced_gpus only
    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        if sampling_with_log_probs:
            next_token_log_probs = torch.log_softmax(next_token_logits, dim=1)

        # pre-process distribution
        with torch.no_grad():
            next_token_scores = logits_processor(input_ids, next_token_logits.detach().clone())
            next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                if sampling_with_log_probs:
                    scores += (next_token_log_probs,)
                else:
                    scores += (next_token_scores,)

            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # sample
        with torch.no_grad():
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)

            # if torch.isnan(probs).any():
            #     breakpoint()
            # if torch.isinf(probs).any():
            #     breakpoint()
            # probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids


class StricterConstrainedBeamSearchScorer(ConstrainedBeamSearchScorer):
    pass


class BartForConditionalGenerationWrapper(BartForConditionalGeneration):
    def generate_with_grad(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        generate_config = GenerationConfig(**kwargs)
        return generate(self,
                        inputs=input_ids,
                        generation_config=generate_config,
                        logits_processor=None,
                        stopping_criteria=None,
                        prefix_allowed_tokens_fn=None,
                        synced_gpus=False,
                        **kwargs,
                        )

    def constrained_beam_search(
            self,
            input_ids: torch.LongTensor,
            constrained_beam_scorer: ConstrainedBeamSearchScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = None,
            **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:

        # TODO: Here we modify the constraints to support our strict constrains: if failed, failed forever,
        #  no chance to keep generating based on failed sequences
        # ori_check_completes_constraints = constrained_beam_scorer.check_completes_constraints

        # ori_check_completes_constraints
        def check_completes_constraints(sequence):
            new_state = constrained_beam_scorer.make_constraint_states(1)[0]
            new_state.init_state()

            if sequence is not None:
                return new_state.constraints[0].check_completes_constraint_by_str(sequence)
                # for token in sequence:
                #     # completes or steps **one** constraint
                #     complete, stepped = new_state.add(token)
                #     if not stepped:
                #         if complete:
                #             assert sequence[-1] == 1
                #         break
                #     # the entire list of constraints are fulfilled
                #     # if new_state.completed:
                #     #     break

            return new_state.completed

        constrained_beam_scorer.check_completes_constraints = check_completes_constraints

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (output_attentions if output_attentions is not None else self.generation_config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate if return_dict_in_generate is not None else self.generation_config.return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(constrained_beam_scorer._beam_hyps)
        num_beams = constrained_beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.")

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)

            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
            strict_forcing = constrained_beam_scorer.constraints[0].strict_forcing
            if strict_forcing:
                is_seq_finished = [False for _ in range(input_ids.shape[0])]
                is_seq_constrained = [False for _ in range(input_ids.shape[0])]
                for input_idx in range(input_ids.shape[0]):
                    for token_idx in range(input_ids.shape[1]):
                        if token_idx > 2:
                            if input_ids[input_idx][token_idx] in {2}:
                                is_seq_finished[input_idx] = True
                    is_seq_constrained[input_idx] = constrained_beam_scorer.constraints[0].check_completes_constraint_by_str(input_ids[input_idx], remove_token_after_eos=False)

                for input_idx in range(next_token_scores.shape[0]):
                    valid_token_set = constrained_beam_scorer.constraints[0].get_valid_token_list(input_ids[input_idx].tolist(), input_ids[input_idx].size(0) - 1, use_cache=False)

                    if is_seq_finished[input_idx]:
                        next_token_scores[input_idx] = 0
                        continue

                    if not is_seq_constrained[input_idx]:
                        next_token_scores[input_idx] = float("-inf")
                        continue

                    if len(valid_token_set) == 0:
                        next_token_scores[input_idx] = float("-inf")
                        continue

                    all_zero_mask = torch.full_like(next_token_scores[0], float("-inf"))
                    all_zero_mask[list(valid_token_set)] = 0
                    next_token_scores[input_idx] = next_token_scores[input_idx] + all_zero_mask

            scores_for_all_vocab = next_token_scores.clone()

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += ((outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,))
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += ((outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,))

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(next_token_scores, 5 * num_beams, dim=1, largest=True, sorted=True)

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = constrained_beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                scores_for_all_vocab,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if constrained_beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = constrained_beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        # constrained_beam_scorer.check_completes_constraints = ori_check_completes_constraints
        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]


class T5ForConditionalGenerationWrapper(T5ForConditionalGeneration):
    def generate_with_grad(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            num_beam_groups: Optional[int] = None,
            diversity_penalty: Optional[float] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            sampling_with_log_probs: Optional[bool] = False,
            **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        return generate(self,
                        input_ids=input_ids,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=do_sample,
                        early_stopping=early_stopping,
                        num_beams=num_beams,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        bad_words_ids=bad_words_ids,
                        bos_token_id=bos_token_id,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        num_return_sequences=num_return_sequences,
                        decoder_start_token_id=decoder_start_token_id,
                        use_cache=use_cache,
                        num_beam_groups=num_beam_groups,
                        diversity_penalty=diversity_penalty,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        output_scores=output_scores,
                        return_dict_in_generate=return_dict_in_generate,
                        sampling_with_log_probs=sampling_with_log_probs,
                        **model_kwargs)


class ForceFromContextStrConstraint(Constraint):
    r"""
    [`Constraint`] enforcing that an ordered sequence of tokens is included in the output.

    Args:
        token_ids (`List[int]`):
            The id of the token that must be generated by the output.
    """

    def __init__(self, context_str, tokenizer, same_token_ids_mapping, seqlen=None, min_answer_token_count=0, cached_tire=None, cached_next_valid_token=None,
                 strict_forcing=False):

        super(Constraint, self).__init__()

        if not isinstance(context_str, str):
            raise ValueError(f"`token_ids` has to be a non-empty list, but is {context_str}.")

        self.strict_forcing = strict_forcing
        self.context_str = context_str
        self.min_answer_token_count = min_answer_token_count
        self.tokenizer = tokenizer
        self.same_token_ids_mapping = same_token_ids_mapping
        self.seqlen = (len(context_str) if seqlen is None else seqlen)
        self.special_id_set = set(self.tokenizer.all_special_ids)

        if cached_tire is not None:
            self.tire = cached_tire
        else:
            tree_candidate_str_list = []
            for start_i in range(len(context_str)):
                tokenized_partial_context = self.tokenizer(context_str[start_i:], return_offsets_mapping=True, return_attention_mask=False, add_special_tokens=False).data
                len_in_str = tokenized_partial_context["offset_mapping"][:self.seqlen][-1][-1]
                if len_in_str > 0:
                    tree_candidate_str_list.append(context_str[start_i:start_i + len_in_str].strip())
            self.tire = marisa_trie.Trie(tree_candidate_str_list)

        self.fulfilled_idx = -1  # the index of the currently fulfilled step
        self.generated_ids = []
        self.cached_next_valid_token = {} if cached_next_valid_token is None else cached_next_valid_token
        self.next_valid_tokens = self.get_valid_token_list(self.generated_ids, self.fulfilled_idx)
        self.completed = False
        # self.reseted = False
        # self.failed = False

    def advance(self):
        if self.completed:
            return None
        # valid_tokens = set(itertools.chain.from_iterable([t] + self.same_token_ids_mapping[t] for t in self.tokenized_suffix_ids[:, 0].unique().tolist()))
        # valid_tokens.update(self.tokenizer.all_special_ids)
        return list(self.next_valid_tokens)

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        if self.completed:
            return False
        if len(self.next_valid_tokens) == 0:
            return False

        # if self.reseted:
        #     return False

        # valid_tokens = self.get_valid_token_list(self.generated_ids + [])
        if token_id in self.next_valid_tokens:
            return True
        if token_id in self.tokenizer.all_special_ids:
            return True
        return False

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.fulfilled_idx += 1
            stepped = True
            # plus 2 for start tokens [2,0]
            if self.fulfilled_idx == (self.seqlen - 1 + 2):
                completed = True
            # if len(self.generated_ids) >= (2 + self.min_answer_token_count) and self.generated_ids[0] == 2 and self.generated_ids[1] == 0 and (2 in self.generated_ids[2 + self.min_answer_token_count:]):
            # if len(self.generated_ids) >= (2 + self.min_answer_token_count) and (self.tokenizer.eos_token_id in self.generated_ids[2 + self.min_answer_token_count:]):
            #     completed = True

            self.generated_ids.append(token_id)
            # if len(self.generated_ids) >= (2 + self.min_answer_token_count):
            if len(self.generated_ids) >= (2 + self.min_answer_token_count) and (self.tokenizer.eos_token_id in self.generated_ids[2 + self.min_answer_token_count:]):
                completed = True
            self.completed = completed

            if token_id not in self.special_id_set or self.fulfilled_idx < 2:
                self.next_valid_tokens = self.get_valid_token_list(self.generated_ids, self.fulfilled_idx)
                # print(self.generated_ids, self.tokenizer.decode(self.generated_ids), self.next_valid_tokens)
                if len(self.next_valid_tokens) == 0 and not self.completed:
                    reset = True
                    self.reset()
                    # self.failed = True

                    # assert False

        else:
            # failed to make progress.
            reset = True
            # self.failed = True
            self.reset()

        # if self.failed:
        #     stepped = False
        # if self.reseted:
        #     return False, False, True
        return stepped, completed, reset

    def get_valid_token_list(self, generated_ids, fulfilled_idx, add_special_tokens=True, use_cache=True):
        cache_key = tuple(generated_ids + [fulfilled_idx])
        if cache_key not in self.cached_next_valid_token:

            if len(generated_ids) == 0 and fulfilled_idx == -1:
                return {self.tokenizer.sep_token_id}
            if len(generated_ids) == 1 and fulfilled_idx == 0:
                return {self.tokenizer.bos_token_id}
            if self.completed:
                return {self.tokenizer.eos_token_id}

            if len(generated_ids) > 2 and generated_ids[-1] == self.tokenizer.eos_token_id:
                return {self.tokenizer.eos_token_id}

            current_str = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            candidate_str_list = list(set(s[len(current_str):].strip() for s in self.tire.keys(current_str)))
            candidate_str_list = [c_str for c_str in candidate_str_list if len(c_str.strip()) > 0]

            if len(candidate_str_list) == 0:
                return {}
            #     for other_token in self.same_token_ids_mapping[generated_ids[-1]]:
            #         if other_token != generated_ids[-1]:
            #             generated_ids_ = generated_ids[:-1] + [other_token]
            #             current_str = self.tokenizer.decode(generated_ids_, skip_special_tokens=True).strip()
            #             candidate_str_list = list(set(s[len(current_str):] for s in self.tire.keys(current_str)))

            # candidate_str_list = [c_str for c_str in candidate_str_list if len(c_str.strip()) > 0]
            # tokenied
            tokenized_suffix_ids_first_word_stride = self.tokenizer(
                list(set(itertools.chain.from_iterable([[s.split()[0][:c_i + 1] for c_i in range(0, len(s.split()[0]))] for s in candidate_str_list]))),
                return_tensors="pt",
                add_special_tokens=False,
                return_attention_mask=False,
                max_length=self.seqlen,
                padding=True,
                truncation=True
            ).data["input_ids"]

            if not self.tokenizer.add_prefix_space:
                candidate_str_list += [" " + c_str for c_str in candidate_str_list if len(c_str) > 0 and not c_str[0].isspace()]

            tokenized_suffix_ids = self.tokenizer(
                candidate_str_list,
                return_tensors="pt",
                add_special_tokens=False,
                return_attention_mask=False,
                max_length=self.seqlen,
                padding=True,
                truncation=True
            ).data["input_ids"]

            next_valid_tokens = tokenized_suffix_ids[:, 0].unique().tolist() + tokenized_suffix_ids_first_word_stride[:, 0].unique().tolist()
            next_valid_tokens = list(set(next_valid_tokens))

            # if len(current_str) > 0 and len(current_str.split()[0].strip()) > 0 and current_str.split()[0].strip() in self.tokenizer.vocab:
            #     next_valid_tokens.append(self.tokenizer.vocab[current_str.split()[0].strip()])
            new_next_valid_tokens = []
            generated_ids_to_be_verified = [self.generated_ids + [valid_token] for valid_token in next_valid_tokens]
            generated_strs_to_be_verified = self.tokenizer.batch_decode(generated_ids_to_be_verified, skip_special_tokens=True)
            for valid_token, str_to_be_verified in zip(next_valid_tokens, generated_strs_to_be_verified):
                if len(self.tire.keys(str_to_be_verified)) > 0:
                    new_next_valid_tokens.append(valid_token)
                else:
                    for same_token_with_or_without_space in self.same_token_ids_mapping[valid_token]:
                        if same_token_with_or_without_space != valid_token:
                            same_str = self.tokenizer.decode(self.generated_ids + [same_token_with_or_without_space], skip_special_tokens=True)
                            if len(self.tire.keys(same_str)) > 0:
                                new_next_valid_tokens.append(same_token_with_or_without_space)

            # next_valid_tokens = next_valid_tokens + same_tokens

            if add_special_tokens:
                next_valid_tokens.append(self.tokenizer.eos_token_id)
            next_valid_tokens = set(next_valid_tokens)

            # if 12873 in next_valid_tokens:
            #     breakpoint()
            if use_cache:
                self.cached_next_valid_token[cache_key] = set(next_valid_tokens)
            else:
                return set(next_valid_tokens)

        return self.cached_next_valid_token[cache_key]

    def reset(self):
        self.fulfilled_idx = -1  # the index of the currently fulfilled step
        self.generated_ids = []

        # self.reseted = True
        # self.next_valid_tokens = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        self.next_valid_tokens = self.get_valid_token_list(self.generated_ids, self.fulfilled_idx)
        self.completed = False

    def remaining(self):
        if self.completed:
            # since this can be completed without reaching max height
            return 0
        else:
            return self.seqlen - (self.fulfilled_idx + 2)

    def copy(self, stateful=False):
        new_constraint = ForceFromContextStrConstraint(self.context_str, self.tokenizer, self.same_token_ids_mapping, self.seqlen, self.min_answer_token_count,
                                                       cached_tire=self.tire, cached_next_valid_token=self.cached_next_valid_token)

        if stateful:
            new_constraint.fulfilled_idx = self.fulfilled_idx  # the index of the currently fulfilled step
            new_constraint.generated_ids += self.generated_ids
            new_constraint.next_valid_tokens = copy.deepcopy(self.next_valid_tokens)
            new_constraint.completed = self.completed

        return new_constraint

    def check_completes_constraint_by_str(self, seq_ids, remove_token_after_eos=True):
        if remove_token_after_eos:
            seq_str = removing_the_tokens_after_eos_and_decode([seq_ids], self.tokenizer)[0].strip()
        else:
            seq_str = self.tokenizer.decode(seq_ids, skip_special_tokens=True)
        candidate_str_list = self.tire.keys(seq_str)
        if len(candidate_str_list) > 0:
            return True
        else:
            return False


def removing_the_tokens_after_eos_and_decode(batch_seq_ids, tokenizer):
    trimmed_batch_seq_ids = []
    for seq_ids in batch_seq_ids:
        if type(seq_ids) is not list:
            seq_ids = seq_ids.squeeze().tolist()

        trimmed_seq_ids = []
        for idx, id_ in enumerate(seq_ids):
            if idx > 2 and id_ == tokenizer.eos_token_id:
                break
            trimmed_seq_ids.append(id_)
        trimmed_batch_seq_ids.append(trimmed_seq_ids)
    return tokenizer.batch_decode(trimmed_batch_seq_ids, skip_special_tokens=True)
