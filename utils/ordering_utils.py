# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code based on https://github.com/airKlizz/passage-ordering/blob/main/training/scripts/models/ordering_utils.py
"""

import logging
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class OrderingMixin:
    """
    A class contraining all of the functions supporting generation, to be used as a mixin in PreTrainedModel.
    """

    def is_sequence_ordering_model(self):
        return False

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def adjust_logits_during_generation(self, logits, **kwargs):
        return logits

    def _use_cache(self, outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        if len(outputs) <= 1 or use_cache is False:
            return False
        return True

    def postprocess_next_sequence_scores(
        self,
        scores,
        decoder_input_id,
        done,
        ordered_sequences,
        batch_size,
        num_beams,
    ):
        """
        Set scores to "-inf" when:
            1- the decoder_input_id is not eos (i.e. doesn't represent a sequence),
            2- the batch is done,
            3- the sequence is already ordered.
        """
        # 1
        scores[decoder_input_id != self.eos_token_id] = float("-inf")
        # 2
        scores[done] = float("-inf")
        # 3
        for idx in range(batch_size * num_beams):
            for sequence in ordered_sequences[idx]:
                scores[idx, sequence] = float("-inf")

        return scores

    @torch.no_grad()
    def order(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        num_beams: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_ids: Optional[List[int]] = None,
        use_cache: Optional[bool] = None,
        specific_head: Optional[int] = None,
        **model_specific_kwargs,
    ) -> torch.LongTensor:

        # We cannot order if the model does not have a LM head
        if not self.is_sequence_ordering_model():
            raise AttributeError(
                "You tried to order sequences with a model that does not support ordering."
                "Please use BartForSequenceOrdering"
            )

        use_cache = False

        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert input_ids is not None, "Input_ids is not defined."
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # batch_size and sequence_length
        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]

        # set effective batch size and effective batch multiplier
        effective_batch_size = batch_size
        effective_batch_mult = 1

        assert (
            decoder_start_token_ids is not None
        ), "decoder_start_token_ids has to be defined for encoder-decoder generation"
        assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
        assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

        # done sentences
        done = torch.zeros((batch_size * num_beams), device=input_ids.device).bool()

        # ordered_sequences
        ordered_sequences = [[] for _ in range(batch_size)]

        # remained sequences to order
        remained_sequences = [torch.nonzero(elem == self.eos_token_id).squeeze(-1).tolist() for elem in input_ids]

        # prediction to range in the input_ids
        pred2range = [
            {l[i]: (1, l[i] + 1) if i == 0 else (l[i - 1] + 1, l[i] + 1) for i in range(len(l))}
            for l in remained_sequences
        ]

        # prediction to the idx of the sentence
        pred2idx = [{x: i for i, x in enumerate(p)} for p in remained_sequences]

        # expand ordered_sequences and remained_sequences
        ordered_sequences = [l.copy() for l in ordered_sequences for _ in range(num_beams)]
        remained_sequences = [l.copy() for l in remained_sequences for _ in range(num_beams)]

        # get encoder and store encoder outputs
        encoder = self.get_encoder()

        encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)

        # create decoder_input_ids and decoder_attention_mask
        decoder_token_ids = decoder_start_token_ids + [-1] * (sequence_length - len(decoder_start_token_ids))
        decoder_input_ids = torch.tensor(
            decoder_token_ids,
            dtype=torch.long,
            device=next(self.parameters()).device,
        ).repeat(effective_batch_size * num_beams, 1)
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, sequence_length)
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, effective_batch_mult * num_beams, sequence_length
        )

        decoder_input_ids = decoder_input_ids.contiguous().view(
            effective_batch_size * num_beams, sequence_length
        )  # shape: (batch_size * num_return_sequences * num_beams, sequence_length)
        input_ids = input_ids.contiguous().view(
            effective_batch_size * num_beams, sequence_length
        )  # shape: (batch_size * num_return_sequences * num_beams, sequence_length)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size * num_beams, sequence_length
        )  # shape: (batch_size * num_return_sequences * num_beams, sequence_length)

        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (
            encoder_outputs[0].index_select(0, expanded_batch_idxs),
            *encoder_outputs[1:],
        )

        if num_beams > 1:
            output = self._order_beam_search(
                decoder_input_ids,
                input_ids=input_ids,
                done=done,
                remained_sequences=remained_sequences,
                pred2range=pred2range,
                pred2idx=pred2idx,
                ordered_sequences=ordered_sequences,
                batch_size=effective_batch_size,
                num_beams=num_beams,
                sequence_length=sequence_length,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                specific_head=specific_head,
                model_specific_kwargs=model_specific_kwargs,
            )
        else:
            output = self._order_no_beam_search(
                decoder_input_ids,
                input_ids=input_ids,
                done=done,
                remained_sequences=remained_sequences,
                pred2range=pred2range,
                pred2idx=pred2idx,
                ordered_sequences=ordered_sequences,
                batch_size=effective_batch_size,
                sequence_length=sequence_length,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                specific_head=specific_head,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output

    def _order_no_beam_search(
        self,
        decoder_input_ids,
        input_ids,
        done,
        ordered_sequences,
        remained_sequences,
        pred2range,
        pred2idx,
        batch_size,
        sequence_length,
        encoder_outputs,
        attention_mask,
        use_cache,
        specific_head,
        model_specific_kwargs,
    ):

        past = None

        decoder_step = 0
        while decoder_step < sequence_length:
            model_inputs = self.prepare_inputs_for_generation(
                decoder_input_ids=decoder_input_ids[:, : decoder_step + 1],
                past=past,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                encoder_outputs=encoder_outputs,
            )
            outputs = self(**model_inputs)

            if specific_head == None:
                next_sequence_logits = outputs.logits[:, -1, :]
            else:
                next_sequence_logits = outputs.head_logits[:, specific_head, -1, :]

            scores = self.postprocess_next_sequence_scores(
                scores=next_sequence_logits,
                decoder_input_id=decoder_input_ids[:, decoder_step],
                done=done,
                ordered_sequences=ordered_sequences,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs.past_key_values

            next_sequence = torch.argmax(scores, dim=-1)
            next_sequence_mask = ~((scores != float("-inf")).any(-1))
            # ignore next token for token != eos (see self.postprocess_next_sequence_scores)
            next_sequence[next_sequence_mask] = -100

            for batch in range(batch_size):
                prediction = int(next_sequence[batch])
                # ignore next token
                if prediction == -100:
                    continue
                # get the next sequence ids from input_ids
                begin, end = pred2range[batch][prediction]
                next_sequence_ids = input_ids[batch][begin:end]
                # add next_sequence_ids to the decoder_input_ids
                decoder_input_ids[
                    batch, decoder_step + 1 : decoder_step + 1 + next_sequence_ids.size(0)
                ] = next_sequence_ids[: decoder_input_ids.size(1) - (decoder_step + 1)]

                ordered_sequences[batch].append(prediction)
                # remove sequence from remained_sequences
                remained_sequences[batch].remove(prediction)
                # done if all sequences ordered
                if len(remained_sequences[batch]) == 0 and done[batch] == False:
                    done[batch] = True
                    # replace -1 by pad tokens to avoid errors
                    decoder_input_ids[batch][decoder_input_ids[batch] == -1] = self.pad_token_id

            decoder_step = decoder_step + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if done.all() == True:
                break

        # get the sequence idx and add to results
        results = [[pred2idx[batch][pred] for pred in ordered_sequences[batch]] for batch in range(batch_size)]
        return results

    def _order_beam_search(
        self,
        decoder_input_ids,
        input_ids,
        done,
        ordered_sequences,
        remained_sequences,
        pred2range,
        pred2idx,
        batch_size,
        num_beams,
        sequence_length,
        encoder_outputs,
        attention_mask,
        use_cache,
        specific_head,
        model_specific_kwargs,
    ):

        # scores for each sentence in the beam
        beam_scores = torch.ones((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)
        # steps for each sentence in the beam
        beam_steps = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_steps = beam_steps.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = None

        decoder_step = 0
        while decoder_step < sequence_length:

            model_inputs = self.prepare_inputs_for_generation(
                decoder_input_ids=decoder_input_ids[:, : decoder_step + 1],
                past=past,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                encoder_outputs=encoder_outputs,
            )

            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, sequence_length)
            if specific_head == None:
                next_sequence_logits = outputs.logits[:, -1, :]
            else:
                next_sequence_logits = outputs.head_logits[
                    :, specific_head, -1, :
                ]  # (batch_size * num_beams, sequence_length)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs.past_key_values

            scores = F.log_softmax(next_sequence_logits, dim=-1)  # (batch_size * num_beams, sequence_length)

            scores = self.postprocess_next_sequence_scores(
                scores=scores,
                decoder_input_id=decoder_input_ids[:, decoder_step],
                done=done,
                ordered_sequences=ordered_sequences,
                batch_size=batch_size,
                num_beams=num_beams,
            )

            next_sequence_mask = ~((scores != float("-inf")).any(-1))
            # for each beam, set the score of the first token to the beam score
            # if there is no next sequence scores (i.e. not eos or beam done).
            # this is to compare beam scores after
            for idx, score in enumerate(scores):
                if not (score == float("-inf")).all():
                    continue
                scores[idx, 0] = beam_scores[idx]

            assert scores.shape == (
                batch_size * num_beams,
                sequence_length,
            ), "Shapes of scores: {} != {}".format(scores.shape, (batch_size * num_beams, sequence_length))

            # compute the next_scores for each beam.
            # the score is the mean of all sequences scores.
            # this does not follow the original beam search algorithm
            # but it is to handle the issue that every beam does not have the same number of sentences.
            next_scores = (
                scores + (beam_steps[:, None].expand_as(scores) * beam_scores[:, None].expand_as(scores))
            ) / (
                beam_steps[:, None].expand_as(scores) + 1
            )  # (batch_size * num_beams, sequence_length)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(
                batch_size, num_beams * sequence_length
            )  # (batch_size, num_beams * sequence_length)

            # make sure that only next sequences of the first beam are considered to avoid sampling the exact same sequences num_beams times
            if beam_steps.sum() == 0:
                next_scores.view(batch_size, num_beams, sequence_length)[:, 1:] += -1e9

            topk_next_scores, topk_next_sequences = torch.topk(
                next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            assert topk_next_scores.size() == topk_next_sequences.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(topk_next_sequences[batch_idx], topk_next_scores[batch_idx])
                ):

                    # once the beam is full, don't add more tokens to it.
                    if len(next_sent_beam) == num_beams:
                        break

                    # once all ordering possibilities have been explored
                    # pad if bean is not full
                    if beam_token_score == float("-inf"):
                        next_sent_beam.extend(
                            [(-1e9, self.pad_token_id, False, 0)] * (num_beams - len(next_sent_beam))
                        )
                        break

                    # get beam and token IDs
                    beam_id = (beam_token_id // sequence_length).item()
                    token_id = (beam_token_id % sequence_length).item()

                    effective_beam_id = batch_idx * num_beams + beam_id

                    # if we are done with this sentence, add a pad token
                    if done[effective_beam_id]:
                        next_sent_beam.append(
                            (beam_token_score, self.pad_token_id, False, effective_beam_id)
                        )  # pad the batch
                        continue

                    # get new_sequence.
                    # True if the token_id referes to a new sequence
                    # False if not
                    new_sequence = not (
                        (
                            next_scores.view(batch_size * num_beams, sequence_length)[effective_beam_id]
                            != float("-inf")
                        ).sum()
                        == 1
                        and token_id == 0
                    )

                    # set token_id to the next token_id of the sequence if it is not a new_sequence
                    if not new_sequence:
                        token_id = decoder_input_ids[effective_beam_id, decoder_step + 1]

                    # add next predicted token
                    next_sent_beam.append((beam_token_score, token_id, new_sequence, effective_beam_id))

                # update next beam content
                assert (
                    len(next_sent_beam) == num_beams
                ), f"Beam should always be full ({len(next_sent_beam)}/{num_beams})"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_new_sequence = input_ids.new([x[2] for x in next_batch_beam])
            beam_idx = input_ids.new([x[3] for x in next_batch_beam])

            beam_steps = beam_steps[beam_idx]
            beam_steps = beam_steps.new(
                [step + 1 if x[2] == True else step for step, x in zip(beam_steps, next_batch_beam)]
            )

            # re-order according to the beam idx
            done = done[beam_idx]
            ordered_sequences = [ordered_sequences[i].copy() for i in beam_idx]
            remained_sequences = [remained_sequences[i].copy() for i in beam_idx]

            # re-order batch and update current length
            decoder_input_ids = decoder_input_ids[beam_idx, :]
            for idx in range(batch_size * num_beams):
                batch_idx = idx // num_beams
                if beam_new_sequence[idx] == False:
                    decoder_input_ids[idx, decoder_step + 1] = beam_tokens[idx]
                else:
                    next_sequence_pred = int(beam_tokens[idx])
                    # get the next sequence ids from input_ids
                    begin, end = pred2range[batch_idx][next_sequence_pred]
                    next_sequence_ids = input_ids[idx][begin:end]
                    # add next_sequence_ids to the decoder_input_ids
                    decoder_input_ids[
                        idx, decoder_step + 1 : decoder_step + 1 + next_sequence_ids.size(0)
                    ] = next_sequence_ids[: decoder_input_ids.size(1) - (decoder_step + 1)]
                    ordered_sequences[idx].append(next_sequence_pred)
                    # remove sequence from remained_sequences
                    remained_sequences[idx].remove(next_sequence_pred)
                    # Check if the beam is done
                    done[idx] = done[idx] or len(remained_sequences[idx]) == 0

            decoder_step = decoder_step + 1

            # stop when we are done with each sentence
            if done.all() == True:
                break

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

        # find the best beam for each batch
        best_beam = beam_scores.view(batch_size, num_beams).argmax(-1)
        ordered_sequences = [
            ordered_sequences[idx]
            for idx in (best_beam + torch.arange(0, batch_size, device=best_beam.device) * num_beams)
        ]

        # get the sequence idx and add to results
        results = [[pred2idx[batch][pred] for pred in ordered_sequences[batch]] for batch in range(batch_size)]
        return results
