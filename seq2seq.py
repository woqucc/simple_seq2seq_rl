import itertools
import math
import sys
from statistics import mean

import accelerate.utils
from accelerate import Accelerator

import os

import json

import pandas as pd
import torch

import numpy as np

from pathlib import Path

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from evaluation import rouge_scores, rouge_score, f1_score, f1_scores, exact_match_scores
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig, BartConfig, get_linear_schedule_with_warmup, get_constant_schedule

from my_generation_utils_v4260 import ForceFromContextStrConstraint, BartForConditionalGenerationWrapper, removing_the_tokens_after_eos_and_decode
import src.options
import src.util


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, accelerator: Accelerator, checkpoint_path):
    torch.manual_seed(accelerator.process_index + opt.seed)  # different seed for different sampling depending on global_rank
    # train_sampler = RandomSampler(train_dataset) if not opt.is_distributed else DistributedSampler(train_dataset)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=opt.num_workers,
        collate_fn=collator
    )
    print(train_dataloader, len(train_dataloader))


    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    epoch_num = opt.total_steps / len(train_dataloader)
    # print(train_dataloader, len(train_dataloader))

    if accelerator.is_main_process:
        logger.info("-" * 50)
        logger.info(f"per gpu bs {opt.per_gpu_batch_size}, final bs {opt.accumulation_steps * opt.per_gpu_batch_size * accelerator.num_processes}\n"
                    f"source/target max length {opt.source_max_length}/{opt.target_max_length}\n"
                    f"Total samples({accelerator.process_index}):\t{len(train_dataset)}\n"
                    f"Total steps({accelerator.process_index}):\t{opt.total_steps}, about {epoch_num:.2f} epochs\n"
                    f"Loader len({len(train_dataloader)}), warmup steps({accelerator.process_index}):\t{opt.warmup_steps}")

    for epoch_i in range(math.ceil(epoch_num)):
        tqdm_train_dataloader = tqdm(train_dataloader, disable=not accelerator.is_main_process, total=opt.total_steps, initial=step)
        train_loss_list = []

        for i, batch in enumerate(tqdm_train_dataloader):
            step += 1
            with accelerator.accumulate(model):
                (idx, labels, labels_masks, input_ids, input_mask) = batch
                model_output = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    labels=labels,
                    return_dict=True,
                )

                train_loss = model_output.loss
                accelerator.backward(train_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            train_loss_list.append(train_loss.item())

            if step % 10 == 0:
                tqdm_train_dataloader.set_description(f"({accelerator.process_index})loss: {mean(train_loss_list):.4f}, lr:  {scheduler.get_last_lr()[0]}, {scheduler.scheduler.last_epoch}")

            if step % opt.eval_freq == 0 and opt.eval_during_training:
                dev_f1, dev_rougel = eval(model, eval_dataset, tokenizer, collator, opt, accelerator, step)
                if train_loss_list:
                    logger.info(f"training loss:{mean(train_loss_list):.4f}")
                    train_loss_list = []

                model.train()

            if step % opt.save_freq == 0:
                train_loss_list = []
                if accelerator.is_main_process:
                    src.util.save(model, optimizer, scheduler, opt, checkpoint_path, f"checkpoint-{step}")

                accelerator.wait_for_everyone()

            if step >= opt.total_steps:
                break

    accelerator.wait_for_everyone()


def eval(model, dataset, tokenizer, collator, opt, accelerator: Accelerator, step=None):
    sampler = SequentialSampler(dataset)
    logger.info(f"Using {sampler} as sampler for evaluation.")

    eval_dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=opt.num_workers,
        collate_fn=collator)

    model.eval()
    total = 0
    rouge1_list, rouge2_list, rougel_list = [], [], []
    f1_list, em_list = [], []


    pred_answers, gold_answers = [], []
    eval_dataloader = accelerator.prepare(eval_dataloader)

    eval_dataloader_tqdm = tqdm(eval_dataloader, disable=not accelerator.is_main_process)

    logger.info(f"accelerator device: {accelerator.device}, model device: {model.device}, dataset len: {len(dataset)}")

    with torch.no_grad():

        same_token_ids_mapping = torch.load("/home/li-shao-bo/output/same_token_ids_mapping.pt")

        def constrained_model_generate(context_str, **kwargs):
            constraint_for_single_sample = ForceFromContextStrConstraint(context_str, tokenizer, same_token_ids_mapping, opt.target_max_length,
                                                                         min_answer_token_count=opt.min_answer_token_count, strict_forcing=kwargs.pop("strict_forcing"))
            kwargs["constraints"] = [constraint_for_single_sample]
            if type(model) is DistributedDataParallel:
                return model.module.generate(**kwargs)
            else:
                return model.generate(**kwargs)

        def model_generate(**kwargs):
            if type(model) is DistributedDataParallel:
                return model.module.generate(**kwargs)
            else:
                return model.generate(**kwargs)

        for i, batch in enumerate(eval_dataloader_tqdm):


            (dataset_idx_list, _, _, input_ids, input_mask) = batch

            outputs = model_generate(

                input_ids=input_ids,
                attention_mask=input_mask,
                max_length=opt.target_max_length,
                num_beams=opt.num_beams
            )

            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for batch_idx, prediction in enumerate(predictions):
                prediction = prediction.strip()

                example = dataset[dataset_idx_list[batch_idx]]
                gold_answer = example["target"]
                qid = example["id"]
                context = example["context"]
                if prediction not in context and opt.use_constrained_decoding:
                    print(f"Going to regenerate {qid}: {prediction}")
                    outputs = constrained_model_generate(context_str=context,
                                                         input_ids=input_ids[batch_idx:batch_idx + 1],
                                                         attention_mask=input_mask[batch_idx:batch_idx + 1],
                                                         max_length=opt.target_max_length,
                                                         num_beams=opt.constrained_num_beams,
                                                         strict_forcing=False)

                    prediction = removing_the_tokens_after_eos_and_decode(outputs, tokenizer)[0]

                    print(f"new prediction: {prediction}")

                if prediction not in context and opt.use_constrained_decoding:
                    print(f"Going to regenerate {qid}: {prediction} AGAIN with strict forcing in context!")
                    outputs = constrained_model_generate(context_str=context,
                                                         input_ids=input_ids[batch_idx:batch_idx + 1],
                                                         attention_mask=input_mask[batch_idx:batch_idx + 1],
                                                         max_length=opt.target_max_length,
                                                         num_beams=opt.constrained_num_beams,
                                                         strict_forcing=True)

                    prediction = removing_the_tokens_after_eos_and_decode(outputs, tokenizer)[0]

                    print(f"new prediction: {prediction}")

                pred_answers.append((qid, prediction))
                gold_answers.append((qid, gold_answer))

                answer_rouge_scores = rouge_score(prediction, gold_answer)
                answer_f1_score = f1_score(prediction, gold_answer)

                rougel_list.append(answer_rouge_scores['rouge-l']["f"])

                f1_list.append(answer_f1_score)

                eval_dataloader_tqdm.set_description(f"f1/rougel: {mean(f1_list):.4f}/{mean(rougel_list):.4f}")

    rouge1_list, total = src.util.weighted_average(np.mean(f1_list), total, opt)
    rougel_list, total = src.util.weighted_average(np.mean(rougel_list), total, opt)
    if accelerator.use_distributed:

        pred_answers = src.util.all_gather(pred_answers)
        gold_answers = src.util.all_gather(gold_answers)
        pred_answers = list(itertools.chain.from_iterable(pred_answers))
        gold_answers = list(itertools.chain.from_iterable(gold_answers))

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        pred_answers = {qid: answer for qid, answer in pred_answers}
        gold_answers = {qid: answer for qid, answer in gold_answers}
        assert len(dataset) == len(pred_answers) == len(gold_answers)

        key_list = list(pred_answers.keys())

        answer_rouge_scores = rouge_scores([pred_answers[k] for k in key_list], [gold_answers[k] for k in key_list])
        char_answer_rouge_scores = rouge_scores([" ".join(pred_answers[k]) for k in key_list], [" ".join(gold_answers[k]) for k in key_list])

        answer_f1_score = mean(f1_scores([pred_answers[k] for k in key_list], [gold_answers[k] for k in key_list]))
        answer_em_score = mean(exact_match_scores([pred_answers[k] for k in key_list], [gold_answers[k] for k in key_list]))

        answer_rouge_1_score = mean([s['rouge-1']['f'] for s in answer_rouge_scores])
        answer_rouge_2_score = mean([s['rouge-2']['f'] for s in answer_rouge_scores])
        answer_rouge_l_score = mean([s['rouge-l']['f'] for s in answer_rouge_scores])

        char_answer_rouge_1_score = mean([s['rouge-1']['f'] for s in char_answer_rouge_scores])
        char_answer_rouge_2_score = mean([s['rouge-2']['f'] for s in char_answer_rouge_scores])
        char_answer_rouge_l_score = mean([s['rouge-l']['f'] for s in char_answer_rouge_scores])

        logger.info(f"rouge-1/2/l:\t{answer_rouge_1_score:.4f}/{answer_rouge_2_score:.4f}/{answer_rouge_l_score:.4f}")
        logger.info(f"char-rouge-1/2/l:\t{char_answer_rouge_1_score:.4f}/{char_answer_rouge_2_score:.4f}/{char_answer_rouge_l_score:.4f}")
        logger.info(f"f1/em:\t {answer_f1_score:.4f}/{answer_em_score:.4f}")

        pred_save_path = os.path.join(opt.checkpoint_dir, f'prediction-{step}-bs{opt.num_beams}-cbs{opt.constrained_num_beams}.json' if step is not None else (f'prediction-bs{opt.num_beams}-cbs{opt.constrained_num_beams}-constrained.json' if opt.use_constrained_decoding else f'prediction-bs{opt.num_beams}.json'))

        eval_save_path = os.path.join(opt.checkpoint_dir, f'evaluations-{step}-bs{opt.num_beams}-cbs{opt.constrained_num_beams}.log' if step is not None else 'evaluations.log')

        logger.info(f"Saving predictions at {pred_save_path}")
        with open(pred_save_path, 'w') as f:
            json.dump(pred_answers, f, indent=4, ensure_ascii=False)
        with open(eval_save_path, 'a') as of:
            of.write(pd.DataFrame({
                f'step' if step is not None else 'None': {
                    "rouge-1": answer_rouge_1_score, "rouge-2": answer_rouge_2_score, "rouge-l": answer_rouge_l_score,
                    "char-rouge-1": char_answer_rouge_1_score, "char-rouge-2": char_answer_rouge_2_score, "char-rouge-l": char_answer_rouge_l_score,
                    "f1": answer_f1_score, "em": answer_em_score,
                }
            }).to_string() + "\n")

    accelerator.wait_for_everyone()

    return f1_list, rougel_list


class S2SDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 training=True,
                 source_prefix="",
                 target_prefix="",
                 source_key="question",
                 target_key="answers",
                 ):
        with open(data_path, 'r') as fin:
            self.data_list = json.load(fin)

        # self.data_list = data_list
        self.training = training

        self.source_prefix = source_prefix
        self.target_prefix = target_prefix

        self.source_key = source_key
        self.target_key = target_key

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        example = self.data_list[index]

        source = example[self.source_key]
        target = example[self.target_key][0]

        sample = {
            'index': index,
            'id': example["id"],
            'source': (self.source_prefix + " " + source).strip(),
            'target': (self.target_prefix + " " + target).strip(),
        }
        if "context" in example:
            sample["context"] = example["context"]
        return sample


class Collator(object):
    def __init__(self, tokenizer, source_max_length, target_max_length):
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __call__(self, batch):
        assert (batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]

        tokenized_target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.target_max_length if self.target_max_length > 0 else None,
            # pad_to_max_length=True,
            padding="max_length",
            # return_tensors='pt',
            truncation=True if self.target_max_length > 0 else False,
            verbose=False,
            return_offsets_mapping=True
        )

        target_ids = torch.tensor(tokenized_target.input_ids, dtype=torch.long)
        target_mask = torch.tensor(tokenized_target.attention_mask, dtype=torch.bool)
        target_ids = target_ids.masked_fill(~target_mask, -100)

        source = [ex['source'] for ex in batch]
        tokenized_source = self.tokenizer.batch_encode_plus(
            source,
            max_length=self.source_max_length if self.source_max_length > 0 else None,
            # pad_to_max_length=True,
            padding="max_length",
            # return_tensors='pt',
            truncation=True if self.source_max_length > 0 else False,
            verbose=False,
            return_offsets_mapping=True
        )

        source_ids = torch.tensor(tokenized_source.input_ids, dtype=torch.long)
        source_mask = torch.tensor(tokenized_source.attention_mask, dtype=torch.bool)

        return (index, target_ids, target_mask, source_ids, source_mask)


if __name__ == "__main__":
    options = src.options.Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)


    assert not opt.use_checkpoint, "This version does not support checkpointing"
    mixed_precision = "fp16" if opt.fp16 else None
    accelerator = Accelerator(gradient_accumulation_steps=opt.accumulation_steps, mixed_precision=mixed_precision)
    accelerate.utils.set_seed(opt.seed)

    if accelerator.is_main_process:
        print("-" * 50)
        print(opt)
        print("using fp16:", accelerator.use_fp16)
        print("-" * 50)

    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    accelerator.wait_for_everyone()

    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = src.util.init_logger(
        accelerator.is_main_process,
        accelerator.use_distributed,
        checkpoint_path / 'run.log'
    )
    if accelerator.use_distributed:
        logger.info("Using DDP")

    pretrained_model_path = opt.pretrained_model_path
    config = AutoConfig.from_pretrained(pretrained_model_path)

    if type(config) is BartConfig:
        model_class = BartForConditionalGenerationWrapper
    else:
        model_class = T5ForConditionalGeneration

    # load data
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=True)
    collator = Collator(tokenizer, opt.source_max_length, opt.target_max_length)

    eval_dataset = S2SDataset(opt.eval_data, training=False, source_prefix=opt.source_prefix, target_prefix=opt.target_prefix)
    step = 0
    if not os.path.exists(os.path.join(opt.checkpoint_dir, "pytorch_model.bin")) and opt.train:
        model = model_class.from_pretrained(pretrained_model_path)
        model = model.to(accelerator.device)
    else:
        model = model_class.from_pretrained(opt.checkpoint_dir)
        model = model.to(accelerator.device)
        print(f"model loaded from {os.path.join(opt.checkpoint_dir, 'pytorch_model.bin')}, is training: {opt.train}")
    if opt.train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        if opt.scheduler == "linear":
            # scheduler = src.util.WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=opt.total_steps, min_ratio=0., fixed_lr=opt.fixed_lr)
            opt.warmup_steps = min(opt.warmup_steps, opt.total_steps)
            # TODO: I don't know what's wrong with the accelerate lib, but it need to multiply the steps to get right learning rate.
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_steps * accelerator.num_processes, num_training_steps=opt.total_steps * accelerator.num_processes)

        elif opt.scheduler == 'fixed':

            scheduler = get_constant_schedule(optimizer)
        else:
            assert False, f"Unsupported scheduler name {opt.scheduler}"

        if os.path.exists(os.path.join(opt.checkpoint_dir, "optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, "optimizer.pt")))
            print(f"optimizer loaded from {os.path.join(opt.checkpoint_dir, 'optimizer.pt')}")
        if os.path.exists(os.path.join(opt.checkpoint_dir, "scheduler.pt")):
            step = scheduler.last_epoch
            scheduler.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, "scheduler.pt")))
            print(f"scheduler loaded from {os.path.join(opt.checkpoint_dir, 'scheduler.pt')}")

    if opt.train:

        train_dataset = S2SDataset(opt.train_data, source_prefix=opt.source_prefix, target_prefix=opt.target_prefix, training=True)
        train(model,
              optimizer,
              scheduler,
              step,
              train_dataset,
              eval_dataset,
              opt,
              collator,
              accelerator,
              checkpoint_path)

    else:
        eval(model,
             eval_dataset,
             tokenizer,
             collator,
             opt,
             accelerator)
