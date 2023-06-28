# simple_seq2seq_rl

Here we put the source code that can be used to reproduce the works in the paper "Modeling Extractive Question Answering Using Encoder-Decoder Models with Constrained Decoding and Evaluation-Based Reinforcement Learning." The code is based on the project in  https://github.com/facebookresearch/FiD, thanks to the authors of FiD.


## Training 

`seq2seq_rl.py` is a simple implementation of the RL-enhanced sequence-to-sequence model. The RL enable the model to be supervised by the pre-defined rewards. In the experiments, we use some overlap-based evaluation metrics for MRC as the rewards, such as EM, F1, and ROUGE-L.

Train on single GPU (24GB memory), batch_size = 16 (per_gpu_batch_size) * 2 (accumulation_steps) 
```
python -m accelerate.commands.launch --mixed_precision fp16 seq2seq_rl.py --train_data data/squad_train.json --eval_data data/squad_validation.json --pretrained_model_path /DIR_OF_FACEBOOK_BART_BASE_MODEL/facebook_bart-base --per_gpu_batch_size 16 --accumulation_steps 2 --checkpoint_dir OUTPUT_PATH --lr 5e-5 --optim adamw --scheduler linear --weight_decay 0.01 --source_max_length 384 --target_max_length 16 --total_steps 100000 --eval_freq 10000 --save_freq 10000 --num_workers 0 --train --eval_during_training --warmup_steps 500 --num_sample_sequences 16 --text_loss_weight 1 --baseline_reward_scale 1 --num_sample_sequences 4 --reward_name f1
```

There are two extra parameters other than the conventional hyper-parameters in sequence-to-sequence models:

- num_sample_sequences is the number of the sequences sampled from the decoder to calculate the rewards against the baseline sequence.
- reward_name denotes the reward function. The choices are: `rougel`, `f1`, `exact_match`, and `exact_match_f1`. 

## Evaluation

`seq2seq.py` implements the constrained decoding for the sequence-to-sequence text generation model. The constrained decoding makes each generation a substring of the context. Use the flag `--use_constrained_decoding` to turn on the constrained decoding. `constrained_num_beams` is the beam size in constrained decoding.

Evaluate the checkpoints:
```
python seq2seq.py --eval_data data/squad_validation.json --pretrained_model_path /home/li-shao-bo/models/facebook_bart-base --per_gpu_batch_size 1 --checkpoint_dir OUTPUT_PATH/checkpoint-100000 --source_max_length 384 --target_max_length 16 --num_workers 0 --min_answer_token_count 0 --num_beams 4 --use_constrained_decoding --constrained_num_beams 4 
```

The currently constrained decoding is slow, so it is only activated when the normal decoding gives unexpected results (i.e., the generation answer span does not exist in the context).
