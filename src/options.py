import argparse
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--total_steps', type=int, default=1000)
        self.parser.add_argument('--num_epoch', type=int, default=-1)

        self.parser.add_argument('--scheduler_steps', type=int, default=None,
                                 help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--teacher_forcing_rate', type=float, default=0.9, help='TF rate for citation gate')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')
        self.parser.add_argument('--fp16', action='store_true')

    def add_eval_options(self):
        self.parser.add_argument('--write_results', action='store_true', help='save results')
        self.parser.add_argument('--write_crossattention_scores', action='store_true',
                                 help='save dataset with cross-attention scores')
        self.parser.add_argument('--prefix', type=str, default='')
        self.parser.add_argument("--gold_file_path", type=str, default="")
        self.parser.add_argument("--mapping_file", type=str, default="")

    def add_reader_options(self):
        self.parser.add_argument('--train_data', type=str, default='none', help='path of train text data')
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of text data')

        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--pretrained_model_path', type=str, default='')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')

        self.parser.add_argument('--source_max_length', type=int, default=200,
                                 help='maximum number of tokens in text segments (question+passage)')

        self.parser.add_argument('--target_max_length', type=int, default=-1,
                                 help='maximum number of tokens used to train the model, no truncation if -1')

        self.parser.add_argument('--num_sample_sequences', type=int, default=1)
        self.parser.add_argument('--text_loss_weight', type=float, default=1)
        self.parser.add_argument('--baseline_reward_scale', type=float, default=1)

        self.parser.add_argument('--num_beams', type=int, default=4)
        self.parser.add_argument('--constrained_num_beams', type=int, default=4)


        self.parser.add_argument('--rl_algorithm', type=str, default="normalize", choices=["self_critic", "A2C", "normalize"])
        self.parser.add_argument('--reward_name', type=str, default="f1", choices=["f1", "exact_match", "rougel", "exact_match_f1"])
        self.parser.add_argument('--use_constrained_decoding', action='store_true')
        self.parser.add_argument('--min_answer_token_count', type=int, default=0)

        self.parser.add_argument('--no_title', action='store_true', help='article titles not included in passages')

        self.parser.add_argument('--n_context', type=int, default=None)
        self.parser.add_argument('--n_sentence', type=int, default=None)
        self.parser.add_argument('--n_cite_sampling', type=int, default=2)

        self.parser.add_argument('--question_prefix', type=str, default='question:')
        self.parser.add_argument('--passage_prefix', type=str, default='context:')
        self.parser.add_argument('--title_prefix', type=str, default='title:')

        self.parser.add_argument('--source_prefix', type=str, default='')
        self.parser.add_argument('--target_prefix', type=str, default='')

        self.parser.add_argument('--mode', type=str, default="binary", choices=["binary", "em", "transparent", "interpolation"])

        self.parser.add_argument('--fixed_encoder', action='store_true', help='remove the gradient in encoder')
        self.parser.add_argument('--fixed_cite_decoder', action='store_true', help='remove the gradient in encoder')
        self.parser.add_argument('--fixed_text_decoder', action='store_true', help='remove the gradient in encoder')

    def add_retriever_options(self):
        self.parser.add_argument('--train_data', type=str, default='none', help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--indexing_dimension', type=int, default=768)
        self.parser.add_argument('--no_projection', action='store_true',
                                 help='No addition Linear layer and layernorm, only works if indexing size equals 768')
        self.parser.add_argument('--question_maxlength', type=int, default=40,
                                 help='maximum number of tokens in questions')
        self.parser.add_argument('--passage_maxlength', type=int, default=200,
                                 help='maximum number of tokens in passages')
        self.parser.add_argument('--no_question_mask', action='store_true')
        self.parser.add_argument('--no_passage_mask', action='store_true')
        self.parser.add_argument('--extract_cls', action='store_true')
        self.parser.add_argument('--no_title', action='store_true', help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int, default=1)

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default=None, help='path for retraining')

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int,
                                 help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1,
                                 help="For distributed training: local_rank")
        self.parser.add_argument("--is_distributed", action='store_true')
        self.parser.add_argument("--main_port", type=int, default=-1,
                                 help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        # training parameters
        self.parser.add_argument('--eval_freq', type=int, default=500,
                                 help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--save_freq', type=int, default=5000,
                                 help='save model every <save_freq> steps during training')
        self.parser.add_argument('--num_workers', type=int, default=16)
        self.parser.add_argument('--eval_during_training', action='store_true')
        self.parser.add_argument('--train', action='store_true')

        self.parser.add_argument('--eval_output_dir', type=str, default=None)
        self.parser.add_argument('--continue_step', type=int, default=None)

    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir) / opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir / 'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        return opt


def get_options(use_reader=False,
                use_retriever=False,
                use_optim=False,
                use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_retriever:
        options.add_retriever_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()
