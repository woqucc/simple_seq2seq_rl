import os
import errno
import pickle

import torch
import sys
import logging
import json
from pathlib import Path
import torch.distributed as dist
import csv

from torch.distributed import get_world_size

logger = logging.getLogger(__name__)


def init_logger(is_main=True, is_distributed=False, filename=None, level=logging.INFO):
    if is_distributed:
        torch.distributed.barrier()

    logging.root.setLevel(logging.NOTSET if is_main else logging.WARN)
    logger = logging.getLogger()
    formatter = logging.Formatter("[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    return logger


def get_checkpoint_path(opt):
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, checkpoint_exists


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, opt, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path)
    epoch_path = os.path.join(path, name)  # "step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    model_to_save.save_pretrained(epoch_path)

    torch.save(opt, os.path.join(epoch_path, "training_args.bin"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(epoch_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(epoch_path, "scheduler.pt"))
    print("Model checkpoint saved to", epoch_path)


def load(model_class, dir_path, opt, reset_params=False, **kwargs):
    epoch_path = os.path.realpath(dir_path)
    logger.info("Loading %s" % epoch_path)
    if str(model_class) == "<class '__main__.FiDBartDualDecoder'>":
        if "from_pretrained_params" in kwargs:
            model = model_class.from_pretrained(epoch_path, **kwargs["from_pretrained_params"])
    else:
        model = model_class.from_pretrained(epoch_path)
    model = model.to(opt.device)
    opt_checkpoint, step, best_eval_metric = None, None, None

    if os.path.exists(os.path.join(epoch_path, "optimizer.pth.tar")):
        optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")

        logger.info("loading checkpoint %s" % optimizer_path)
        checkpoint = torch.load(optimizer_path, map_location=opt.device)
        opt_checkpoint = checkpoint["opt"]
        step = checkpoint["step"]
        if "best_eval_metric" in checkpoint:
            best_eval_metric = checkpoint["best_eval_metric"]
        else:
            best_eval_metric = checkpoint["best_dev_em"]
        if not reset_params:
            optimizer, scheduler = set_optim(opt_checkpoint, model)
            scheduler.load_state_dict(checkpoint["scheduler"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            optimizer, scheduler = set_optim(opt, model)

    elif os.path.exists(os.path.join(epoch_path, "optimizer.pt")):
        optimizer, scheduler = set_optim(opt, model)
        optimizer.load_state_dict(torch.load(os.path.join(epoch_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(epoch_path, "scheduler.pt")))


    else:
        optimizer, scheduler = set_optim(opt, model)

    if step is None and os.path.basename(dir_path).startswith("checkpoint-"):
        step = int(os.path.basename(dir_path).replace("checkpoint-", ""))
    return model, optimizer, scheduler, opt_checkpoint, step, best_eval_metric


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
                   1.0 + (self.min_ratio - 1) * (step - self.warmup_steps) / float(max(1.0, self.scheduler_steps - self.warmup_steps)),
                   )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return 1.0


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def set_optim(opt, model):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        if opt.scheduler_steps is None:
            scheduler_steps = opt.total_steps
        else:
            scheduler_steps = opt.scheduler_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=scheduler_steps, min_ratio=0., fixed_lr=opt.fixed_lr)
    return optimizer, scheduler


def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
    return x


def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def max_all(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return x


def weighted_average(x, count, opt):
    if not opt.is_distributed:
        return x, count
    t_loss = torch.tensor([x * count], device=opt.device)
    t_total = torch.tensor([count], device=opt.device)
    t_loss = sum_main(t_loss, opt)
    t_total = sum_main(t_total, opt)
    return (t_loss / t_total).item(), t_total.item()


def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    with open(output_path, 'w') as outfile:
        for path in files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def save_distributed_dataset(data, opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    write_path = dir_path / 'tmp_dir'
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f'{opt.global_rank}.json'
    with open(tmp_path, 'w') as fw:
        json.dump(data, fw)
    if opt.is_distributed:
        torch.distributed.barrier()
    if opt.is_main:
        final_path = dir_path / 'dataset_wscores.json'
        logger.info(f'Writing dataset with scores at {final_path}')
        glob_path = write_path / '*'
        results_path = write_path.glob('*.json')
        alldata = []
        for path in results_path:
            with open(path, 'r') as f:
                data = json.load(f)
            alldata.extend(data)
            path.unlink()
        with open(final_path, 'w') as fout:
            json.dump(alldata, fout, indent=4)
        write_path.rmdir()


def load_passages(path):
    if not os.path.exists(path):
        logger.info(f'{path} does not exist')
        return
    logger.info(f'Loading passages from: {path}')
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    passages.append((row[0], row[1], row[2]))
                except:
                    logger.warning(f'The following input line has not been correctly loaded: {row}')
    return passages


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
