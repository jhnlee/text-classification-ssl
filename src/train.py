import argparse
import glob
import logging
import os
import pickle
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    BertForSequenceClassification,
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from data_utils import AugmentedDataset
from utils import (
    rotate_checkpoints,
    mask_tokens,
    set_seed,
    load_and_cache_examples,
    get_tsa_threshold,
    ResultWriter,
    random_swap,
    random_deletion,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
}


def train(
    args,
    labeled_dataset,
    unlabeled_dataset,
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    eval_dataset=None,
) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    def collate(data):
        """
        Dimension of each elements in minibatch
        sentence(tensor) : N x L
        augmented sentences(tensor) : N x L
        labels(tensor) : N
        """
        sentences, aug_sentences, labels, sentence_logit, = list(zip(*data))
        return (
            pad_sequence(sentences, batch_first=True, padding_value=tokenizer.pad_token_id),
            pad_sequence(aug_sentences, batch_first=True, padding_value=tokenizer.pad_token_id),
            torch.tensor(labels),
            torch.stack(sentence_logit, 0),
        )

    # labeled_sampler = SubsetRandomSampler(labeled_idx)
    # unlabeled_sampler = SubsetRandomSampler(unlabeled_idx)
    labeled_sampler = (
        RandomSampler(labeled_dataset)
        if args.local_rank == -1
        else DistributedSampler(labeled_dataset)
    )
    unlabeled_sampler = (
        RandomSampler(unlabeled_dataset)
        if args.local_rank == -1
        else DistributedSampler(unlabeled_dataset)
    )

    labeled_dataloader = DataLoader(
        labeled_dataset,
        sampler=labeled_sampler,
        batch_size=args.labeled_batch_size,
        collate_fn=collate,
    )
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset,
        sampler=unlabeled_sampler,
        batch_size=args.unlabeled_batch_size,
        collate_fn=collate,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (len(labeled_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = (
            len(labeled_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        )

    args.warmup_steps = int(args.warmup_percent * t_total)
    if args.do_sgd == 1:
        optimizer = SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.sgd_momentum,
            nesterov=True,
        )
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=args.adam_epsilon)
        # optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.99), eps=args.adam_epsilon)

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    # Loss function for crossentropy & consistency loss
    sup_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    unsup_loss_fct = torch.nn.KLDivLoss(reduction="none")

    # Train!
    logger.info("***** Running Consistency training *****")
    logger.info("  Num labeled examples = %d", len(labeled_dataset))
    logger.info("  Num unlabeled examples = %d", len(unlabeled_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Labeled batch size = %d",
        args.labeled_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info(
        "  Unlabeled batch size = %d",
        args.unlabeled_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    best_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_val_loss = 1e10
    best_val_acc = 0

    model.zero_grad()

    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)

    unlabeled_iter = iter(unlabeled_dataloader)

    stop_iter = False

    for _ in train_iterator:
        # epoch_iterator = tqdm(labeled_dataloader, desc="Iteration")
        if stop_iter == True:
            break

        for step, labeled_batch in enumerate(labeled_dataloader):

            (labeled_inputs, _, label, _) = labeled_batch
            try:
                (unlabeled_inputs, unlabeled_aug_inputs, _, sentence_logit) = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_dataloader)
                (unlabeled_inputs, unlabeled_aug_inputs, _, sentence_logit) = next(unlabeled_iter)

            (labeled_inputs, label, unlabeled_inputs, unlabeled_aug_inputs, sentence_logit,) = [
                x.to(args.device)
                for x in (
                    labeled_inputs,
                    label,
                    unlabeled_inputs,
                    unlabeled_aug_inputs,
                    sentence_logit,
                )
            ]

            model.train()
            # Supervised loss(CrossEntropy)
            logit_sup = model(input_ids=labeled_inputs)[0]
            loss_sup = sup_loss_fct(logit_sup, label)
            if args.do_tsa:
                tsa_threshold = get_tsa_threshold(
                    schedule=args.tsa,
                    global_step=global_step,
                    num_train_steps=t_total,
                    start=1.0 / logit_sup.shape[-1],
                    end=1,
                )
                larger_than_threshold = torch.exp(-loss_sup) > tsa_threshold
                loss_sup_mask = torch.ones_like(label, dtype=torch.float32) * ~larger_than_threshold
                loss = (loss_sup * loss_sup_mask).sum() / torch.max(
                    loss_sup_mask.sum(), torch.tensor(1.0).to(args.device)
                )
            else:
                loss = loss_sup.mean()

            # logger.info("sup loss = %.3f", loss.item())

            # Unsupervised loss(KL Divergence)
            if args.do_unsup:

                # ramp up unsupervised loss coefficient
                unsup_coef = (global_step / t_total) * args.unsup_coefficient
                unsup_coef = args.unsup_coefficient
                # logger.info("coef = %.3f", unsup_coef)

                if args.do_kldiv:
                    with torch.no_grad():
                        model.eval()
                        softmax_temp = args.softmax_temp if args.softmax_temp > 0 else 1.0
                        output_unlabeled = model(input_ids=unlabeled_inputs)[0]
                        output_unlabeled = F.softmax(output_unlabeled / softmax_temp, dim=-1)

                        # confidence-based data filtering
                        if args.confidence_threshold != -1:
                            loss_unsup_mask = (
                                torch.max(output_unlabeled, dim=-1)[0] > args.confidence_threshold
                            ).to(torch.float32)
                        else:
                            loss_unsup_mask = torch.ones_like(output_unlabeled, dtype=torch.bool)
                        loss_unsup_mask = loss_unsup_mask.to(args.device)

                    model.train()
                    output_unlabeled_aug = model(input_ids=unlabeled_aug_inputs)[0]
                    output_unlabeled_aug = F.log_softmax(
                        model(input_ids=unlabeled_aug_inputs)[0], dim=-1
                    )
                    loss_unsup = unsup_loss_fct(output_unlabeled_aug, output_unlabeled).sum(-1)

                    # (1) Divied into the length (the way original uda paper did)
                    # loss_unsup = (loss_unsup * loss_unsup_mask).mean()

                    # (2) Divide into the number of unmasked values
                    loss_unsup = (loss_unsup * loss_unsup_mask).sum() / torch.max(
                        loss_unsup_mask.sum(), torch.tensor(1.0).to(args.device)
                    )
                elif args.do_pseudo:
                    with torch.no_grad():
                        model.eval()
                        output_unlabeled = model(input_ids=unlabeled_inputs)[0].softmax(dim=-1)
                        confidence = output_unlabeled.max(dim=-1)[0]
                        pseudo_label = output_unlabeled.argmax(dim=-1)
                        conf_threshold = 0.8 if args.do_tsa else 0.95
                        conf_mask = (confidence > conf_threshold).to(torch.float32)

                    model.train()
                    output_unlabeled_aug = model(input_ids=unlabeled_aug_inputs)[0]
                    loss_unsup = sup_loss_fct(output_unlabeled_aug, pseudo_label)
                    loss_unsup = (loss_unsup * conf_mask).sum() / torch.max(
                        conf_mask.sum(), torch.tensor(1.0).to(args.device)
                    )

                # logger.info("unsup loss = %.3f", loss_unsup.item())
                loss += unsup_coef * loss_unsup

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    results = evaluate(args, model, tokenizer, eval_dataset)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / args.logging_steps, global_step
                    )
                    logging_loss = tr_loss

                    if results["loss"] > 3:
                        logger.info(" Evaluation metric diverges. Exit Training ")
                        stop_iter = True
                        break

                    # if results["loss"] < best_val_loss:
                    if results["accuracy"] > best_val_acc:
                        best_val_acc = results["accuracy"]
                        best_val_loss = results["loss"]
                        best_step = global_step
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                    elif results["accuracy"] == best_val_acc:
                        if results["loss"] < best_val_loss:
                            best_val_acc = results["accuracy"]
                            best_val_loss = results["loss"]
                            best_step = global_step
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(args.output_dir)
                            tokenizer.save_pretrained(args.output_dir)
                            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                    logger.info("*** best acc : %s ***", best_val_acc)
                    logger.info("*** best loss : %s ***", best_val_loss)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_val_acc, best_val_loss, best_step


def evaluate(
    args, model: BertForSequenceClassification, tokenizer: BertTokenizer, datasets, prefix=""
) -> Dict:
    eval_output_dir = args.output_dir
    if not os.path.isdir(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    def collate(data):
        sentences, labels = list(zip(*data))
        return (
            pad_sequence(sentences, batch_first=True, padding_value=tokenizer.pad_token_id),
            torch.tensor(labels),
        )

    eval_sampler = SequentialSampler(datasets)
    eval_dataloader = DataLoader(
        datasets, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )
    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(datasets))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    cls_preds = None

    total_preds = []
    total_class = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        inputs, class_labels = batch

        inputs = inputs.to(args.device)
        class_labels = class_labels.to(args.device)

        with torch.no_grad():
            outputs = model(input_ids=inputs, labels=class_labels)
            loss, cls_scores = outputs[:2]
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

        cls_preds = torch.softmax(cls_scores, dim=1).detach().cpu().argmax(axis=1)
        class_labels = class_labels.detach().cpu()

        total_preds.append(cls_preds)
        total_class.append(class_labels)

    total_preds = torch.cat(total_preds)
    total_class = torch.cat(total_class)

    eval_loss = eval_loss / nb_eval_steps
    result = {
        "loss": eval_loss,
        "accuracy": (total_preds == total_class).sum().item() / len(total_preds),
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file) which should be labeled.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--experiments_dir", type=str, required=True, help="The experiment directory for test.",
    )
    parser.add_argument(
        "--num_labeled_examples", type=int, required=True, help="The number of labeled examples.",
    )
    parser.add_argument(
        "--num_augmentation", default=9, type=int, help="Number of augmentation.",
    )
    parser.add_argument(
        "--mlm_prob", default=0.15, type=float, help="MLM probability.",
    )
    parser.add_argument(
        "--num_mlm_repeat", default=5, type=int, help="Num repeat of mlm for each sentence.",
    )
    parser.add_argument(
        "--train_max_len", default=128, type=int, help="Maximum sequence length.",
    )
    parser.add_argument(
        "--eval_max_len", default=512, type=int, help="Maximum sequence length.",
    )
    parser.add_argument(
        "--num_labels", default=2, type=int, help="Number of class labels.",
    )

    # Other parameters
    parser.add_argument("--use_bt", action="store_true", help="Use back-translated data")
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--test_data_file", default=None, type=str, help="test data file path.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--pretrained_mlm_dir", default=None, type=str, help="The pretrained mlm model checkpoint.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--labeled_batch_size",
        default=4,
        type=int,
        help="Batch size of labeled data for training.",
    )
    parser.add_argument(
        "--do_unsup", action="store_true", help="Whether to do consistency training.",
    )
    parser.add_argument(
        "--do_kldiv",
        action="store_true",
        help="Whether to use kl divergence loss for consistency training.",
    )
    parser.add_argument(
        "--do_pseudo",
        action="store_true",
        help="Whether to use pseudo label loss for consistency training.",
    )
    parser.add_argument(
        "--unsup_ratio",
        default=3,
        type=float,
        help="The batch size for the unsupervised loss is unsup_ratio * train_batch_size.",
    )
    parser.add_argument(
        "--do_tsa", action="store_true", help="Whether to apply training signal annealing.",
    )
    parser.add_argument(
        "--tsa", default="linear_schedule", type=str, help="Training signal annealing.",
    )
    parser.add_argument(
        "--softmax_temp",
        default=0.85,
        type=float,
        help="Softmax temperature for unsupervised loss.",
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.45,
        type=float,
        help="Confidence thresholds for unsupervised loss.",
    )
    parser.add_argument(
        "--unsup_coefficient", default=1.0, type=float, help="Coefficient for unsupervised loss.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=128,
        type=int,
        help="Batch size of labeled data for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--p_random_swap",
        default=0.1,
        type=float,
        help="Percentage of Random swap in the sentence.",
    )
    parser.add_argument(
        "--p_random_deletion",
        default=0.1,
        type=float,
        help="Percentage of Random deletion in the sentence.",
    )
    parser.add_argument("--do_sgd", default=1, type=int, help="Whether to use sgd.")
    parser.add_argument("--sgd_momentum", default=0.9, type=float, help="Momentum for sgd.")
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_percent",
        default=0.1,
        type=float,
        help="Percentage of linear warmup over warmup_steps.",
    )
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=50, help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    args = parser.parse_args()

    if args.eval_data_file is None:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.labeled_batch_size = args.labeled_batch_size // torch.distributed.get_world_size()
        args.n_gpu = 1
    args.device = device

    args.unlabeled_batch_size = int(args.labeled_batch_size * args.unsup_ratio)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    now = datetime.now()
    date = "%s-%s %s:%s" % (now.month, now.day, now.hour, now.minute)
    log_dir = os.path.join(args.output_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filehandler = logging.FileHandler(os.path.join(args.output_dir, "log/" + date + ".log"))
    logger.addHandler(filehandler)

    # Set seed
    set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model_path = args.pretrained_mlm_dir
    model = BertForSequenceClassification.from_pretrained(model_path, config=config).to(args.device)
    labeled_dataset = AugmentedDataset(
        tokenizer,
        args.train_data_file,
        args.num_labeled_examples,
        args.num_augmentation,
        args.mlm_prob,
        args.num_mlm_repeat,
        args.train_max_len,
        args.use_bt,
    )
    labeled_dataset.aug_sentence, labeled_dataset.sentence, labeled_dataset.label = [
        x[: args.num_labeled_examples]
        for x in (labeled_dataset.aug_sentence, labeled_dataset.sentence, labeled_dataset.label)
    ]
    unlabeled_dataset = AugmentedDataset(
        tokenizer,
        args.train_data_file,
        args.num_labeled_examples,
        args.num_augmentation,
        args.mlm_prob,
        args.num_mlm_repeat,
        args.train_max_len,
        args.use_bt,
    )
    unlabeled_dataset.aug_sentence, unlabeled_dataset.sentence, unlabeled_dataset.label = [
        x[args.num_labeled_examples :]
        for x in (
            unlabeled_dataset.aug_sentence,
            unlabeled_dataset.sentence,
            unlabeled_dataset.label,
        )
    ]
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    eval_dataset.max_len = args.eval_max_len

    if args.local_rank == 0:
        torch.distributed.barrier()

    global_step, tr_loss, best_val_acc, best_val_loss, best_step = train(
        args, labeled_dataset, unlabeled_dataset, model, tokenizer, eval_dataset
    )

    logger.info(
        " global_step = %s, average loss = %s, best_acc = %s, best_loss = %s, best_step = %s",
        global_step,
        tr_loss,
        best_val_acc,
        best_val_loss,
        best_step,
    )
    if args.local_rank in [-1, 0]:
        # Test
        writer = ResultWriter(args.experiments_dir)
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        args.eval_data_file = args.test_data_file
        test_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
        test_dataset.max_len = args.eval_max_len
        results = evaluate(args, model, tokenizer, test_dataset)
        results.update(
            {
                "best_step": best_step,
                "num_labeled": args.num_labeled_examples,
                "mlm_prob": args.mlm_prob,
                "mlm_repeat": args.num_mlm_repeat,
                "unsup_ratio": args.unsup_ratio,
                "confidence_threshold": args.confidence_threshold,
                "softmax_temp": args.softmax_temp,
            }
        )
        writer.update(args, **results)
    else:
        writer = ResultWriter(args.experiments_dir)
        results = {}
        results.update(
            {"best_step": best_step, "loss": 3, "accuracy": 0.5,}
        )
        writer.update(args, **results)


if __name__ == "__main__":
    main()
