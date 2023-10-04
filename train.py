import collections
import logging

import mindspore.dataset
import torch.nn

from model import BertForQuestionAnswering
from mindnlp._legacy.amp import auto_mixed_precision
from mindnlp.metrics import Accuracy
from mindnlp.engine import Trainer
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindspore import nn
from tqdm import tqdm




def train(train_dataset, test_dataset, config):
    logging.info('##start train##')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    model = auto_mixed_precision(model, 'O1')
    def get_linear_lr(lr, total_step):
        lrs = []
        for current_step in range(total_step):
            factor = max(0.0, 1 - current_step / total_step)
            lrs.append(lr * factor)
        return lrs

    # 自定义线性动态学习率
    linear_lr = get_linear_lr(lr=config.learning_rate,
                              total_step=train_dataset.get_dataset_size() * config.epochs)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=linear_lr)
    def forward_fn(input_ids, seg_ids, padding_mask, label):
        loss, start_logits, end_logits = model(input_ids=input_ids,
                                               attention_mask=padding_mask,
                                               token_type_ids=seg_ids,
                                               position_ids=None,
                                               start_positions=label[:, 0],
                                               end_positions=label[:, 1])
        return loss, start_logits, end_logits

    grad_fn = mindspore.ops.value_and_grad(forward_fn,
                                           None,
                                           optimizer.parameters,
                                           has_aux=True)

    def train_step(input_ids, seg_ids, padding_mask, label):
        (loss, start_logits, end_logits), gradients = grad_fn(input_ids, seg_ids, padding_mask, label)
        optimizer(gradients)
        return loss, start_logits, end_logits

    for epoch in range(config.epochs):
        model.set_train()
        with tqdm(total=train_dataset.get_dataset_size()) as progress:
            progress.set_description(f'Epoch {epoch}')
            loss_total = 0
            for input_ids, label, seg_ids in train_dataset.create_tuple_iterator():
                input_ids = input_ids.astype(mindspore.int32)
                label = label.astype(mindspore.int32)
                seg_ids = seg_ids.astype(mindspore.int32)
                padding_mask = (input_ids != 0)
                loss, start_logits, end_logits = train_step(input_ids, seg_ids, padding_mask, label)
                loss_total += loss
                acc_start = (start_logits.argmax(1) == label[:, 0]).astype(mindspore.float32).mean()
                acc_end = (end_logits.argmax(1) == label[:, 1]).astype(mindspore.float32).mean()
                acc = (acc_start + acc_end) / 2
                progress.set_postfix(train_loss=loss, train_acc=acc)
                progress.update(1)
        progress.close()
        mindspore.save_checkpoint(model, config.model_checkpoint_dir+f'epoch_{epoch}')

def evaluate(eval_dataset, model, inference=False):
    model.set_train(model=False)
    acc_sum, n = 0.0, 0
    all_results = collections.defaultdict(list)
    for data in eval_dataset.create_dict_iterator():
        input_ids = data['input_ide'].astype(mindspore.int64)
        seg_ids = data['seg_ids'].astype(mindspore.int64)
        label = data['label'].astype(mindspore.int32)
        padding_mask = (input_ids != 0)
        start_logits, end_logits = model(input_ids=input_ids,
                                         attention_mask=padding_mask,
                                         token_type_ids=seg_ids,
                                         position_ids=None)
        if not inference:
            acc_sum_start = (start_logits == label[:, 0]).astype(mindspore.float32).item()
            acc_sum_end = (end_logits.argmax(1) == label[:, 1]).astype(mindspore.float32).item()
            acc_sum += (acc_sum_start, acc_sum_end)
            n += len(label)
        model.set_train()
        return acc_sum / (2 * n)

