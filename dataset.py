import mindspore
from mindnlp.dataset.question_answer.squad1 import SQuAD1
from mindnlp.transforms.tokenizers import BertTokenizer
from mindnlp.transforms import PadTransform
from mindspore import dataset as ds
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from logs import logger
import logging
import os


class LoadSQuAD1DatasetForBert:
    def __init__(self,
                 raw_data,
                 batch_size=None,
                 max_seq_len=384,
                 doc_stride=128,
                 **kwargs):
        self.raw_data = raw_data
        self.preprocessed_data = None
        self.postprocessed_data = None
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        self.pad_ids = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.cls_ids = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_ids = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.is_train = kwargs.pop('is_train', True)
        self.is_infer = kwargs.pop('is_infer', False)

    def _get_answer_start_end(self, context, answers):
        start_idx = None
        end_idx = None
        for i in range(len(context) - len(answers) + 1):
            if context[i] != answers[0]:
                continue
            for j in range(len(answers)):
                if answers[j] != context[i + j]:
                    break
                start_idx = i
                end_idx = i + j
            if end_idx - i + 1 == len(answers):
                return context, answers, i, end_idx
        if start_idx is None or end_idx is None:
            start_idx, end_idx = -1, -1
        return context, answers, start_idx, end_idx

    def _delete_cls_sep(self, token_ids):
        return token_ids[1:-1]

    def _slide(self, context_ids, question_ids_len):
        slided_data = []
        rest_len = self.max_seq_len - question_ids_len - 1
        context_ids_len = len(context_ids)
        context_ids = context_ids.astype(mindspore.int64)
        # logging.debug(f'文本长度为{context_ids_len}, 剩余长度为{rest_len}')
        if context_ids_len > rest_len:
            # logging.debug('\n##进入滑动窗口##')
            start_pos, end_pos = 0, rest_len
            while True:
                # logging.debug(f'窗口起始位置：{start_pos}, 结束位置：{end_pos - 1}')
                ids_temp = context_ids[start_pos:end_pos]
                slided_data.append([ids_temp, start_pos, end_pos - 1])
                if end_pos >= context_ids_len:
                    break
                start_pos += self.doc_stride
                end_pos += self.doc_stride
                if end_pos >= context_ids_len:
                    end_pos = context_ids_len
        else:
            slided_data.append([context_ids, 0, context_ids_len-1])
        return slided_data

    def data_pre_process(self):
        if self.raw_data is None:
            raise ValueError('原始数据不存在')
        data = self.raw_data
        data = data.rename(input_columns=['answer_start'], output_columns=['sb'])
        data = data.map(operations=[self.tokenizer, self._delete_cls_sep],
                        input_columns=["context"])
        data = data.map(operations=[self.tokenizer],
                        input_columns=["question"])
        data = data.map(operations=[self.tokenizer, self._delete_cls_sep],
                        input_columns=["answers"])

        data = data.map(operations=[self._get_answer_start_end],
                        input_columns=["context", "answers"],
                        output_columns=["context", "answers", "answer_start", "answer_end"],
                        column_order=["id", "context", "question", "answers",
                                      "answer_start", "answer_end"])
        self.preprocessed_data = data

    def data_slide_and_yield(self):
        for row in self.preprocessed_data.create_dict_iterator():
            question_ids = row['question'].asnumpy()  # question_ids = [CLS] + question + [SEP]
            question_ids_len = len(row['question'])
            answer_start, answer_end = int(row['answer_start'].asnumpy()), \
                int(row['answer_end'].asnumpy())
            slided_data = self._slide(row['context'], question_ids_len)
            for slided_context, slided_start, slided_end in slided_data:
                new_start_pos, new_end_pos = 0, 0
                slided_context = np.concatenate((slided_context.asnumpy(),
                                                 np.array([self.sep_ids])), axis=0)
                input_ids = np.concatenate((question_ids, slided_context), axis=0)
                seg_ids = np.concatenate((np.zeros(shape=(question_ids_len,)),
                                          np.ones(shape=(self.max_seq_len - question_ids_len,))),
                                         axis=0)
                # if self.is_infer:

                if slided_start <= answer_start and answer_end <= slided_end:
                    # logging.debug(f'滑动窗口存在答案，窗口为：{slided_start}-{slided_end}')
                    new_start_pos = answer_start - slided_start
                    new_end_pos = answer_end - slided_start
                    new_start_pos += question_ids_len
                    new_end_pos += question_ids_len
                else:
                    # logging.debug(f'该窗口不存在答案，窗口为：{slided_start}-{slided_end}，'
                    #               f'答案为{answer_start}-{answer_end}')
                    pass

                label_ids = np.array([new_start_pos, new_end_pos])
                yield input_ids, seg_ids, label_ids #,row['answers']

    def data_post_process(self):
        if self.preprocessed_data is None:
            raise ValueError('数据未预处理，请先进行预处理')
        data = ds.GeneratorDataset(
            source=self.data_slide_and_yield,
            column_names=["input_ids", "seg_ids", "label", "answers"])
        pad_op = PadTransform(self.max_seq_len,
                              pad_value=self.tokenizer.token_to_id('[PAD]'))
        data = data.map(operations=[pad_op], input_columns=['input_ids'])
        if not self.is_infer:
            data = data.batch(self.batch_size, drop_remainder=True)
        self.postprocessed_data = data


def cache(func):
    """
      本修饰器的作用是将Sload_squad1_dataset方法处理后的结果进行缓存，下次使用时可直接载入！
    """
    def wrapper(*args, **kwargs):
        config, is_train = kwargs['config'], kwargs['is_train']
        cache_path = config.data_cache_path
        data_path = cache_path + ('train' if is_train else 'dev')
        if not os.path.exists(data_path):
            logging.info(f'缓存文件{data_path} 不存在，重新处理并缓存！')
            data = func(*args, **kwargs)
            data.save(data_path)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            data = ds.MindDataset(dataset_files=data_path)
        return data

    return wrapper


@cache
def get_new_data(raw_data, config, is_train):
    if is_train:
        logging.info(f'处理原始训练数据集，数据集大小{raw_data.get_dataset_size()}')
    else:
        logging.info(f'处理原始测试数据集，数据集大小{raw_data.get_dataset_size()}')
    squad1_loader = LoadSQuAD1DatasetForBert(raw_data,
                                             config.batch_size,
                                             config.max_seq_len,
                                             config.doc_stride,
                                             is_train=is_train)
    squad1_loader.data_pre_process()
    squad1_loader.data_post_process()
    new_data = squad1_loader.postprocessed_data
    for _ in tqdm(new_data.create_dict_iterator(),
                     desc='正在遍历每个样本（问题)'):
        pass
    logging.info(f'new_data_size: {new_data.get_dataset_size()}')
    return new_data


def load_squad1_dataset(config):
    logging.info('##正在加载数据集##')
    raw_dataset_train, raw_dataset_test = SQuAD1(config.dataset_dir)
    new_dataset_train = get_new_data(raw_data=raw_dataset_train,
                                     config=config, is_train=True)
    new_dataset_eval = get_new_data(raw_data=raw_dataset_test,
                                    config=config, is_train=False)
    logging.info(f'处理后训练数据集大小：{new_dataset_train.get_dataset_size()}')
    logging.info(f'处理后验证数据集大小：{new_dataset_eval.get_dataset_size()}')
    return new_dataset_train, new_dataset_eval

if __name__ == '__main__':
    """
    检查数据集是否正确加载
    """
    import transformers
    logger.logger_init(log_level=logging.INFO)
    config = EasyDict(
        {
            'dataset_dir': '.mindnlp/datasets/SQuAD1',
            'data_cache_path': '.mindnlp/datasets/cache/',
            # 'batch_size': 12,
            # 'max_seq_len': 128,
            # 'doc_stride': 64
        }
    )
    train_dataset, test_dataset = load_squad1_dataset(config)
    print(train_dataset.get_col_names())
    np.set_printoptions(threshold=np.inf)
    tokenizer = transformers.BertTokenizer.from_pretrained('D:\\PyCharm 2023.1.3\\PyCharm_Projects\\BertForQuestionAnswerByMindspore\\.mindnlp\models\\bert-base-uncased')
    i = 0

    def show_data(index):
        print(data['input_ids'].shape)
        print(data['input_ids'][index].asnumpy())
        print(tokenizer.convert_ids_to_tokens(data['input_ids'][index].asnumpy()))
        print(tokenizer.decode(data['input_ids'][index].asnumpy()))
        # print(tokenizer.convert_ids_to_tokens(data['answers'][0].asnumpy()))
        # print(data['seg_ids'].shape)
        # print(data['seg_ids'].asnumpy())
        # print(data['answers'])
        # print(data['seg_ids'])
        print(data['label'][index])

    for data in tqdm(train_dataset.create_dict_iterator(), ncols=80,
                     desc='正在遍历每个样本（问题)',
                     total=train_dataset.get_dataset_size()):
            for j in range(12):
                if data['label'][j, 0] == 0:
                    assert j > 0 and j < 11
                    show_data(j-1)
                    show_data(j)
                    show_data(j+1)
                    assert 0

