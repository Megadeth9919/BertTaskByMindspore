import logging
import os

from logs import logger
from train import train
from infer import inference
from mindnlp.models import BertConfig
import dataset

class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.project_dir, '.mindnlp', 'datasets', 'SQuAD1')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.pretrained_model_name = 'bert-base-uncased'
        self.pretrained_model_dir = os.path.join(self.project_dir, '.mindnlp', 'models', self.pretrained_model_name)
        self.dataset_dir = '.mindnlp/datasets/SQuAD1/'
        self.model_checkpoint_path = '.mindnlp/models/checkpoint/'
        self.data_cache_path = '.mindnlp/datasets/cache/'
        self.max_answer_len = 30
        self.max_query_len = 64
        self.max_seq_len = 384
        self.doc_stride = 128
        self.batch_size = 12
        self.learning_rate = 3.5e-5
        self.epochs = 1
        logger.logger_init(log_file_name='qa', log_level=logging.INFO,
                           log_dir=self.logs_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


if __name__ == '__main__':
    config = ModelConfig()
    dataset_train, dataset_eval = dataset.load_squad1_dataset(config)
    train(dataset_train, dataset_eval, config)
    # inference(config)


