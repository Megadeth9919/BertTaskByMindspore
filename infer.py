from dataset import LoadSQuAD1DatasetForBert
from mindspore import dataset as ds
from model import BertForQuestionAnswering
import mindspore
import gradio as gr


class Infer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def infer_step(self, context, question):
        def yield_data():
            for _ in range(1):
                yield 0, context, question, 0, 0

        infer_data = ds.GeneratorDataset(yield_data,
                                         column_names=["id", "context", "question",
                                                       "answers", "answer_start"])
        data_loader = LoadSQuAD1DatasetForBert(infer_data,
                                               self.config.batch_size,
                                               self.config.max_seq_len,
                                               self.config.doc_stride,
                                               is_infer=True)
        data_loader.data_pre_process()
        data_loader.data_post_process()
        infer_data = data_loader.postprocessed_data
        for data in infer_data.create_dict_iterator():
            input_ids = data['input_ids'].astype(mindspore.int32)
            seg_ids = data['seg_ids'].astype(mindspore.int32)
            label = data['label'].astype(mindspore.int32)
            padding_mask = (input_ids != 0)
            _, start_logits, end_logits = self.model(input_ids=input_ids,
                                                     attention_mask=padding_mask,
                                                     token_type_ids=seg_ids,
                                                     position_ids=None,
                                                     start_positions=label[:, 0],
                                                     end_positions=label[:, 1])
            return f'{start_logits.argmax(1)}'


def inference(config):
    model = BertForQuestionAnswering(config)
    parm_dict = mindspore.load_checkpoint(config.model_checkpoint_path)
    parm_not_load, _ = mindspore.load_param_into_net(model, parm_dict)
    model.set_train(False)
    infer = Infer(config, model)
    demo = gr.Interface(
        fn=infer.infer_step,
        inputs=[gr.Textbox(lines=5, label='context'),
                gr.Textbox(lines=2, label='question')],
        outputs='text',
        flagging_options=None,
    )
    demo.launch()


# if __name__ == '__main__':
    # inference()
