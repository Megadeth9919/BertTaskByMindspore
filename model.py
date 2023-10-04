from mindnlp.models.bert.bert import BertPreTrainedModel
from mindnlp.models import BertModel
from mindspore import nn, ops

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, 2)

    def construct(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                start_positions=None,
                end_positions=None):
        outputs = self.bert(
           input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids,
           position_ids=position_ids)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(axis=-1, output_num=2)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if end_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = ops.clip_by_value(start_positions, 0, ignored_index)
            end_positions = ops.clip_by_value(end_positions, 0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs