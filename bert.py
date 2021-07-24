from transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss



class BertForSequenceClassificationUserDefined(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        e1_pos=None,
        e2_pos=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        e_pos_outputs = []
        sequence_output = outputs[0]
        for i in range(0, len(e1_pos)):
            e1_pos_output_i = sequence_output[i, e1_pos[i].item(), :]
            e2_pos_output_i = sequence_output[i, e2_pos[i].item(), :]
            e_pos_output_i = torch.cat((e1_pos_output_i, e2_pos_output_i), dim=0)
            e_pos_outputs.append(e_pos_output_i)
        e_pos_output = torch.stack(e_pos_outputs)

        e_pos_output = self.dropout(e_pos_output)
        logits = self.classifier(e_pos_output)

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # outputs = (loss,) + outputs

        # return outputs  # (loss), logits, (hidden_states), (attentions)
        return loss, logits