import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from dataclasses import dataclass
from transformers.file_utils import ModelOutput

from model.heinsen_routing import Routing




class Multi_label_RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        if self.config.special_token_type:
          x = features
        else:
          x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@dataclass
class Multi_label_SequenceClassifierOutput(ModelOutput):
  loss: Optional[torch.FloatTensor] = None
  Type_logits: torch.FloatTensor = None
  polarity_logits: torch.FloatTensor = None
  Tense_logits: torch.FloatTensor = None
  Certainty_logits: torch.FloatTensor = None
  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  attentions: Optional[Tuple[torch.FloatTensor]] = None



class Multi_label_RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        config.num_labels = 4
        self.Type_label_classifier = Multi_label_RobertaClassificationHead(config)
        config.num_labels = 3
        self.polarity_label_classifier = Multi_label_RobertaClassificationHead(config)
        config.num_labels = 3
        self.Tense_label_classifier = Multi_label_RobertaClassificationHead(config)
        config.num_labels = 2
        self.Certainty_label_classifier = Multi_label_RobertaClassificationHead(config)

        self.type_net = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )

        self.polarity_net = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )

        self.tense_net = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )

        self.certainty_net = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # return_outputs: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.config.special_token_type:
          batch_size, seq_size = input_ids.shape

          cls_flag = input_ids == 0
          type_special_flag = input_ids == self.config.tokenizer_type_token_id
          polarity_special_flag = input_ids == self.config.tokenizer_polarity_token_id
          tense_special_flag = input_ids == self.config.tokenizer_tense_token_id
          certainty_special_flag = input_ids == self.config.tokenizer_certainty_token_id

          type_sequence_output = sequence_output[cls_flag+type_special_flag]
          polarity_sequence_output = sequence_output[cls_flag+polarity_special_flag]
          tense_sequence_output = sequence_output[cls_flag+tense_special_flag]
          certainty_sequence_output = sequence_output[cls_flag+certainty_special_flag]

          type_sequence_output = type_sequence_output.view(batch_size, -1, self.config.hidden_size)
          polarity_sequence_output = polarity_sequence_output.view(batch_size, -1, self.config.hidden_size)
          tense_sequence_output = tense_sequence_output.view(batch_size, -1, self.config.hidden_size)
          certainty_sequence_output = certainty_sequence_output.view(batch_size, -1, self.config.hidden_size)

          type_special_hidden_states = self.type_net(type_sequence_output).view(batch_size,-1)
          polarity_special_hidden_states = self.polarity_net(polarity_sequence_output).view(batch_size,-1)
          tense_special_hidden_states = self.tense_net(tense_sequence_output).view(batch_size,-1)
          certainty_special_hidden_states = self.certainty_net(certainty_sequence_output).view(batch_size,-1)

          Type_logits = self.Type_label_classifier(type_special_hidden_states)
          polarity_logits = self.polarity_label_classifier(polarity_special_hidden_states)
          Tense_logits = self.Tense_label_classifier(tense_special_hidden_states)
          Certainty_logits = self.Certainty_label_classifier(certainty_special_hidden_states)
        
        else:
          Type_logits = self.Type_label_classifier(sequence_output)
          polarity_logits = self.polarity_label_classifier(sequence_output)
          Tense_logits = self.Tense_label_classifier(sequence_output)
          Certainty_logits = self.Certainty_label_classifier(sequence_output)

        loss = None
        if labels is not None:
          type_loss_fct = nn.CrossEntropyLoss()
          polarity_loss_fct = nn.CrossEntropyLoss()
          Tense_loss_fct = nn.CrossEntropyLoss()
          Certainty_loss_fct = nn.CrossEntropyLoss()

          type_loss = type_loss_fct(Type_logits.view(-1, 4), labels[:, 0].view(-1))
          polarity_loss = polarity_loss_fct(polarity_logits.view(-1, 3), labels[:, 1].view(-1))
          Tense_loss = Tense_loss_fct(Tense_logits.view(-1, 3), labels[:, 2].view(-1))
          Certainty_loss = Certainty_loss_fct(Certainty_logits.view(-1, 2), labels[:, 3].view(-1))

          # loss = type_loss*0.25 + polarity_loss*0.25 + Tense_loss*0.25 + Certainty_loss*0.25
          loss = type_loss*(4/12) + polarity_loss*0.25 + Tense_loss*0.25 + Certainty_loss*(2/12)


        if not return_dict:
            output = (Type_logits,polarity_logits,Tense_logits,Certainty_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        

        return Multi_label_SequenceClassifierOutput(
            loss=loss,
            Type_logits=Type_logits,
            polarity_logits=polarity_logits,
            Tense_logits=Tense_logits,
            Certainty_logits=Certainty_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Multi_label_Heinsen_routing_RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        config.num_labels = 4
        self.Type_label_classifier = Routing(self.config, self.config.num_labels)
        config.num_labels = 3
        self.polarity_label_classifier = Routing(self.config, self.config.num_labels)
        config.num_labels = 3
        self.Tense_label_classifier = Routing(self.config, self.config.num_labels)
        config.num_labels = 2
        self.Certainty_label_classifier = Routing(self.config, self.config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # return_outputs: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        Type_logits = self.Type_label_classifier(sequence_output)
        polarity_logits = self.polarity_label_classifier(sequence_output)
        Tense_logits = self.Tense_label_classifier(sequence_output)
        Certainty_logits = self.Certainty_label_classifier(sequence_output)

        loss = None
        if labels is not None:
          type_loss_fct = nn.CrossEntropyLoss()
          polarity_loss_fct = nn.CrossEntropyLoss()
          Tense_loss_fct = nn.CrossEntropyLoss()
          Certainty_loss_fct = nn.CrossEntropyLoss()

          type_loss = type_loss_fct(Type_logits.view(-1, 4), labels[:, 0].view(-1))
          polarity_loss = polarity_loss_fct(polarity_logits.view(-1, 3), labels[:, 1].view(-1))
          Tense_loss = Tense_loss_fct(Tense_logits.view(-1, 3), labels[:, 2].view(-1))
          Certainty_loss = Certainty_loss_fct(Certainty_logits.view(-1, 2), labels[:, 3].view(-1))

          # loss = type_loss*0.25 + polarity_loss*0.25 + Tense_loss*0.25 + Certainty_loss*0.25
          loss = type_loss*(4/12) + polarity_loss*0.25 + Tense_loss*0.25 + Certainty_loss*(2/12)


        if not return_dict:
            output = (Type_logits,polarity_logits,Tense_logits,Certainty_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        

        return Multi_label_SequenceClassifierOutput(
            loss=loss,
            Type_logits=Type_logits,
            polarity_logits=polarity_logits,
            Tense_logits=Tense_logits,
            Certainty_logits=Certainty_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# https://soohee410.github.io/multilabel
class Heinsen_routing_RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.Routing_head = Routing(self.config, self.config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # return_outputs: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.Routing_head(sequence_output)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()

          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Multi_label_Hidden_states_RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*4, config.hidden_size*4)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size*4, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# https://soohee410.github.io/multilabel
class Multi_label_Hidden_states_RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        config.num_labels = 4
        self.Type_label_classifier = Multi_label_Hidden_states_RobertaClassificationHead(config)
        config.num_labels = 3
        self.polarity_label_classifier = Multi_label_Hidden_states_RobertaClassificationHead(config)
        config.num_labels = 3
        self.Tense_label_classifier = Multi_label_Hidden_states_RobertaClassificationHead(config)
        config.num_labels = 2
        self.Certainty_label_classifier = Multi_label_Hidden_states_RobertaClassificationHead(config)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # return_outputs: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        all_hidden_states = torch.stack(outputs['hidden_states'])

        concatenate_pooling = torch.cat(
          (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]),-1
        )

        concatenate_pooling = concatenate_pooling[:, 0]

        Type_logits = self.Type_label_classifier(concatenate_pooling)
        polarity_logits = self.polarity_label_classifier(concatenate_pooling)
        Tense_logits = self.Tense_label_classifier(concatenate_pooling)
        Certainty_logits = self.Certainty_label_classifier(concatenate_pooling)

        outputs.hidden_states = None
        loss = None
        if labels is not None:
          type_loss_fct = nn.CrossEntropyLoss()
          polarity_loss_fct = nn.CrossEntropyLoss()
          Tense_loss_fct = nn.CrossEntropyLoss()
          Certainty_loss_fct = nn.CrossEntropyLoss()

          type_loss = type_loss_fct(Type_logits.view(-1, 4), labels[:, 0].view(-1))
          polarity_loss = polarity_loss_fct(polarity_logits.view(-1, 3), labels[:, 1].view(-1))
          Tense_loss = Tense_loss_fct(Tense_logits.view(-1, 3), labels[:, 2].view(-1))
          Certainty_loss = Certainty_loss_fct(Certainty_logits.view(-1, 2), labels[:, 3].view(-1))

          loss = type_loss*(4/12) + polarity_loss*0.25 + Tense_loss*0.25 + Certainty_loss*(2/12)


        if not return_dict:
            output = (Type_logits,polarity_logits,Tense_logits,Certainty_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        

        return Multi_label_SequenceClassifierOutput(
            loss=loss,
            Type_logits=Type_logits,
            polarity_logits=polarity_logits,
            Tense_logits=Tense_logits,
            Certainty_logits=Certainty_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )