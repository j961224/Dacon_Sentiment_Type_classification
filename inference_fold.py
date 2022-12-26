import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
from utils.datasets import Dacon_Dataset
from utils.preprocessor import Preprocessor
from utils.encoder import Encoder
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
    Trainer
)
from args import (MyTrainingArguments, ModelArguments, DataTrainingArguments)
from model.roberta import Multi_label_RobertaForSequenceClassification, Multi_label_Hidden_states_RobertaForSequenceClassification, Multi_label_Heinsen_routing_RobertaForSequenceClassification
from utils.trainer import Multi_label_Trainer


def main():
  parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, MyTrainingArguments)
  )

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model_args, data_args, training_args = parser.parse_args_into_dataclasses()

  dirs = os.listdir(model_args.PLM)
  dirs = sorted(dirs)

  final_type_output_pred = []
  final_polarity_output_pred = []
  final_tense_output_pred = []
  final_certainty_output_pred = []
  for i in range(len(dirs)):
    model_d = os.path.abspath(os.path.join(model_args.PLM, dirs[i]))

    tokenizer = AutoTokenizer.from_pretrained(model_d)

    data_args.data_type = "test"
    config = AutoConfig.from_pretrained(model_d)
    config.special_token_type = data_args.special_token_type

    loader = Dacon_Dataset(data_dir = data_args.data_dir, data_type = data_args.data_type, dataset_type = '', speical_token_use = data_args.special_token_type)

    dset = loader.load_datasets()
    final_prediction_id = dset['test']['sentence_id']
    dset = dset['test']
    print(dset['sentence'][0])

    preprocessor = Preprocessor(train_flag=False, dataset_type='')
    dset = dset.map(preprocessor, batched=True, num_proc=4,remove_columns=dset.column_names)
    print(dset)

    encoder = Encoder(tokenizer, data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)

    model = Multi_label_RobertaForSequenceClassification.from_pretrained(model_d, config=config)
    # model = Multi_label_Heinsen_routing_RobertaForSequenceClassification.from_pretrained(model_args.PLM, config=config)

    Type_label = {'대화형': 0, '추론형': 1, '사실형': 2, '예측형': 3}
    polarity_label = {'부정': 0, '긍정': 1, '미정': 2}
    Tense_label = {'과거': 0, '미래': 1, '현재': 2}
    Certainty_label = {'불확실': 0, '확실': 1}
    Type_label = {v:k for k,v in Type_label.items()}
    polarity_label = {v:k for k,v in polarity_label.items()}
    Tense_label = {v:k for k,v in Tense_label.items()}
    Certainty_label = {v:k for k,v in Certainty_label.items()}
    print(Type_label)
    print(polarity_label)
    print(Tense_label)
    print(Certainty_label)

    trainer = Multi_label_Trainer(                       
      model=model,                         
      args=training_args,                  
      data_collator=data_collator,
    )

    outputs = trainer.predict(dset)
    type_prob = F.softmax(torch.Tensor(outputs[0][0]), dim=-1)
    polarity_prob = F.softmax(torch.Tensor(outputs[0][1]), dim=-1)
    tense_prob = F.softmax(torch.Tensor(outputs[0][2]), dim=-1)
    certainty_prob = F.softmax(torch.Tensor(outputs[0][3]), dim=-1)

    if i==0:
      final_type_output_pred.append(type_prob)
      final_polarity_output_pred.append(polarity_prob)
      final_tense_output_pred.append(tense_prob)
      final_certainty_output_pred.append(certainty_prob)
    else:
      final_type_output_pred[0]+=type_prob
      final_polarity_output_pred[0]+=polarity_prob
      final_tense_output_pred[0]+=tense_prob
      final_certainty_output_pred[0]+=certainty_prob


  final_type_output_pred = np.argmax(final_type_output_pred[0], axis=1)
  final_polarity_output_pred = np.argmax(final_polarity_output_pred[0], axis=1)
  final_tense_output_pred = np.argmax(final_tense_output_pred[0], axis=1)
  final_certainty_output_pred = np.argmax(final_certainty_output_pred[0], axis=1)

  final_prediction = []
  for type_pred, polarity_pred, tense_pred, certainty_pred in zip(final_type_output_pred, final_polarity_output_pred, final_tense_output_pred, final_certainty_output_pred):
    one_predict = Type_label[type_pred.numpy().tolist()] + '-' + polarity_label[polarity_pred.numpy().tolist()] + '-' +Tense_label[tense_pred.numpy().tolist()] + '-' + Certainty_label[certainty_pred.numpy().tolist()]
    final_prediction.append(one_predict)
  
  final_submission = pd.DataFrame(columns = ['ID','label'])
  final_submission['ID'] = final_prediction_id
  final_submission['label'] = final_prediction

  
  if not os.path.exists('./results'):
    os.makedirs('./results')
  submission_save_path = os.path.join('./results',f'submission_predict_ensemble.csv')
  final_submission.to_csv(submission_save_path, index=False)


if __name__ == "__main__" :
    main()