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

  tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)

  data_args.data_type = "test"
  config = AutoConfig.from_pretrained(model_args.PLM)
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

  model = Multi_label_RobertaForSequenceClassification.from_pretrained(model_args.PLM, config=config)
  # model = Multi_label_Heinsen_routing_RobertaForSequenceClassification.from_pretrained(model_args.PLM, config=config)

  data_df = pd.read_csv(f'{data_args.data_dir}/train.csv')
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
  type_preds = np.argmax(outputs[0][0], axis=1)
  polarity_preds = np.argmax(outputs[0][1], axis=1)
  tense_preds = np.argmax(outputs[0][2], axis=1)
  certainty_preds = np.argmax(outputs[0][3], axis=1)


  final_prediction = []
  for type_pred, polarity_pred, tense_pred, certainty_pred in zip(type_preds, polarity_preds, tense_preds, certainty_preds):
    one_predict = Type_label[type_pred] + '-' + polarity_label[polarity_pred] + '-' +Tense_label[tense_pred] + '-' + Certainty_label[certainty_pred]
    final_prediction.append(one_predict)
  
  final_submission = pd.DataFrame(columns = ['ID','label'])
  final_submission['ID'] = final_prediction_id
  final_submission['label'] = final_prediction

  
  if not os.path.exists('./results'):
    os.makedirs('./results')
  submission_save_path = os.path.join('./results',f'submission_predict.csv')
  final_submission.to_csv(submission_save_path, index=False)


if __name__ == "__main__" :
    main()