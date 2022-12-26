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
    Trainer,
    RobertaForSequenceClassification
)
from args import (MyTrainingArguments, ModelArguments, DataTrainingArguments)
from model.roberta import Multi_label_RobertaForSequenceClassification, Multi_label_Hidden_states_RobertaForSequenceClassification
from utils.trainer import Multi_label_Trainer


def main():
  parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, MyTrainingArguments)
  )

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model_args, data_args, training_args = parser.parse_args_into_dataclasses()

  tokenizer = AutoTokenizer.from_pretrained(model_args.Type_PLM)

  data_args.data_type = "test"
  config = AutoConfig.from_pretrained(model_args.Type_PLM)

  loader = Dacon_Dataset(data_args.data_dir, data_args.data_type, '', data_args.special_token_type)
  # type_loader = Dacon_Dataset(data_args.data_dir, data_args.data_type, 'type', data_args.special_token_type)

  dset = loader.load_datasets()
  final_prediction_id = dset['test']['sentence_id']
  dset = dset['test']
  print(dset)

  # preprocessor = Preprocessor(train_flag=False, dataset_type='')
  preprocessor = Preprocessor(train_flag=False, dataset_type='type')
  dset = dset.map(preprocessor, batched=True, num_proc=4,remove_columns=dset.column_names)
  print(dset)

  encoder = Encoder(tokenizer, data_args.max_length)
  # dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
  dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)


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

  config = AutoConfig.from_pretrained(model_args.Type_PLM)
  model = RobertaForSequenceClassification.from_pretrained(model_args.Type_PLM, config=config)
  # model = RobertaForSequenceClassification_special_token.from_pretrained(model_args.Type_PLM, config=config)

  trainer = Trainer(                       
    model=model,                         
    args=training_args,                  
    data_collator=data_collator,
  )

  # outputs = trainer.predict(dset)
  outputs = trainer.predict(dset)
  type_preds = np.argmax(outputs[0], axis=1)

  config = AutoConfig.from_pretrained(model_args.Polarity_PLM)
  # tokenizer = AutoTokenizer.from_pretrained(model_args.Polarity_PLM)
  model = RobertaForSequenceClassification.from_pretrained(model_args.Polarity_PLM, config=config)
  # model = RobertaForSequenceClassification_special_token.from_pretrained(model_args.Polarity_PLM, config=config)

  # polarity_loader = Dacon_Dataset(data_args.data_dir, data_args.data_type, 'polarity', data_args.special_token_type)
  # polarity_dset = polarity_loader.load_datasets()
  # polarity_dset = polarity_dset['test']
  # preprocessor = Preprocessor(train_flag=False, dataset_type='polarity')
  # polarity_dset = polarity_dset.map(preprocessor, batched=True, num_proc=4,remove_columns=polarity_dset.column_names)
  # encoder = Encoder(tokenizer, data_args.max_length)
  # polarity_dset = polarity_dset.map(encoder, batched=True, num_proc=4, remove_columns=polarity_dset.column_names)
  # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)


  trainer = Trainer(                       
    model=model,                         
    args=training_args,                  
    data_collator=data_collator,
  )

  # outputs = trainer.predict(dset)
  outputs = trainer.predict(dset)
  polarity_preds = np.argmax(outputs[0], axis=1)

  config = AutoConfig.from_pretrained(model_args.Tense_PLM)
  # tokenizer = AutoTokenizer.from_pretrained(model_args.Tense_PLM)
  model = RobertaForSequenceClassification.from_pretrained(model_args.Tense_PLM, config=config)
  # model = RobertaForSequenceClassification_special_token.from_pretrained(model_args.Tense_PLM, config=config)

  # tense_loader = Dacon_Dataset(data_args.data_dir, data_args.data_type, 'tense', data_args.special_token_type)
  # tense_dset = tense_loader.load_datasets()
  # tense_dset = tense_dset['test']
  # preprocessor = Preprocessor(train_flag=False, dataset_type='tense')
  # tense_dset = tense_dset.map(preprocessor, batched=True, num_proc=4,remove_columns=tense_dset.column_names)
  # encoder = Encoder(tokenizer, data_args.max_length)
  # tense_dset = tense_dset.map(encoder, batched=True, num_proc=4, remove_columns=tense_dset.column_names)
  # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)

  trainer = Trainer(                       
    model=model,                         
    args=training_args,                  
    data_collator=data_collator,
  )

  # outputs = trainer.predict(dset)
  outputs = trainer.predict(dset)
  tense_preds = np.argmax(outputs[0], axis=1)

  config = AutoConfig.from_pretrained(model_args.Certainty_PLM)
  # tokenizer = AutoTokenizer.from_pretrained(model_args.Certainty_PLM)
  model = RobertaForSequenceClassification.from_pretrained(model_args.Certainty_PLM, config=config)
  # model = RobertaForSequenceClassification_special_token.from_pretrained(model_args.Certainty_PLM, config=config)

  # certainty_loader = Dacon_Dataset(data_args.data_dir, data_args.data_type, 'certainty', data_args.special_token_type)
  # certainty_dset = certainty_loader.load_datasets()
  # certainty_dset = certainty_dset['test']
  # preprocessor = Preprocessor(train_flag=False, dataset_type='certainty')
  # certainty_dset = certainty_dset.map(preprocessor, batched=True, num_proc=4,remove_columns=certainty_dset.column_names)
  # encoder = Encoder(tokenizer, data_args.max_length)
  # certainty_dset = certainty_dset.map(encoder, batched=True, num_proc=4, remove_columns=certainty_dset.column_names)
  # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)


  trainer = Trainer(                       
    model=model,                         
    args=training_args,                  
    data_collator=data_collator,
  )

  # outputs = trainer.predict(dset)
  outputs = trainer.predict(dset)
  certainty_preds = np.argmax(outputs[0], axis=1)

  final_prediction = []
  for type_pred, polarity_pred, tense_pred, certainty_pred in zip(type_preds, polarity_preds, tense_preds, certainty_preds):
    one_predict = Type_label[type_pred] + '-' + polarity_label[polarity_pred] + '-' +Tense_label[tense_pred] + '-' + Certainty_label[certainty_pred]
    final_prediction.append(one_predict)
  
  final_submission = pd.DataFrame(columns = ['ID','label'])
  final_submission['ID'] = final_prediction_id
  final_submission['label'] = final_prediction

  
  if not os.path.exists('./results'):
    os.makedirs('./results')
  submission_save_path = os.path.join('./results',f'submission_predict_no_multi.csv')
  final_submission.to_csv(submission_save_path, index=False)


if __name__ == "__main__" :
    main()