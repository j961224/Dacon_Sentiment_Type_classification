import os
import math
import torch
import random
import numpy as np
from transformers import (
  Trainer, 
  HfArgumentParser, 
  AutoTokenizer, 
  AutoConfig, 
  DataCollatorWithPadding, 
  AutoModelForSequenceClassification,
  RobertaForSequenceClassification,
  T5Tokenizer,  
)
from functools import partial
from datasets import load_metric
from sklearn.model_selection import StratifiedKFold, KFold
from args import (MyTrainingArguments, ModelArguments, DataTrainingArguments)

from utils.datasets import Dacon_Dataset
from utils.preprocessor import Preprocessor
from utils.encoder import Encoder
from sklearn.metrics import f1_score

from model.roberta import Multi_label_RobertaForSequenceClassification, Multi_label_Hidden_states_RobertaForSequenceClassification, RobertaForSequenceClassification_special_token
from utils.trainer import Rdrop_Trainer
from datasets import Dataset, DatasetDict

class Oversampling:
  def __init__(self, dataset, seed):
    self.dataset = dataset
    self.seed = seed
  
  def oversampler(self, label_number, oversampling_scale):
    label = self.dataset['labels']
    input_ids = self.dataset['input_ids']
    attention_mask = self.dataset['attention_mask']

    level = list(filter(lambda x: label[x] == label_number, range(len(label))))
    sampling_level = [label_number] * len(level) * oversampling_scale
    sampling_input_ids = [input_ids[i] for i in level] * oversampling_scale
    sampling_attention_mask = [attention_mask[i] for i in level] * oversampling_scale

    label.extend(sampling_level)
    input_ids.extend(sampling_input_ids)
    attention_mask.extend(sampling_attention_mask)

    dataset = Dataset.from_dict({
      "labels": label,
      "input_ids": input_ids,
      "attention_mask": attention_mask
    })

    dataset.shuffle(self.seed)
    return dataset


def seed_everything(seed):
  os.environ["PYTHONHASHSEED"] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  np.random.default_rng(seed)
  random.seed(seed)

def compute_metrics(EvalPrediction):
  preds, labels = EvalPrediction

  pred = np.argmax(preds, axis=1)

  f1_metric = load_metric('f1')    
  f1 = f1_metric.compute(predictions = pred, references = labels, average="weighted")

  return f1

def main():
  print(f"# of CPU : {os.cpu_count()}")
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  parser = HfArgumentParser(
    (ModelArguments, DataTrainingArguments, MyTrainingArguments)
  )
  model_args, data_args, training_args = parser.parse_args_into_dataclasses()
  seed_everything(training_args.seed)

  tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)

  if data_args.special_token_type:
    if data_args.dataset_type=='type':
      special_token = '[type_token] '
    elif data_args.dataset_type=='polarity':
      special_token = '[polarity_token] '
    elif data_args.dataset_type=='tense':
      special_token = '[tense_token] '
    else:
      special_token = '[certainty_token] '
    special_tokens_dict = {'additional_special_tokens': [special_token]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
  
  loader = Dacon_Dataset(data_args.data_dir, data_args.data_type, data_args.dataset_type, data_args.special_token_type)

  
  dset = loader.load_datasets()
  dset = dset['train'].shuffle(training_args.seed)
  print(dset)

  preprocessor = Preprocessor(train_flag=True, dataset_type = data_args.dataset_type)
  dset = dset.map(preprocessor, batched=True, num_proc=4,remove_columns=dset.column_names)
  print(dset)

  config = AutoConfig.from_pretrained(model_args.PLM)
  if data_args.special_token_type:
    config.special_token_type = data_args.special_token_type
    config.tokenizer_special_token_id = tokenizer.encode(tokenizer.additional_special_tokens[0])[1]

  if data_args.dataset_type=='type':
    config.num_labels = 4
  elif data_args.dataset_type=='certainty':
    config.num_labels = 2
  else:
    config.num_labels = 3

  encoder = Encoder(tokenizer, data_args.max_length)
  dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
  print(dset)
  
  data_collator =DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)
  skf = StratifiedKFold(n_splits=5, shuffle=True)

  for i, (train_idx, valid_idx) in enumerate(skf.split(dset, dset['labels'])):

    train_dataset = dset.select(train_idx.tolist())
    valid_dataset = dset.select(valid_idx.tolist())

    print("Before"*30)
    print(train_dataset)
    if data_args.use_oversampling:
      if data_args.dataset_type=='type':
        oversampling = Oversampling(train_dataset, training_args.seed)
        train_dataset = oversampling.oversampler(3, 5)

        oversampling = Oversampling(train_dataset, training_args.seed)
        train_dataset = oversampling.oversampler(0, 3)
      elif data_args.dataset_type=='polarity':
        oversampling = Oversampling(train_dataset, training_args.seed)
        train_dataset = oversampling.oversampler(0, 5)

        oversampling = Oversampling(train_dataset, training_args.seed)
        train_dataset = oversampling.oversampler(2, 10)
      
      elif data_args.dataset_type=='tense':
        oversampling = Oversampling(train_dataset, training_args.seed)
        train_dataset = oversampling.oversampler(1, 2)
      
      else:
        oversampling = Oversampling(train_dataset, training_args.seed)
        train_dataset = oversampling.oversampler(0, 5)
      print("After"*30)
      print(train_dataset)

    print(valid_dataset)


    if training_args.max_steps == -1:
      name = f"EP_Fold{i}:{training_args.num_train_epochs}_"
    else:
      name = f"MS_Fold{i}:{training_args.max_steps}_"
    name += f"LR:{training_args.learning_rate}_BS:{training_args.per_device_train_batch_size}_WR:{training_args.warmup_ratio}_WD:{training_args.weight_decay}_{training_args.use_rdrop}_{data_args.dataset_type}"

    if data_args.special_token_type:
      model = RobertaForSequenceClassification_special_token.from_pretrained(model_args.PLM, config=config)
      model.resize_token_embeddings(len(tokenizer))
      tokenizer.save_pretrained("./checkpoints/"+name)
    else:
      model = RobertaForSequenceClassification.from_pretrained(model_args.PLM, config=config)

    trainer = Rdrop_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,            
        eval_dataset=valid_dataset,             
        data_collator=data_collator,            
        tokenizer=tokenizer,                    
        compute_metrics=compute_metrics,
      )

    trainer.train()
    trainer.evaluate()
    prev_path = model_args.save_path
    model_args.save_path = os.path.join(model_args.save_path, name)
    trainer.save_model(model_args.save_path)
    model_args.save_path = prev_path
    wandb.finish()
    break


if __name__ == "__main__":
  main()