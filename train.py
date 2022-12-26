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

from model.roberta import Multi_label_RobertaForSequenceClassification, Multi_label_Hidden_states_RobertaForSequenceClassification, Multi_label_Heinsen_routing_RobertaForSequenceClassification
from utils.trainer import Multi_label_Trainer, Multi_label_Rdrop_Trainer, Multi_label_Smart_Trainer



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

  type_preds = np.argmax(preds[0], axis=1)
  polarity_preds = np.argmax(preds[1], axis=1)
  tense_preds = np.argmax(preds[2], axis=1)
  certainty_preds = np.argmax(preds[3], axis=1)

  type_f1 = f1_score(labels[:, 0], type_preds, average='weighted')
  polarity_f1 = f1_score(labels[:, 1], polarity_preds, average='weighted')
  tense_f1 = f1_score(labels[:, 2], tense_preds, average='weighted')
  certainty_f1 = f1_score(labels[:, 3], certainty_preds, average='weighted')

  total_f1 = type_f1*0.25 + polarity_f1*0.25 + tense_f1*0.25 + certainty_f1*0.25
  return {"type_f1": type_f1, "polarity_f1": polarity_f1, "tense_f1":tense_f1, "certainty_f1":certainty_f1, \
  "total_f1":total_f1}

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
    special_tokens_dict = {'additional_special_tokens': ['[type_token]','[polarity_token]','[tense_token]','[certainty_token]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
  
  loader = Dacon_Dataset(data_args.data_dir, data_args.data_type, '', data_args.special_token_type)

  
  dset = loader.load_datasets()
  dset = dset['train'].shuffle(training_args.seed)
  print(dset)

  preprocessor = Preprocessor(train_flag=True, dataset_type='')
  dset = dset.map(preprocessor, batched=True, num_proc=4,remove_columns=dset.column_names)
  print(dset)


  encoder = Encoder(tokenizer, data_args.max_length)
  dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
  print(dset)
  
  data_collator =DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)
  # skf = StratifiedKFold(n_splits=5, shuffle=True)
  skf = KFold(n_splits=5, shuffle=True, random_state=training_args.seed)

  for i, (train_idx, valid_idx) in enumerate(skf.split(dset, dset['labels'])):
    print("#########################",i)
    
    train_dataset = dset.select(train_idx.tolist())
    valid_dataset = dset.select(valid_idx.tolist())
    print(train_dataset)
    print(valid_dataset)


    if training_args.max_steps == -1:
      name = f"EP_Fold{i}:{training_args.num_train_epochs}_"
    else:
      name = f"MS_Fold{i}:{training_args.max_steps}_"
    name += f"LR:{training_args.learning_rate}_BS:{training_args.per_device_train_batch_size}_WR:{training_args.warmup_ratio}_WD:{training_args.weight_decay}_{training_args.use_rdrop}"
    
    config = AutoConfig.from_pretrained(model_args.PLM)
    config.num_labels = 64
    config.problem_type = "multi_label_classification"
    config.special_token_type = data_args.special_token_type
    if data_args.special_token_type:
      config.tokenizer_type_token_id = tokenizer.encode(tokenizer.additional_special_tokens[0])[1]
      config.tokenizer_polarity_token_id = tokenizer.encode(tokenizer.additional_special_tokens[1])[1]
      config.tokenizer_tense_token_id = tokenizer.encode(tokenizer.additional_special_tokens[2])[1]
      config.tokenizer_certainty_token_id = tokenizer.encode(tokenizer.additional_special_tokens[3])[1]
      
    # model = Multi_label_Hidden_states_RobertaForSequenceClassification.from_pretrained(model_args.PLM, config=config)
    model = Multi_label_RobertaForSequenceClassification.from_pretrained(model_args.PLM, config=config)
    # model = Multi_label_Heinsen_routing_RobertaForSequenceClassification.from_pretrained(model_args.PLM, config=config)
    if data_args.special_token_type:
      model.resize_token_embeddings(len(tokenizer))
      tokenizer.save_pretrained("./checkpoints/"+name)


    trainer = Multi_label_Rdrop_Trainer(
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


if __name__ == "__main__":
  main()