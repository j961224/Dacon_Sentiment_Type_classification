import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset, DatasetDict

import math

class Dacon_Dataset(Dataset):
  def __init__(self, data_dir, data_type, dataset_type, speical_token_use):

    self.data_type = data_type
    self.dataset_type = dataset_type
    self.speical_token_use = speical_token_use
    
    if data_type in 'test':
      self.data_df = pd.read_csv(f'{data_dir}/test.csv')
    else:
      self.data_df = pd.read_csv(f'{data_dir}/train.csv')
      self.data_df = self.data_df.drop([208, 14989, 2108], axis = 0)
      self.data_df = self.data_df.reset_index(drop=True)
      self.data_df = self.data_df.drop_duplicates(['문장'], keep = 'first', ignore_index = True)

    if data_type not in ['test']:
      self.Type_label = {'대화형': 0, '추론형': 1, '사실형': 2, '예측형': 3}
      self.polarity_label = {'부정': 0, '긍정': 1, '미정': 2}
      self.Tense_label = {'과거': 0, '미래': 1, '현재': 2}
      self.Certainty_label = {'불확실': 0, '확실': 1}

    self.len = self.data_df.shape[0]
  
  def load_datasets(self):
    sentence_id = self.data_df['ID']
    sentence = self.data_df['문장']

    sub_dataset = []

    if self.data_type not in ['test']:
      label = self.data_df['label']

      for idx, one_label in enumerate(label):

        if self.dataset_type=='' and self.speical_token_use:
          one_sentence = '[type_token] + [polarity_token] + [tense_token] + [certainty_token] '
          one_sentence += sentence[idx]
        elif self.dataset_type!='' and self.speical_token_use:
          if self.dataset_type=='type':
            one_sentence = '[type_token] '
            one_sentence += sentence[idx]
          elif self.dataset_type=='polarity':
            one_sentence = '[polarity_token] '
            one_sentence += sentence[idx]
          elif self.dataset_type=='tense':
            one_sentence = '[tense_token] '
            one_sentence += sentence[idx]
          else:
            one_sentence = '[certainty_token] '
            one_sentence += sentence[idx]
        else:
          one_sentence = sentence[idx]

        if self.dataset_type=='':
          multi_label_one_tagging = []
          multi_label_one_tagging.append(self.Type_label[one_label.split("-")[0]])
          multi_label_one_tagging.append(self.polarity_label[one_label.split("-")[1]])
          multi_label_one_tagging.append(self.Tense_label[one_label.split("-")[2]])
          multi_label_one_tagging.append(self.Certainty_label[one_label.split("-")[3]])
          info = {'id' : sentence_id[idx], 'sentence' : one_sentence, 'multi_label' : multi_label_one_tagging}
        elif self.dataset_type=='type':
          info = {'id' : sentence_id[idx], 'sentence' : one_sentence, 'label' : self.Type_label[one_label.split("-")[0]]}
        elif self.dataset_type=='polarity':
          info = {'id' : sentence_id[idx], 'sentence' : one_sentence, 'label' : self.polarity_label[one_label.split("-")[1]]}
        elif self.dataset_type=='tense':
          info = {'id' : sentence_id[idx], 'sentence' : one_sentence, 'label' : self.Tense_label[one_label.split("-")[2]]}
        else:
          info = {'id' : sentence_id[idx], 'sentence' : one_sentence, 'label' : self.Certainty_label[one_label.split("-")[3]]}
        sub_dataset.append(info)
        

      df = pd.DataFrame(sub_dataset)
      dataset = Dataset.from_pandas(df)

      dataset_dict = DatasetDict({"train": dataset})

    else:
      final_sentence = []
      print("##################################"*20,self.dataset_type)
      if self.speical_token_use and self.dataset_type=='':
        # one_sentence = '[type_token] ' + '[polarity_token] '+ '[tense_token] '+ '[certainty_token] ' v1
        one_sentence = '[type_token] + [polarity_token] + [tense_token] + [certainty_token] ' # v0
        final_sentence = [one_sentence+s for s in sentence]
      
      elif self.dataset_type!='' and self.speical_token_use:
        if self.dataset_type=='type':
          final_sentence = ['[type_token] '+s for s in sentence]
        elif self.dataset_type=='polarity':
          final_sentence = ['[polarity_token] '+s for s in sentence]
        elif self.dataset_type=='tense':
          final_sentence = ['[tense_token] '+s for s in sentence]
        else:
          final_sentence = ['[certainty_token] '+s for s in sentence]
      
      else:
        final_sentence = sentence
      dataset = Dataset.from_dict({
        "sentence_id": sentence_id,
        "sentence": final_sentence
      })

      dataset_dict = DatasetDict({"test": dataset})
    return dataset_dict