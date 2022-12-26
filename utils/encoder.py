import re
import torch
import numpy as np

class Encoder :
    def __init__(self, tokenizer, max_input_length: int) :
      self.tokenizer = tokenizer
      self.max_input_length = max_input_length
    
    def __call__(self, examples):
      model_inputs = self.tokenizer(examples['inputs'],
          max_length=self.max_input_length,
          truncation=True,
          return_token_type_ids=False,
      )

      if 'labels' in examples :
          model_inputs['labels'] = examples['labels']
      return model_inputs