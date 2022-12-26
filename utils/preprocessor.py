class Preprocessor:
  def __init__(self, train_flag, dataset_type):
    self.train_flag=train_flag
    self.dataset_type = dataset_type

  def __call__(self, dataset):
    
    dataset['inputs'] = dataset['sentence']
    if self.train_flag == True:
      if self.dataset_type=='':
        dataset['labels'] = dataset['multi_label']
      else:
        dataset['labels'] = dataset['label']
    return dataset