from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DataTrainingArguments:
  max_length: int = field(
    default=315, metadata={"help": "Max length of input sequence"},
  )

  data_dir: str = field(
    default='./data', metadata={"help": "Datasets directory"},
  )

  data_type: str = field(
    default='train', metadata={"help": "Datasets type"},
  )

  dataset_type: str = field(
    default='type', metadata={"help":"Dataset Type"}
  )

  special_token_type: bool = field(
    default= False, metadata = {"help": "add special token"}
  )

  use_oversampling: bool = field(
    default = False, metadata = {"help": "Use oversampling"}
  )
