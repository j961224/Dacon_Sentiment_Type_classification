from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default='./exps', metadata={"help": "checkpoint save directory"}
    )

    use_rdrop: bool = field(
      default=False, metadata={"help":"use r-drop"}
    )

    reg_alpha: float = field(
        default=0.7,
        metadata={
            "help": "alpha value for regularized dropout(default: 0.7)"
        },
    )