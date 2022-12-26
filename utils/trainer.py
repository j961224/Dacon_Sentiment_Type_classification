import torch
import torch.nn as nn
from transformers.trainer_pt_utils import nested_detach
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import Trainer

from typing import Callable
from torch import Tensor
from itertools import count 
from torch.utils.checkpoint import checkpoint

def kl_loss(inputs, target, reduction='batchmean'):
    return F.kl_div(
        F.log_softmax(inputs, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )

def sym_kl_loss(input, target, reduction='sum', alpha=1.0):
    return alpha * F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d

def inf_norm(x):
    return torch.norm(x, p=float('inf'), dim=-1, keepdim=True)

class SMARTLoss(nn.Module):
    
    def __init__(
        self,
        eval_fn: Callable,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        
    def forward(self, embed: Tensor, state: Tensor, data_flag: str) -> Tensor:
        noise = torch.randn_like(embed, requires_grad=True) * self.noise_var

        # Indefinite loop with counter 
        for i in count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise 
            state_perturbed = self.eval_fn(embed_perturbed, data_flag)
            # Return final loss if last step (undetached state)
            if i == self.num_steps: 
                return self.loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)

            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise

            noise_gradient, = torch.autograd.grad(outputs = loss, inputs = noise)
            # Move noise towards gradient to change state as much as possible 
            step = noise + self.step_size * noise_gradient 
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()

class Multi_label_Smart_Trainer(Trainer):
  
  def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

      model.train()
      inputs = self._prepare_inputs(inputs)
      
      loss = self.compute_loss(model, inputs)

      if self.args.n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu parallel training

      if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
          # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
          loss = loss / self.args.gradient_accumulation_steps

      if self.deepspeed:
          # loss gets scaled under gradient_accumulation_steps in deepspeed
          loss = self.deepspeed.backward(loss)
      else:
          loss.backward()

      return loss.detach()
  
  def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False):
      """
      How the loss is computed by Trainer. By default, all models return the loss in the first element.
      Subclass and override for custom behavior.
      """

      # if self.label_smoother is not None and "labels" in inputs:
      if "labels" in inputs:
          labels = inputs['labels'] # inputs.pop("labels")
          pad_mask = labels.unsqueeze(-1).eq(-100) # ignore_index
      else:
          labels = None
      
      outputs = model(**inputs)

      embed = self.model.roberta.embeddings(inputs.input_ids)
      def eval_fn(embed, data_flag):
        outputs = self.model.roberta(inputs_embeds=embed, attention_mask=inputs.attention_mask)
        pooled = outputs[0]
        if data_flag=="type":
          logits = self.model.Type_label_classifier(pooled)
        elif data_flag=='polarity':
          logits = self.model.polarity_label_classifier(pooled)
        elif data_flag=='tense':
          logits = self.model.Tense_label_classifier(pooled)
        elif data_flag=='certainty':
          logits = self.model.Certainty_label_classifier(pooled)

        return logits
      
      smart_loss_fn = SMARTLoss(eval_fn = eval_fn, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)

      # Save past state if it exists
      # TODO: this needs to be fixed and made cleaner later.
      if self.args.past_index >= 0:
          self._past = outputs[self.args.past_index]

      if labels is not None:
          
          type_loss_fct = nn.CrossEntropyLoss()
          polarity_loss_fct = nn.CrossEntropyLoss()
          Tense_loss_fct = nn.CrossEntropyLoss()
          Certainty_loss_fct = nn.CrossEntropyLoss()

          type_loss = type_loss_fct(outputs['Type_logits'].view(-1, 4), labels[:, 0].view(-1))
          polarity_loss = polarity_loss_fct(outputs['polarity_logits'].view(-1, 3), labels[:, 1].view(-1))
          Tense_loss = Tense_loss_fct(outputs['Tense_logits'].view(-1, 3), labels[:, 2].view(-1))
          Certainty_loss = Certainty_loss_fct(outputs['Certainty_logits'].view(-1, 2), labels[:, 3].view(-1))

          loss = type_loss*(4/12) + polarity_loss*0.25 + Tense_loss*0.25 + Certainty_loss*(2/12)
          if return_outputs==False:

            smart_type_loss = smart_loss_fn(embed, outputs['Type_logits'], "type")
            smart_polarity_loss = smart_loss_fn(embed, outputs['polarity_logits'], "polarity")
            smart_Tense_loss = smart_loss_fn(embed, outputs['Tense_logits'], "tense")
            smart_Certainty_loss = smart_loss_fn(embed, outputs['Certainty_logits'], "certainty")

            smart_loss = smart_type_loss * (4/12) +  smart_polarity_loss * 0.25 + smart_Tense_loss * 0.25 + smart_Certainty_loss * (2/12)
            loss += smart_loss*0.05# 0.02(0.7393080677) # 0.05, 0.5

      else:
          # We don't use .loss here since the model may return tuples instead of ModelOutput.
          loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

      return (loss, outputs) if return_outputs else loss



class Multi_label_Trainer(Trainer):
  def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

      model.train()
      inputs = self._prepare_inputs(inputs)
      
      loss = self.compute_loss(model, inputs)

      if self.args.n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu parallel training

      if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
          # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
          loss = loss / self.args.gradient_accumulation_steps

      if self.deepspeed:
          # loss gets scaled under gradient_accumulation_steps in deepspeed
          loss = self.deepspeed.backward(loss)
      else:
          loss.backward()

      return loss.detach()
  
  def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False):
      """
      How the loss is computed by Trainer. By default, all models return the loss in the first element.
      Subclass and override for custom behavior.
      """

      # if self.label_smoother is not None and "labels" in inputs:
      if "labels" in inputs:
          labels = inputs['labels'] # inputs.pop("labels")
          pad_mask = labels.unsqueeze(-1).eq(-100) # ignore_index
      else:
          labels = None
      
      outputs = model(**inputs)


      # Save past state if it exists
      # TODO: this needs to be fixed and made cleaner later.
      if self.args.past_index >= 0:
          self._past = outputs[self.args.past_index]

      if labels is not None:

          type_loss_fct = nn.CrossEntropyLoss()
          polarity_loss_fct = nn.CrossEntropyLoss()
          Tense_loss_fct = nn.CrossEntropyLoss()
          Certainty_loss_fct = nn.CrossEntropyLoss()

          type_loss = type_loss_fct(outputs['Type_logits'].view(-1, 4), labels[:, 0].view(-1))
          polarity_loss = polarity_loss_fct(outputs['polarity_logits'].view(-1, 3), labels[:, 1].view(-1))
          Tense_loss = Tense_loss_fct(outputs['Tense_logits'].view(-1, 3), labels[:, 2].view(-1))
          Certainty_loss = Certainty_loss_fct(outputs['Certainty_logits'].view(-1, 2), labels[:, 3].view(-1))

          loss = type_loss*0.25 + polarity_loss*0.25 + Tense_loss*0.25 + Certainty_loss*0.25

      else:
          # We don't use .loss here since the model may return tuples instead of ModelOutput.
          loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
      return (loss, outputs) if return_outputs else loss



class Multi_label_Rdrop_Trainer(Trainer):

  def get_normalized_probs(self, net_output: Dict[str, Union[torch.Tensor, Any]], log_probs=True) -> torch.Tensor:
      logits = net_output
      if log_probs:
          return F.log_softmax(logits, dim=-1)
      else:
          return F.softmax(logits, dim=-1)

  def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

      model.train()
      inputs = self._prepare_inputs(inputs)

      concat_inputs = {
          'input_ids': torch.cat([inputs['input_ids'], inputs['input_ids'].clone()], 0),
          'attention_mask': torch.cat([inputs['attention_mask'], inputs['attention_mask'].clone()], 0),
          'labels': torch.cat([inputs['labels'], inputs['labels'].clone()], 0),
      }
      
      loss = self.compute_loss(model, concat_inputs)

      if self.args.n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu parallel training

      if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
          # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
          loss = loss / self.args.gradient_accumulation_steps

      if self.deepspeed:
          # loss gets scaled under gradient_accumulation_steps in deepspeed
          loss = self.deepspeed.backward(loss)
      else:
          loss.backward()

      return loss.detach()
  
  def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False):
      """
      How the loss is computed by Trainer. By default, all models return the loss in the first element.
      Subclass and override for custom behavior.
      """

      # if self.label_smoother is not None and "labels" in inputs:
      if "labels" in inputs:
          labels = inputs['labels'] # inputs.pop("labels")
          Type_pad_mask = labels[:, 0].unsqueeze(-1).eq(-100) # ignore_index
          polarity_pad_mask = labels[:, 1].unsqueeze(-1).eq(-100)
          Tense_pad_mask = labels[:, 2].unsqueeze(-1).eq(-100)
          Certainty_pad_mask = labels[:, 3].unsqueeze(-1).eq(-100)
      else:
          labels = None
      
      outputs = model(**inputs)


      # Save past state if it exists
      # TODO: this needs to be fixed and made cleaner later.
      if self.args.past_index >= 0:
          self._past = outputs[self.args.past_index]

      if labels is not None:

          

          Type_loss = self.label_smoothed_nll_loss(outputs['Type_logits'], labels[:, 0], epsilon=0.1 if self.label_smoother else 0) 
          Type_kl_loss = self.compute_kl_loss(outputs['Type_logits'], pad_mask=Type_pad_mask)

          polarity_loss = self.label_smoothed_nll_loss(outputs['polarity_logits'], labels[:, 1], epsilon=0.1 if self.label_smoother else 0) 
          polarity_kl_loss = self.compute_kl_loss(outputs['polarity_logits'], pad_mask=polarity_pad_mask)

          Tense_loss = self.label_smoothed_nll_loss(outputs['Tense_logits'], labels[:, 2], epsilon=0.1 if self.label_smoother else 0) 
          Tense_kl_loss = self.compute_kl_loss(outputs['Tense_logits'], pad_mask=Tense_pad_mask)

          Certainty_loss = self.label_smoothed_nll_loss(outputs['Certainty_logits'], labels[:, 3], epsilon=0.1 if self.label_smoother else 0) 
          Certainty_kl_loss = self.compute_kl_loss(outputs['Certainty_logits'], pad_mask=Certainty_pad_mask)

          loss = Type_loss*(4/12) + polarity_loss*0.25 + Tense_loss*0.25 + Certainty_loss*(2/12)

          loss += self.args.reg_alpha * (Type_kl_loss * (4/12) + polarity_kl_loss * 0.25 + Tense_kl_loss * 0.25 + Certainty_kl_loss * (2/12))

      else:
          # We don't use .loss here since the model may return tuples instead of ModelOutput.
          loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
      return (loss, outputs) if return_outputs else loss
    
  def compute_kl_loss(self, net_output: Dict[str, Union[torch.Tensor, Any]], pad_mask=None, reduce=True) -> torch.Tensor:
    net_prob = self.get_normalized_probs(net_output, log_probs=True)
    net_prob_tec = self.get_normalized_probs(net_output, log_probs=False)
    if net_prob.size(0) == 3 or net_prob.size(0) == 5:
      return 0
    p, q = torch.split(net_prob, net_prob.size(0)//2, dim=0)
    p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0)//2, dim=0)
    
    p_loss = F.kl_div(p, q_tec, reduction='none') # ToDo nn.KLDivLoss(reduction='batchmean') v2 / none(v0)
    q_loss = F.kl_div(q, p_tec, reduction='none') # ToDo nn.KLDivLoss(reduction='batchmean')
    
    if pad_mask is not None:
        pad_mask, _ = torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    if reduce:
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
  
  def label_smoothed_nll_loss(self, model_output: Dict[str, Union[torch.Tensor, Any]], labels: torch.Tensor, epsilon: float) -> torch.Tensor:
      logits = model_output
      log_probs = -F.log_softmax(logits, dim=-1)
      if labels.dim() == log_probs.dim() - 1:
          labels = labels.unsqueeze(-1)

      padding_mask = labels.eq(-100)
      # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
      # will ignore them in any case.
      labels = torch.clamp(labels, min=0)
      nll_loss = log_probs.gather(dim=-1, index=labels)
      # works for fp16 input tensor too, by internally upcasting it to fp32
      smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

      nll_loss.masked_fill_(padding_mask, 0.0)
      smoothed_loss.masked_fill_(padding_mask, 0.0)
      
      nll_loss = nll_loss.sum()
      smoothed_loss = smoothed_loss.sum()
      eps_i = epsilon / log_probs.size(-1)
      return (1. - epsilon) * nll_loss + eps_i * smoothed_loss


class Rdrop_Trainer(Trainer):
    
  def get_normalized_probs(self, net_output: Dict[str, Union[torch.Tensor, Any]], log_probs=True) -> torch.Tensor:
      logits = net_output["logits"] if isinstance(net_output, dict) else net_output[0]
      if log_probs:
          return F.log_softmax(logits, dim=-1)
      else:
          return F.softmax(logits, dim=-1)
      
  
  def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
      if not self.args.use_rdrop:
          return super().training_step(model, inputs)
          
      model.train()
      inputs = self._prepare_inputs(inputs)
      concat_inputs = {
          'input_ids': torch.cat([inputs['input_ids'], inputs['input_ids'].clone()], 0),
          'attention_mask': torch.cat([inputs['attention_mask'], inputs['attention_mask'].clone()], 0),
          'labels': torch.cat([inputs['labels'], inputs['labels'].clone()], 0),
      }
      
      loss = self.compute_loss(model, concat_inputs)

      if self.args.n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu parallel training

      if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
          # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
          loss = loss / self.args.gradient_accumulation_steps

      if self.deepspeed:
          # loss gets scaled under gradient_accumulation_steps in deepspeed
          loss = self.deepspeed.backward(loss)
      else:
          loss.backward()

      return loss.detach()
  
  
  def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False):
      """
      How the loss is computed by Trainer. By default, all models return the loss in the first element.
      Subclass and override for custom behavior.
      """
      if not self.args.use_rdrop and self.args.label_smoothing_factor == 0:
          return super().compute_loss(model, inputs)

      elif not self.args.use_rdrop and self.args.label_smoothing_factor != 0:
          assert "labels" in inputs
          labels = inputs["labels"]
          outputs = model(**inputs)
          # Save past state if it exists
          # TODO: this needs to be fixed and made cleaner later.
          if self.args.past_index >= 0:
              self._past = outputs[self.args.past_index]

          if labels is not None:
              loss = self.label_smoother(outputs, labels)
          else:
              # We don't use .loss here since the model may return tuples instead of ModelOutput.
              loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

          return (loss, outputs) if return_outputs else loss

      else:
          # if self.label_smoother is not None and "labels" in inputs:
          if "labels" in inputs:
              labels = inputs['labels'] # inputs.pop("labels")
              pad_mask = labels.unsqueeze(-1).eq(-100) # ignore_index
          else:
              labels = None
          
          outputs = model(**inputs)
          
          # Save past state if it exists
          # TODO: this needs to be fixed and made cleaner later.
          if self.args.past_index >= 0:
              self._past = outputs[self.args.past_index]

          if labels is not None:
              # loss = self.label_smoother(outputs, labels)
              
              # nll loss original version
              loss = self.label_smoothed_nll_loss(outputs, labels,
                                                  epsilon=0.1 if self.label_smoother else 0) 
              
              kl_loss = self.compute_kl_loss(outputs, pad_mask=pad_mask)
              loss += self.args.reg_alpha * kl_loss

          else:
              # We don't use .loss here since the model may return tuples instead of ModelOutput.
              loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

          return (loss, outputs) if return_outputs else loss

  def compute_kl_loss(self, net_output: Dict[str, Union[torch.Tensor, Any]], pad_mask=None, reduce=True) -> torch.Tensor:
      net_prob = self.get_normalized_probs(net_output, log_probs=True)
      net_prob_tec = self.get_normalized_probs(net_output, log_probs=False)
      if net_prob.size(0) == 3 or net_prob.size(0) == 5:
        return 0
      p, q = torch.split(net_prob, net_prob.size(0)//2, dim=0)
      p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0)//2, dim=0)
      
      p_loss = F.kl_div(p, q_tec, reduction='none') # ToDo nn.KLDivLoss(reduction='batchmean') v2 / none(v0)
      q_loss = F.kl_div(q, p_tec, reduction='none') # ToDo nn.KLDivLoss(reduction='batchmean')
      
      if pad_mask is not None:
          pad_mask, _ = torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
          p_loss.masked_fill_(pad_mask, 0.)
          q_loss.masked_fill_(pad_mask, 0.)

      if reduce:
          p_loss = p_loss.mean()
          q_loss = q_loss.mean()

      loss = (p_loss + q_loss) / 2
      return loss
  
  def label_smoothed_nll_loss(self, model_output: Dict[str, Union[torch.Tensor, Any]], labels: torch.Tensor, epsilon: float) -> torch.Tensor:
      logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
      log_probs = -F.log_softmax(logits, dim=-1)
      if labels.dim() == log_probs.dim() - 1:
          labels = labels.unsqueeze(-1)

      padding_mask = labels.eq(-100)
      # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
      # will ignore them in any case.
      labels = torch.clamp(labels, min=0)
      nll_loss = log_probs.gather(dim=-1, index=labels)
      # works for fp16 input tensor too, by internally upcasting it to fp32
      smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

      nll_loss.masked_fill_(padding_mask, 0.0)
      smoothed_loss.masked_fill_(padding_mask, 0.0)
      
      nll_loss = nll_loss.sum()
      smoothed_loss = smoothed_loss.sum()
      eps_i = epsilon / log_probs.size(-1)
      return (1. - epsilon) * nll_loss + eps_i * smoothed_loss
