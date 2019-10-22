import torch
from torch.optim.optimizer import Optimizer, required

class QModelOpt(Optimizer):
  r"""Implements gradient descent with momentum  but with learning rate
  and momentum parameter set in order to optimise the Quadratic model that one gets 
  with the full fisher.
  """

  def __init__(self, params, lr=required, momentum=0.9, dampening=0,
               weight_decay=0, nesterov=False):
      if lr is not required and lr < 0.0:
          raise ValueError("Invalid learning rate: {}".format(lr))
      if momentum < 0.0:
          raise ValueError("Invalid momentum value: {}".format(momentum))
      if weight_decay < 0.0:
          raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

      defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                      weight_decay=weight_decay, nesterov=nesterov)
      self.lr = lr
      self.momentum = momentum     
          
      if nesterov and (momentum <= 0 or dampening != 0):
          raise ValueError("Nesterov momentum requires a momentum and zero dampening")
      super(QModelOpt, self).__init__(params, defaults)
      self.n = len(self.param_groups[0]['params'])
      print(self.n)
      self.prev_update_list = [None] * self.n


  def __setstate__(self, state):
      super(QModelOpt, self).__setstate__(state)
      for group in self.param_groups:
          group.setdefault('nesterov', False)

  def step(self, closure=None):
      """Performs a single optimization step.
      Arguments:
          closure (callable, optional): A closure that reevaluates the model
              and returns the loss.
      """
      loss = None
      if closure is not None:
          loss = closure()

      for group in self.param_groups:
          weight_decay = group['weight_decay']
          momentum = group['momentum']
          dampening = group['dampening']
          nesterov = group['nesterov']

          for i, p in enumerate(group['params']):
              if p.grad is None:
                  continue
              d_p = p.grad.data
              if weight_decay != 0:
                  d_p.add_(weight_decay, p.data)
              
              # Apply learning rate
              d_p.mul_(-1 * self.lr) # d_p is now alpha * Delta from KFAC paper
              if momentum != 0:
                  param_state = self.state[p]
                  if 'momentum_buffer' not in param_state:
                      buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                  else:
                      buf = param_state['momentum_buffer'] # buf is delta_0
                      buf.mul_(self.momentum).add_(1 - dampening, d_p) # this updates param_state['momentum_buffer']
                      # buf is now final update, and is stored in param_state['momentum_buffer'] for next time

                  self.prev_update_list[i] = torch.clone(buf).detach()
                  d_p = buf
              p.data.add_(d_p)
              
      return loss

