import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from  torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.module import Module

from torch.autograd import Variable
from torch.autograd.function import Function
import numpy as np

import scipy
from scipy.linalg import cho_factor, cho_solve

default_np_dtype = np.float32

def regularized_cholesky_factor(mat, lambda_, inverse_method='cpu',
                        use_cuda=True):
  # calculates and returns the regularised cholesky decomposition of a matrix
  assert mat.shape[0] == mat.shape[1]
  ii = torch.eye(mat.shape[0])
  if use_cuda:
    ii = ii.cuda()
  regmat = mat + lambda_ * ii

  if inverse_method == 'cpu':
    #regmat = regmat.cpu()
    #if mat.shape[0]<=1005:
      #print(np.linalg.cond(regmat.numpy()))
    #cho_factor = torch.cholesky(regmat, upper = False)
    cho_factor = torch.cholesky(regmat.cpu(), upper = False)
    if use_cuda:
      cho_factor = cho_factor.cuda()
  elif inverse_method == 'gpu':
    assert use_cuda
    cho_factor = torch.cholesky(regmat, upper = False) 
  else:
    assert False, 'unknown inverse_method ' + str(INVERSE_METHOD)

  return cho_factor

def ng_init(s1, s2, bias = True): # uniform weight init from Ng UFLDL, and zero biases
  r = 1 / np.sqrt(s1)
  if bias:
    s1 += 1
  flat = np.random.random(s1*s2)*2*r-r
  weights = flat.reshape([s1, s2])
  #biases = np.zeros((1, s2))
  #combo = np.concatenate((weights, biases))
  return weights.astype(default_np_dtype)
  
def jl_kfac_gaussian_proj_mats(fs, q = 2000, use_cuda = True):
  n = len(fs) - 1
  A_Proj = [None] * n
  B_Proj = [None] * n
  for i in range(n):
    a = np.random.normal(scale = q**(-1/2), size = (q, fs[i] + 1))
    b = np.random.normal(scale = q**(-1/2), size = (q, fs[i+1]))
    #a = gaussian_random_matrix(q, fs[i] + 1)
    #b = gaussian_random_matrix(q, fs[i+1])
    
    a = torch.FloatTensor(a)
    b = torch.FloatTensor(b)
    
    if use_cuda:
      a = a.cuda()
      b = b.cuda()
    
    A_Proj[i] = a
    B_Proj[i] = b
    
  return A_Proj, B_Proj
  


class JlNet(nn.Module):
  def __init__(self, out_dim = 10, in_channel = 1, img_sz = 32, hidden_dim = 256, use_cuda = True, pre_reg_mat_decay = 0.9, initial_damping = 1, proj_dim = 1000):
    super(JlNet, self).__init__()
    self.in_dim = in_channel * img_sz * img_sz
    self.fs = [self.in_dim, hidden_dim, hidden_dim, out_dim]
    self.n = len(self.fs) - 1
    self.proj_dim = proj_dim
    self.U_proj = [None] * 1
    self.grads_proj = [None] * 1
    self.pre_proj = [None] * 1
    self.initialize = [True] * 1
    self.initialize_list = [True] * self.n
    self.pre_reg_mat = [None] * 1
    self.AU_AU = [None] * 1
    self.precon_update_list = [None] * self.n
    self.grad_list = [None] * self.n 
    self.Bs = [None] * self.n
    self.As = [None] * self.n
    self.acts_proj = [None] * self.n
    self.preact_grads_proj = [None] * self.n
    self.pre_reg_mat_decay = pre_reg_mat_decay
    self.damping = self.initial_damping = initial_damping
    self.use_cuda = use_cuda
##############################################################
    class JlngAddmm(Function):
      @staticmethod
      def _get_output(ctx, arg, inplace=False):
        if inplace:
          ctx.mark_dirty(arg)
          return arg
        else:
          return arg.new().resize_as_(arg)
    
      @staticmethod
      def forward(ctx, add_matrix, matrix1, matrix2, beta=1, alpha=1, inplace=False, i=0):
        ctx.save_for_backward(matrix1, matrix2)
        ctx.idx = i
        ctx.mode = 'U_proj_capture'
        output = JlngAddmm._get_output(ctx, add_matrix, inplace=inplace)
        return torch.addmm(beta, add_matrix, alpha,
                           matrix1, matrix2, out=output)

      @staticmethod
      def backward(ctx, grad_output):
        # grad_output is pre-activation derivatives
        matrix1, matrix2 = ctx.saved_variables # matrix1 is activations, matrix2 is weights
        i = ctx.idx                            # i is block index
        grad_matrix1 = grad_matrix2 = None
        B = grad_output.data
        A = matrix1.data
        batch_size = A.shape[0]
        batch_size_sqrt = np.sqrt(batch_size)
        # A has shape batch_size x w_in 
        # B has shape batch_size x w_out
        if ctx.mode == 'U_proj_capture':
          #if i == self.n - 1:
            #self.U_proj[0] = ((self.A_proj[i] @ A.t()) * (self.B_proj[i] @ (B.t() * batch_size)))/ batch_size_sqrt
				    
          #else:
            #self.U_proj[0]  += ((self.A_proj[i] @ A.t()) * (self.B_proj[i] @ (B.t() * batch_size))) / batch_size_sqrt

          # update the pre_reg_mat at i==0 i.e. when U_proj complete
          #if i == 0:
            
            #if self.initialize[0]:
              #print('acidic')
              #self.pre_reg_mat[0] = self.U_proj[0] @ self.U_proj[0].t()
              #self.initialize[0] = False
            #else:
              #self.pre_reg_mat[0] = self.pre_reg_mat_decay * self.pre_reg_mat[0] + (1 - self.pre_reg_mat_decay) * (self.U_proj[0] @ self.U_proj[0].t()) # shape q x q
        
          if self.initialize_list[i]:
            #print(i)

            self.Bs[i] = B.t() @ B     # store pre-activation derivatives and activations for later

            self.As[i] = A.t() @ A
            self.initialize_list[i] = False
          else:    
            #print('lol')
            self.Bs[i] = self.pre_reg_mat_decay * self.Bs[i] + (1 - self.pre_reg_mat_decay) * B.t() @ B
            self.As[i] = self.pre_reg_mat_decay * self.As[i] + (1 - self.pre_reg_mat_decay) * A.t() @ A	
          ctx.mode = 'grads_proj_capture'
            
        elif ctx.mode == 'grads_proj_capture':
          
          if i == self.n - 1:
            self.grads_proj[0] = torch.mean((self.A_proj[i] @ A.t()) * (self.B_proj[i] @ (B.t() * batch_size)), dim = 1)
            #alt = A.t() @ B
            #alt = A_proj[i] @ alt @ B_proj[i].t()
            #print(i, torch.sum((torch.diag(alt) - grads_proj[0])**2))
          else:
            self.grads_proj[0] += torch.mean((self.A_proj[i] @ A.t()) * (self.B_proj[i] @ (B.t() * batch_size)), dim = 1)
            
          self.acts_proj[i] = (self.A_proj[i] @ (self.As[i]/ batch_size)) # shape q x w_in
          self.preact_grads_proj[i] = (self.B_proj[i] @ (self.Bs[i] * batch_size)) # shape q x w_out
          
          if i == self.n-1:
            self.AU_AU[0] = ((self.A_proj[i] @ self.acts_proj[i].t()) * (self.B_proj[i] @ self.preact_grads_proj[i].t()))
          else:
            self.AU_AU[0] += ((self.A_proj[i] @ self.acts_proj[i].t()) * (self.B_proj[i] @ self.preact_grads_proj[i].t()))
          ctx.mode = 'jlng'
        elif ctx.mode == 'jlng':
          grads_pre_fisher = A.t() @ B # already normalised as B is divided by batch_size
          # only compute the grad coefficients once

          if i == self.n-1:
            approx_kmat_cho = regularized_cholesky_factor(self.AU_AU[0], lambda_ = self.damping)
            #approx_kmat_cho = regularized_cholesky_factor(self.pre_reg_mat[0], lambda_ = self.damping)
            temp = torch.from_numpy(cho_solve((approx_kmat_cho.cpu().numpy(), True), self.grads_proj[0].view(-1, 1).cpu().numpy()))        
            if self.use_cuda:
              temp = temp.cuda()
              
            #grad_coeffs[0] = U_proj[0].t() @ temp # no ema method
            self.pre_proj[0] = self.AU_AU[0] @ temp #  
            #self.pre_proj[0] = self.pre_reg_mat[0] @ temp
          #temp = (Bs[i] / batch_size_sqrt) * grad_coeffs[0] # no ema method
          #grad_subtract = As[i].t() @ temp 

          #preact_grads_temp = preact_grads_proj[i] * temp[0] # shape q x out, full kfac method
          #grad_subtract = acts_proj[i].t() @ preact_grads_temp 
          
          test = self.A_proj[i] * self.grads_proj[0].view(-1, 1)
          grads_pre_fisher_test = (test.t() @ self.B_proj[i]) * self.proj_dim
          
          #grads_pre_fisher_norm = torch.sum(grads_pre_fisher ** 2)
          #grads_pre_fisher_test_norm = torch.sum(grads_pre_fisher_test ** 2)
          
          #norm_ratio = torch.sqrt(grads_pre_fisher_norm/grads_pre_fisher_test_norm)
          
          A_temp = self.A_proj[i] * self.pre_proj[0]
          grad_subtract = (A_temp.t() @ self.B_proj[i]) * self.proj_dim #* norm_ratio
          
          update = grads_pre_fisher_test - grad_subtract 
          #if i == 0:
            #update_norm = torch.sum(update ** 2)
            #print(norm_ratio)

            #print(i, (torch.sum(grad_subtract**2)/grads_pre_fisher_norm).item())
            #print(i, (grads_pre_fisher_test_norm/grads_pre_fisher_norm).item())
            #print((update_norm / grads_pre_fisher_norm).item())
            #print(i, (torch.sum(update * grads_pre_fisher)/torch.sqrt(update_norm)/torch.sqrt(grads_pre_fisher_norm)).item())
          grad_matrix2 = Variable(update) / self.damping

          #print(torch.sum(grad_subtract **2), torch.sum(grads_pre_fisher **2), torch.sum(grad_matrix2 **2))
          self.precon_update_list[i] = -1 * torch.clone(grad_matrix2).detach()
          self.grad_list[i] = torch.clone(grads_pre_fisher).detach()
          ctx.mode = 'standard'

        elif ctx.mode == 'standard':
          grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        else:
          assert False, 'unknown mode '+mode

        if ctx.needs_input_grad[1]:
          grad_matrix1 = torch.mm(grad_output, matrix2.t())      
        
        return None, grad_matrix1, grad_matrix2, None, None, None, None

    def jlng_matmul(mat1, mat2, i = 0):
      output = Variable(mat1.data.new(mat1.data.size(0), mat2.data.size(1)))
      return JlngAddmm.apply(output, mat1, mat2, 0, 1, True, i)

    class JlLinear(Module):
        r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
        Shape:
            - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
              additional dimensions and :math:`H_{in} = \text{in\_features}`
            - Output: :math:`(N, *, H_{out})` where all but the last dimension
              are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
        Attributes:
            weight: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in\_features}}`
            bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                    If :attr:`bias` is ``True``, the values are initialized from
                    :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                    :math:`k = \frac{1}{\text{in\_features}}`
        Examples::
            >>> m = nn.Linear(20, 30)
            >>> input = torch.randn(128, 20)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([128, 30])
        """
        __constants__ = ['bias', 'in_features', 'out_features']

        def __init__(self, in_features, out_features, bias=True, layer_idx = 0, use_cuda = True):
            super(JlLinear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weights = Parameter(torch.from_numpy(ng_init(in_features, out_features, bias = bias)))
            self.layer_idx = layer_idx
            self.use_cuda = use_cuda
            if bias:
              self.bias = True
            else:
              self.register_parameter('bias', None)
            
        def forward(self, input):
            if self.bias:
                ones_input = torch.ones(input.shape[0], 1)
                if self.use_cuda:
                    ones_input = ones_input.cuda()
                input= torch.cat((input, ones_input), dim = 1)
                    
            return jlng_matmul(input, self.weights, self.layer_idx)
            
        def extra_repr(self):
            return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None
            )
############################################################################
    self.JlLinear = JlLinear
    self.linear = nn.Sequential(
          JlLinear(self.in_dim, hidden_dim, layer_idx = 0, use_cuda = self.use_cuda), 
          nn.ReLU(inplace = True),
          JlLinear(hidden_dim, hidden_dim, layer_idx = 1, use_cuda = self.use_cuda),
          nn.ReLU(inplace = True),
    )
    self.last = JlLinear(hidden_dim, out_dim, layer_idx = 2, use_cuda = self.use_cuda)    
    
  def new_proj(self):
    self.A_proj, self.B_proj = jl_kfac_gaussian_proj_mats(self.fs, q = self.proj_dim, use_cuda = self.use_cuda)
    self.initialize[0] = True
    
  def reset_damping(self):
    self.damping = self.initial_damping
    
  def reset_kfac_ema(self):
    self.initialize_list = [True] * self.n
    
  def features(self, x):
    x = self.linear(x.view(-1, self.in_dim))
    return x
    
  def logits(self, x):
    x = self.last(x)
    return x
    
  def forward(self, x):
    x = self.features(x)
    x = self.logits(x)
    return x
          
def JlNet100():
    return JlNet(hidden_dim=100)


def JlNet400():
    return JlNet(hidden_dim=400)


def JlNet1000():
    return JlNet(hidden_dim=1000)


def JlNet2000():
    return JlNet(hidden_dim=2000)


def JlNet5000():
    return JlNet(hidden_dim=5000)

  
  
