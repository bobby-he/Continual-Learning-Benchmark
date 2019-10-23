import time
import torch

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum().item()
            res.append(correct_k*100.0 / batch_size)

        if len(res)==1:
            return res[0]
        else:
            return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Timer(object):
    """
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval
        
def ip(list1, list2):
  # Inner product of lists or tuples of tensors.
  
  return sum(tuple(torch.sum(tensor1 * tensor2) 
                        for tensor1, tensor2 in zip(list1, list2)))
                        
def _two_by_two_solve(m, vec):
  """Solve a 2x2 system by direct inversion.
  Args:
    m: A length 2 list of length 2 lists, is a 2x2 matrix of [[a, b], [c, d]].
    vec: The length 2 tensor, a vector of [e, f].
  Returns:
    matmul(m^{-1}, vec).
  """
  a = m[0][0]
  b = m[0][1]
  c = m[1][0]
  d = m[1][1]
  inv_m_det = 1.0 / (a * d - b * c)
  m_inverse = torch.tensor([
      [d * inv_m_det, -b * inv_m_det],
      [-c * inv_m_det, a * inv_m_det]
  ])
  return torch.mm(m_inverse, vec)
  
def _eval_quadratic_no_c(m, vec):
  return 0.5 * vec.t() @ m @ vec
