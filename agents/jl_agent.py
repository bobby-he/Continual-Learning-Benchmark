from __future__ import print_function
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer, ip, _two_by_two_solve, _eval_quadratic_no_c, regularized_cholesky_factor
from optimizers.optimizers import QModelOpt
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve

class JlNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, agent_config):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super(JlNN, self).__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(self.config['out_dim'])>1 else False  # A convenience flag to indicate multi-head/task
        self.first_update = True
        if agent_config['gpuid'][0] >= 0:
            self.gpu = True
        else:
            self.gpu = False
        
        self.model = self.create_model()
        self.criterion_fn = nn.CrossEntropyLoss()
        self.sig_fn = torch.sigmoid
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
        self.init_optimizer()
        self.reset_optimizer = True
        self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
                                    # Set a interger here for the incremental class scenario
        self.adjust_lr_and_momentum = True

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr']}
        self.optimizer = QModelOpt(**optimizer_arg)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              #gamma=0.1)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()
        model.damping = model.initial_damping = cfg['damping']
        #print(model.damping)
        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features
        
        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task,out_dim in cfg['out_dim'].items():
            model.last[task] = model.JlLinear(n_feat,out_dim, layer_idx = 2, use_cuda = self.gpu)
            model.fs[-1] = out_dim
        model.sample_new_proj()
        model.new_proj()
        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return out

    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            output = self.predict(input)

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc(output, target, task, acc)

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
              .format(acc=acc,time=batch_timer.toc()))
        return acc.avg

    def criterion(self, preds, targets, tasks, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.
        if self.multihead:
            loss = 0
            for t,t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i]==t]  # The index of inputs that matched specific task
                if len(inds)>0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)  # restore the loss from average
            loss /= len(targets)  # Average the total loss by the mini-batch size
        else:
            pred = preds['All']
            if isinstance(self.valid_out_dim, int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                pred = preds['All'][:,:self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
        return loss
        
    def gradient_capture(self, inputs, targets, tasks):
        output_logits = self.forward(inputs)
        self.optimizer.zero_grad()
        true_loss = self.criterion(output_logits, targets, tasks)
        self.model.backward_mode = 'grads_capture'
        true_loss.backward()
        
    def update_model(self, inputs, targets, tasks):
        batch_size = len(targets)
        #print('batch_size {}'.format(batch_size))
        output_logits = self.forward(inputs)
        with torch.no_grad():
          output_probs = self.sig_fn(output_logits['All'])
          #print('hehe')
          mft_sqrt_factor = torch.sqrt(output_probs) # this is cheating, only for the not multihead case
          
        # capturing U_proj
        self.optimizer.zero_grad()
        model_dist = torch.distributions.multinomial.Categorical(logits = output_logits['All'].data)
        model_samples = model_dist.sample()
        sampled_loss = self.criterion(output_logits, model_samples, tasks)
        self.model.backward_mode = 'U_proj_capture'
        sampled_loss.backward(retain_graph = True)
        
        # capturing grad_proj
        self.optimizer.zero_grad()
        true_loss = self.criterion(output_logits, targets, tasks)
        self.model.backward_mode = 'grads_proj_capture'
        true_loss.backward(retain_graph = True)
        
        # actually compute updates
        self.optimizer.zero_grad()
        self.model.backward_mode = 'jlng'
        true_loss.backward(retain_graph = True)
        
            # Do the orthogonal projection
        for prev_task in range(self.model.task_id):
            #print('ok')
            inner_prod_grad = ip(self.model.task_gradients[prev_task], self.model.precon_update_list)
            #inner_prod_natural = ip(self.model.task_natural_gradients[prev_task], self.model.precon_update_list)

            for group in self.optimizer.param_groups:
            
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    p.grad.data -= (inner_prod_grad / self.model.task_grad_ips[prev_task]) * self.model.task_gradients[prev_task][i] #\
                    #p.grad.data -= (inner_prod_natural/self.model.task_natural_grad_ips[prev_task]) * self.model.task_natural_gradients[prev_task][i]
                    
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
            
                self.model.precon_update_list[i] = -1 * torch.clone(p.grad.data).detach()
        
        if self.adjust_lr_and_momentum:            
            self.model.backward_mode = 'standard'   
            # compute learning rate and momentum parameter
            dummy = torch.zeros_like(output_logits['All'])
            dummy.requires_grad = True
            if self.gpu:
              dummy = dummy.cuda()
              
            g = torch.autograd.grad(output_logits['All'], self.model.parameters(), grad_outputs = dummy, create_graph = True)
            
            jvp_precon = torch.autograd.grad(g, dummy, grad_outputs = self.model.precon_update_list, retain_graph = True)
            
            with torch.no_grad():
              mft_precon_temp = mft_sqrt_factor * jvp_precon[0]
              mft_precon = mft_precon_temp - output_probs * torch.sum(mft_precon_temp, dim = 1, keepdim = True)
              
              b_11 = self.config['reg_coef'] * ip(self.model.precon_update_list, self.model.precon_update_list)
              m_11 = torch.sum(mft_precon * mft_precon) / batch_size
              c_1 = ip(self.model.grad_list, self.model.precon_update_list)
            
            if self.first_update:
              print('yo')
              alpha = -1 * c_1 / (b_11 + m_11)
              print(self.model.precon_update_list)
              print(alpha)
              #print(alpha)
              self.optimizer.momentum = 0
              assert (alpha>=0), 'Looks like you have a negative learning rate noob!'
              #self.optimizer.lr = alpha
              self.optimizer.lr = 0.1

              #assert (self.optimizer.prev_update_list[0] is None)
              self.optimizer.step()
              self.first_update = False
              
            else:
              jvp_prev = torch.autograd.grad(g, dummy, grad_outputs = self.optimizer.prev_update_list)

              with torch.no_grad():
                b_21 = self.config['reg_coef'] * ip(self.model.precon_update_list, self.optimizer.prev_update_list)
                b_22 = self.config['reg_coef'] * ip(self.optimizer.prev_update_list, self.optimizer.prev_update_list)
            
                mft_prev_temp = mft_sqrt_factor * jvp_prev[0]
                mft_prev = mft_prev_temp - output_probs * torch.sum(mft_prev_temp, dim = 1, keepdim = True)
              
              m_21 = torch.sum(mft_precon * mft_prev) / batch_size
              m_22 = torch.sum(mft_prev * mft_prev) / batch_size
              
              c_2 = ip(self.model.grad_list, self.optimizer.prev_update_list)
              self.c = torch.tensor([[c_1], [c_2]])
              m = [[m_11 + b_11, m_21 + b_21],
                  [m_21 + b_21, m_22 + b_22]]
              self.b = [[b_11, b_21], [b_21, b_22]]
              self.sol = -1. * _two_by_two_solve(m, self.c)
              
              alpha = self.sol[0, 0]
              momentum = self.sol[1, 0]

        #self.optimizer.lr = alpha
        #self.optimizer.momentum = momentum
        self.optimizer.lr = 0.1
        self.optimizer.momentum = 0.9
        self.optimizer.step()
        
        for prev_task in range(self.model.task_id):
            for i in range(self.model.n):                    
                self.model.stored_task_gradients[prev_task][i] += torch.clone(self.model.stored_kfac_As[prev_task][i] @ self.optimizer.prev_update_list[i] @ self.model.stored_kfac_Bs[prev_task][i]).detach()  
                self.model.task_gradients[prev_task][i] = torch.clone(self.model.stored_task_gradients[prev_task][i]).detach()              
        #self.gram_schmidt()
            print(self.model.stored_task_gradients[prev_task])
        for task in range(self.model.task_id):
            for prev_task in range(task):
                inner_prod_grad = ip(self.model.stored_task_gradients[task], self.model.stored_task_gradients[prev_task])
                for i in range(self.model.n):
                    self.model.task_gradients[task][i] -= inner_prod_grad / self.model.task_grad_ips[prev_task] * self.model.stored_task_gradients[prev_task][i]
            self.model.task_grad_ips[task] = ip(self.model.task_gradients[task], self.model.task_gradients[task])        
            
        return true_loss.detach(), output_logits
    
    def gram_schmidt(self):
        for task in range(self.model.task_id):
            for prev_task in range(task):
                inner_prod_grad = ip(self.model.stored_task_gradients[task], self.model.stored_task_gradients[prev_task])
                for i in range(self.model.n):
                    self.model.task_gradients[task][i] -= inner_prod_grad / self.model.task_grad_ips[prev_task] * self.model.stored_task_gradients[prev_task][i]
            self.model.task_grad_ips[task] = ip(self.model.task_gradients[task], self.model.task_gradients[task])        
    
    def update_damping(self, inputs, targets, tasks, old_loss):

        with torch.no_grad():
          new_output_logits = self.model(inputs)
        new_loss = self.criterion(new_output_logits, targets, tasks)
        new_loss0 = new_loss.data
        qmodel_change = 0.5 * torch.sum(self.sol * self.c)
        qmodel_change -= _eval_quadratic_no_c(torch.tensor(self.b), torch.tensor(self.sol))[0,0]
        loss_change = new_loss - old_loss
        rho = loss_change / qmodel_change
        
        if (rho > 0.75) or (loss_change < 0 and qmodel_change > 0):
          self.model.damping *= self.model.omega
        elif rho < 0.25:
          self.model.damping /= self.model.omega
        
        self.model.damping = max(self.model.damping, self.model.min_damp)

    def learn_batch(self, train_loader, val_loader=None):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        self.model.reset_damping()
        self.model.reset_kfac_ema()
        self.first_update = True
        #self.model.proj_dim = (self.model.task_id+1) * 1000
        print(self.model.proj_dim)
        self.model.sample_new_proj()
        #self.model.new_proj()
        #self.update_projected_grads_and_naturals()
        
        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            for param_group in self.optimizer.param_groups:
                self.log('LR:',param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()

            print('new projection')
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            for i, (input, target, task) in enumerate(train_loader):   
                       
                if i != len(train_loader) - 1:        
                    batch_size = len(target)   
                    #self.model.new_proj()
                    data_time.update(data_timer.toc())  # measure data loading time

                    if self.gpu:
                        input = input.cuda()
                        target = target.cuda()

                    loss, output = self.update_model(input, target, task)
                    #if (i+1) % self.model.damping_update_period == 0 or (i+1) == len(train_loader):
                        #self.update_damping(input, target, task, loss)
                    
                    input = input.detach()
                    target = target.detach()

                    # measure accuracy and record loss
                    acc = accumulate_acc(output, target, task, acc)
                    losses.update(loss, input.size(0))

                    batch_time.update(batch_timer.toc())  # measure elapsed time
                    data_timer.toc()

                    if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                        self.log('[{0}/{1}]\t'
                              '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                              '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                              '{loss.val:.3f} ({loss.avg:.3f})\t'
                              '{acc.val:.2f} ({acc.avg:.2f})'.format(
                            i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, acc=acc))

            self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

            # Evaluate the performance of current task
            if val_loader != None:
                self.validation(val_loader)
        
        # collect the gradients in self.model.stored_task_gradients
        for i, (input, target, task) in enumerate(train_loader):
            #self.model.batch_size = len(target)
            if self.gpu:
                input = input.cuda()
                target = target.cuda()
            self.gradient_capture(input, target, task)                     
        for i in range(self.model.n):
            self.model.stored_task_gradients[self.model.task_id][i] /= len(train_loader.dataset)
            self.model.task_gradients[self.model.task_id][i] = self.model.stored_task_gradients[self.model.task_id][i]
            self.model.stored_kfac_As[self.model.task_id][i] = self.model.As[i] / batch_size
            self.model.stored_kfac_Bs[self.model.task_id][i] = self.model.Bs[i] * batch_size
                
        self.model.task_id += 1
        print('task id', self.model.task_id)	

    def learn_stream(self, data, label):
        assert False,'No implementation yet'

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model,torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self
        
        
    def update_projected_grads_and_naturals(self):
        # capture the projected gradients and natural gradients
        for prev_id in range(self.model.task_id):    
          for i in range(self.model.n):       
            acts_proj = self.model.A_proj[i] @ self.model.stored_kfac_As[prev_id][i]
            preact_grads_proj = self.model.B_proj[i] @ self.model.stored_kfac_Bs[prev_id][i]
            if i == 0:
              grads_proj = torch.diag(self.model.A_proj[i] @ self.model.stored_task_gradients[prev_id][i] @ self.model.B_proj[i].t()).unsqueeze(1)
              AU_AU = ((self.model.A_proj[i] @ acts_proj.t()) * (self.model.B_proj[i] @ preact_grads_proj.t())) 
            else:
              AU_AU += ((self.model.A_proj[i] @ acts_proj.t()) * (self.model.B_proj[i] @ preact_grads_proj.t()))    
              grads_proj += torch.diag(self.model.A_proj[i] @ self.model.stored_task_gradients[prev_id][i] @ self.model.B_proj[i].t()).unsqueeze(1)
          
          approx_kmat_cho = regularized_cholesky_factor(AU_AU, lambda_ = self.model.damping)
          temp = torch.from_numpy(cho_solve((approx_kmat_cho.cpu().numpy(), True), grads_proj.cpu().numpy()))
          if self.gpu:
            temp = temp.cuda()
          
          temp = AU_AU @ temp  * self.model.proj_dim
            
          for i in range(self.model.n):       
            A_temp_grad = self.model.A_proj[i] * grads_proj
            self.model.task_gradients[prev_id][i] = A_temp_grad.t() @ self.model.B_proj[i] * self.model.proj_dim
            
            A_temp_natural = self.model.A_proj[i] * temp
            #self.model.task_natural_gradients[prev_id][i] = A_temp_natural.t() @ self.model.B_proj[i]
            self.model.task_natural_gradients[prev_id][i] = self.model.task_gradients[prev_id][i] - A_temp_natural.t() @ self.model.B_proj[i]
        
        # gram schmidt now 
        for task_id in range(self.model.task_id):
          for prev_id in range(task_id):
              
              # calculate the inner product
              inner_prod_grad_grad = ip(self.model.task_gradients[task_id], self.model.task_gradients[prev_id])
              inner_prod_grad_precon = ip(self.model.task_gradients[task_id], self.model.task_natural_gradients[prev_id])
              inner_prod_precon_grad = ip(self.model.task_natural_gradients[task_id], self.model.task_gradients[prev_id])
              inner_prod_precon_precon = ip(self.model.task_natural_gradients[task_id], self.model.task_natural_gradients[prev_id])
              # project onto orthogonal 
              for i in range(self.model.n):
                  self.model.task_gradients[task_id][i] -= (inner_prod_grad_grad / self.model.task_grad_ips[prev_id]) * self.model.task_gradients[prev_id][i]  
                  self.model.task_gradients[task_id][i] -= (inner_prod_grad_precon / self.model.task_natural_grad_ips[prev_id]) * self.model.task_natural_gradients[prev_id][i]
                  
                  self.model.task_natural_gradients[task_id][i] -= (inner_prod_precon_grad / self.model.task_grad_ips[prev_id]) * self.model.task_gradients[prev_id][i]
                  self.model.task_natural_gradients[task_id][i] -= (inner_prod_precon_precon / self.model.task_natural_grad_ips[prev_id]) * self.model.task_natural_gradients[prev_id][i]
                  
          self.model.task_grad_ips[task_id] = ip(self.model.task_gradients[task_id], self.model.task_gradients[task_id])
          cur_task_grad_precon_ip = ip(self.model.task_gradients[task_id], self.model.task_natural_gradients[task_id])
          #print(cur_task_grad_precon_ip)
          #print(ip(self.model.task_natural_gradients[task_id], self.model.task_natural_gradients[task_id]))
          for i in range(self.model.n):
              self.model.task_natural_gradients[task_id][i] -= (cur_task_grad_precon_ip / self.model.task_grad_ips[task_id]) * self.model.task_gradients[task_id][i]    
            
          self.model.task_natural_grad_ips[task_id] = ip(self.model.task_natural_gradients[task_id], self.model.task_natural_gradients[task_id])
          #print(ip(self.model.task_natural_gradients[task_id], self.model.task_gradients[task_id]))
          #print(self.model.task_natural_grad_ips[task_id])
          

def accumulate_acc(output, target, task, meter):
    if 'All' in output.keys(): # Single-headed model
        meter.update(accuracy(output['All'], target), len(target))
    else:  # outputs from multi-headed (multi-task) model
        for t, t_out in output.items():
            inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
            if len(inds) > 0:
                t_out = t_out[inds]
                t_target = target[inds]
                meter.update(accuracy(t_out, t_target), len(inds))

    return meter
