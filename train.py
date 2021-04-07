import torch
import torch.nn as nn
import torchvision
import os
import torch.nn.functional as F
from PIL import Image
from utils.utils import iou, AverageMeter, debug_val_example
from tqdm import tqdm



class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        self.loss = nn.NLLLoss(weight)
        
    def forward(self, outputs, targets):
        outputs, _ = outputs
        return self.loss(F.log_softmax(outputs, dim=1), targets)
    
def train_one_epoch(model, criterion, optimizer, data_loader, device):
    
    top1 = AverageMeter('Acc@1', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')
    train_loss = 0.0
    for data in tqdm(data_loader):
        
        model.to(device) 
        
        image , target = data['image'].to(device, dtype=torch.float), data['label'].to(device)
        
        output = model(image)
        
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.cpu().detach().numpy()    
        
        acc1 = iou(output, target)
        
        top1.update(acc1, image.size(0))
        avgloss.update(loss, image.size(0))
        # if cnt%10 == 0:
        #     #debug val example checks and prints a random debug example every
        #     #10 batches.
        #     #it uses cpu, so might be slow. 
        #     debug_val_example(model, debug_data_loader)
        #     print('Loss', avgloss.avg)
            

        #     print('Training: * Acc@1 {top1.avg:.3f}'
        #                 .format(top1=top1))
            
    return avgloss.avg, top1.avg


def validate_model(model, criterion, valid_loader, device):

  top1 = AverageMeter('Acc@1', ':6.2f')
  jacc1 = AverageMeter('Jacc_sim@1', ':6.2f')
  avgloss = AverageMeter('Loss', '1.5f')
  val_loss = 0.0

  model.eval()
  with torch.no_grad():
    for data in valid_loader:
      image , target = data['image'].to(device, dtype=torch.float), data['label'].to(device)
      output = model(image)
    
      loss = criterion(output, target)
    
    
      val_loss += loss.cpu().detach().numpy()    
    
      acc1, jacc = iou(output, target)
      
      top1.update(acc1, image.size(0))
      avgloss.update(loss, image.size(0)) 
      jacc1.update(jacc, image.size(0))                             
              
  return avgloss.avg , top1.avg, jacc1.avg
      
        
            
        
        
        
    
    