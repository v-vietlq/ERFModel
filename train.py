import torch
import torch.nn as nn
import torchvision
import os
import torch.nn.functional as F
from PIL import Image
from utils.utils import iou, AverageMeter, debug_val_example



class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        self.loss = nn.NLLLoss(weight)
        
    def forward(self, outputs, targets):
        outputs, _ = outputs
        return self.loss(F.log_softmax(outputs, dim=1), targets)
    
def train_one_epoch(model, criterion, optimizer, data_loader,debug_data_loader, device, ntrain_batches):
    
    top1 = AverageMeter('Acc@1', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')
    train_loss = 0.0
    cnt = 0
    for data in data_loader:
        cnt += 1
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
        if cnt%10 == 0:
            #debug val example checks and prints a random debug example every
            #10 batches.
            #it uses cpu, so might be slow. 
            debug_val_example(model, debug_data_loader)
            print('Loss', avgloss.avg)
            

            print('Training: * Acc@1 {top1.avg:.3f}'
                        .format(top1=top1))
            
        if cnt >= ntrain_batches:
            return avgloss.avg, top1.avg
        
        
            
        
        
        
    
    