import torch

torch.multiprocessing.set_start_method('spawn')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from tqdm import tqdm
from torch.utils.data import DataLoader 

from src.datasets import NuscenesRangeViewDataset
from src.models.lasernet_seg import LaserNetSeg
from src.losses import FocalLoss
from src.settings import DATASET_PATH
from src.utils import accuracy

# train_dataset = NuscenesRangeViewDataset(data_root=DATASET_PATH, n=(0, 4032))
# val_dataset = NuscenesRangeViewDataset(data_root=DATASET_PATH, n=(4032, 5120))

train_dataset = NuscenesRangeViewDataset(data_root=DATASET_PATH, n=(0, 8064))
val_dataset = NuscenesRangeViewDataset(data_root=DATASET_PATH, n=(8064, 9152))

train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=0)

EPOCHS = 2000
LEARNING_RATE = 0.001
    
lasernet_seg = torch.nn.DataParallel(LaserNetSeg(), device_ids=[0, 1])
loss = FocalLoss(reduction='mean')
optimizer = torch.optim.Adam(lasernet_seg.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)  # todo

lasernet_seg.zero_grad()
loss.zero_grad()
optimizer.zero_grad()

losses_train = []
accs_train = []

losses_val = []
accs_val = []
for epoch in tqdm(range(EPOCHS)):
    
    print("|--- Epoch", epoch, "---|")
    
    for batch_rv, batch_labels, _ in tqdm(train_dataloader):
        
        batch_pointclass_preds = lasernet_seg(x=batch_rv)
        
        L_train = loss(batch_pointclass_preds, batch_labels)
                
        if torch.isnan(L_train):
            print("L_train had value of nan") 
            break
            
        lasernet_seg.zero_grad()
        L_train.backward()
        optimizer.step()
        
        # save loss and accuracy
        losses_train.append(L_train.item())
        accs_train.append(accuracy(batch_pointclass_preds, batch_labels).item())

            
    with torch.no_grad():
        for batch_rv, batch_labels, _ in tqdm(val_dataloader):
            
            batch_pointclass_preds = lasernet_seg(x=batch_rv)
            
            L_val = loss(batch_pointclass_preds, batch_labels)
            
            if torch.isnan(L_val):
                print("L_val had value of nan") 
                break
                
            losses_val.append(L_val.item())
            accs_val.append(accuracy(batch_pointclass_preds, batch_labels).item())

        
    if torch.isnan(L_train) or torch.isnan(L_val):
        break

    if epoch % 5 == 0:
        print("train_loss", L_train.item(), "train_acc",  accs_train[-1], "val_loss", L_val.item(), "val_acc",  accs_val[-1])
        torch.save(lasernet_seg, f'lasernet_seg-d{len(train_dataset)}-b64-e{epoch}-adam-lr{LEARNING_RATE}-schplat')
       
    lr_scheduler.step(L_val)