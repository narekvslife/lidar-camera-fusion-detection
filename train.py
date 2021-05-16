import torch

torch.multiprocessing.set_start_method('spawn')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from tqdm import tqdm

from src.datasets import NuscenesRangeViewDataset
from src.models.lasernet import LaserNet
from src.losses import LaserNetLoss
from src.settings import DATASET_PATH

from torch.utils.data import DataLoader 

train_dataset = NuscenesRangeViewDataset(data_root=DATASET_PATH, n=(0, 4032))
val_dataset = NuscenesRangeViewDataset(data_root=DATASET_PATH, n=(4032, 5120))

# train_dataset = NuscenesRangeViewDataset(data_root=DATASET_PATH, n=(0, 8064))
# val_dataset = NuscenesRangeViewDataset(data_root=DATASET_PATH, n=(8064, 9152))

train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=0)

EPOCHS = 2000

lasernet = torch.nn.DataParallel(LaserNet(), device_ids=[0, 1])
loss = LaserNetLoss(focal_loss_reduction='mean')
optimizer = torch.optim.Adam(lasernet.parameters(), lr=0.002)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)


lasernet.zero_grad()
loss.zero_grad()
optimizer.zero_grad()

train_losses = []
train_accs = []

val_losses = []
val_accs = []
for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    for batch_rv, batch_labels, batch_target_bbs in tqdm(train_dataloader, leave=True):
        
        batch_pointclass_preds, batch_bb_param_preds, batch_log_std_preds = lasernet(x=batch_rv)
                
        L_train = loss(batch_pointclass_preds, batch_bb_param_preds, batch_log_std_preds,
                 batch_labels,           batch_target_bbs)
        
        train_losses.append(L_train.item())
        
        lasernet.zero_grad()
        L_train.backward()
        optimizer.step()

    with torch.no_grad():
        for batch_rv, batch_labels, batch_target_bbs in tqdm(val_dataloader):
            
            batch_pointclass_preds, batch_bb_param_preds, batch_log_std_preds = lasernet(x=batch_rv)
            
            L_val = loss(batch_pointclass_preds, batch_bb_param_preds, batch_log_std_preds,
                         batch_labels,           batch_target_bbs)
        
            val_losses.append(L_val.item())

    torch.save(lasernet, f'lasernet-d{len(train_dataset)}-b64-e{epoch}-adam-lr002-sch095e1')
    print('------------------------')
    print('|epoch|', epoch, "|train_loss|", L_train.item(), "|val_loss|", L_val.item())
    print('------------------------')
    lr_scheduler.step()

