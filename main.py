import os
import torch
import torchvision
from torchvision.utils import make_grid, save_image
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset import *
from model import *

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--data_dir', type=str, default='./path/to/data')
    parser.add_argument('--saved_models', type=str, default='./saved_models')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--print_freq', type=int,default=5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--lambd', type=float, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.saved_models):
        os.mkdir(args.saved_models)
    print(args)

    TL_loss = []
    RL_loss = []
    iters = []

    data_loader = get_dataloader(args)

    ex_train = iter(data_loader)
    batch_train = ex_train.next()
    print(batch_train['anchor_image'].shape, batch_train['positive_image'].shape, batch_train['negative_image'].shape)
    save_image(make_grid(torch.cat([batch_train['anchor_image'],
                                    batch_train['anchor_augment'],
                                    batch_train['positive_image'],
                                    batch_train['negative_image']], dim=0),
                         nrow=args.batchsize),
               '/kaggle/working/batch_train.png')

    print('-' * 50)

    model = DMTNet_Model(args)
    model = model.to(device)

    for epoch in range(args.max_epochs):
        running_TL, running_RL = 0.0, 0.0
        for ii, batch in enumerate(data_loader):

            model.train()
            triplet_loss, recons_loss = model.train_model(batch)

            running_TL += triplet_loss
            running_RL += recons_loss
            
            if(ii+1)%10 == 0:
                # print step stats
                print(f'Epoch: {epoch+1}/{args.max_epochs} |' \
                    f'Step: {ii+1}/{len(data_loader)} |' \
                    f'Triplet Loss: {triplet_loss:.4f} |'\
                    f'Recons Loss: {recons_loss:.4f}')

        # print epoch stats
        print(f'\nEpoch stats: Epoch {epoch+1}/{args.max_epochs} |'\
              f'Avg. Triplet Loss: {running_TL/len(data_loader)} |'\
              f'Avg. Recons Loss: {running_RL/len(data_loader)}\n')

        iters.append(epoch+1)
        TL_loss.append(running_TL/len(data_loader))
        RL_loss.append(running_RL/len(data_loader))
        
        if (epoch+1) % 10 == 0:
            torch.save(model.encoder.state_dict(), f'./{args.saved_models}/RN18_Encoder+Recons_pretrained.pth')
            
        if (epoch+1) % 40 == 0:
            torch.save({'model': model.state_dict(), 'epochs': epoch+1}, f'./{args.saved_models}/RN18_Encoder+Recons_pretrained_{epoch+1}epochs.pt')
            print(f">>> Model saved at {epoch+1} Epoch for Feature Visualization")

    loss_per_epoch = {'Epochs' : iters, 'TL Loss' : TL_loss, 'RL Loss' : RL_loss}
    df = pd.DataFrame(loss_per_epoch)
    df.to_csv(f'./{args.saved_models}/Epoch_Loss.csv')

    print(f"Training completed for {args.max_epochs} Epochs !!")
    total_epoch = args.max_epochs
    torch.save(model.encoder.state_dict(), f"./{args.saved_models}/RN18_Encoder+Recons_{total_epoch}epochs.pth")
