import torch
import torch.nn as nn
import torch.optim as optim

from Network_modules import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DMTNet_Model(nn.Module):
    def __init__(self, args):
        super(DMTNet_Model, self).__init__()
        self.args = args
        self.encoder = RN18_Encoder()
        self.decoder = UNet_Decoder()
        self.triplet_loss = nn.TripletMarginLoss(margin=args.margin)
        self.optimizer = optim.SGD(self.parameters(), lr=args.learning_rate)


    def train_model(self, batch):

        self.train()
        self.optimizer.zero_grad()
        triplet_loss, recons_loss = 0.0, 0.0

        ### Triplet loss ###
        anchor_feat = self.encoder(batch['anchor_image'].to(device), pool=True)     # [N,512]
        positive_feat = self.encoder(batch['positive_image'].to(device), pool=True) # [N,512]
        negative_feat = self.encoder(batch['negative_image'].to(device), pool=True) # [N,512]

        triplet_loss += self.triplet_loss(anchor_feat, positive_feat, negative_feat)


        ### Reconstruction loss ###
        anchor_fmap = self.encoder(batch['anchor_augment'].to(device), pool = True)     # [N,512,8,8]
        recons_img = self.decoder(anchor_fmap) # [N,3,256,256]

        recons_loss += F.mse_loss(batch['anchor_image'].to(device), recons_img)

        (self.args.lambd * triplet_loss + recons_loss).backward()
        self.optimizer.step()

        return triplet_loss.item(), recons_loss.item()


    def validate_model(self):
        pass