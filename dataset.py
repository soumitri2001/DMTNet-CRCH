import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

random.seed(16)


class CRC_Dataset(Dataset):
    def __init__(self, args):
        super(CRC_Dataset, self).__init__()
        self.args = args
        self.transformations = transforms.Compose([transforms.Resize((256,256)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])
                                                   ])

        self.rand_transforms = transforms.Compose([transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(0.180)),
                                                                                      transforms.RandomVerticalFlip(),
                                                                                      transforms.RandomHorizontalFlip()]),
                                                   transforms.ToTensor(),
                                                   transforms.RandomErasing(p=0.8),
                                                   transforms.ToPILImage(),
                                                   transforms.Resize((256, 256)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])
                                                   ])

        self.df = pd.DataFrame(columns=['filepath', 'label'])

        data_root = os.path.join(args.data_dir, 'train')
        dir2label = {'Anger':0, 'Boredom':1, 'Disgust':2, 'Fear':3, 'Happiness':4, 'Neutral':5, 'Sadness':6}
        for dir in os.listdir(data_root):
            dir_path = os.path.join(data_root, dir)
            if os.path.isdir(dir_path):
                for img in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img)
                    img_label = dir2label[dir]
                    self.df = self.df.append({'filepath': img_path, 'label': img_label}, ignore_index=True)

        print(f"DML Training dataset contains {len(self.df)} images !!")




    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        sample = {}

        # 1. get the file paths
        anchor_path = self.df.iloc[index]['filepath']

        positive_path, negative_path = None, None

        remaining_list = list(range(len(self.df)))
        remaining_list.remove(index)

        while True:
            p_rand = random.randint(0, len(remaining_list) - 1)
            if p_rand != index and self.df.iloc[p_rand]['label'] == self.df.iloc[index]['label']: # same label
                positive_path = self.df.iloc[p_rand]['filepath']
                remaining_list.remove(p_rand)
                break

        while True:
            n_rand = random.randint(0, len(remaining_list) - 1)
            if n_rand != index and self.df.iloc[n_rand]['label'] != self.df.iloc[index]['label']: # different label
                negative_path = self.df.iloc[n_rand]['filepath']
                remaining_list.remove(n_rand)
                break

        # verification
        assert (positive_path is not None and negative_path is not None)
        assert (self.df.iloc[index]['label'] == self.df.iloc[p_rand]['label'] and self.df.iloc[index]['label'] != self.df.iloc[n_rand]['label'])

        # 2. get the respective images
        anchor_image_ = Image.open(anchor_path).convert('RGB')
        positive_image = Image.open(positive_path).convert('RGB')
        negative_image = Image.open(negative_path).convert('RGB')


        # 3. apply transforms
        anchor_image = self.transformations(anchor_image_)
        positive_image = self.transformations(positive_image)
        negative_image = self.transformations(negative_image)

        anchor_augment = self.rand_transforms(anchor_image_)


        # 4. form the sample dict
        sample = {'anchor_image' : anchor_image, 'anchor_path' : anchor_path, 'anchor_augment' : anchor_augment,
                  'positive_image' : positive_image, 'positive_path' : positive_path,
                  'negative_image': negative_image, 'negative_path': negative_path
                  }

        return sample



def get_dataloader(args):
    dset = CRC_Dataset(args)
    data_loader = DataLoader(dset, batch_size=args.batchsize, shuffle=True, num_workers=8)
    return data_loader