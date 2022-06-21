import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from dataset import FFHQ
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
from glob import glob
from torchvision.utils import make_grid
from torchvision.utils import save_image
import argparse
import logging
import sys

from mask_unet import PConvUNet
from mask_loss import InpaintingLoss, VGG16FeatureExtractor

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_args_parser():
    parser = argparse.ArgumentParser('Mask Generator From Image', add_help=False)
    parser.add_argument("--epochs", type=int, default=1500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--data_path", type=str, default="/home/s20225004/FFHQ", help="dataset path")
    parser.add_argument("--output_dir", type=str, default=".", help="dataset path")
    parser.add_argument("--mode", type=str, default="train", help="executing mode: train or test")
    parser.add_argument("--test_weight_path", type=str, default=".", help="test weight path")
    return parser

class Trainer(object):
    def __init__(self, dataset, criterion, epochs, batch_size, lr, save_path="./outputs"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self._build_model()
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate, betas=(0.9, 0.999))
        self.save_path = save_path
        self.gray_transform = transforms.Grayscale()
        self.criterion = criterion.to(self.device)

    def _build_model(self):
        net = PConvUNet(finetune=True, layer_size=7, in_ch=1)
        self.net = net.to(self.device)
        self.net.train()
        print('Finish build model.')

    def train(self):
        total_step = 0
        for epoch in range(self.epochs):

            batch_loss = 0
            for batch_idx, (input, mask, _) in enumerate(self.dataloader):
                input, mask= input.to(self.device), mask.to(self.device)

                gray_input = self.gray_transform(input)
                recon_x = self.net(gray_input)
                loss = self.criterion(recon_x, mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item() / len(self.dataloader)


                if total_step % 10 == 0:
                    logging.info("Epoch [{:3}/{:3}] Step [{:10}/{:10}] Loss: {}".format(epoch, self.epochs, batch_idx, total_step, batch_loss))
                    grid = make_grid(torch.cat([input[:8], mask[:8], recon_x[:8]], dim=0))
                    save_image(grid, os.path.join(self.save_path, "image_epoch{}_step{}.png".format(str(epoch).zfill(3), str(total_step))))
                total_step += 1

                if total_step % 100 == 0:
                    torch.save(self.net.state_dict(), os.path.join(self.save_path, "weight_epoch{}_step{}.pth".format(str(epoch).zfill(3), str(total_step)))) #Change this path

        print('Finish training.')


class Tester(object):
    def __init__(self, dataset, weight_path=None, batch_size=1, save_path="."):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.save_path = save_path
        self._build_model()

        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.datalen = dataset.__len__()
        self.mse_all_img = []

        # Load of pretrained_weight file
        self.weight_PATH = weight_path
        self.net.load_state_dict(torch.load(weight_path))

        print("self.device: ", self.device)
        self.gray_transform = transforms.Grayscale()

        print("weight path: ", weight_path)
        print("output path: ", self.save_path)
        print("Testing Init...")

    def _build_model(self):
        net = PConvUNet(finetune=True, layer_size=7, in_ch=1)
        self.net = net.to(self.device)
        self.net.eval()
        print('Finish build model.')

    def test(self):
        for _, (input, name) in enumerate(self.dataloader):
            input = input.to(self.device)
            gray_input = self.gray_transform(input)

            recon_x = self.gray_transform(self.net(gray_input))
            mean_value = recon_x[0].mean()
            recon_x[0][recon_x[0] <= mean_value] = 0
            recon_x[0][recon_x[0] > mean_value] = 255

            save_image(recon_x[0], os.path.join(self.save_path, name[0]))

        print("Testing is completed.")

def main(args):
    if not os.path.isdir("./outputs"):
        os.mkdir("./outputs")

    if not os.path.isdir("./outputs/models"):
        os.mkdir("./outputs/models")

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    if args.mode == "train":
        img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])

        mask_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])

        dataset = FFHQ(data_root=args.data_path, img_transform=img_tf, mask_transform=mask_tf, data='train')
        criterion = InpaintingLoss(extractor=VGG16FeatureExtractor())
        trainer = Trainer(dataset, criterion, args.epochs, args.batch_size, args.lr, save_path=args.output_dir)
        trainer.train()
        
    elif args.mode == "test":
        img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            ])

        mask_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            ])
        dataset = FFHQ(data_root=args.data_path, img_transform=img_tf, mask_transform=mask_tf, data='test')
        tester = Tester(dataset, weight_path=args.test_weight_path, batch_size=1, save_path=args.output_dir)
        tester.test()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
