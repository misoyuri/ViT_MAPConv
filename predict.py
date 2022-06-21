import argparse
from distutils.util import strtobool
import os

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from src.model_MAPConv import VisionFaceTransformer
from torchvision.utils import save_image
from mask_src.mask_unet import PConvUNet
from glob import glob
from torchvision.utils import make_grid
from torchvision.utils import save_image

def main(args):
    # Define the used device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the model
    print("Loading the Model...")
    inpaint_model = VisionFaceTransformer()
    inpaint_model.load_state_dict(torch.load(args.model, map_location=device)['model'])
    inpaint_model = inpaint_model.to(device)
    inpaint_model.eval()

    mask_net = PConvUNet(finetune=True, layer_size=7, in_ch=1)
    mask_net.load_state_dict(torch.load(args.mask_model))
    mask_net = mask_net.to(device)
    mask_net.eval()
    
    image_paths = glob('{}/*.png'.format(args.img))
    
    for idx in range(len(image_paths)):
        img_path = image_paths[idx]

        
        # Loading Input and Mask
        print("Loading the inputs...")
        org = Image.open(img_path)
        inp = TF.to_tensor(org.convert('RGB'))
        gray_input = TF.to_tensor(org.convert('L'))

        # Model prediction
        print("Model Prediction...")
        with torch.no_grad():
            inp_ = inp.unsqueeze(0).to(device)
            gray_input = gray_input.unsqueeze(0).to(device)

            if args.resize:
                org_size = inp_.shape[-2:]
                inp_ = F.interpolate(inp_, size=224)

            mask_ = mask_net(gray_input)
            mean_value = mask_[0].mean()
            mask_[0][mask_[0] <= mean_value] = 0.0
            mask_[0][mask_[0] > mean_value] = 1.0

            raw_out, _ = inpaint_model(inp_, mask_)
        if args.resize:
            raw_out = F.interpolate(raw_out, size=org_size)

        # Post process
        raw_out = raw_out.to(torch.device('cpu')).squeeze()
        raw_out = raw_out.clamp(0.0, 1.0)

        mask_ = mask_.to(torch.device('cpu')).squeeze()
        out = mask_ * inp + (1 - mask_) * raw_out

        # Saving an output image
        img_name = img_path.split('/')[-1]

        file_save_path = os.path.join(args.output_dir, img_name)
        save_image(raw_out, file_save_path)
        print("Saving the output... {}".format(img_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the inputs")
    parser.add_argument('--img', type=str, default="examples/img0.jpg")
    parser.add_argument('--model', type=str, default="./saved_models/epoch010_70000.pth")
    parser.add_argument('--mask_model', type=str, default="./saved_models/weight_epoch001_step3200.pth")
    parser.add_argument('--output_dir', type=str, default="./outputs")
    parser.add_argument('--resize', type=strtobool, default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
