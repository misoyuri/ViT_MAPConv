import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image


def evaluate(model, dataset, device, filename, experiment=None):
    print('Start the evaluation')
    model.eval()
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output_phase1, output_phase2, _ = model(image.to(device), mask.to(device))
    output_phase1 = output_phase1.to(torch.device('cpu'))
    output_phase2 = output_phase2.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output_phase2

    grid = make_grid(torch.cat([image, mask, output_phase1, output_phase2, output_comp, gt], dim=0))
    save_image(grid, filename)
    if experiment is not None:
        experiment.log_image(filename, filename)

