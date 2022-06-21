from torch.utils.data import DataLoader

from .utils import save_ckpt, to_items
from .evaluate_TwoPhase import evaluate


class Trainer(object):
    def __init__(self, step, config, device, model, dataset_train,
                 dataset_val, criterion, optimizer, experiment):
        self.stepped = step
        self.config = config
        self.device = device
        self.model = model
        self.dataloader_train = DataLoader(dataset_train,
                                           batch_size=config.batch_size,
                                           shuffle=True)
        self.dataset_val = dataset_val
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluate = evaluate
        self.experiment = experiment

    def iterate(self):
        print('Start the training')
        for epoch in range(self.config.max_epoch):
            step = 0
            for _, (input, mask, gt) in enumerate(self.dataloader_train):
                loss_dict = self.train(step+self.stepped, input, mask, gt)
                # report the loss
                if step % self.config.log_interval == 0:
                    self.report(step+self.stepped, loss_dict)

                # evaluation
                if (step+self.stepped + 1) % self.config.vis_interval == 0 \
                        or step == 0 or step + self.stepped == 0:
                    # set the model to evaluation mode
                    self.model.eval()
                    self.evaluate(self.model, self.dataset_val, self.device,
                                '{}/val_vis/epoch{}_{}.png'.format(self.config.ckpt,
                                                                    str(epoch).zfill(3), step+self.stepped),
                                self.experiment)

                # save the model
                if (step+self.stepped + 1) % self.config.save_model_interval == 0 \
                        or (step + 1) == self.config.max_iter:
                    print('Saving the model...')
                    save_ckpt('{}/models/epoch{}_{}.pth'.format(self.config.ckpt,
                                                                str(epoch).zfill(3), step+self.stepped + 1),
                            [('model', self.model)],
                            [('optimizer', self.optimizer)],
                            step+self.stepped + 1)

                if step >= self.config.max_iter:
                    break
                step += 1
            
            self.stepped += step

    def train(self, step, input, mask, gt):
        # set the model to training mode
        self.model.train()

        # send the input tensors to cuda
        input = input.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)

        # model forward
        output_phase1, output_phase2, _ = self.model(input, mask)
        loss_dict = self.criterion(input, mask, output_phase1, output_phase2, gt)
        loss = 0.0
        for key, val in loss_dict.items():
            coef = getattr(self.config, '{}_coef'.format(key))
            loss += coef * val

        # updates the model's params
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict['total'] = loss
        return to_items(loss_dict)

    def report(self, step, loss_dict):
        print('[STEP: {:>6}] | Valid Loss: {:.6f} | Hole Loss: {:.6f}'\
              '| Perc Loss: {:.6f} | Style Loss: {:.6f}'.format(
                        step, loss_dict['valid_phase1'], loss_dict['hole_phase1'],
                        loss_dict['perc_phase1'], loss_dict['style_phase1']))

        print('[STEP: {:>6}] | Valid Loss: {:.6f} | Hole Loss: {:.6f}'\
              '| Perc Loss: {:.6f} | Style Loss: {:.6f} '\
              '| TV Loss: {:.6f} | Total Loss: {:.6f}'.format(
                        step, loss_dict['valid_phase2'], loss_dict['hole_phase2'],
                        loss_dict['perc_phase2'], loss_dict['style_phase2'],
                         loss_dict['tv_phase2'], loss_dict['total']))

        if self.experiment is not None:
            self.experiment.log_metrics(loss_dict, step=step)