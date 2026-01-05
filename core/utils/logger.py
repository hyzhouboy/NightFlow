from torch.utils.tensorboard import SummaryWriter
from loguru import logger as loguru_logger
import torch
import numpy as np
import torchvision.utils as vutils

def tensor2numpy(var_dict):
    for key, vars in var_dict.items():
        if isinstance(vars, np.ndarray):
            var_dict[key] = vars
        elif isinstance(vars, torch.Tensor):
            var_dict[key] = vars.data.cpu().numpy()
        else:
            raise NotImplementedError("invalid input type for tensor2numpy")

    return var_dict

class Logger:
    def __init__(self, model, scheduler, cfg):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.cfg = cfg
        if self.cfg.log_dir is None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(self.cfg.log_dir)

    def _print_training_status(self):
        # metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        # metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        metrics_str = ""
        for k in sorted(self.running_loss.keys()):
            metrics_str += ("  {:s}:{:.4f}, ").format(k, self.running_loss[k]/self.cfg.sum_freq)
        
        # print the training status
        print(training_str + metrics_str)

        # if self.writer is None:
        #     self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/self.cfg.sum_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.cfg.sum_freq == self.cfg.sum_freq-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        # if self.writer is None:
        #     self.writer = SummaryWriter()
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def save_image(self, mode_tag, images_dict):
        images_dict = tensor2numpy(images_dict)

        for tag, values in images_dict.items():
            if not isinstance(values, list) and not isinstance(values, tuple):
                values = [values]
            for idx, value in enumerate(values):
                if len(value.shape) == 3:
                    value = value[:, np.newaxis, :, :]
                value = value[:1]
                value = torch.from_numpy(value)

                image_name = '{}/{}'.format(mode_tag, tag)
                if len(values) > 1:
                    image_name = image_name + "_" + str(idx)
                self.writer.add_image(image_name, vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True),
                                self.total_steps)


    def close(self):
        self.writer.close()

