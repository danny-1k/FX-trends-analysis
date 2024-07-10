import os
import torch
from tqdm import tqdm
from utils.utils import RunningAverager
from torchvision.utils import make_grid



class Trainer:
    def __init__(
            self, 
            model, 
            optimiser, 
            lossfn, 
            train, 
            test, 
            writer, 
            save_dir, 
            image_size=(64, 128),
            device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.model = model
        self.model.to(device)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.optimiser = optimiser
        self.train = train
        self.test = test
        self.writer = writer
        self.save_dir = save_dir
        self.image_size = image_size
        self.lossfn = lossfn
        self.device = device

        self.train_loss = RunningAverager(smooth=0.3)
        self.test_loss = RunningAverager(smooth=0.3)

        self.best_loss = float("inf")


    def run(self, epochs):

        for epoch in range(epochs):
            self._run_train_once(epoch)
            self._run_test_once(epoch)

            self._log_training_stats(epoch)

            if self.test_loss.value < self.best_loss:
                self._save_checkpoint(epoch)
                self.best_loss = self.test_loss.value


            self.train_loss.reset()
            self.test_loss.reset()


    def _run_train_once(self, epoch):
        self.model.train()

        for x in tqdm(self.train):
            self.optimiser.zero_grad()

            x = x.to(self.device)

            z, p = self.model(x)

            loss = self.lossfn(z, p, x)

            loss.backward()
            self.optimiser.step()

            self.train_loss.update(loss.item())

        self.writer.add_scalar(tag="Train/Loss", scalar_value=self.train_loss.value, global_step=epoch)
        self.writer.add_image(tag="Train/Predictions", img_tensor=make_grid(p.view(p.shape[0], 1, *self.image_size)), global_step=epoch)
        self.writer.add_image(tag="Train/GroundTruth", img_tensor=make_grid(x.view(x.shape[0], 1, *self.image_size)), global_step=epoch)


    @torch.no_grad()
    def _run_test_once(self, epoch):
        self.model.eval()

        for x in tqdm(self.test):
            x = x.to(self.device)

            z, p = self.model(x)

            loss = self.lossfn(z, p, x)

            self.test_loss.update(loss.item())

        self.writer.add_scalar(tag="Test/Loss", scalar_value=self.test_loss.value, global_step=epoch)
        self.writer.add_image(tag="Test/Predictions", img_tensor=make_grid(p.view(p.shape[0], 1, *self.image_size)), global_step=epoch)
        self.writer.add_image(tag="Test/GroundTruth", img_tensor=make_grid(x.view(x.shape[0], 1, *self.image_size)), global_step=epoch)


    def _log_training_stats(self, epoch):
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and not isinstance(parameter.grad, type(None)):
                self.writer.add_histogram(name, parameter, epoch)
                self.writer.add_histogram(f"{name}.grad", parameter.grad, epoch)

    
    def _save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "train_loss": self.train_loss.value,
            "test_loss": self.test_loss.value
        }

        torch.save(checkpoint, os.path.join(self.save_dir, "checkpoint.pt"))

        
class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run_train_once(self, epoch):
        self.model.train()

        for x1, x2, label in tqdm(self.train):
            self.optimiser.zero_grad()

            x1 = x1.to(self.device)
            x2 = x2.to(self.device)

            z1, p1 = self.model(x1)
            z2, p2 = self.model(x2)

            loss = self.lossfn(z1, z2, p1, p2, x1, x2, label)

            loss.backward()
            self.optimiser.step()

            self.train_loss.update(loss.item())

        self.writer.add_scalar(tag="Train/Loss", scalar_value=self.train_loss.value, global_step=epoch)
        self.writer.add_image(tag="Train/Predictions1", img_tensor=make_grid(p1.view(p1.shape[0], 1, *self.image_size)), global_step=epoch)
        self.writer.add_image(tag="Train/Predictions2", img_tensor=make_grid(p1.view(p2.shape[0], 1, *self.image_size)), global_step=epoch)
        self.writer.add_image(tag="Train/GroundTruth1", img_tensor=make_grid(x1.view(x1.shape[0], 1, *self.image_size)), global_step=epoch)
        self.writer.add_image(tag="Train/GroundTruth2", img_tensor=make_grid(x2.view(x2.shape[0], 1, *self.image_size)), global_step=epoch)
    
    @torch.no_grad()
    def _run_test_once(self, epoch):
        self.model.eval()

        for x1, x2, label in tqdm(self.test):
            self.optimiser.zero_grad()

            x1 = x1.to(self.device)
            x2 = x2.to(self.device)

            z1, p1 = self.model(x1)
            z2, p2 = self.model(x2)

            loss = self.lossfn(z1, z2, p1, p2, x1, x2, label)

            self.test_loss.update(loss.item())

        self.writer.add_scalar(tag="Test/Loss", scalar_value=self.test_loss.value, global_step=epoch)
        self.writer.add_image(tag="Test/Predictions1", img_tensor=make_grid(p1.view(p1.shape[0], 1, *self.image_size)), global_step=epoch)
        self.writer.add_image(tag="Test/Predictions2", img_tensor=make_grid(p1.view(p2.shape[0], 1, *self.image_size)), global_step=epoch)
        self.writer.add_image(tag="Test/GroundTruth1", img_tensor=make_grid(x1.view(x1.shape[0], 1, *self.image_size)), global_step=epoch)
        self.writer.add_image(tag="Test/GroundTruth2", img_tensor=make_grid(x2.view(x2.shape[0], 1, *self.image_size)), global_step=epoch)