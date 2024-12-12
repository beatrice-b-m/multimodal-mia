import os
# import sys
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
# from functools import partialmethod
import wandb

def seed_torch(seed: int):
    """
    Seed all torch random number generators and set the deterministic flag.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



class ModelTrainer:
    def __init__(self, model: torch.nn.Module | dict, device: str, 
                 loader_dict: dict, criterion: torch.nn.Module | None, 
                 optimizer: torch.nn.Module, save_dir: str, param_dict: dict,
                 metric_collection=None):
        self.model = model
        self.device = device
        self.loader_dict = loader_dict
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_dir = save_dir
        self._params = param_dict
        # self._patience = self._params['monitor_patience']
        self.metric_collection = metric_collection

        self._best = None
        self._best_save_path = None
        self._monitor_is_loss = None
        self._train_log = None
        self._epochs_wo_improvement = None
        self._phase_func = None
        self._scaler = None

        self._init_save_dir()
        self._select_phase_func()

    def _init_save_dir(self):
        if self.save_dir is not None:
            if self._params["monitor_metric"] == 'loss':
                self._best = np.inf
                self._monitor_is_loss = True
            else:
                self._best = -np.inf
                self._monitor_is_loss = False

            # if save dir doesn't exist, make it
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            self._best_save_path = os.path.join(self.save_dir, 'best.pth')

    def _select_phase_func(self):
        # select epoch phase function
        phase_func_dict = {
            "vae": self._vae_epoch_phase,
            "autoencoder": self._autoencoder_epoch_phase,
            "decoder": self._decoder_epoch_phase
        }
        self._phase_func = phase_func_dict[self._params["model_type"]]

    def start(self):
        # init value at training start
        self._train_log = []
        self._epochs_wo_improvement = 0

        # init the gradient scaler if using mixed precision training
        if self._params.get('mixed_precision', False):
            self._scaler = torch.amp.GradScaler(self.device)

        # iterate over epochs
        for epoch in range(self._params["n_epochs"]):
            # check if there have been too many epochs without val improvement
            if self.out_of_patience():
                print("\nending training early...\n")
                break
            
            epoch_metrics_dict = dict()

            print(f"\nepoch {epoch} {'-' * 30}")

            # if vae training, step with loss
            if self._params["model_type"] == "vae":
                self.criterion.step(epoch)
                print(f'using beta: {self.criterion._b}')

            # perform train/val phases
            for phase in ['train', 'val']:
                phase_metrics_dict = self._phase_func(phase)

                # add phase dict to the epoch metrics dict
                epoch_metrics_dict[phase] = phase_metrics_dict
                
                # if we want to save the model and the phase is val
                if (phase == "val") and (self.save_dir is not None):
                    # extract the candidate for the new best metric
                    best_candidate = phase_metrics_dict[self._params["monitor_metric"]]
                    
                    if self._is_candidate_best(best_candidate):
                        self._best = best_candidate
                        print(f"saving model with best {self._params['monitor_metric']} '{self._best:.4f}'...")
                        self._epochs_wo_improvement = 0
                        
                        # get checkpoint
                        torch.save(self.model.state_dict(), self._best_save_path)

                    # if this epoch was not the best so far and early stopping is enabled
                    elif self._params["monitor_patience"] is not None:
                        # increment the counter
                        self._epochs_wo_improvement += 1

            # format the metric names for wandb and log the epoch
            if self._params.get("enable_wandb", 1):
                train_dict = {f"train/{k}":v for k, v in epoch_metrics_dict["train"].items()}
                val_dict = {f"val/{k}":v for k, v in epoch_metrics_dict["val"].items()}
                wandb.log({**train_dict, **val_dict})

            # add current epoch metrics object to the log list
            self._train_log.append(epoch_metrics_dict)

    def _vae_epoch_phase(self, phase: str):
        # select current dataloader from the loader dict
        phase_loader = self.loader_dict[phase]
        phase_size = len(phase_loader)

        # if phase is training, set model to training mode
        # otherwise set model to eval mode
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        # init the running loss for the epoch
        running_loss = 0.0

        # iterate over data in current phase loader
        with tqdm(phase_loader, unit="batch", total=phase_size) as epoch_iter:
            for batch, data in enumerate(epoch_iter):
                # unpack data and send them to the device
                X, _ = data
                X = X.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # track history if in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                        # get model outputs for the current batch
                        preds, mean, logvar = self.model(X)

                        # get the reconstruction loss for the current batch preds
                        # use the input images as the ground truth
                        # monitor loss is unweighted by the beta scheduler and should only be used for model performance tracking
                        weighted_loss, monitor_loss = self.criterion(preds, X, mean, logvar)

                    running_loss += monitor_loss.item()

                    # update metric collection
                    # metric_collection.update(preds, y)

                    # if train phase, backpropogate and step with the optimizer
                    if phase == 'train':
                        # loss.backward()
                        # optimizer.step()
                        self._scaler.scale(weighted_loss).backward()
                        self._scaler.step(self.optimizer)
                        self._scaler.update()

                # update metrics in tqdm display after each 10% chunk
                # or if in val/test phase, update on the last batch
                if ((phase == 'train') & (batch % (max(phase_size // 10, 1)) == 0)) |\
                ((phase != 'train') & (batch == (phase_size - 1))):
                    # compute the metrics and output them to a dict                
                    metrics_dict = {"loss": running_loss / (batch + 1)}
                    # metrics_dict.update({k:v.item() for k, v in metric_collection.compute().items()})

                    # unpack metrics dict into a new dict with the phase label
                    epoch_iter.set_postfix({**{"phase": phase}, **metrics_dict})

        # reset the metric collection at the end of the current phase
        # metric_collection.reset()
        
        return metrics_dict

    def _autoencoder_epoch_phase(self, phase: str):
        # select current dataloader from the loader dict
        phase_loader = self.loader_dict[phase]
        phase_size = len(phase_loader)

        # if phase is training, set model to training mode
        # otherwise set model to eval mode
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        # init the running loss for the epoch
        running_loss = 0.0

        # iterate over data in current phase loader
        with tqdm(phase_loader, unit="batch", total=phase_size) as epoch_iter:
            for batch, data in enumerate(epoch_iter):
                # unpack data and send them to the device
                X, _ = data
                X = X.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # track history if in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                        # get model outputs for the current batch
                        preds = self.model(X)

                        # get the reconstruction loss for the current batch preds
                        # use the input images as the ground truth
                        loss = self.criterion(preds, X)

                    running_loss += loss.item()

                    # update metric collection
                    # metric_collection.update(preds, y)

                    # if train phase, backpropogate and step with the optimizer
                    if phase == 'train':
                        # loss.backward()
                        # optimizer.step()
                        self._scaler.scale(loss).backward()
                        self._scaler.step(self.optimizer)
                        self._scaler.update()

                # update metrics in tqdm display after each 10% chunk
                # or if in val/test phase, update on the last batch
                if ((phase == 'train') & (batch % (max(phase_size // 10, 1)) == 0)) |\
                ((phase != 'train') & (batch == (phase_size - 1))):
                    # compute the metrics and output them to a dict                
                    metrics_dict = {"loss": running_loss / (batch + 1)}
                    # metrics_dict.update({k:v.item() for k, v in metric_collection.compute().items()})

                    # unpack metrics dict into a new dict with the phase label
                    epoch_iter.set_postfix({**{"phase": phase}, **metrics_dict})

        # reset the metric collection at the end of the current phase
        # metric_collection.reset()
        
        return metrics_dict
    

    def _decoder_epoch_phase(self, phase):
        # select current dataloader from the loader dict
        phase_loader = self.loader_dict[phase]
        phase_size = len(phase_loader)

        # if phase is training, set model to training mode
        # otherwise set model to eval mode
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        # init the running loss for the epoch
        running_loss = 0.0

        # iterate over data in current phase loader
        with tqdm(phase_loader, unit="batch", total=phase_size) as epoch_iter:
            for batch, batch_data in enumerate(epoch_iter):
                # unpack data and send them to the device
                batch_imgs, batch_captions = batch_data
                batch_imgs = batch_imgs.to(self.device)

                # tokenize captions and get input ids and attention masks
                batch_tokens = self.model.tokenize(batch_captions)
                batch_tokens = {k:v.to(self.device) for k,v in batch_tokens.items()}

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # track history if in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                        outputs = self.model(batch_imgs, batch_tokens)
        
                    running_loss += outputs.loss.item()

                    # update metric collection
                    # metric_collection.update(preds, y)

                    # if train phase, backpropogate and step with the optimizer
                    if phase == 'train':
                        # loss.backward()
                        # optimizer.step()
                        self._scaler.scale(outputs.loss).backward()
                        self._scaler.step(self.optimizer)
                        self._scaler.update()

                # update metrics in tqdm display after each 10% chunk
                # or if in val/test phase, update on the last batch
                if ((phase == 'train') & (batch % (max(phase_size // 10, 1)) == 0)) |\
                ((phase != 'train') & (batch == (phase_size - 1))):
                    # compute the metrics and output them to a dict                
                    metrics_dict = {"loss": running_loss / (batch + 1)}
                    # metrics_dict.update({k:v.item() for k, v in metric_collection.compute().items()})

                    # unpack metrics dict into a new dict with the phase label
                    epoch_iter.set_postfix({**{"phase": phase}, **metrics_dict})

        # reset the metric collection at the end of the current phase
        # metric_collection.reset()
        
        return metrics_dict


    def _is_candidate_best(self, candidate):
        if self._monitor_is_loss:
            return candidate < self._best
        else:
            return candidate > self._best

    def out_of_patience(self):
        patience = self._params['monitor_patience']
        if patience is None: # infinite patience
            return False
        elif self._epochs_wo_improvement > patience: # out of patience
            return True
        else: # remaining patience
            return False

