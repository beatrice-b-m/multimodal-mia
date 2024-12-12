import os
# import sys
import wandb
import torch

# from opacus.validators import ModuleValidator
# from opacus import PrivacyEngine

# from torchvision.datasets import CIFAR10
# from torchvision.transforms import v2
# from torchmetrics import MetricCollection
# from torchmetrics.classification import Accuracy, AUROC, F1Score, Precision, Recall

# scripts
from training.models import get_model_dict
from training.interface import handle_params
from training.backend import ModelTrainer#train_model
from training.data import get_transform_list, get_dataloader
from training.vae import KullbackLeiblerReconstructionLoss

def seed_torch(seed: int):
    """
    Seed all torch random number generators and set the deterministic flag.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    
def train_wrapper(param_dict: dict): 
    # -----------------------------------------------------------------------
    # limit GPU availability
    print(f"using GPU: {param_dict['gpu_num']}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param_dict["gpu_num"])
    print(f"CUDA available?: {torch.cuda.is_available()}")

    # -----------------------------------------------------------------------
    # get pytorch device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------------------------
    # prepare data loaders
    # define constant args
    const_dict = {
        "transform_list": get_transform_list(), 
        "batch_size": param_dict["batch_size"], 
        "data_dir": "data/mscoco", 
        "seed": param_dict["seed"], 
        "is_distributed": False, 
        "num_workers": 18, 
        "pin_memory": True
    }
    # get dataloaders and add to the loader_dict
    loader_dict = {
        "train": get_dataloader(**{**const_dict, "dataset_type": "train2014", "shuffle": True}),
        "val": get_dataloader(**{**const_dict, "dataset_type": "val2014"})
    }

    # -----------------------------------------------------------------------
    # set seed (we set a separate seed for the data so this can be whatever we want)
    seed_torch(param_dict["seed"])

    # -----------------------------------------------------------------------
    # define metric collection
    # num_classes = len(class_dict)
    # task_type = 'multiclass' if num_classes > 2 else 'binary'

    # # define captioning metrics
    # metric_collection = MetricCollection({
    #     'acc': Accuracy(task=task_type, num_classes=num_classes),
    #     'auc': AUROC(task=task_type, num_classes=num_classes),
    #     'prec': Precision(task=task_type, num_classes=num_classes),
    #     'rec': Recall(task=task_type, num_classes=num_classes),
    #     'f1': F1Score(task=task_type, num_classes=num_classes, average='micro')
    # })
    # metric_collection.to(device)
    metric_collection = None

    # -----------------------------------------------------------------------
    # define model
    model_dict = get_model_dict() # this was wrapped as a func so the model dict can be retrieved in the eval script
    model_func = model_dict[param_dict["model"]]
    model = model_func(param_dict) if param_dict["model_type"] == "decoder" else model_func()

    # -----------------------------------------------------------------------
    loss_func_dict = {
        "MSE":torch.nn.MSELoss(),
        "L1":torch.nn.L1Loss()
    }
    
    # define loss and optimizer
    if param_dict["model_type"] == "autoencoder":
        criterion = loss_func_dict[param_dict["loss_function"]]
        
    elif param_dict["model_type"] == "vae":
        # get joint kl divergence/l1 loss for the variational autoencoder
        criterion = KullbackLeiblerReconstructionLoss(
            recon_loss=loss_func_dict[param_dict["loss_function"]], 
            n_epochs=param_dict["n_epochs"], 
            b_cycle_time=param_dict["b_cycle_time"], 
            b_peak_time=param_dict["b_peak_time"]
        )

    else:
        criterion = torch.nn.CrossEntropyLoss()

    # params taken from https://arxiv.org/pdf/2206.13424v3 (made for resnet18)
    # will these still work???
    if param_dict['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=param_dict["learning_rate"])
    elif (param_dict['optimizer'] == 'adamw') & (param_dict['model_type'] == 'decoder'):
        # use the default gpt2 params if training a decoder
        optimizer = torch.optim.AdamW(model.parameters(), lr=param_dict["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    elif param_dict['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=param_dict["learning_rate"])
    elif param_dict['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=param_dict["learning_rate"], weight_decay=5e-4, momentum=0.9)
    else:
        raise ValueError(f"unrecognized optimizer '{param_dict['optimizer']}'")

    # -----------------------------------------------------------------------
    # get model save dir
    model_save_dir = param_dict.get("model_save_dir", None)
    if model_save_dir is None:
        base_path = "./logs"
        model_save_dir = os.path.join(base_path, param_dict["test_name"], param_dict["model"])
    print(f"saving model to: {model_save_dir}")
    
    # -----------------------------------------------------------------------
    # send the model to the gpu for training
    model = model.to(device)
    
    # initialize wandb to log the training
    wandb.init(
        # set the wandb project where this run will be logged
        project=param_dict["test_name"],
        # track hyperparameters and run metadata
        config=param_dict
    )

    try:
        # define the model trainer and start the training loop
        trainer = ModelTrainer(
            model=model, 
            device=device, 
            loader_dict=loader_dict, 
            criterion=criterion, 
            optimizer=optimizer, 
            save_dir=model_save_dir, 
            param_dict=param_dict, 
            metric_collection=metric_collection
        )
        trainer.start()

        # extract best metric
        best_metric = trainer._best

    except Exception as e:
        print(f"caught exception during training:\n{e}")
        best_metric = 0.0

    # end the wandb run manually in case this is being used for tuning
    wandb.finish()
    return best_metric

def get_param_dict():
    # wrapped in a function so the params can easily be synced with the
    # scheduler script
    return {
        "test_name": {
            "argname": "tn",
            "dtype": "str",
            "choices": None,
            "default": "cs573-project",
        },
        "gpu_num": {
            "argname": "g",
            "dtype": "int",
            "choices": [n for n in range(8)],
            "default": None,
        },
        "model": {
            "argname": "m",
            "dtype": "str",
            "choices": list(get_model_dict().keys()),
            "default": "rn18_ae",
        },
        "model_type": {
            "argname": "mt",
            "dtype": "str",
            "choices": ["autoencoder", "vae", "decoder"],
            "default": "autoencoder",
        },
        "optimizer": {
            "argname": "o",
            "dtype": "str",
            "choices": ['adam', 'adamw', 'sgd'],
            "default": 'sgd',
        },
        "seed": {
            "argname": "s",
            "dtype": "int",
            "choices": None,
            "default": 13,
        },
        "batch_size": {
            "argname": "b",
            "dtype": "int",
            "choices": None,
            "default": 200,
        },
        "learning_rate": {
            "argname": "lr",
            "dtype": "float",
            "choices": None,
            "default": 0.1,
        },
        "n_epochs": {
            "argname": "e",
            "dtype": "int",
            "choices": None,
            "default": 50,
        },
        "monitor_patience": {
            "argname": "p",
            "dtype": "int",
            "choices": None,
            "default": 10,
        },
        "model_save_dir": {
            "argname": "d",
            "dtype": "str",
            "choices": None,
            "default": "None",
        },
        "proj_up_ver": {
            "argname": "puv",
            "dtype": "str",
            "choices": ["A", "B"],
            "default": "A",
        },
        "autoencoder_weights": {
            "argname": "aew",
            "dtype": "str",
            "choices": None,
            "default": "None",
        },
        "captioner_weights": { # only considered during captioner training TODO: make consistent for all model types to load from partially trained checkpoint
            "argname": "capw",
            "dtype": "str",
            "choices": None,
            "default": "None",
        },
        "freeze_encoder": { # only considered during captioner training
            "argname": "fe",
            "dtype": "int",
            "choices": [0, 1],
            "default": 1,
        },
        "mixed_precision": {
            "argname": "mp",
            "dtype": "int",
            "choices": [0, 1],
            "default": 1,
        },
        "loss_function": {
            "argname": "lo",
            "dtype": "str",
            "choices": ["MSE", "L1"],
            "default": "L1",
        },
        "autoencoder": { # only relevant for embedding-type models
            "argname": "ae",
            "dtype": "str",
            "choices": None,
            "default": "None",
        },
        "b_cycle_time": { # only relevant for vae-type models
            "argname": "bct",
            "dtype": "int",
            "choices": None,
            "default": 5,
        },
        "b_peak_time": { # only relevant for vae-type models
            "argname": "bpt",
            "dtype": "int",
            "choices": None,
            "default": 2,
        },
    }

    
if __name__ == "__main__":
    # get the model parameter definitions then handle args/user input
    param_dict = get_param_dict()
    model_param_dict = handle_params(param_dict, args=True, confirm=True)
    model_param_dict.update({
        "monitor_metric": "loss",
    })
    
    # pass the model parameters to the train wrapper
    train_wrapper(model_param_dict)

    print("\nTraining complete.")
    