from bayes_opt import BayesianOptimization, acquisition, SequentialDomainReductionTransformer
import random
from pathlib import Path
import json
import os
import numpy as np

# scripts
from train import train_wrapper, get_param_dict
from training.interface import handle_params

class HyperparameterOptimizer:
    def __init__(self, name: str, pbounds: dict, seed: int = 13, verbose: int = 2, domain_reduction: bool = True, enable_log: bool = True):
        self.name = name
        self.pbounds = pbounds
        self.seed = seed
        self.verbose = verbose
        self.bounds_transformer = None

        if domain_reduction:
           self.bounds_transformer = self._get_bounds_transformer()
        self._acq = acquisition.UpperConfidenceBound(kappa=2.5)
        self._opt = self._get_optimizer()

        self._i = 0
        self.enable_log = enable_log
        self.log = []

    def register(self, params: dict, target: float):
        self._opt.register(params=params, target=target)
        print(f"registered {params} -> {target}")
        if self.enable_log:
            self.log.append({"run": self._i, "params": params, "target": target})
        self._i += 1
    
    def suggest(self):
        return self._opt.suggest()
    
    def suggest_random(self):
        suggested_params = dict()

        for param, (bound_min, bound_max) in self.pbounds.items():
            suggested_params[param] = random.uniform(bound_min, bound_max)
            
        return suggested_params

    def save_log(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.name}.log")
        print(f"saving log to {save_path}")
        with open(save_path, 'w') as f:
            for iteration in self.log:
                json.dump(iteration, f)
                f.write('\n')

    @staticmethod
    def _get_bounds_transformer(**kwargs):
        return SequentialDomainReductionTransformer(**kwargs)
    
    # @staticmethod
    # def _get_utility(**kwargs):
    #     return UtilityFunction(**kwargs)
    
    def _get_optimizer(self):
        return BayesianOptimization(
            f=None,
            acquisition_function=self._acq,
            pbounds=self.pbounds,
            verbose=self.verbose,
            random_state=self.seed,
            bounds_transformer=self.bounds_transformer
        )


def parse_opt_params(opt_param_dict: dict):
    cat_var_dict = dict()
    out_param_dict = opt_param_dict.copy()

    for k, v in opt_param_dict.items():
        # code categorical vars with lists
        if isinstance(v, list):
            # send the list v to the categorical vars dict
            cat_var_dict[k] = v
            # set range that results in an even likelihood for all categories when
            # int(suggested_v) is applied
            out_param_dict[k] = (0.0, len(v)-1e-5)

    return out_param_dict, cat_var_dict


def fix_categorical_vars(run_params: dict, cat_dict: dict):
    out_run_params = run_params.copy()
    
    cat_var_list = list(cat_dict.keys())
    if len(cat_var_list) < 1:
        pass
        
    else:
        for var_name, v_float in run_params.items():
            var_category_list = cat_dict.get(var_name, None)
            if var_category_list is None:
                continue
            else:
                # cast v float to an int then use it as the index for the
                # category list
                out_run_params[var_name] = var_category_list[int(v_float)]

    return out_run_params


def opt_loop(model_dict: dict, cat_dict: dict, optimizer: HyperparameterOptimizer, 
             init_run_list: list[dict]=None):
    if init_run_list is None:
        init_run_list = []

    # build a list to track if each run is guided or not
    is_run_guided_list = [False]*model_dict["n_init"] + [True]*model_dict["n_guided"]

    for i, run_is_guided in enumerate(is_run_guided_list):
        print(f"\n{'guided ' if run_is_guided else ''}tuning iteration {i} {'-'*40}")
        # get run parameters
        if run_is_guided: # if guided, get suggestion from gaussian process
            run_params = optimizer.suggest()
        elif len(init_run_list): # or if there are entries in init_run_list
            run_params = init_run_list.pop(0)
        else: # otherwise randomly generate some parameters to try
            run_params = optimizer.suggest_random()

        fixed_params = fix_categorical_vars(run_params, cat_dict)
        # copy the base model params dict and overwrite any tuned params
        full_run_params = model_dict.copy()
        full_run_params.update(fixed_params)

        # train the model and get its best eval, then register it
        run_eval = train_wrapper(full_run_params)
        optimizer.register(run_params, run_eval)
        


def opt_wrapper(opt_param_dict: dict, init_run_list: list, const_params: dict | None = None):
    # disable wandb printing so our log is a little quieter for tuning
    os.environ["WANDB_SILENT"] = "true"
    
    # handle general params
    # chosen hparams for optimization will automatically override these
    param_dict = get_param_dict()
    param_dict.update({
        "n_init": { # n initial iterations
            "argname": "ni",
            "dtype": "int",
            "choices": None,
            "default": 10,
        },
        "n_guided": { # n guided iterations
            "argname": "ng",
            "dtype": "int",
            "choices": None,
            "default": 20,
        },
    })
    
    model_param_dict = handle_params(param_dict, args=True, confirm=True)
    model_param_dict.update({
        "monitor_metric": "loss",
        "monitor_patience": 5,
        'mixed_precision': True
    })
    if const_params is not None:
        model_param_dict.update(const_params)

    # parse categorical variables to assign a float interval to each
    # (our optimizer can only work with floats, we'll use the cat_var_dict
    # to parse them back)
    opt_bounds_dict, cat_var_dict = parse_opt_params(opt_param_dict)

    print(f"\n\noptimizer bounds:\nraw:\t{opt_param_dict}\nparsed:\t{opt_bounds_dict}")
    optimizer = HyperparameterOptimizer(
        name=model_param_dict["test_name"], 
        pbounds=opt_bounds_dict, 
        seed=model_param_dict["seed"]
    )

    # run the optimization loop
    opt_loop(
        model_dict=model_param_dict, 
        cat_dict=cat_var_dict, 
        optimizer=optimizer,    
        init_run_list=init_run_list
    )

    # save the optimizer log
    optimizer.save_log(save_dir=f"./logs/tuning/{optimizer.name}")

    best_eval = -np.inf
    best_iter_dict = None
    
    print("\ntuning history:")
    for iter_dict in optimizer.log:
        param_list = [f"{k}: {v:.4f}" for k,v in iter_dict['params'].items()]
        print(f"run: {iter_dict['run']}, {', '.join(param_list)}, target: {iter_dict['target']:.4f}")
        
        if iter_dict['target'] > best_eval:
            best_eval = iter_dict['target']
            best_iter_dict = iter_dict

    print("\nretraining best model...")
    print(best_iter_dict)
    # retrain the best model with the highest performing params
    # to ensure we've saved it
    fixed_best_params = fix_categorical_vars(best_iter_dict['params'], cat_var_dict)
    full_best_params = model_param_dict.copy()
    full_best_params.update(fixed_best_params)
    full_best_params.update({
        "model_save_dir": f"./logs/tuning/{optimizer.name}"
    })

    # train the model and get its best eval, then register it
    _ = train_wrapper(full_best_params)
    
    print(f"\ntuning completed and best model saved to {full_best_params['model_save_dir']}.\n\n")
    
    return optimizer
    
