import argparse
import subprocess
import re

# this script handles the command line interface for the training script
# given a parameter dictionary (shown below) it queries the user to configure
# an output dictionary

def handle_params(param_dict: dict, args: bool = True, confirm: bool = True):
    # init value to None for all inner parameter dicts
    for pname, pdict in param_dict.items():
        pdict['value'] = None

    # if args, handle any command line arguments
    if args:
        args_dict = handle_param_args(param_dict)
        
        for argname, argval in args_dict.items():
            if argname not in param_dict.keys():
                continue
            elif argval is not None:
                param_dict[argname]["value"] = argval
            elif (param_dict[argname]["default"] is not None) and (args_dict['use_defaults'] == True):
                param_dict[argname]["value"] = param_dict[argname]["default"]
    # otherwise instantiate this object as an empty dict so we don't get errors 
    else:
        args_dict = dict()
    
        
    if args_dict.get('skip_cli', False) != True:
        # get cli input for parameters with no set value
        out_param_dict = handle_param_cli(param_dict, confirm=confirm)

    else:
        type_checker = InputTypeChecker()
        out_param_dict = dict()
        fail_list = list()

        # extract the values we've defined for our params and make sure they've all been set
        for pname, pdict in param_dict.items():
            pval = pdict['value']
            
            if pval is None:
                fail_list.append(pname)
            else:
                # cast the param value to the right format
                pval = type_checker[pdict['dtype']].format(pval)
                out_param_dict[pname] = pval

        if fail_list:
            raise ValueError(f"`skip_cli` is True but there are still undefined parameters! {fail_list}")

    return out_param_dict

def handle_param_args(param_dict: dict):
    # define command line args with argparser
    parser = argparse.ArgumentParser() # offer some way to set this?

    # extract the name and the dict for each parameter
    for pname, pdict in param_dict.items():
        # check if the current param has defined choices and try to map its elements to strings
        pchoices = pdict.get('choices', None)
        pchoices = list(map(str, pchoices)) if pchoices is not None else None
        
        # format them into an argument
        parser.add_argument(
            f"-{pdict['argname']}", 
            f"--{pname}",
            choices=pchoices
        )

    # add additional arguments here
    parser.add_argument('-ud', '--use_defaults', default=False, action='store_true') #
    parser.add_argument('-sc', '--skip_cli', default=False, action='store_true')

    # parse the args and output them as a dict
    args = parser.parse_args()
    return args.__dict__

def handle_param_cli(param_dict: dict, confirm: bool = True):
    type_checker = InputTypeChecker()
    
    out_param_dict = dict()

    for param_name, param_details in param_dict.items():
        # skip any parameters which have already been defined
        if param_details['value'] is not None:
            out_param_dict[param_name] = type_checker[param_details['dtype']].format(param_details['value'])
            continue

        # if we're selecting a gpu, print the available memory on each card
        if param_name == "gpu_num":
            print_gpu_memory()
            
        # if the parameter has defaults/defined choices format them for printing
        choices_str = " " + str(param_details['choices']) if param_details["choices"] is not None else ""
        defaults_str = f" [default: {param_details['default']}]" if param_details["default"] is not None else ""
        input_string = f"Select the {param_name}{choices_str}{defaults_str}: "
        
        # while valid input has not been received
        while True:
            selection = input(input_string).strip()

            # handle defaults
            if (selection == "") & (param_details["default"] is not None):
                selection = param_details["default"]
                break
            else:
                # this should throw an error if we don't have a handler for the dtype
                if not type_checker[param_details['dtype']].validate:
                    print(f"Please enter a valid {param_details['dtype']}")
                    continue

                # cast the selection string to the right format
                selection = type_checker[param_details['dtype']].format(selection)

                # if a list of choices is defined and our selection isn't in it try again
                if (param_details["choices"] is not None) and (selection not in param_details["choices"]):
                    print(f"Please choose a {param_details['dtype']} in {param_details['choices']}")
                    continue
                else:
                    break

        out_param_dict[param_name] = selection

    # if user confirmation is not required, just return the out dict
    if not confirm:
        return out_param_dict
    else:
        # confirm parameters with user
        print("\nTraining model with parameters:")
        for param_name, param_value in out_param_dict.items():
            print(f"{param_name}: {param_value}")
    
        # pause until user accepts parameters
        while True:
            confirmed = input("Continue? [y/n] ").strip().lower()
            if confirmed == "n":
                return handle_param_cli(param_dict, confirm=confirm)
            elif confirmed == "y":
                return out_param_dict


    

class InputTypeChecker:
    def __init__(self):
        self.handler_dict = {
            "int": IntHandler,
            "float": FloatHandler,
            "str": StrHandler,
        }

    def __getitem__(self, key: str):
        return self.handler_dict[key]


class IntHandler:
    @staticmethod
    def validate(value: str):
        try:
            int(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def format(value: str):
        return int(value)

class FloatHandler:
    @staticmethod
    def validate(value: str):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def format(value: str):
        return float(value)

class StrHandler:
    @staticmethod
    def validate(value: str):
        return True

    @staticmethod
    def format(value: str):
        # for strings we should handle the edge case where our default should be None
        # code these as default: "None" then fix here
        return None if value == "None" else value

def get_gpu_memory():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'])
    lines = result.decode('utf-8').strip().split('\n')
    gpus = []
    for line in lines:
        values = line.split(', ')
        gpu = {
            'index': int(values[0]),
            'total': int(values[1]),
            'used': int(values[2]),
            'free': int(values[3])
        }
        gpus.append(gpu)
    return gpus

def print_gpu_memory():
    gpus = get_gpu_memory()
    print("GPU Memory Usage:")
    print("┌───────┬─────────────┬─────────────┬─────────────┐")
    print("│ GPU   │ Total       │ Used        │ Free        │")
    print("├───────┼─────────────┼─────────────┼─────────────┤")
    for gpu in gpus:
        print(f"│ {gpu['index']:<5} │ {gpu['total']:>8} MB │ {gpu['used']:>8} MB │ {gpu['free']:>8} MB │")
    print("└───────┴─────────────┴─────────────┴─────────────┘")

if __name__ == "__main__":
    def get_param_dict():
        # model params definitions should be wrapped in a function so they can easily be shared
        # between scripts (e.g. imported by the scheduler to auto-determine hyperparam. ranges)
        return {
            "gpu_num": {
                "argname": "g",
                "dtype": "int",
                "choices": [n for n in range(4)],
                "default": None,
            },
            "dataset": {
                "argname": "d",
                "dtype": "str",
                "choices": ["datasetA", "datasetB"],
                "default": "datasetA",
            },
            "model": {
                "argname": "m",
                "dtype": "str",
                "choices": ["modelA", "modelB", "modelC"],
                "default": None,
            },
            "learning_rate": {
                "argname": "l",
                "dtype": "float",
                "choices": None,
                "default": 0.0002,
            },
            "n_epochs": {
                "argname": "e",
                "dtype": "int",
                "choices": None,
                "default": 100,
            },
        }
    # get the model parameter definitions then handle args/user input
    param_dict = get_param_dict()
    model_param_dict = handle_params(param_dict, args=True, confirm=True)

    print(f"\n\nmodel parameter output:\n{model_param_dict}")
    