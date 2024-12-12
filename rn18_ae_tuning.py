from training.optimizer import opt_wrapper

"""
script run with:
python rn18_ae_tuning.py -tn="cs573-proj-ae-rn18-sgd" -g 7 -m rn18_ae -e 15 -b 200 -o sgd -ni 4 -ng 6 -ud -sc
python rn18_ae_tuning.py -tn="cs573-proj-ae-rn18-adam" -g 6 -m rn18_ae -e 15 -b 200 -o adam -ni 4 -ng 6 -ud -sc

version A:
python train.py -tn="cs573-proj-ae-rn18-sgd" -g 7 -m rn18_ae -e 20 -lr 0.2544 -p 2 -d "logs/tuned/rn18_ae/" -b 200 -o sgd -ud -sc
python train.py -tn="cs573-proj-ae-rn18-sgd" -g 5 -m rn18_ae -e 20 -lr 0.2544 -p 2 -d "logs/tests/rn18_ae_l1/" -b 200 -o sgd -lo L1 -ud -sc

version B:
python train.py -tn="cs573-proj-ae-rn18-b-sgd" -g 5 -m rn18_ae_b -e 20 -lr 0.1 -p 2 -d "logs/tests/rn18_ae_b_l1/" -b 124 -o sgd -lo L1 -ud -sc

version C:
python train.py -tn="cs573-proj-ae-rn18-c-sgd" -g 5 -m rn18_ae_c -e 20 -lr 0.1 -p 2 -d "logs/tests/rn18_ae_c_l1/" -b 124 -o sgd -lo L1 -ud -sc
python train.py -tn="cs573-proj-ae-rn18-c-sgd" -g 5 -m rn18_ae_c -e 20 -lr 0.1 -p 2 -d "logs/tests/rn18_ae_c_fp_l1/" -b 124 -o sgd -lo L1 -mp 0 -ud -sc
"""

if __name__ == "__main__":
    # define params to optimize
    opt_param_dict = {
        "learning_rate": (0.00001, 0.5)
    }

    # define any initial runs
    init_run_list = [
        {"learning_rate": 0.1},
        {"learning_rate": 0.01},
        {"learning_rate": 0.001},
        {"learning_rate": 0.0001},
    ]

    const_params = {
        "monitor_patience": 2,
        "model_save_dir": "logs/tuned/rn18_ae/"
    }

    opt_obj = opt_wrapper(opt_param_dict, init_run_list, const_params=const_params)

    opt_obj = opt_wrapper(opt_param_dict, init_run_list)
    