from training.optimizer import opt_wrapper

"""
script run with:
python rn18_ae_embed_tuning.py -tn="cs573-proj-ae-embed-rn18-sgd" -g 7 -m rn18_ae_embed -e 15 -b 200 -o sgd -ni 4 -ng 6 -ud -sc

python train.py -tn="cs573-proj-ae-embed-rn18" -g 6 -m rn18_ae_embed -e 20 -p 2 -b 200 -o sgd -lr 0.2544 -puv A -ud -sc -fae 1 -aew logs/tuned/rn18_ae/best.pth
python train.py -tn="cs573-proj-ae-embed-rn18" -g 6 -m rn18_ae_embed -e 20 -p 2 -b 200 -o sgd -lr 0.2544 -puv A -ud -sc -fae 1 -aew logs/tests/rn18_ae_c_l1/best.pth

logs/tests/rn18_ae_c_l1/best.pth
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
        "proj_up_ver": "A",
        # "autoencoder_weights": "PATH/TO/WEIGHTS",
        "monitor_patience": 2
    }

    opt_obj = opt_wrapper(opt_param_dict, init_run_list, const_params=const_params)
    