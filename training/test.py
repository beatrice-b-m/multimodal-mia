

if __name__ == "__main__":
    # scripts
    from .data import get_transform_list, get_dataset, CombinedDataset, Subset
    from .models import get_distilgpt2_srn18_vae
    import random
    import time

    # prepare data ----------------------------------------------------------------------------------
    SEED = 13
    DATA_DIR = "data/mscoco"

    transform_list = get_transform_list()
    # get datasets
    train_ds = get_dataset(transform_list, dataset_type="train2014", data_dir=DATA_DIR, seed=SEED)
    val_ds = get_dataset(transform_list, dataset_type="val2014", data_dir=DATA_DIR, seed=SEED)
    test_ds = get_dataset(transform_list, dataset_type="test2014", data_dir=DATA_DIR, seed=SEED)

    random_gen = random.Random(SEED)
    n_subset = 500
    train_idxs = random_gen.sample(range(len(train_ds)), n_subset)
    test_idxs = random_gen.sample(range(len(test_ds)), n_subset)

    # sample subsets
    train_subset = Subset(train_ds, train_idxs)
    test_subset = Subset(test_ds, test_idxs)

    mia_ds = CombinedDataset(train_subset, test_subset, 1, 0)

    # prepare model ---------------------------------------------------------------------------------
    device = 'cuda'
    param_dict = {
        'captioner_weights': "logs/tests/srn18_vae_dgpt2_z3/best.pth",
        'mixed_precision': 1,
        'freeze_encoder': 0, # only affects things if ur finetuning the model
    }
    model = get_distilgpt2_srn18_vae(param_dict)
    model.eval()
    model = model.to(device)

    # generate captions -----------------------------------------------------------------------------
    print('\n\nsingle image test:')
    start_time = time.time()
    image, _ = val_ds[0]
    single_output = model.generate_caption(image)
    print(f"elapsed: {time.time() - start_time}s")
    print(f"{len(single_output) = }")
    print(f"{single_output = }")

    print('\n\nbatch image test')
    batch_outputs = model.generate_batch_captions(mia_ds)
    print(f"{len(batch_outputs) = }")
    # print(f"{batch_outputs = }")