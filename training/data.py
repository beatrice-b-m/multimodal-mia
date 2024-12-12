from torch import float32, Generator
from torch.utils.data import Dataset, DataLoader#, random_split
from torchvision.transforms import v2
from torch.utils.data.distributed import DistributedSampler

from pycocotools.coco import COCO
import random
import cv2 as cv



class COCODataset(Dataset):
    def __init__(self, coco: COCO, base_path: str, transform=None, seed=2196):
        self.coco = coco
        self.base_path = base_path
            
        self.transform = transform
        self._random = random.Random(seed)

        self.img_meta_list = self.coco.dataset['images']

    def __len__(self):
        return len(self.img_meta_list)

    def __getitem__(self, idx):
        img_meta = self.img_meta_list[idx]
        
        img = self._load_img(self.base_path + img_meta['file_name'])

        if self.transform is not None:
            img = self.transform(img)

        caption_id = self.coco.getAnnIds(imgIds = img_meta['id'])
        caption = self._process_caption(self._random.choice(self.coco.loadAnns(caption_id))['caption'])
        
        return img, caption
    
    def _process_caption(self, caption: str):
        return caption.strip()

    def _load_img(self, img_path):
        return cv.imread(img_path)

    def to_loader(self, batch_size: int, **kwargs):
        return DataLoader(
            self, 
            batch_size=batch_size, 
            **kwargs
        )


def get_transform_list(enabled_list: list | None = None, max_size: int = 128):
    transform_dict = {
        "normalize": [
            v2.Normalize(
                (104.0396, 114.1143, 120.1350), # calculated on train subset
                (15.9597, 14.3134, 14.4428)
            )
        ],
        "reformat": [
            v2.ToImage(),
            v2.ToDtype(float32, scale=True),
        ],
        "resize": [
            # resize the shortest side to max_size px, maintaining aspect ratio
            # requires torchvision 0.19 or later
            v2.Resize(None, max_size=max_size),
            # pad to max_size x max_size with black (0) padding
            v2.CenterCrop((max_size, max_size)),  
        ],
    }
    # by default use all
    if enabled_list is None:
        enabled_list = list(transform_dict.keys())

    # only retain selected transform types
    transform_dict = {k:v for k,v in transform_dict.items() if k in enabled_list}
    # unpack transforms and compose
    return v2.Compose([transform for transform_type in transform_dict.values() for transform in transform_type])

def get_dataset(transform_list, dataset_type: str, data_dir: str, seed: int):
    # data_type: e.g. 'train2014'
    # data_dir: e.g. 'data/mscoco'

    # define paths and load coco api dataset object
    caption_path = f"{data_dir}/annotations/captions_{dataset_type}.json"
    image_dir = f"{data_dir}/images/{dataset_type}/"
    coco_dataset = COCO(caption_path)

    # then pass to a pytorch dataset subclass
    return COCODataset(coco=coco_dataset, base_path=image_dir, transform=transform_list, seed=seed)

def get_dataloader(transform_list, batch_size: int, dataset_type: str, data_dir: str, seed: int, is_distributed: bool = False, **kwargs):
    # load the dataset
    ds = get_dataset(transform_list=transform_list, dataset_type=dataset_type, data_dir=data_dir, seed=seed)

    # check if dataset is for train or validation
    is_train = "train" in dataset_type
    is_val = "val" in dataset_type

    # if using distributed training, modify the dataloader kwargs to add a distributed sampler and disable standard shuffling
    if is_distributed:
        # shuffle should only be True for the train set
        # drop_last should only be True for the validation set
        sampler = DistributedSampler(ds, shuffle=is_train, seed=seed, drop_last=is_val)
        kwargs.update({
            "sampler": sampler, 
            "shuffle": False
        })

    return DataLoader(ds, batch_size=batch_size, **kwargs)


# def get_loader_dict(transform_list, batch_size: int, sampler: DistributedSampler | None = None, 
#                     data_dir: str = 'data/mscoco', data_seed: int = 81693):

    

#     # load train dataset
#     val_data_type='val2014'
#     val_captions = f"{data_dir}/annotations/captions_{val_data_type}.json"
#     val_images_dir = f"{data_dir}/images/{val_data_type}/"
#     coco_val = COCO(val_captions)

    
#     val_ds = COCODataset(coco=coco_val, base_path=val_images_dir, transform=transform_list, seed=data_seed)

#     # further split the train dataset into train/validation sets
#     # so we can better control the degree of overfitting for our target model
#     # train_ds, val_ds = random_split(train_val_ds, [0.8, 0.2], generator=split_gen)
#     print(f"train set: {len(train_ds)}")
#     print(f"val set: {len(val_ds)}")

#     if sampler is not None:
#         data_loader_args = {"batch_size": batch_size, "shuffle": False, "sampler": sampler, "num_workers": 6, "pin_memory": True}
#     else:
#         data_loader_args = {"batch_size": batch_size, "shuffle": True, "num_workers": 6, "pin_memory": True}

#     # send split datasets to loaders
#     train_loader = DataLoader(
#         train_ds,
#         **data_loader_args
#     )
#     val_loader = DataLoader(
#         val_ds,
#         **data_loader_args
#     )
#     print(f"train loader: {len(train_loader)}")
#     print(f"val loader: {len(val_loader)}")
#     return {"train": train_loader, "val": val_loader}


if __name__ == "__main__":
    transform_list = get_transform_list()
    # loader_dict = get_loader_dict(transform_list, batch_size=200)

    print('dataloaders loaded successfully...')