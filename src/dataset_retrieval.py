import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from pathlib import Path

unseen_classes = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]

unseen_classes_tu = [
    'flying_saucer',
    'fire_hydrant',
    'walkie_talkie',
    'umbrella',
    'couch',
    'duck',
    'bicycle',
    'tablelamp',
    'castle',
    'snowman',
    'door_handle',
    'calculator',
    'traffic_light',
    'foot',
    'snail',
    'eyeglasses',
    'backpack',
    'swan',
    'pear',
    'hedgehog',
    'knife',
    'dolphin',
    'sailboat',
    'cabinet',
    'octopus',
    'purse',
    'snake',
    'truck',
    'bookshelf',
    'rollerblades'
]

unseen_classes_qd = [
    "bat",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "skyscraper",
    "tree",
    "windmill",
    "feather",
    "campfire",
    "palm tree",
    "fire_hydrant",
    "bread",
    "beach",
    "megaphone",
    "cactus",
    "zebra",
    "tiger",
    "shark",
    "frog",
    "banana",
    "cake",
    "hamburger",
    "fan"
    ]

class Sketchy(torch.utils.data.Dataset):

    def __init__(
        self, opts, transform, mode='train',
            instance_level=False, used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.instance_level = instance_level
        self.return_orig = return_orig

        self.all_categories = os.listdir(os.path.join(
            self.opts.data_dir, 'sketchy', 'Sketchy', 'sketch', 'tx_000000000000'
        ))
        if mode == 'val':
            self.all_categories = unseen_classes
        elif self.opts.data_split > 0:
            tmp = len(self.all_categories)
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
            print(f"[INFO] {len(self.all_categories)} out of {tmp} categories used")
        else:
            self.all_categories = list(set(self.all_categories) - set(unseen_classes))

        self.all_sketches_path = []
        # self.all_photos_path = {} # training
        self.all_photos_path = [] # evaluating
        self.all_photos_path_neg = {} # evaluating

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(
                self.opts.data_dir, 'sketchy', 'Sketchy', 'sketch', 'tx_000000000000', category, '*.png'
            )))
            # self.all_photos_path[category] = glob.glob(os.path.join(
            #     self.opts.data_dir, 'sketchy', 'Sketchy', 'extended_photo', category, '*.jpg'
            # )) # training
            self.all_photos_path.extend(glob.glob(os.path.join(
                self.opts.data_dir, 'sketchy', 'Sketchy', 'extended_photo', category, '*.jpg'
            )))
            self.all_photos_path_neg[category] = glob.glob(os.path.join(
                self.opts.data_dir, 'sketchy', 'Sketchy', 'extended_photo', category, '*.jpg'
            )) # training

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]
        sk_path = filepath

        filepath, filename = os.path.split(filepath)
        category = os.path.split(filepath)[-1]

        if self.instance_level:
            img_path = os.path.join(
                self.opts.data_dir, 'Sketchy', 'photo', 'tx_000000000000', category, filename.split('-')[0]+'.jpg'
            )
            neg_path = np.random.choice(self.all_photos_path[category])

        else:
            # img_path = np.random.choice(self.all_photos_path[category])#training
            try: #evaluating
                img_path = self.all_photos_path[index]
            except:
                img_path = self.all_photos_path[-1]
            neg_path = np.random.choice(self.all_photos_path_neg[np.random.choice(self.all_categories)])
        photo_path, _ = os.path.split(img_path)
        photo_cat = os.path.split(photo_path)[-1]

        sk_data = Image.open(sk_path).convert('RGB')
        img_data = Image.open(img_path).convert('RGB')
        neg_data = Image.open(neg_path).convert('RGB')

        sk_data = ImageOps.pad(sk_data, size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(img_data, size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(neg_data, size=(self.opts.max_size, self.opts.max_size))

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, photo_cat, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms

class TU(torch.utils.data.Dataset):

    def __init__(
        self, opts, transform, mode='train',
            instance_level=False, used_cat=None, return_orig=False):
        
        self.opts = opts
        self.transform = transform
        self.instance_level = instance_level
        self.return_orig = return_orig

        self.all_categories = [i.stem for i in Path("/data/dataset/tuberlin-ext/train/sketch").iterdir()]
        
        if mode == 'val':
            self.all_categories = unseen_classes_tu
        elif self.opts.data_split > 0:
            tmp = len(self.all_categories)
            np.random.shuffle(self.all_categories)
            if used_cat is None: # train
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else: # valid setting, but never used
                self.all_categories = list(set(self.all_categories) - set(used_cat))
            print(f"[INFO] {len(self.all_categories)} out of {tmp} categories used")
        else:
            self.all_categories = list(set(self.all_categories) - set(unseen_classes_tu))

        self.all_sketches_path = []
        # self.all_photos_path = {} # training
        self.all_photos_path = [] # evaluating
        self.all_photos_path_neg = {} # evaluating

        split = 'test' if mode=='val' else 'train'
        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(
                self.opts.data_dir, 'tuberlin-ext', split, 'sketch', category, '*.jpg'
            )))
            
            # =================== training
            # self.all_photos_path[category] = glob.glob(os.path.join(
            #     self.opts.data_dir, 'tuberlin-ext', split, 'photo', category, '*.jpg'
            # ))
            # ============================= 
            
            # =================== evaluating (for performance check)
            self.all_photos_path.extend(glob.glob(os.path.join(
                self.opts.data_dir, 'tuberlin-ext', split, 'photo', category, '*.jpg'
            )))
            self.all_photos_path_neg[category] = glob.glob(os.path.join(
                self.opts.data_dir, 'tuberlin-ext', split, 'photo', category, '*.jpg'
            ))
            # ============================= 

    def __len__(self):
        # return len(self.all_sketches_path)
        return len(self.all_photos_path)

    def __getitem__(self, index):
        # =================== training
        # filepath = self.all_sketches_path[index]
        # sk_path = filepath

        # filepath, filename = os.path.split(filepath)
        # category = os.path.split(filepath)[-1]

        # img_path = np.random.choice(self.all_photos_path[category])
        # neg_path = np.random.choice(self.all_photos_path[np.random.choice(self.all_categories)])
        # ============================= 
        
        # =================== evaluating
        img_path = self.all_photos_path[index]
        filepath, filename = os.path.split(img_path)
        photo_cat = os.path.split(filepath)[-1]
        try:
            sk_path = self.all_sketches_path[index]
        except:
            sk_path = self.all_sketches_path[-1]
        neg_path = np.random.choice(self.all_photos_path_neg[np.random.choice(self.all_categories)])
        # sk_path, _ = os.path.split(sk_path)
        category = os.path.split(sk_path)[-2].split("/")[-1]
        # print(category)
        # ============================= 

        sk_data = Image.open(sk_path).convert('RGB')
        img_data = Image.open(img_path).convert('RGB')
        neg_data = Image.open(neg_path).convert('RGB')

        sk_data = ImageOps.pad(sk_data, size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(img_data, size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(neg_data, size=(self.opts.max_size, self.opts.max_size))

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        # =================== training
        # return (sk_tensor, img_tensor, neg_tensor, category, filename)
        
        # =================== evaluating
        return (sk_tensor, img_tensor, neg_tensor, category, photo_cat, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms
    
class QuickDraw(torch.utils.data.Dataset):

    def __init__(
        self, opts, transform, mode='train',
            instance_level=False, used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.instance_level = instance_level
        self.return_orig = return_orig

        self.all_categories = os.listdir(os.path.join(
            self.opts.data_dir, 'QuickDraw-Extended', 'QuickDraw_sketches_final'
        ))
        if mode == 'val':
            self.all_categories = unseen_classes_qd
        elif self.opts.data_split > 0: # never used while reproducing
            tmp = len(self.all_categories)
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
            print(f"[INFO] {len(self.all_categories)} out of {tmp} categories used")
        else:
            self.all_categories = list(set(self.all_categories) - set(unseen_classes_qd))

        self.all_sketches_path = []
        self.all_photos_path = {} # training
        # self.all_photos_path = [] # evaluating
        # self.all_photos_path_neg = {} # evaluating

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(
                self.opts.data_dir, 'QuickDraw-Extended', 'QuickDraw_sketches_final', category, '*.png'
            )))
            # =================== training
            self.all_photos_path[category] = glob.glob(os.path.join(
                self.opts.data_dir, 'QuickDraw-Extended', 'QuickDraw_images_final', category, '*.jpg'
            ))
            # ============================= 
            
            # =================== evaluating (for performance check)
            # self.all_photos_path.extend(glob.glob(os.path.join(
            #     self.opts.data_dir, 'QuickDraw-Extended', 'QuickDraw_images_final', category, '*.jpg'
            # )))
            # self.all_photos_path_neg[category] = glob.glob(os.path.join(
            #     self.opts.data_dir, 'QuickDraw-Extended', 'QuickDraw_images_final', category, '*.jpg'
            # ))
            # ============================= 

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]
        sk_path = filepath

        filepath, filename = os.path.split(filepath)
        category = os.path.split(filepath)[-1]

        # =================== training
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(self.all_categories)])
        # ============================= 
        
        # =================== evaluating
        # try:
        #     img_path = self.all_photos_path[index]
        # except:
        #     img_path = self.all_photos_path[-1]
        # neg_path = np.random.choice(self.all_photos_path_neg[np.random.choice(self.all_categories)])
        # photo_path, _ = os.path.split(img_path)
        # photo_cat = os.path.split(photo_path)[-1]
        # ============================= 

        sk_data = Image.open(sk_path).convert('RGB')
        img_data = Image.open(img_path).convert('RGB')
        neg_data = Image.open(neg_path).convert('RGB')

        sk_data = ImageOps.pad(sk_data, size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(img_data, size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(neg_data, size=(self.opts.max_size, self.opts.max_size))

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        # =================== training
        return (sk_tensor, img_tensor, neg_tensor, category, filename)
        
        # =================== evaluating
        # return (sk_tensor, img_tensor, neg_tensor, category, photo_cat, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm
    dataset_cls = Sketchy if opts.dataset == 0 else TU if opts.dataset == 1 else QuickDraw
    dataset_transforms = dataset_cls.data_transform(opts)

    dataset_train = dataset_cls(opts, dataset_transforms, mode='train',
        instance_level=opts.instance_level, return_orig=True)

    dataset_val = dataset_cls(opts, dataset_transforms, mode='val',
        instance_level=opts.instance_level, used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, filename,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
