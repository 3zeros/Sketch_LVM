
from pathlib import Path
    

# 2. load test loader
# val_dataset = Sketchy(opts, dataset_transforms, mode='val', instance_level=False, used_cat=train_dataset.all_categories, return_orig=False)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=workers)

# calculate mAP, prec

def evaluate(opts):
    # load ckpt
    ckpt_path = Path("./saved_models")/opts.exp_name/opts.epoch_to_load
    print(ckpt_path)
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Sketch-based OD')

    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--epoch_to_load', type=str, default='last.ckpt')
    
    # parser.add_argument('--data_dir', type=str, default='/data/dataset/')
    # parser.add_argument('--instance_level', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=12)

    opts = parser.parse_args()
    evaluate(opts)

