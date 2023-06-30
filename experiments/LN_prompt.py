import os
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
sys.path.append("..")
from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy
from experiments.options import opts
def count_parameters(model, verbose=False):
    visual_cnt=0; text_cnt=0; etc_cnt=0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if verbose is True: print(f"\t{name}\t{p.numel()}")
            if name.startswith("clip.visual"):
                visual_cnt+=p.numel()
            elif name.startswith("clip.transformer"):
                text_cnt+=p.numel()
            else:
                etc_cnt+=p.numel()
    if verbose is True: print(f"visual: {visual_cnt}, text: {text_cnt}, etc: {etc_cnt}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train',
        instance_level=opts.instance_level, return_orig=False)

    val_dataset = Sketchy(opts, dataset_transforms, mode='val',
        instance_level=opts.instance_level, used_cat=train_dataset.all_categories, return_orig=False)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}-{top10:.2f}",
        mode='min',
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    trainer = Trainer(gpus=-1,
        min_epochs=1, max_epochs=2000,
        benchmark=True,
        logger=logger,
        # val_check_interval=10, 
        # accumulate_grad_batches=1,
        check_val_every_n_epoch=opts.check_val_every_n_epoch,
        resume_from_checkpoint=ckpt_path,
        callbacks=[checkpoint_callback]
    )

    if ckpt_path is None:
        model = Model()
    else:
        print ('resuming training from %s'%ckpt_path)
        model = Model().load_from_checkpoint(ckpt_path)

    trainable_param = count_parameters(model, verbose=False)
    
    print ('beginning training...good luck...')
    trainer.fit(model, train_loader, val_loader)
