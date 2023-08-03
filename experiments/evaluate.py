
from pathlib import Path
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
sys.path.append("..")
from src.model_LN_prompt import Model
from experiments.options import opts
from src.dataset_retrieval import Sketchy, TU, QuickDraw

device = torch.cuda.current_device()

def evaluate(opts):
    
    # 1. load ckpt
    ckpt_path = Path("./saved_models") / opts.exp_name / "last.ckpt"
    # ckpt_path = Path("./saved_models") / opts.exp_name / "epoch=55-top10=0.00.ckpt"
    model = Model().load_from_checkpoint(ckpt_path).to(device)
    model.eval()
    
    # 2. load dataloader
    ds_cls = Sketchy if opts.dataset==0 else TU if opts.dataset==1 else QuickDraw
    dataset_transforms = ds_cls.data_transform(opts)
    val_dataset = ds_cls(opts, dataset_transforms, mode='val', instance_level=opts.instance_level, return_orig=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    
    # 3. inference
    val_step_outputs_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            for i, batched_item in enumerate(batch):
                if isinstance(batched_item, torch.Tensor):
                    batch[i] = batched_item.to(device)
            val_step_outputs = model.validation_step(batch, 0)
            val_step_outputs_list.append(val_step_outputs)
    
    sketch_emb, photo_emb, cat, photo_cat = model.validation_epoch_end(val_step_outputs_list)
    
    save_path = Path("./output") / opts.exp_name
    if save_path.exists() is False:
        save_path.mkdir()
    
    torch.save(sketch_emb, save_path / "sketch_emb")
    torch.save(photo_emb, save_path / "photo_emb")
    torch.save(cat, save_path / "cat")
    torch.save(photo_cat, save_path / "photo_cat")
    print(sketch_emb.shape, photo_emb.shape, cat.shape, photo_cat.shape)
    
    return

evaluate(opts)

