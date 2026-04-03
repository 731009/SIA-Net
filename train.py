
import os
import argparse
import argparse
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
from utils import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from timm.utils import AverageMeter
from accelerate import Accelerator
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
import logging
from rich.logging import RichHandler
import hydra
import numpy as np

from torch.utils.data import DataLoader
import random


import matplotlib.pyplot as plt
import copy


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ! solve warning


import itertools
class TwoStreamBatchSampler(torch.utils.data.sampler.Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def weights_init_normal(init_type):
    def init_func(m):
        classname = m.__class__.__name__
        gain = 0.02

        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    return init_func

def get_logger(config):
    file_handler = logging.FileHandler(os.path.join(config.hydra_path, f"{config.job_name}.log"))
    rich_handler = RichHandler()

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False
    log.info("Successfully create rich logger")

    return log

# ---------------- Ramp-up ----------------
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
def get_current_consistency_weight(epoch, consistency=0.1, rampup=200):
    return consistency * sigmoid_rampup(epoch, rampup)

# ---------------- EMA ----------------
def update_ema_variables(student_model, teacher_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# ---------------- ALSEMA----------------
def update_ema_by_loss(student_model,
                       teacher_model,
                       global_step,
                       curr_loss,
                       prev_loss=None,
                       base_alpha=0.999,
                       beta=1.0,
                       use_sample_variance=False):


    warmup_cap = 1.0 - 1.0 / (global_step + 1)

    if prev_loss is None:
        alpha = min(warmup_cap, base_alpha * 0.5)
    else:
        cl = float(curr_loss) if torch.is_tensor(curr_loss) else curr_loss
        pl = float(prev_loss) if torch.is_tensor(prev_loss) else prev_loss
        delta = cl - pl

        if use_sample_variance: #two choices
            var2 = (delta * delta) / 2.0    
        else:
            var2 = (delta * delta) / 4.0    

        s = 1.0 / (1.0 + torch.exp(torch.tensor(-beta * var2)))
        alpha = float(min(warmup_cap, base_alpha * s))

    one_minus_alpha = 1.0 - alpha
    with torch.no_grad():
        for ema_param, param in zip(teacher_model.parameters(), student_model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=one_minus_alpha)

    return alpha  




def train(config, student_model, logger):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
        # *  acceleretor create
    accelerator = Accelerator()
    # * init averageMeter
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

# ----------------init rich progress----------------
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )
# ----------------* set optimizer----------------
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config.init_lr)

# ----------------* set scheduler strategy---------------
    if config.use_scheduler:
        scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

#---------------- * load model----------------
    if config.load_mode == 1:  # * load weights from checkpoint
        logger.info(f"load model from: {os.path.join(config.ckpt, config.latest_checkpoint_file)}")
        ckpt = torch.load(
            os.path.join(config.ckpt, config.latest_checkpoint_file), map_location=lambda storage, loc: storage
        )
        student_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        if config.use_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
        # elapsed_epochs = 0
    else:
        elapsed_epochs = 0

# ----------------teacher----------------
    teacher_model = copy.deepcopy(student_model)
    for p in teacher_model.parameters():
        p.requires_grad = False

    student_model.train()

    writer = SummaryWriter(config.hydra_path)

    # from utils.loss_function import  DiceLoss

    mse_loss_fn = nn.MSELoss().to(accelerator.device)
    bce_loss_fn = nn.BCEWithLogitsLoss().to(accelerator.device)
    
    writer = SummaryWriter(config.data_path + '/log')

    from SIA_dataloader import SIADataLoader
    dataset = SIADataLoader(config).queue_dataset
    
    labeled_idxs = list(range(0, config.labeled_num))
    unlabeled_idxs = list(range(config.labeled_num, len(dataset)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, 
        batch_size=config.batch_size, 
        secondary_batch_size=config.batch_size-config.labeled_bs
        )

    train_loader = DataLoader(dataset,
                               batch_sampler=batch_sampler,
                               shuffle=False,
                               num_workers=0,
                               pin_memory=False
                               )

    accelerator = Accelerator()

    train_loader, student_model, optimizer, scheduler = accelerator.prepare(
        train_loader, student_model, optimizer, scheduler
        )
    teacher_model = accelerator.prepare(teacher_model)

    epochs = config.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)


    epoch_tqdm = progress.add_task(description="[red]epoch progress", total=epochs)
    batch_tqdm = progress.add_task(description="[blue]batch progress", total=len(train_loader))

    writer = SummaryWriter(config.hydra_path)
    loss_meter = AverageMeter()
    iteration = 0
    
    progress.start()

    all_epochs = []
    all_avg_losses = []

    for epoch in range(1, epochs + 1):
        progress.update(epoch_tqdm, completed=epoch)
        epoch += elapsed_epochs
        prev_loss = None   
        num_iters = 0
       
        load_meter = AverageMeter()
        train_time = AverageMeter()
        load_start = time.time()  

        total_loss = 0.0
        total_supervised_loss = 0.0
        total_consistency_loss = 0.0
        total_consistency_weight = 0.0
        total_cons_loss = 0.0

        for i, batch in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                progress.update(batch_tqdm, completed=i + 1)  
                
                optimizer.zero_grad()
                train_start = time.time()
                load_time = time.time() - load_start

                x = batch["source"]["data"]
                if "IAM" in config.module_list:
                    intensity_x = batch["intensity_source"]["data"]
                else:
                    intensity_x = x
                if "SAM" in config.module_list:
                    spatial_x = batch["spatial_source"]["data"]
                else:
                    spatial_x = x

                gt = batch["gt"]["data"]
                sia_gt = batch["sia_gt"]["data"]

                x = x.type(torch.FloatTensor).to(accelerator.device)
                spatial_x = spatial_x.type(torch.FloatTensor).to(accelerator.device)
                intensity_x = intensity_x.type(torch.FloatTensor).to(accelerator.device)
                gt = gt.type(torch.FloatTensor).to(accelerator.device)
                sia_gt = sia_gt.type(torch.FloatTensor).to(accelerator.device)

                unlabeled_x = x[config.labeled_bs:]#args.labeled_bs:=2
                unlabeled_spatial_x = spatial_x[config.labeled_bs:]
                unlabeled_intensity_x = intensity_x[config.labeled_bs:]


                # ---- Student  ----
                pred_s, spatial_s, intensity_s = student_model(x, spatial_x, intensity_x)
                pred_s_soft= torch.softmax(pred_s,dim=1)
                # mask = torch.sigmoid(pred_s.clone())  # TODO should use softmax, because it returns two probability (sum = 1)
                # mask[mask > 0.5] = 1
                # mask[mask <= 0.5] = 0

                # ---- Teacher  ----
                with torch.no_grad():
                    pred_t, spatial_t, intensity_t = teacher_model(unlabeled_x, unlabeled_spatial_x, unlabeled_intensity_x)
                    pred_t_soft= torch.softmax(pred_t,dim=1)
                # mask2 = torch.sigmoid(pred_t.clone())  # TODO should use softmax, because it returns two probability (sum = 1)
                # mask2[mask2 > 0.5] = 1
                # mask2[mask2 <= 0.5] = 0

                gt_ = gt
                gt_back = torch.zeros_like(gt_)
                gt_back[gt_ == 0] = 1
                gt_ = torch.cat([gt_back, gt_], dim=1) 

                loss_bce = bce_loss_fn(pred_s_soft[:config.labeled_bs], gt_[:config.labeled_bs])
                sup_loss=loss_bce


                consistency_weight = get_current_consistency_weight(
                    epoch, consistency=config.consistency, rampup=config.consistency_rampup
                )
                cons_loss = mse_loss_fn(torch.sigmoid(pred_s[config.labeled_bs:]), torch.sigmoid(pred_t))
                # consistencyLoss = torch.mean(
                #     (outputs_soft[args.labeled_bs:] - ema_output_soft)**2)
                if "SAM" in config.module_list:
                    cons_loss += 0.5 * mse_loss_fn( spatial_s[config.labeled_bs:],sia_gt[config.labeled_bs:])
                if "IAM" in config.module_list:
                    cons_loss += 0.5 * mse_loss_fn( intensity_s[config.labeled_bs:],sia_gt[config.labeled_bs:])
                consistency_loss=consistency_weight * cons_loss

                loss = sup_loss + consistency_loss

                accelerator.backward(loss)
                progress.refresh()           

            optimizer.step()

            #ema
            # update_ema_variables(student_model, teacher_model, config.ema_alpha, iteration)#progress.refresh()
           
            update_ema_by_loss(
                student_model, teacher_model,
                global_step= iteration,
                curr_loss=loss.item(),         
                prev_loss=prev_loss,           
                base_alpha=0.999,  
                beta=1.0,    
                use_sample_variance=True      
            )
            prev_loss = loss.item()

            num_iters += 1
            iteration += 1


            writer.add_scalar("Train/Loss", loss.item(), iteration)
            temp_file_base = os.path.join(config.hydra_path, "train_temp")
            os.makedirs(temp_file_base, exist_ok=True)
            loss_meter.update(loss.item(), x.size(0))

            logger.info(
                f"\nEpoch: {epoch} | Batch: {i} \n"
                f"Loss: {loss.item():.6f} | Supervised Loss: {sup_loss.item():.6f} | Consistency Loss: {consistency_loss.item():.6f} | Consistency weight: {consistency_weight:.6f}  |  cons_loss: {cons_loss.item():.6f}\n"
                )

            load_start = time.time()

            train_time.update(time.time() - train_start)
            load_meter.update(load_time)

            total_loss += loss.item()
            total_supervised_loss += sup_loss.item()
            total_consistency_loss += consistency_loss.item()
            total_consistency_weight += consistency_weight
            total_cons_loss += cons_loss.item()

            avg_loss = total_loss / num_iters
            avg_supervised_loss = total_supervised_loss / num_iters
            avg_consistency_loss = total_consistency_loss / num_iters
            avg_consistency_weight = total_consistency_weight / num_iters
            avg_cons_loss = total_cons_loss / num_iters 

        if config.use_scheduler: 
            scheduler.step()
            logger.info(f"Learning rate:  {scheduler.get_last_lr()[0]}")

        writer.add_scalar('info/avg_loss', avg_loss, epoch)
        writer.add_scalar('info/avg_supervised_loss', avg_supervised_loss, epoch)
        writer.add_scalar('info/avg_consistency_loss', avg_consistency_loss, epoch)
        writer.add_scalar('info/avg_consistency_weight', avg_consistency_weight, epoch)
        writer.add_scalar('info/avg_cons_loss', avg_cons_loss, epoch)

        logging.info(
            f"\nEpoch: {epoch}        Iteration: {num_iters}   "
            f"zijiavg_loss: {avg_loss:.6f}, zijiavg_supervised_loss: {avg_supervised_loss:.6f}, zijiavg_consistency_loss: {avg_consistency_loss:.6f} ,zijiavg_consistency_weight: {avg_consistency_weight:.6f},zijiavg_cons_loss: {avg_cons_loss:.6f} \n"
        )

        logger.info(f"\nEpoch {epoch} " f"Loss Avg:  {loss_meter.avg}\n" f"Dice Avg:  {dice_meter.avg}\n")
        
        scheduler_dict = scheduler.state_dict() if config.use_scheduler else None
        
        all_epochs.append(epoch)
        all_avg_losses.append(loss_meter.avg)
        
        # Store latest checkpoint in each epoch
        scheduler_dict = scheduler.state_dict() if config.use_scheduler else None
        torch.save(
            {
                "student": student_model.state_dict(),
                "teacher": teacher_model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch,
            },
            os.path.join(config.hydra_path, config.latest_checkpoint_file),
        )

        # Save checkpoint
        if epoch % config.epochs_per_checkpoint == 0:
            torch.save(
                {
                    "model": student_model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler_dict,
                    "epoch": epoch,
                },
                os.path.join(config.hydra_path, f"checkpoint_{epoch:04d}.pt"),
            )

        def save_loss_curve(epochs, losses, save_dir):
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
            plt.title('Training Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.grid(True)
            plt.legend()
            
            save_path = os.path.join(save_dir, "loss_curve.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Loss curve saved to {save_path}")
        
    save_loss_curve(all_epochs, all_avg_losses, config.hydra_path)

    # progress.stop()
    writer.close()
         

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    if isinstance(config.patch_size, str):
        assert (
            len(config.patch_size.split(",")) <= 3
        ), f'patch size can only be one str or three str but got {len(config.patch_size.split(","))}'
        if len(config.patch_size.split(",")) == 3:
            config.patch_size = tuple(map(int, config.patch_size.split(",")))
        else:
            config.patch_size = int(config.patch_size)

    #os["CUDA_AVAILABLE_DEVICES"] = config.gpu
    # * model selection
    if config.network == "res_unet":
        from models.three_d.residual_unet3d import UNet
        model = UNet(in_channels=config.in_classes, n_classes=config.out_classes, base_n_filter=32)

    elif config.network == "unet":
        from models.three_d.unet3d import UNet3D  # * 3d unet
        model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)

    elif config.network == "er_net":
        from models.three_d.ER_net import ER_Net
        model = ER_Net(classes=config.out_classes, channels=config.in_classes)

    elif config.network == "re_net":
        from models.three_d.RE_net import RE_Net
        model = RE_Net(classes=config.out_classes, channels=config.in_classes)

    elif config.network == "IS":
        from models.three_d.IS import IS
        model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes)
        
    elif config.network == "unetr":
        from models.three_d.unetr import UNETR
        model = UNETR()

    elif config.network == "densenet":
        from models.three_d.densenet3d import SkipDenseNet3D
        model = SkipDenseNet3D(in_channels=config.in_classes, classes=config.out_classes)

    elif config.network == "vtnet":
        from models.three_d.vtnet import VTUNet
        model = VTUNet(num_classes=config.out_classes, input_dim=config.in_classes)

    elif config.network == "vnet":
        from models.three_d.vnet3d import VNet
        model = VNet(in_channels=config.in_classes, classes=config.out_classes)

    elif config.network == "densevoxelnet":
        from models.three_d.densevoxelnet3d import DenseVoxelNet
        model = DenseVoxelNet(in_channels=config.in_classes, classes=config.out_classes)


    elif config.network == "SIABETA":
        from models.three_d.SIA_Unet_BETA import SIAUNet as SIA_BETA

        model = SIA_BETA(
            in_channels=config.in_classes, out_channels=config.out_classes, init_features=32, module_list=config.module_list
        )
    #vnet
    elif config.network == "SIA_VNET":
        from models.three_d.sia_vnet import SIAVNet as SIA_VNET

        model = SIA_VNET(
            in_channels=config.in_classes, classes=config.out_classes, module_list=config.module_list
        )
    #csrnet
    elif config.network == "SIA_CSRNET":
        from models.three_d.sia_csrnet import SIACSRNet as SIA_CSRNET

        model = SIA_CSRNET(
            in_channels=config.in_classes, out_channels=config.out_classes,init_features=64, module_list=config.module_list
        )

    elif config.network == "csrnetse":
        from models.three_d.csrnet_SE import CSRNet
        model = CSRNet(in_channels=config.in_classes, out_channels=config.out_classes)

    model.apply(weights_init_normal(config.init_type))


    # * create logger
    logger = get_logger(config)
    info = "\nParameter Settings:\n"
    for k, v in config.items():
        info += f"{k}: {v}\n"
    logger.info(info)

    train(config, model, logger)
    logger.info(f"tensorboard file saved in:{config.hydra_path}")



if __name__ == "__main__":
    main()
