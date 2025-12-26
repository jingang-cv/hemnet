import torch
import numpy as np
import os
import random
import time
import argparse
import copy
import logging
import math

from scipy import io as sio
from scipy import signal
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from archs.cross_baseline_cat import BioPhysNet
from datasets.rppg_datasets_reconstruction import COHFACE, PURE, UBFC, VIPL, MMSEHR
from losses.NPLoss import Neg_Pearson
from losses.CELoss import TorchLossComputer
from utils.utils import AvgrageMeter, cxcorr_align, pearson_correlation_coefficient

def set_seed(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _init_fn(seed=92):
    np.random.seed(seed)


class RecLoss:
    def __init__(self, priorWeight=1e-4, appearanceWeight=1e-3, shadingWeight=1e-3, sparsityWeight=1e-7, size=64):
        self.priorWeight = priorWeight
        self.appearanceWeight = appearanceWeight
        self.shadingWeight = shadingWeight
        self.sparsityWeight = sparsityWeight
        self.size = size

    def __call__(self, rgb, shade, spec, b, shading, mask, x):
        scale = torch.sum(shade * shading * mask, (1, 2)) / (torch.sum(shade * shade * mask, (1, 2)) + 1e-9)
        scaledShading = torch.reshape(scale, (-1, 1, 1)) * shade
        alpha = (shading - scaledShading) * mask
        priorLoss = torch.sum(b ** 2) * self.priorWeight / x.shape[0]

        originalImage = torch.clone(x)

        delta = ((originalImage - rgb) ** 2) * torch.reshape(mask, (-1, 1, self.size, self.size))
        appearanceLoss = torch.sum(delta ** 2 / (self.size * self.size)) * 255 * 255 * self.appearanceWeight / x.shape[0]
        shadingLoss = torch.sum(alpha ** 2) * self.shadingWeight / x.shape[0]
        sparsityLoss = torch.sum(spec) * self.sparsityWeight / x.shape[0]
        return priorLoss, appearanceLoss, shadingLoss, sparsityLoss

def kl_loss_func(rPPG, labels, clip_avg_hr):
    clip_average_HR = (clip_avg_hr - 40)
    kl_loss = 0.0
    fre_loss = 0.0
    train_mae = 0.0
    for bb in range(rPPG.shape[0]):
        kl_loss_temp, fre_loss_temp, train_mae_temp = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(rPPG[bb],clip_average_HR[bb],30,1.0)
        kl_loss = kl_loss + kl_loss_temp
        fre_loss = fre_loss + fre_loss_temp
        train_mae = train_mae + train_mae_temp
    fre_loss = fre_loss / rPPG.shape[0]
    train_mae = train_mae / rPPG.shape[0]
    kl_loss = kl_loss / rPPG.shape[0]
    return kl_loss, fre_loss, train_mae

class Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rec_loss = RecLoss(priorWeight=args.proir_weight, appearanceWeight=args.appearance_weight, shadingWeight=args.shading_weight, sparsityWeight=args.sparsity_weight)
        self.np_loss = Neg_Pearson()

    def forward(self, sample, output):
        ecg = sample["ecg"]
        clip_avg_hr = sample["clip_avg_hr"]
        args = self.args
        step_log = {}
        loss = 0.0
        if args.np_loss:
            _nploss = self.np_loss(output["rppg"], ecg)
            step_log["nploss"] = _nploss.item()
            loss += _nploss * args.np_weight
        kl_loss, peak_loss, train_mae = kl_loss_func(output["rppg"], ecg, clip_avg_hr)
        step_log["kl_loss"] = kl_loss.item()
        step_log["peak_loss"] = peak_loss.item()
        step_log["train_mae"] = train_mae.item()
        if args.cn_loss:
            loss += (kl_loss + peak_loss) * args.cn_weight
        if args.rec_loss:
            image = output["image"]
            rgb, shade, spec, b, shading_gt, mask_gt = output["rgb"], output["shade"], output["spec"], output["b"], sample["shading"], sample["seg"]
            shading_gt = shading_gt.reshape(-1, shading_gt.shape[2], shading_gt.shape[3])
            mask_gt = mask_gt.reshape(-1, mask_gt.shape[2], mask_gt.shape[3])
            priorLoss, appearanceLoss, shadingLoss, sparsityLoss = self.rec_loss(rgb, shade, spec, b, shading_gt, mask_gt, image)
            step_log["prior"] = priorLoss.item()
            step_log["appear"] = appearanceLoss.item()
            step_log["shading"] = shadingLoss.item()
            step_log["sparsity"] = sparsityLoss.item()
            loss += priorLoss + appearanceLoss + shadingLoss + sparsityLoss
            step_log["loss"] = loss.item()
        return loss, step_log

class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_dim = args.input_dim
        self.video_feature_extractor = BioPhysNet(frames=160, device=self.device).to(self.device)
        self.run_date = time.strftime('%m%d_%H%M', time.localtime(time.time()))
        self.save_path = f'{args.save_path}/{self.run_date}'
        print(f"exp save path: {self.save_path}")
        # VIPL dataset
        if args.dataset == 'VIPL':
            self.train_dataset = VIPL(train=True, T=args.num_rppg, norm_type=args.norm_type, aug=args.aug, w=args.w, h=args.w, fold=args.K)
            self.val_dataset_video = VIPL(train=False, T=-1, norm_type=args.norm_type, aug="",w=args.w, h=args.w, fold=args.K)
            self.val_dataset_clip = VIPL(train=False, T=args.num_rppg, norm_type=args.norm_type, aug="",w=args.w, h=args.w, fold=args.K)
        elif args.dataset == 'UBFC':
            self.train_dataset = UBFC(train=True, T=args.num_rppg, norm_type=args.norm_type, aug=args.aug, w=args.w, h=args.w)
            self.val_dataset_video = UBFC(train=False, T=-1, norm_type=args.norm_type, aug="",w=args.w, h=args.w)
            self.val_dataset_clip = UBFC(train=False, T=args.num_rppg, norm_type=args.norm_type, aug="",w=args.w, h=args.w)
        elif args.dataset == 'PURE':
            self.train_dataset = PURE(train=True, T=args.num_rppg, norm_type=args.norm_type, aug=args.aug, w=args.w, h=args.w)
            self.val_dataset_video = PURE(train=False, T=-1, norm_type=args.norm_type, aug="",w=args.w, h=args.w)
            self.val_dataset_clip = PURE(train=False, T=args.num_rppg, norm_type=args.norm_type, aug="",w=args.w, h=args.w)
        elif args.dataset == 'COHFACE':
            self.train_dataset = COHFACE(train=True, T=args.num_rppg, norm_type=args.norm_type, aug=args.aug, w=args.w, h=args.w)
            self.val_dataset_video = COHFACE(train=False, T=-1, norm_type=args.norm_type, aug="",w=args.w, h=args.w)
            self.val_dataset_clip = COHFACE(train=False, T=args.num_rppg, norm_type=args.norm_type, aug="",w=args.w, h=args.w)
        elif args.dataset == 'MMSEHR':
            self.train_dataset = MMSEHR(train=True, T=args.num_rppg, norm_type=args.norm_type, aug=args.aug, w=args.w, h=args.w)
        else:
            print("dataset not supported")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=_init_fn)
        self.val_dataloader_video = DataLoader(self.val_dataset_video, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=_init_fn)
        self.val_dataloader_clip = DataLoader(self.val_dataset_clip, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=_init_fn)

        ## optimizer


        self.optimizer = torch.optim.Adam(self.video_feature_extractor.parameters(),lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        # self.criterion_Pearson = Neg_Pearson()
        self.criterion = Loss(self.args)

        ## constant
        self.bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        self.best_epoch = 0
        self.best_val_mae = 1000
        self.best_val_rmse = 1000
        self.best_val_sd = 1000
        self.best_r = 1000
        self.frame_rate = 30 # 测试

    def prepare_train(self, start_epoch, continue_log, only_val=False):
        if start_epoch != -1:
            self.save_path = self.args.save_path + '/' + continue_log

        self.save_ckpt_path = f'{self.save_path}/ckpt'
        self.save_rppg_path = f'{self.save_path}/rppg'
        if not only_val:
            if not os.path.exists(self.save_ckpt_path):
                os.makedirs(self.save_ckpt_path)
            if not os.path.exists(self.save_rppg_path):
                os.makedirs(self.save_rppg_path)

            logging.basicConfig(filename=f'./logs/{self.args.train_model}_{self.args.dataset}_{self.args.num_rppg}_{self.run_date}.log',\
                                format='%(message)s', filemode='w')
            self.logger = logging.getLogger(f'./logs/{self.args.train_model}_{self.args.dataset}_{self.args.num_rppg}_{self.run_date}')
            self.logger.setLevel(logging.INFO)
            self.logger.info(f"exp save path: {self.save_path}")

            ## save proj_file to save_path
            cur_file = os.getcwd()
            cur_file_name = cur_file.split('/')[-1]
            os.system(f'cp -r {cur_file} {self.save_path}/{cur_file_name}')

        if start_epoch != -1:
            if not os.path.exists(f'{self.save_ckpt_path}/video_feature_extractor_{start_epoch }.pth'):
                raise Exception(f'video_feature_extractor ckpt file {start_epoch} not found')
            self.video_feature_extractor.load_state_dict(torch.load(f'{self.save_ckpt_path}/video_feature_extractor_{start_epoch}.pth'))

        ## block gradient for codebook module (codebook and decoder)
        self.video_feature_extractor.train()

    def evaluate_video(self, epoch = 0, only_val=False):

        save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
        hr_gt = []
        hr_backbone = []
        hr_mask = []
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()

        with torch.no_grad():

            for i, sample_batched in enumerate(tqdm(self.val_dataloader_video)):
                # get the inputs
                inputs, ecg = sample_batched['video'].cuda(), sample_batched['ecg'].cuda()
                if "seg" in sample_batched.keys():
                    mask = sample_batched['seg'].cuda()

                num_clip = 3 
                input_len = inputs.shape[2]
                input_len = input_len - input_len % (num_clip * 8)
                clip_len = input_len // num_clip

                inputs = inputs[:, :, :input_len, :, :]
                ecg = ecg[:, :input_len]

                new_args = copy.deepcopy(self.args)
                new_args.num_rppg = clip_len

                val_video_feature_extractor = BioPhysNet(frames=clip_len, device=self.device).to(self.device)
                val_video_feature_extractor.load_state_dict(torch.load(f'{self.save_ckpt_path}/video_feature_extractor_{epoch}.pth', weights_only=True))
                val_video_feature_extractor.eval()

                psd_total = 0
                psd_gt_total = 0
                psd_mask_total = 0
                for idx in range(num_clip):

                    inputs_iter = inputs[:, :, idx*clip_len : (idx+1)*clip_len, :, :]
                    ecg_iter = ecg[:, idx*clip_len : (idx+1)*clip_len]
                    input_dict = {"video":inputs_iter} 
                    if "seg" in sample_batched.keys():
                        mask_clip = mask[:, idx*clip_len : (idx+1)*clip_len, :, :].to(self.device)
                        input_dict["seg"] = mask_clip

                    psd_gt = TorchLossComputer.complex_absolute(ecg_iter, self.frame_rate, bpm_range)
                    psd_gt_total += psd_gt.view(-1).max(0)[1].cpu() + 40

                    outputs = val_video_feature_extractor(input_dict)
                    rPPG = outputs['rppg']

                    psd = TorchLossComputer.complex_absolute(rPPG[0], self.frame_rate, bpm_range)
                    psd_total += psd.view(-1).max(0)[1].cpu() + 40

                hr_backbone.append(psd_total / num_clip)
                hr_mask.append(psd_mask_total / num_clip)
                hr_gt.append(psd_gt_total / num_clip)

        ## save the results
        b, a = signal.butter(2, [0.67 / 15, 3 / 15], 'bandpass')
        # 使用 lfilter 函数进行滤波
        rPPG_np = rPPG[0].cpu().data.numpy()
        rPPG_np = signal.lfilter(b, a, rPPG_np)
        rPPG[0] = torch.from_numpy(rPPG_np).cuda()
        results_rPPG = []
        y1 = rPPG[0].cpu().data.numpy()
        y2 = ecg_iter[0].cpu().data.numpy()
        results_rPPG.append(y1)
        results_rPPG.append(y2)
        sio.savemat(os.path.join(save_path_epoch, f'test_rPPG.mat'), {'results_rPPG': results_rPPG})
        # show the ecg images
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        psd_pred = TorchLossComputer.complex_absolute(rPPG[0], self.frame_rate, self.bpm_range)
        psd_gt = TorchLossComputer.complex_absolute(ecg_iter[0], self.frame_rate, self.bpm_range)
        ax[0].set_title('rPPG')
        ax[0].plot(y1, label='rPPG')
        ax[0].plot(y2, label='ecg')
        ax[0].legend()
        ax[1].set_title('psd')
        ax[1].plot(psd_pred[0].cpu().data.numpy(), label='pred')
        ax[1].plot(psd_gt[0].cpu().data.numpy(), label='gt')
        ax[1].legend()
        fig.savefig(os.path.join(save_path_epoch, f'test_rPPG.png'))
        plt.close(fig)

        backbone_mae = np.mean(np.abs(np.array(hr_gt) - np.array(hr_backbone)))
        backbone_rmse = np.sqrt(np.mean(np.square(np.array(hr_gt) - np.array(hr_backbone))))
        cur_sd = np.std(np.array(hr_gt) - np.array(hr_backbone))
        cur_r = pearson_correlation_coefficient(np.array(hr_gt), np.array(hr_backbone))


        cur_mae = backbone_mae
        cur_rmse = backbone_rmse

        if cur_mae < self.best_val_mae:
            self.best_val_mae = cur_mae
            self.best_val_rmse = cur_rmse
            self.best_best_sd = cur_sd
            self.best_r = cur_r
            self.best_epoch = epoch
            # save the model
            torch.save(self.video_feature_extractor.state_dict(), os.path.join(self.save_ckpt_path, f'video_feature_extractor_best.pth'))

        if not only_val:
            self.logger.info(f'evaluate epoch {epoch}, total val {len(hr_gt)} ----------------------------------')
            self.logger.info(f'video-level mae of backbone: {backbone_mae}, video-level rmse: {backbone_rmse}, video-level sd: {cur_sd}, video-level r: {cur_r}')
            self.logger.info(f'video-level best mae of vq: {self.best_val_mae:.2f}, best rmse: {self.best_val_rmse:.2f}, best epoch: {self.best_epoch}')
            self.logger.info(f'best results: {self.best_val_mae:.2f}, {self.best_val_rmse:.2f}, {self.best_best_sd:.2f}, {self.best_r:.2f}, {self.best_epoch}')
            self.logger.info(f'------------------------------------------------------------------')
        else:
            print(f'evaluate epoch {epoch}, total val {len(hr_gt)} ----------------------------------')
            print(f'video-level mae of backbone: {backbone_mae}, video-level rmse: {backbone_rmse}, video-level sd: {cur_sd}, video-level r: {cur_r}')
            print(f'video-level best mae of vq: {self.best_val_mae:.2f}, best rmse: {self.best_val_rmse:.2f}, best epoch: {self.best_epoch}')
            print(f'best results: {self.best_val_mae:.2f}, {self.best_val_rmse:.2f}, {self.best_best_sd:.2f}, {self.best_r:.2f}, {self.best_epoch}')
            print(f'------------------------------------------------------------------')

    def train(self, start_epoch=-1, continue_log=''):

        self.prepare_train(start_epoch=start_epoch, continue_log=continue_log)
        self.logger.info(f'prepare train, load ckpt and block gradient, start_epoch: {start_epoch}')

        echo_batches = self.args.echo_batches
        gamma = self.args.gamma
        step_size = self.args.step_size
        eval_step = self.args.eval_step
        lr = self.args.lr * (gamma ** ((start_epoch+1) // step_size))
        optimizer = self.optimizer
        scheduler = self.scheduler
        
        for epoch in range(start_epoch+1, self.args.epochs):
            if epoch % step_size == step_size - 1:
                lr *= gamma
            loss_rPPG_avg = AvgrageMeter()
            loss_peak_avg = AvgrageMeter()
            loss_kl_avg = AvgrageMeter()
            loss_hr_mae = AvgrageMeter()
            save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
            if not os.path.exists(save_path_epoch):
                os.makedirs(save_path_epoch)
            self.logger.info(f'train epoch: {epoch} lr: {lr}')
            with tqdm(range(len(self.train_dataloader))) as pbar:
                for i, sample_batched in zip(pbar, self.train_dataloader):
                    inputs, ecg = sample_batched['video'].cuda(), sample_batched['ecg'].cuda()
                    clip_average_HR  = sample_batched['clip_avg_hr'].cuda()
                    for k, v in sample_batched.items():
                        if torch.is_tensor(v):
                            sample_batched[k] = v.cuda()

                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.video_feature_extractor(sample_batched)
                    
                    if self.args.align:
                        cxcorr, max_idx, min_idx, rppg = cxcorr_align(outputs["rppg"], sample_batched["ecg"])
                        outputs["rppg"] = rppg
                    rPPG = outputs['rppg']

                    ## calculate loss
                    loss, step_log = self.criterion(sample_batched, outputs)
                    
                    clip_average_HR = (clip_average_HR - 40)  

                    ## update loss saver
                    n = inputs.size(0)
                    loss_rPPG_avg.update(step_log["nploss"], n)
                    loss_peak_avg.update(step_log["peak_loss"], n)
                    loss_kl_avg.update(step_log["kl_loss"], n)
                    loss_hr_mae.update(step_log["train_mae"], n)

                    loss.backward()
                    optimizer.step()

                    ## 判断loss为nan时，打印信息
                    if torch.isnan(loss).any():
                        print(f'loss is nan, epoch: {epoch}, mini-batch: {i}')
                        print(f'loss: {loss}, loss_rPPG: {step_log["nploss"]}, loss_peak: {step_log["peak_loss"]}, loss_kl: {step_log["kl_loss"]}')
                        # 打印输入是否有nan
                        print(f"video: {torch.isnan(inputs).any()}, ecg: {torch.isnan(ecg).any()}, {sample_batched['path']}")
                        exit()


                    if i % echo_batches == echo_batches - 1:  # info every mini-batches
                        self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.5f}, NP_loss = {loss_rPPG_avg.avg:.4f}, ' \
                            f'fre_loss = {loss_peak_avg.avg:.4f}, hr_mae = {loss_hr_mae.avg:.2f}')

                        # save the ecg images
                        b, a = signal.butter(2, [0.67 / 15, 3 / 15], 'bandpass')
                        # 使用 lfilter 函数进行滤波
                        rPPG_np = rPPG[0].cpu().data.numpy()
                        rPPG_np = signal.lfilter(b, a, rPPG_np)
                        rPPG[0] = torch.from_numpy(rPPG_np).cuda()
                        results_rPPG = []
                        y1 = rPPG[0].cpu().data.numpy()
                        y2 = ecg[0].cpu().data.numpy()
                        results_rPPG.append(y1)
                        results_rPPG.append(y2)
                        sio.savemat(os.path.join(save_path_epoch, f'minibatch_{i+1:0>4}_rPPG.mat'), {'results_rPPG': results_rPPG})
                        # show the ecg images
                        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
                        psd_pred = TorchLossComputer.complex_absolute(rPPG[0], self.frame_rate, self.bpm_range)
                        psd_gt = TorchLossComputer.complex_absolute(ecg[0], self.frame_rate, self.bpm_range)
                        ax[0].set_title('rPPG')
                        ax[0].plot(y1, label='rPPG')
                        ax[0].plot(y2, label='ecg')
                        ax[0].legend()
                        ax[1].set_title('psd')
                        ax[1].plot(psd_pred[0].cpu().data.numpy(), label='pred')
                        ax[1].plot(psd_gt[0].cpu().data.numpy(), label='gt')
                        ax[1].legend()
                        fig.savefig(os.path.join(save_path_epoch, f'minibatch_{i+1:0>4}_rPPG.png'))
                        plt.close(fig)
                    pbar.set_description(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.4f}, NP_loss = {loss_rPPG_avg.avg:.3f}, ' \
                                f'fre_loss = {loss_peak_avg.avg:.3f}, hr_mae = {loss_hr_mae.avg:.2f}')
            scheduler.step()

            # save the model
            torch.save(self.video_feature_extractor.state_dict(), os.path.join(self.save_ckpt_path, f'video_feature_extractor_{epoch}.pth'))

            # evaluate the model
            if epoch % eval_step == eval_step - 1:
                self.evaluate_video(epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## general parameters
    parser.add_argument('--num_rppg', type=int, default=160, help='the number of rPPG')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='the drop rate of CodeResNet')
    parser.add_argument('--train_model', type=str, default='hemnet', help='train_model = [hemnet]')
    parser.add_argument('--dataset', type=str, default='PURE')
    ### add for codephys
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--echo_batches', type=int, default=400, help='the number of mini-batches to print the loss')
    parser.add_argument('--save_path', type=str, default="/data/chushuyang/hemnet_exp_results", help='the path to save the model [ckpt, code, visulization]')

    parser.add_argument('--align', type=bool, default=False)
    parser.add_argument('--proir_weight', default=1e-4)
    parser.add_argument('--appearance_weight', default=1e-3)
    parser.add_argument('--shading_weight', default=1e-3)
    parser.add_argument('--sparsity_weight', default=1e-7)

    parser.add_argument('--np_loss', default=True)
    parser.add_argument('--np_weight', default=0.01)
    parser.add_argument('--cn_loss', default=True)
    parser.add_argument('--cn_weight', default=1.0)
    parser.add_argument('--rec_loss', default=True)

    ### model parameters
    parser.add_argument('--input_dim', type=int, default=3, help='the number of input channels')
    
    ### datasets
    parser.add_argument('--norm_type', type=str, default='reconstruct')
    parser.add_argument('--aug', type=str, default='') # figsc
    parser.add_argument('--w', type=int, default=64, help='')
    parser.add_argument('--K', type=int, default=1, help="fold")

    args = parser.parse_args()

    set_seed(92)
    hemnet_trainer = Trainer(args)
    hemnet_trainer.prepare_train(start_epoch=-1, continue_log='', only_val=False) # NOTE: WHETHER TO CONTINUE TRAINING
    hemnet_trainer.train(start_epoch=-1, continue_log='') # NOTE: WHETHER TO CONTINUE TRAINING



