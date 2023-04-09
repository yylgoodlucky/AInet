import torch
import numpy as np
import argparse, time, os, random, cv2
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim import lr_scheduler
from torch.autograd import Variable

from dataset.base_datasets import AInet_dataset
from model.AInet import AInet_Generator, AInet_Discriminator, GANLoss
from utils import save_checkpoint, trans_to_cuda
# from torch.utils.tensorboard import SummaryWriter
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preprocessed_dir", type=str, help="Root folder of the preprocessed dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, required=True)

    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--overlay", type=int, default=3)
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--lamb', type=int, default=10)
    
    parser.add_argument('--lr_policy', type=str, default='plateau', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')   
    parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=10, help='# of iter to linearly decay learning rate to zero')
    return parser.parse_args()

config = parse_args()
save_log_path = os.path.join(config.checkpoint_dir, 'log.txt')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(42)

def initialize_weights( net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)
    
def save_sample_images(fake_sequences, real_sequences, epoch, step, checkpoint_dir):
    b, ct, h, w = fake_sequences.shape
    g = (fake_sequences.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    gt = (real_sequences.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    folder = os.path.join(checkpoint_dir, "sample_epoch{:02d}_step{:05d}".format(epoch, step))
    if not os.path.exists(folder): os.mkdir(folder)
    fake = g.reshape(h*b, w, ct)
    real = gt.reshape(h*b, w, ct)
    img = np.concatenate((real, fake), axis=1)
    for t in range(0,ct,3):
        cv2.imwrite("{}/{}_{}.jpg".format(folder, t, ct), img[:, :, t:t+3])

def get_scheduler(optimizer, config):
    if config.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + config.epoch_count - config.niter) / float(config.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iters, gamma=0.1)
    elif config.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.0001, patience=2)
    elif config.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler    

def update_learning_rate(scheduler, optimizer):
    metrics = 0
    if config.lr_policy == "plateau":
        scheduler.step(metrics)
    else:
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
    print('==========================')
    
def train(config):
    start_epoch = 0
    print('===> Loading datasets')

    train_data_loader = DataLoader(AInet_dataset(config.preprocessed_dir, 'train', window_size=config.window_size, step=config.step),
        batch_size=config.batch_size,
        num_workers=config.num_thread,
        shuffle=False, drop_last=True,
        
        )
    test_data_loader = DataLoader(AInet_dataset(config.preprocessed_dir, "test", window_size=config.window_size, step=config.step), 
            batch_size=config.batch_size,
            num_workers=config.num_thread,
            shuffle=False, drop_last=True
            )
    initialize_weights(AInet_Generator())
    initialize_weights(AInet_Discriminator(input_nc=30))
    
    device = torch.device("cuda" if config.cuda else "cpu")
    
    Generator = trans_to_cuda(AInet_Generator())
    Discriminator = trans_to_cuda(AInet_Discriminator(input_nc=30))
    
    optimizer_G = torch.optim.Adam(Generator.parameters(),
                lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_D = torch.optim.Adam(Discriminator.parameters(),
                lr=config.lr, betas=(config.beta1, config.beta2))
    
    net_g_scheduler = get_scheduler(optimizer_G, config)
    net_d_scheduler = get_scheduler(optimizer_D, config)
    
    L1loss = torch.nn.L1Loss().to(device)
    GANloss = GANLoss().to(device)
    
    for epoch in range(start_epoch, config.niter + config.niter_decay + 1):
        print("Starting Epoch: {}".format(epoch))
        # prog_bar = tqdm(enumerate(train_data_loader))
        
        for step, (single_img, input_mfcc, img_seq) in tqdm(enumerate(train_data_loader)):
            ti = time.time()
            single_img = Variable(torch.Tensor(single_img.float())).cuda()
            audio = Variable(torch.Tensor(input_mfcc.float())).cuda()
            img_seq = Variable(torch.Tensor(img_seq.float())).cuda()
            
            Generator.train()
            fake_img_seq = Generator(single_img, audio)
            
            # (1) updata D network
            optimizer_D.zero_grad()
            
            fake_sequences = torch.cat([fake_img_seq[:, i] for i in range(fake_img_seq.size(1))], dim=1)
            real_sequences = torch.cat([img_seq[:, i] for i in range(img_seq.size(1))], dim=1)
            
            # train with fake
            fake_img = torch.cat((fake_sequences, real_sequences), 1)
            pred_fake = Discriminator.forward(fake_img.detach())
            loss_d_fake = GANloss(pred_fake, False)
            
            # train with real
            real_img = torch.cat((fake_sequences, real_sequences), 1)
            pred_real = Discriminator.forward(real_img)
            loss_d_real = GANloss(pred_real, True)

            # Combined D loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward(retain_graph=True)
            optimizer_D.step()
            
            # (2) updata G network
            optimizer_G.zero_grad()
            
            # First, G(A) should fake the discriminator
            fake_sequences = torch.cat([fake_img_seq[:, i] for i in range(fake_img_seq.size(1))], dim=1)
            real_sequences = torch.cat([img_seq[:, i] for i in range(img_seq.size(1))], dim=1)

            fake_img = torch.cat((fake_sequences, real_sequences), 1)
            Discriminator.train()
            pred_fake = Discriminator.forward(fake_img)
            loss_g_gan = GANloss(pred_fake, True)
            
            # Second, G(A) = real
            loss_g_l1 = L1loss(fake_img_seq, img_seq)
            
            loss_g = loss_g_gan + loss_g_l1 * config.lamb
        
            loss_g.backward(retain_graph=True)
            
            optimizer_G.step()
            
            print("===> Epoch[{:02d}]({:04d}/{:04d}): Loss_l1: {:.4f} Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, step, len(train_data_loader), loss_g_l1.item(), loss_d.item(), loss_g.item()))
            
            if step == 1 or step % config.checkpoint_interval == 0:
                save_checkpoint(Generator, optimizer_G, step, config.checkpoint_dir, epoch)
                
            if step % config.eval_interval == 0:
                save_sample_images(fake_sequences, real_sequences, epoch, step, config.checkpoint_dir)
                with torch.no_grad():
                    eval_model(test_data_loader, step, Generator, epoch)
    
        update_learning_rate(net_g_scheduler, optimizer_G)
        update_learning_rate(net_d_scheduler, optimizer_D)
        

def eval_model(test_data_loader, step, model, epoch):
    eval_steps = 500
    print("Evaluating for {} steps".format(eval_steps))
    l1_losses = []
    step = 0
    
    for step, (single, input_mfcc, seq) in tqdm(enumerate(test_data_loader)):
        step += 1
        model.eval()
        
        single = torch.Tensor(single.float()).cuda()
        audio = torch.Tensor(input_mfcc.float()).cuda()
        seq = torch.Tensor(seq.float()).cuda()
    
        fake_seq = model(single, audio)
        
        L1loss = torch.nn.L1Loss()
        l1_loss = L1loss(seq, fake_seq)
        l1_losses.append(l1_loss)
        
        # if step > eval_steps: 
    averaged_l1_loss = sum(l1_losses) / len(l1_losses)
    print('L1_loss: {}'.format(averaged_l1_loss))
    
    log = open(save_log_path, mode="a", encoding="utf-8")
    message = 'Evaluation step {} (epoch {}), L1: {} '.format(
        step, epoch,
        sum(l1_losses) / len(l1_losses))

    print(message, file=log)
    log.close()
    
    return sum(l1_losses) / len(l1_losses)


def main(config):
    
    train(config)

if __name__ == "__main__":
    main(config)
