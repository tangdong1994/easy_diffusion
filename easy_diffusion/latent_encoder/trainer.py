import os
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import trange, tqdm
from easy_diffusion.loss import LPIPSWithDiscriminator
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class AutoEncoderTrainer(object):
    def __init__(self, model, learning_rate, lr_g_factor, scheduler_config=None, image_shape=(3,512,512), device=None, disc_start=50001):
        self.image_shape = image_shape
        if not device:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor
        self.disc_start=disc_start
        self.loss = self.init_loss()
        [self.opt_ae, self.opt_disc], self.scheduler = self.init_optimizer()
        
    def init_optimizer(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.model.encoder.parameters())+
                                  list(self.model.decoder.parameters())+
                                  list(self.model.quant_conv.parameters())+
                                  list(self.model.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    
    def init_loss(self, kl_weight=0.000001, disc_weight=0.5):
        return LPIPSWithDiscriminator(disc_start=self.disc_start, kl_weight=kl_weight, disc_weight=disc_weight, device=self.device)
        
    def train(self, train_dataset, epoches, log_path, batch_size, val_dataset=None, snapshot_freq=1000,print_step=1000,resume_training=False,ema=True, num_workers=os.cpu_count()):
        start_epoch, step, global_step = 0, 0, 0 
        try:
            if resume_training:
                states = torch.load(os.path.join(log_path, "ckpt.pth"))
                self.model.load_state_dict(states[0])

                #states[1]["param_groups"][0]["eps"] = 0.00000001
                self.opt_ae.load_state_dict(states[1])
                for sta in self.opt_ae.state.values():
                    for k, v in sta.items():
                        if torch.is_tensor(v):
                            sta[k] = v.cuda()
                self.opt_disc.load_state_dict(states[2])
                for sta in self.opt_disc.state.values():
                    for k, v in sta.items():
                        if torch.is_tensor(v):
                            sta[k] = v.cuda()
                start_epoch = states[3]
                global_step = states[4]
        except Exception as e:
            print("resume_training happened error!", e)
        tqdm_epoch = trange(start_epoch, epoches)
        train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                multiprocessing_context=mp.get_context('spawn') if num_workers>0 else None,
                collate_fn=train_dataset.collate_fn if hasattr(train_dataset,'collate_fn') else None
            )
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                multiprocessing_context=mp.get_context('spawn') if num_workers>0 else None,
                collate_fn=val_dataset.collate_fn if hasattr(val_dataset,'collate_fn') else None
            )
        else:
            val_loader = None
        self.model.to(self.device)
        for epoch in tqdm_epoch:
            self.model.train()
            ave_loss_ae = 0.
            num_items_ae = 0.
            ave_loss_disc = 0.
            num_items_disc = 0.
            for idx, batch in enumerate(tqdm(train_loader)):
                self.model.train()
                batch = batch.to(self.device)
                xrec, posterior = self.model(batch)
                if global_step%2 == 0:
                    # autoencode
                    loss, log_dict_ae = self.loss(batch, 
                                                  xrec, 
                                                  posterior,
                                                  0, 
                                                  global_step,
                                                  last_layer=self.model.decoder.conv_out.weight, split="train")
                    self.opt_ae.zero_grad()
                    loss.backward()
                    self.opt_ae.step()
                    ave_loss_ae += loss.item() * batch.shape[0]
                    num_items_ae += batch.shape[0]

                if global_step%2 == 1:
                    # discriminator
                    loss, log_dict_disc = self.loss(batch, xrec, posterior, 1, global_step,
                                                    last_layer=self.model.decoder.conv_out.weight, split="train")
                    self.opt_disc.zero_grad()
                    loss.backward()
                    self.opt_disc.step()
                    ave_loss_disc += loss.item() * batch.shape[0]
                    num_items_disc += batch.shape[0]
                self.global_step = global_step
                if global_step%print_step==0 or global_step%(print_step+1)==0:
                    if global_step%2 == 0:
                        print("{} loss: {:5f}".format("autoencoder ",loss))
                        # print(log_dict_ae)
                    if global_step%2 == 1:
                        print("{} loss: {:5f}".format("gan ",loss))
                        # print(log_dict_disc)
                # save model during training
                global_step += 1
                if global_step % snapshot_freq == 0 or global_step == 1:
                    if not os.path.exists(log_path):
                        os.makedirs(log_path)
                    states = [
                            self.model.state_dict(),
                            self.opt_ae.state_dict(),
                            self.opt_disc.state_dict(),
                            epoch,
                            global_step,  
                        ]
#                     if ema:
#                         states.append(ema_helper.state_dict())
#                     torch.save(
#                             states,
#                             os.path.join(log_path, "ckpt_{}.pth".format(step)),
#                         )
                    torch.save(states, os.path.join(log_path, "ckpt.pth"))
            tqdm_epoch.set_description('AutoEncoder Average Loss: {:5f}, disc Loss: {:5f}'.format(ave_loss_ae / num_items_ae, ave_loss_disc/num_items_disc))
            if val_dataset:
                self.validate(val_loader)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        states = [
                self.model.state_dict(),
                self.opt_ae.state_dict(),
                self.opt_disc.state_dict(),
                epoch,
                global_step,  
                ]
#                     if ema:
#                         states.append(ema_helper.state_dict())
#                     torch.save(
#                             states,
#                             os.path.join(log_path, "ckpt_{}.pth".format(step)),
#                         )
        torch.save(states, os.path.join(log_path, "ckpt.pth"))
    
    def validate(self, data_loader):
        self.model.eval()
        ave_loss_ae = 0.
        num_items = 0.
        ave_loss_disc = 0.
        for idx, batch in enumerate(tqdm(data_loader)): 
            batch = batch.to(self.device)
            xrec, posterior = self.model(batch)
            aeloss, log_dict_ae = self.loss(batch, xrec, posterior, 0, self.global_step,
                                                    last_layer=self.model.decoder.conv_out.weight, split="val")
            discloss, log_dict_disc = self.loss(batch, xrec, posterior, 1, self.global_step,
                                                    last_layer=self.model.decoder.conv_out.weight, split="val")
            ave_loss_ae += aeloss.item() * batch.shape[0]
            ave_loss_disc += discloss.item() * batch.shape[0]
            num_items += batch.shape[0]
        print('validate AutoEncoder Average Loss: {:5f}, disc Loss: {:5f}'.format(ave_loss_ae / num_items, ave_loss_disc/num_items))
        x = next(iter(data_loader))
        x = x[:8]
        x_hat, _ = self.model(x.to(self.device))
        plt.figure(figsize=(6,6.5))
        plt.axis('off')
        plt.imshow(make_grid(x[:64,:,:,:].cpu()).permute([1,2,0]), vmin=0., vmax=1.)
        plt.title("Original")

        plt.figure(figsize=(6,6.5))
        plt.axis('off')
        plt.imshow(make_grid(x_hat[:64,:,:,:].cpu()).permute([1,2,0]), vmin=0., vmax=1.)
        plt.title("AE Reconstructed")
        plt.show()
            
        