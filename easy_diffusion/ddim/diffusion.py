import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import time
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR
from torchvision.utils import make_grid
from .loss import noise_estimation_loss_v2 as loss_func, normal_kl, mean_flat, discretized_gaussian_log_likelihood
from .ema import EMAHelper
from .config_type import ModelMeanType, ModelVarType, LossType
from easy_diffusion.aiposdatasetlmdb import AiposFoodDatasetLmdbV2,Mess_Image
import torch.multiprocessing as mp



def collate_fn(batch):
    img, label = zip(*batch)  # transposed
    return torch.stack(img, 0), torch.cat(label, 0)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def data_transform(X, uniform_dequantization=True, 
                   gaussian_dequantization=False,
                  rescaled=True,
                  has_image_mean=False):
    if uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if rescaled:
        X = 2 * X - 1.0
#     elif logit_transform:
#         X = logit_transform(X)

#     if has_image_mean:
#         return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(X,uniform_dequantization=False, 
                   gaussian_dequantization=False,
                  rescaled=True,
                  has_image_mean=False):
#     if has_image_mean:
#         X = X + config.image_mean.to(X.device)[None, ...]

    if False:
        X = torch.sigmoid(X)
    elif rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


class Diffusion(object):
    def __init__(self, model_var_type="fixedlarge",image_shape=(3,64,64), device=None):
        self.model_var_type = ModelVarType.FIXED_LARGE
        self.model_mean_type = ModelMeanType.EPSILON
        self.image_shape = image_shape
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        betas = get_beta_schedule(
            beta_schedule='linear',
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000
        )
        
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        
        self.num_timesteps = betas.shape[0]
        
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            torch.cat((self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]),0)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        if self.model_var_type == ModelVarType.FIXED_LARGE:
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == ModelVarType.FIXED_SMALL:
            self.logvar = posterior_variance.clamp(min=1e-20).log()
            
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(
        self, model_output, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        # model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    torch.cat((self.posterior_variance[1].unsqueeze(0), self.betas[1:]),0),
                    torch.log(torch.cat((self.posterior_variance[1].unsqueeze(0), self.betas[1:]),0))
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
        
    def train(self, model, dataset, leanrning_rate, epoches, log_path,batch_size,snapshot_freq=1000,print_step=1000, resume_training=False,ema=True,ema_rate=0.999, vlb_need=True):
        num_workers = 0
        data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                multiprocessing_context=mp.get_context('spawn') if num_workers>0 else None,
#                 prefetch_factor=4,
                collate_fn=dataset.collate_fn if hasattr(dataset,'collate_fn') else None
            )
        
        # optimizer = Adam(model.parameters(), lr=leanrning_rate)
        optimizer = AdamW(model.parameters(), lr=leanrning_rate, weight_decay=0)
#         optimizer = Adam(model.parameters(), lr=leanrning_rate, weight_decay=0,
#                           betas=(0.9, 0.999), amsgrad=False,
#                           eps=0.00000001)
        # optimizer = SGD(score_model.parameters(), lr=lr, momentum=0.8)
        # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch:  0.90 ** epoch)
        if ema:
            ema_helper = EMAHelper(mu=ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
        start_epoch, step = 0, 0 
        if resume_training:
            states = torch.load(os.path.join(log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = 0.00000001
            optimizer.load_state_dict(states[1])
            for sta in optimizer.state.values():
                for k, v in sta.items():
                    if torch.is_tensor(v):
                        sta[k] = v.cuda()
            start_epoch = states[2]
            step = states[3]
            if ema:
                ema_helper.load_state_dict(states[4])
        tqdm_epoch = trange(start_epoch, epoches)

        for epoch in tqdm_epoch:
            model.to(self.device)
            model.train()
            avg_loss = 0.
            num_items = 0
            data_start = time.time()
            data_time = 0
            idxx = 0
            print("epoch {}".format(epoch),end=" ")
            for x, y in tqdm(data_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()

                x = x.to(self.device)
                if y is not None:
                    if len(y.shape)==2:
                        y = y.unsqueeze(0)
                    y = y.to(self.device).float()
                x = data_transform(x)
                # e = torch.randn_like(x)
                e = torch.randn(x.shape, device=x.device)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                x_t = self.q_sample(x, t, noise=e)
                model_output = model(x_t, t.float(), y=y)
                if vlb_need:
                    batch_size, channel = x_t.shape[:2]
                    if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                        assert model_output.shape == (batch_size, channel * 2, *x_t.shape[2:])
                        model_output, model_var_values = torch.split(model_output, channel, dim=1)
                        # Learn the variance using the variational bound, but don't let
                        # it affect our mean prediction.
                        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                    else:
                        frozen_out = model(x, t.float(), y=y)
                        
                    vb_loss = self._vb_terms_bpd(
                                model_output=frozen_out,
                                x_start=x,
                                x_t=x_t,
                                t=t,
                                clip_denoised=False
                                )["output"].mean()
                    loss = loss_func(model_output, e) + vb_loss
                else:
                    loss = loss_func(model_output, e)
                # loss = loss_registry[config.model.type](model, x, t, e, b) 
                # loss = loss_func(model, x, t, e, y, b)
                optimizer.zero_grad()
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1
                    )
                except Exception:
                    pass
                optimizer.step()
                if ema:
                    ema_helper.update(model)
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
                step += 1
                if idxx%print_step==0:
                    print(" loss: {:5f}".format(loss), end=" ")
                idxx +=1
                if step % snapshot_freq == 0 or step == 1:
                    if not os.path.exists(log_path):
                        os.makedirs(log_path)
                    states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                    if ema:
                        states.append(ema_helper.state_dict())
                    torch.save(
                            states,
                            os.path.join(log_path, "ckpt_{}.pth".format(step)),
                        )
                    torch.save(states, os.path.join(log_path, "ckpt.pth"))
            # scheduler.step()
            # lr_current = scheduler.get_last_lr()[0]
            # print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
            print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, leanrning_rate))
            # Print the averaged training loss so far.
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            # Update the checkpoint after each epoch of training.
            model.eval()
            if y is not None:
                self.visual_process(model,y[0])
            else:
                self.visual_process(model,None)
        states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
        if ema:
            states.append(ema_helper.state_dict())
        torch.save(
                    states,
                    os.path.join(log_path, "ckpt_{}.pth".format(step)),
                    )
        torch.save(states, os.path.join(log_path, "ckpt.pth"))
            # self.visual_process(model,None)
            
    def sample_image(self, x, model, y, sample_type="generalized", skip_type="uniform",timesteps=1000,last=True,skip=None,eta=0.0):
            skip = skip if skip else 1

            if sample_type == "generalized":
                if skip_type == "uniform":
                    skip = self.num_timesteps // timesteps#100#timesteps
                    seq = range(0, self.num_timesteps, skip)
                elif skip_type == "quad":
                    seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), timesteps
                        )
                        ** 2
                    )
                    seq = [int(s) for s in list(seq)]
                else:
                    raise NotImplementedError
                from .denoising import generalized_steps

                xs = generalized_steps(x, seq, model, self.betas,y, eta=eta)
                x = xs
            elif sample_type == "ddpm_noisy":
                if skip_type == "uniform":
                    skip = self.num_timesteps // timesteps
                    seq = range(0, self.num_timesteps, skip)
                elif skip_type == "quad":
                    seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), timesteps
                        )
                        ** 2
                    )
                    seq = [int(s) for s in list(seq)]
                else:
                    raise NotImplementedError
                from .denoising import ddpm_steps

                x = ddpm_steps(x, seq, model, self.betas, y)
            else:
                raise NotImplementedError
            if last:
                x = x[0][-1]
            return x

    def test(self):
            pass

        
    def visual_process(self, model, label_embeding, seed=None, sample_type="generalized", timesteps=100):
            rand_seed = np.random.randint(0,1000)
            if seed is not None:
                rand_seed = seed
            
            torch.manual_seed(rand_seed)
            # torch.cuda.manual_seed()为当前GPU设置随机种子
            torch.cuda.manual_seed(rand_seed)
            # a = np.random.randint(0,300)
            # digit = a #@param {'type':'integer'}
            sample_batch_size = 1 #@param {'type':'integer'}
            num_steps = 1000 #@param {'type':'integer'}
            # score_model.eval()
            ## Generate samples using the specified sampler.
            x = torch.randn(
            sample_batch_size,
            self.image_shape[0],
            self.image_shape[1],
            self.image_shape[2],
            device=self.device,
        )
            with torch.no_grad():
                # y = label_embeding.repeat(sample_batch_size,1)
                if label_embeding is not None:
                    y = label_embeding.unsqueeze(0)
                else:
                    y = None
                
#                 y = y.to(self.device)
#                 y=None
                samples = self.sample_image(x, model, y=y,sample_type=sample_type, timesteps=timesteps)
            samples = inverse_data_transform(samples)#[inverse_data_transform(y) for y in samples]

            ## Sample visualization.
            #samples = samples.clamp(0.0, 1.0)
            import matplotlib.pyplot as plt
            sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

            plt.figure(figsize=(6,6))
            plt.axis('off')
            plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
            plt.show()

            
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
