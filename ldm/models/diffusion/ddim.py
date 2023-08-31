"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               global_strength=1,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        #import pdb; pdb.set_trace()
        C, H, W = shape
        size = (batch_size, C, H, W) # channel=4
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    global_strength=global_strength
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None,
                      global_strength=1):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        #TODO 改这里？
        #import pdb; pdb.set_trace()
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, mininterval=600)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            # cond len=3
            #import pdb; pdb.set_trace()
            # if mask is not None:  #在隐空间加soft mask
            #     img[2,:,:,:]= img[2,:,:,:] * mask[:4, :, :]
            #     img[3,:,:,:]= img[3,:,:,:] * mask[-4:, :, :]
            
            # if mask is not None:
            #     assert x0 is not None
            #     img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? attention mask应该不是直接加这里？
            #     img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]
            #输入的cond还是原尺寸
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold, 
                                      global_strength=global_strength)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None,global_strength=1):
        b, *_, device = *x.shape, x.device
        
        #TODO 应该是在这里，想办法把之前生成的attention mask加进去 8.3
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):  #走这里
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        #for i in range(len(c[k])):
                            #import pdb; pdb.set_trace()
                            # if k=='local_control':
                            #     zero_tensor = torch.zeros((4, 3, 256, 1024)).cuda()
                            #     unconditional_conditioning[k][i] = torch.cat((unconditional_conditioning[k][i], zero_tensor), dim=1)
                            #c_in[k] = torch.cat([unconditional_conditioning[k][i],c[k][i]])
                            
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))] #i只为0
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])  #[B,12,256,1024]
            #import pdb; pdb.set_trace()
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in, global_strength).chunk(2) #这一步关键,均在隐空间 shape [2, 4, 32, 128],已经加上所有的
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output  #这里

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs) #[B, 4, 32, 128]

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()  #torch.Size([2, 4, 32, 128]) 1/8大小 基于t时刻带噪声的图像x预测了t0时刻未添加噪声的图像pred_x0
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise  #这就是sample之后的sample_img,全部在隐空间 t-1时刻的降噪图像x_prev
        return x_prev, pred_x0  #均为[B,4,32,128] x_prev即作为此轮的预测结果进行输出，而pred_x0作为ddim_sampling函数中建立中间体intermediates所需要的list内容
    
    # 修改自 self attention add attention的表述
    def attention_masking(
        self, x, t, attn_map, prev_noise, blur_sigma, model_kwargs=None,
    ):
        """
        Apply the self-attention mask to produce bar{x_t}

        :param x: the predicted x_0 [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param attn_map: the attention map tensor at time t.
        :param prev_noise: the previously predicted epsilon to inject
            the same noise as x_t.
        :param blur_sigma: a sigma of Gaussian blur.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: the bar{x_t}
        """
        import pdb; pdb.set_trace()
        B, C, H, W = x.shape
        assert t.shape == (B,)
        
        if self.sel_attn_depth in [0, 1, 2] or self.sel_attn_block == "middle":
            attn_res = 8
        elif self.sel_attn_depth in [3, 4, 5]:
            attn_res = 16
        elif self.sel_attn_depth in [6, 7, 8]:
            attn_res = 32
        else:
            raise ValueError("sel_attn_depth must be in [0, 1, 2, 3, 4, 5, 6, 7, 8]")
        # TODO： 移动到我的模型上
        # Generating attention mask
        attn_mask = attn_map.reshape(B, self.num_heads, attn_res ** 2, attn_res ** 2).mean(1, keepdim=False).sum(1, keepdim=False) > 1.0  # [16, 64]
        attn_mask = attn_mask.reshape(B, attn_res, attn_res).unsqueeze(1).repeat(1, 3, 1, 1).int().float()   #[16, 3, 8, 8]
        attn_mask = F.interpolate(attn_mask, (H, W))   #[16, 3, 256, 256] 插值为正常的图片尺寸
        import pdb; pdb.set_trace()
        vis(attn_mask,'attn_mask.png')
        # Gaussian blur
        transform = T.GaussianBlur(kernel_size=31, sigma=blur_sigma)
        x_curr = transform(x)
        vis(x_curr, 'x_curr1.png')
        # Apply attention masking
        x_curr = x_curr * (attn_mask) + x * (1 - attn_mask)
        vis(x_curr, 'x_curr2.png')
        # Re-inject the noise
        x_curr = self.q_sample(x_curr, t, noise=prev_noise)
        vis(x_curr, 'x_curr3.png')
        return x_curr
    # 修改自 self attention add attention的表述
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        attn_guidance=True,
        guidance_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param attn_guidance: if True, self-attention guidance is given.
        :param guidance_kwargs: if not None, a dict of the parameters of self-attention guidance.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """

        guide_scale = guidance_kwargs['guide_scale']
        guide_start = guidance_kwargs['guide_start']
        blur_sigma = guidance_kwargs['blur_sigma']
        
        if attn_guidance and guide_start>t[0]:
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            # Get the original output: eps(x_t)
            cond_eps = out['eps']
            
            mask_blurred = self.attention_masking(
                out['pred_xstart'],
                t,
                out['attn_map'],  #关注这个，改为自己的
                prev_noise=cond_eps,
                blur_sigma=blur_sigma,
                model_kwargs=model_kwargs,
            )
            mask_out = self.p_mean_variance(
                model,
                mask_blurred,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            # Get the original output: eps(bar{x_t})
            import pdb; pdb.set_trace()
            uncond_eps = mask_out['eps']   #[16,3,256,256] 直接把这个替换为自己的,做一个可视化？
            
            
            #vis(uncond_eps)
            # Get the guided eps: eps(bar{x_t}) + s * (eps(x_t) - eps(bar{x_t}))
            guided_eps = uncond_eps + guide_scale * (cond_eps - uncond_eps)

            def process_xstart(x):
                if denoised_fn is not None:
                    x = denoised_fn(x)
                if clip_denoised:
                    return x.clamp(-1, 1)
                return x
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=guided_eps)
            )

            final_out = {}
            final_out["mean"], _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
            final_out["variance"] = out["variance"]

            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            if cond_fn is not None: #不走
                final_out["mean"] = self.condition_mean(
                    cond_fn, final_out, x, t, model_kwargs=model_kwargs
                )
            
            sample = final_out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            #vis(out["pred_xstart"])
            return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        
    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]
        #import pdb; pdb.set_trace()
        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image',mininterval=600):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps,mininterval=600)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
