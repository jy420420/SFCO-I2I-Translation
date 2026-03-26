import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss  
import util.util as util
from models.hDCE import PatchHDCELoss
from models.SRC import SRC_Loss

try:
    import torchvision.models as tv_models
except Exception:
    tv_models = None

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def get_lambda(alpha=1.0, size=None, device=None):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

def get_spa_lambda(alpha=1.0, size=None, device=None):
    if alpha > 0.:
        lam = torch.from_numpy(np.random.beta(alpha, alpha, size=size)).float().to(device)
    else:
        lam = 1.
    return lam

class CUTModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0)
        parser.add_argument('--lambda_HDCE', type=float, default=1.0)
        parser.add_argument('--lambda_SRC', type=float, default=1.0)
        parser.add_argument('--dce_idt', action='store_true')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'])
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07)
        parser.add_argument('--num_patches', type=int, default=256)
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--use_curriculum', action='store_true')
        parser.add_argument('--HDCE_gamma', type=float, default=1)
        parser.add_argument('--HDCE_gamma_min', type=float, default=1)
        parser.add_argument('--step_gamma', action='store_true')
        parser.add_argument('--step_gamma_epoch', type=int, default=200)
        parser.add_argument('--no_Hneg', action='store_true')

        parser.add_argument('--use_prob_sampling', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--prob_rho', type=float, default=0.7)
        parser.add_argument('--prob_eps', type=float, default=1e-6)
        parser.add_argument('--prior_sigma', type=float, default=1.5)

        parser.add_argument('--lambda_phase', type=float, default=0.0)
        parser.add_argument('--phase_eps', type=float, default=1e-8)

        parser.add_argument('--lambda_amp', type=float, default=0.0)
        parser.add_argument('--amp_use_log', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--amp_batch_mean_ref', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--amp_area_norm', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--amp_mean_warmup_ratio', type=float, default=0.0)

        parser.add_argument('--debug_nan', type=util.str2bool, nargs='?', const=True, default=False)

        parser.set_defaults(pool_size=0)
        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.train_epoch = None

        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G']

        if opt.lambda_HDCE > 0.0:
            self.loss_names.append('HDCE')
            if opt.dce_idt and self.isTrain:
                self.loss_names += ['HDCE_Y']

        if opt.lambda_SRC > 0.0:
            self.loss_names.append('SRC')

        if opt.lambda_phase > 0:
            self.loss_names.append('Phase')
        if opt.lambda_amp > 0:
            self.loss_names.append('Amp')

        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.alpha = opt.alpha
        if opt.dce_idt and self.isTrain:
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain,
                                      opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain,
                                      opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias,
                                          self.gpu_ids, opt)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            self.criterionHDCE = []
            for _ in self.nce_layers:
                self.criterionHDCE.append(PatchHDCELoss(opt=opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionR = []
            for _ in self.nce_layers:
                self.criterionR.append(SRC_Loss(opt).to(self.device))

        self.vgg_prior = None
        if (self.opt.use_prob_sampling or (self.opt.lambda_amp > 0)):
            if tv_models is None:
                raise ImportError("torchvision is required for VGG prior. Install torchvision or disable prior-dependent modules.")
            vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features
            self.vgg_prior = nn.Sequential(*list(vgg.children())[:16]).to(self.device).eval()
            for p in self.vgg_prior.parameters():
                p.requires_grad = False

        self._phase_w_cache = {}
        self._printed_input_range = False

    def data_dependent_initialize(self, data):
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        if self.opt.isTrain:
            self.compute_D_loss().backward()
            self.compute_G_loss().backward()
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_G_loss()

        if self.opt.debug_nan:
            if torch.is_tensor(self.loss_G) and (torch.isnan(self.loss_G).any() or torch.isinf(self.loss_G).any()):
                print("[WARN] loss_G NaN/Inf")
                for k in ["loss_G_GAN", "loss_HDCE", "loss_HDCE_Y", "loss_SRC", "loss_Phase", "loss_Amp"]:
                    if hasattr(self, k):
                        v = getattr(self, k)
                        try:
                            print(f"  {k}: {float(v)}")
                        except Exception:
                            print(f"  {k}: {v}")

        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.dce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.dce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        if (self.opt.use_prob_sampling or self.opt.lambda_amp > 0) and (not self._printed_input_range):
            mn = float(self.real_A.min().detach().cpu())
            mx = float(self.real_A.max().detach().cpu())
            print(f"[Info] real_A range: min={mn:.3f}, max={mx:.3f} (expect ~[-1,1] for most CUT-style pipelines)")
            self._printed_input_range = True

    def set_epoch(self, epoch):
        self.train_epoch = epoch

    def compute_D_loss(self):
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def _rgb01_for_vgg(self, x):
        x_min = x.amin()
        if x_min < -0.1:
            x01 = (x + 1.0) * 0.5
        else:
            x01 = x
        x01 = x01.clamp(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x01 - mean) / std

    def _gaussian_blur(self, x, sigma=1.5):
        if sigma <= 0:
            return x
        k = int(6 * sigma + 1)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        coords = torch.arange(k, device=x.device, dtype=torch.float32) - pad
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        g_h = g.view(1, 1, 1, k)
        g_v = g.view(1, 1, k, 1)
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        x = F.conv2d(x, g_h)
        x = F.conv2d(x, g_v)
        return x

    @torch.no_grad()
    def compute_Sx(self, x):
        if self.vgg_prior is None:
            raise RuntimeError("VGG prior not initialized.")
        xv = self._rgb01_for_vgg(x)
        feat = self.vgg_prior(xv)
        e = feat.abs().mean(dim=1, keepdim=True)
        e = self._gaussian_blur(e, sigma=self.opt.prior_sigma)
        mn = e.amin(dim=(2, 3), keepdim=True)
        mx = e.amax(dim=(2, 3), keepdim=True)
        S = (e - mn) / (mx - mn + 1e-8)
        return S.clamp(0, 1)

    def _sample_patch_ids_from_prob(self, prob_map, num_patches, rho=0.7, eps=1e-6):
        if prob_map.dim() != 4:
            raise ValueError(f"prob_map must be (B,1,H,W), got {prob_map.shape}")
        pm = prob_map.mean(dim=0, keepdim=False).squeeze(0)
        p = pm.reshape(-1).float()
        p = (p + eps) / (p.sum() + eps * p.numel())
        u = torch.full_like(p, 1.0 / p.numel())
        p_mix = (1.0 - rho) * u + rho * p
        p_mix = torch.clamp(p_mix, min=0)
        p_mix = p_mix / (p_mix.sum() + 1e-12)

        HW = p_mix.numel()
        n_sample = int(min(num_patches, HW))
        idx = torch.multinomial(p_mix, n_sample, replacement=False)
        return idx

    def _phase_inv_weight(self, H, W, device):
        key = (H, W, str(device))
        if key in self._phase_w_cache:
            return self._phase_w_cache[key]
        yy = torch.arange(H, device=device).view(H, 1).expand(H, W)
        xx = torch.arange(W, device=device).view(1, W).expand(H, W)
        cy, cx = H // 2, W // 2
        r = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
        w = 1.0 / (r + 1.0)
        w = w / (w.sum() + 1e-8)
        self._phase_w_cache[key] = w
        return w

    def compute_phase_loss_invfreq(self, x, y):
        eps = self.opt.phase_eps

        gx = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gy = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]

        Fx = torch.fft.fft2(gx, norm='ortho')
        Fy = torch.fft.fft2(gy, norm='ortho')

        mag_x = Fx.abs()
        mag_y = Fy.abs()

        thr = 10.0 * eps
        Ux = torch.where(mag_x > thr, Fx / (mag_x + eps), torch.zeros_like(Fx))
        Uy = torch.where(mag_y > thr, Fy / (mag_y + eps), torch.zeros_like(Fy))

        Ux = torch.fft.fftshift(Ux, dim=(-2, -1))
        Uy = torch.fft.fftshift(Uy, dim=(-2, -1))

        cos = (Ux.real * Uy.real + Ux.imag * Uy.imag).clamp(-1, 1)

        Hh, Ww = cos.shape[-2], cos.shape[-1]
        Winv = self._phase_inv_weight(Hh, Ww, x.device).view(1, 1, Hh, Ww)
        loss = (Winv * (1.0 - cos)).sum(dim=(1, 2, 3))
        return loss.mean()

    def _fft_amp(self, img, use_log=True, eps=1e-8):
        Fimg = torch.fft.fft2(img, norm='ortho')
        A = Fimg.abs()
        if use_log:
            A = torch.log1p(A)
        return A

    def _use_batch_mean_ref_now(self):
        if not self.opt.amp_batch_mean_ref:
            return False
        ratio = float(getattr(self.opt, "amp_mean_warmup_ratio", 0.0))
        if ratio <= 0:
            return True
        if self.train_epoch is None:
            return True
        total = float(getattr(self.opt, "n_epochs", 0) + getattr(self.opt, "n_epochs_decay", 0))
        if total <= 0:
            return True
        return float(self.train_epoch) <= ratio * total

    def compute_amp_loss_soft_fg_bg(self, real_A, fake_B, real_B):
        eps = 1e-8

        Sx = self.compute_Sx(real_A)
        Sy = self.compute_Sx(real_B)
        Mx = F.interpolate(Sx, size=real_A.shape[-2:], mode='bilinear', align_corners=False)
        My = F.interpolate(Sy, size=real_B.shape[-2:], mode='bilinear', align_corners=False)

        fake_fg = fake_B * Mx
        fake_bg = fake_B * (1.0 - Mx)
        src_bg = real_A * (1.0 - Mx)
        tgt_fg = real_B * My

        if self._use_batch_mean_ref_now():
            tgt_fg_ref = tgt_fg.mean(dim=0, keepdim=True).expand_as(tgt_fg)
        else:
            tgt_fg_ref = tgt_fg

        if self.opt.amp_area_norm:
            a_fg = (Mx.mean(dim=(2, 3), keepdim=True) + eps)
            a_bg = ((1.0 - Mx).mean(dim=(2, 3), keepdim=True) + eps)
            fake_fg = fake_fg / a_fg
            fake_bg = fake_bg / a_bg
            src_bg = src_bg / a_bg

            a_tfg = (My.mean(dim=(2, 3), keepdim=True) + eps)
            tgt_fg_ref = tgt_fg_ref / a_tfg

        Af_fg = self._fft_amp(fake_fg, use_log=self.opt.amp_use_log, eps=eps)
        At_fg = self._fft_amp(tgt_fg_ref, use_log=self.opt.amp_use_log, eps=eps)
        Af_bg = self._fft_amp(fake_bg, use_log=self.opt.amp_use_log, eps=eps)
        As_bg = self._fft_amp(src_bg, use_log=self.opt.amp_use_log, eps=eps)

        def cos_dist(a, b):
            a = a.flatten(1)
            b = b.flatten(1)
            a = a / (a.norm(dim=1, keepdim=True) + eps)
            b = b / (b.norm(dim=1, keepdim=True) + eps)
            return (1.0 - (a * b).sum(dim=1)).mean()

        return cos_dist(Af_fg, At_fg) + cos_dist(Af_bg, As_bg)

    def compute_G_loss(self):
        fake = self.fake_B

        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        fake_B_feat = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and getattr(self, "flipped_for_equivariance", False):
            fake_B_feat = [torch.flip(fq, [3]) for fq in fake_B_feat]
        real_A_feat = self.netG(self.real_A, self.nce_layers, encode_only=True)

        if self.opt.use_prob_sampling:
            Sx = self.compute_Sx(self.real_A)
            sample_ids = []
            for fk in real_A_feat:
                H, W = fk.shape[-2], fk.shape[-1]
                pm = F.interpolate(Sx, size=(H, W), mode='bilinear', align_corners=False)
                idx = self._sample_patch_ids_from_prob(pm, self.opt.num_patches,
                                                      rho=self.opt.prob_rho, eps=self.opt.prob_eps)
                sample_ids.append(idx)

            real_A_pool, _ = self.netF(real_A_feat, self.opt.num_patches, sample_ids)
            fake_B_pool, _ = self.netF(fake_B_feat, self.opt.num_patches, sample_ids)
        else:
            fake_B_pool, sample_ids = self.netF(fake_B_feat, self.opt.num_patches, None)
            real_A_pool, _ = self.netF(real_A_feat, self.opt.num_patches, sample_ids)

        if self.opt.dce_idt:
            idt_B_feat = self.netG(self.idt_B, self.nce_layers, encode_only=True)
            if self.opt.flip_equivariance and getattr(self, "flipped_for_equivariance", False):
                idt_B_feat = [torch.flip(fq, [3]) for fq in idt_B_feat]
            real_B_feat = self.netG(self.real_B, self.nce_layers, encode_only=True)
            idt_B_pool, _ = self.netF(idt_B_feat, self.opt.num_patches, sample_ids)
            real_B_pool, _ = self.netF(real_B_feat, self.opt.num_patches, sample_ids)

        self.loss_SRC, weight = self.calculate_R_loss(real_A_pool, fake_B_pool, epoch=self.train_epoch)

        if self.opt.lambda_HDCE > 0.0:
            self.loss_HDCE = self.calculate_HDCE_loss(real_A_pool, fake_B_pool, weight)
        else:
            self.loss_HDCE = 0.0

        self.loss_HDCE_Y = 0.0
        if self.opt.dce_idt and self.opt.lambda_HDCE > 0.0:
            _, weight_idt = self.calculate_R_loss(real_B_pool, idt_B_pool, only_weight=True, epoch=self.train_epoch)
            self.loss_HDCE_Y = self.calculate_HDCE_loss(real_B_pool, idt_B_pool, weight_idt)
            loss_HDCE_both = (self.loss_HDCE + self.loss_HDCE_Y) * 0.5
        else:
            loss_HDCE_both = self.loss_HDCE

        self.loss_Phase = 0.0
        if self.opt.lambda_phase > 0:
            self.loss_Phase = self.compute_phase_loss_invfreq(self.real_A, self.fake_B) * self.opt.lambda_phase

        self.loss_Amp = 0.0
        if self.opt.lambda_amp > 0:
            self.loss_Amp = self.compute_amp_loss_soft_fg_bg(self.real_A, self.fake_B, self.real_B) * self.opt.lambda_amp

        self.loss_G = self.loss_G_GAN + loss_HDCE_both + self.loss_SRC + self.loss_Phase + self.loss_Amp
        return self.loss_G

    def calculate_HDCE_loss(self, src, tgt, weight=None):
        n_layers = len(self.nce_layers)
        total_HDCE_loss = 0.0
        for f_q, f_k, crit, w in zip(tgt, src, self.criterionHDCE, weight):
            if self.opt.no_Hneg:
                w = None
            loss = crit(f_q, f_k, w) * self.opt.lambda_HDCE
            total_HDCE_loss += loss.mean()
        return total_HDCE_loss / n_layers

    def calculate_R_loss(self, src, tgt, only_weight=False, epoch=None):
        n_layers = len(self.nce_layers)
        total_SRC_loss = 0.0
        weights = []
        for f_q, f_k, crit in zip(tgt, src, self.criterionR):
            loss_SRC, weight = crit(f_q, f_k, only_weight, epoch)
            total_SRC_loss += loss_SRC * self.opt.lambda_SRC
            weights.append(weight)
        return total_SRC_loss / n_layers, weights

    def calculate_Patchloss(self, src, tgt, num_patch=4):
        feat_org = self.netG(src, mode='encoder')
        if self.opt.flip_equivariance and getattr(self, "flipped_for_equivariance", False):
            feat_org = torch.flip(feat_org, [3])

        N, C, H, W = feat_org.size()
        ps = H // num_patch
        lam = get_spa_lambda(self.alpha, size=(1, 1, num_patch ** 2), device=feat_org.device)
        feat_org_unfold = F.unfold(feat_org, kernel_size=(ps, ps), padding=0, stride=ps)

        rndperm = torch.randperm(feat_org_unfold.size(2))
        feat_prm = feat_org_unfold[:, :, rndperm]
        feat_mix = lam * feat_org_unfold + (1 - lam) * feat_prm
        feat_mix = F.fold(feat_mix, output_size=(H, W), kernel_size=(ps, ps), padding=0, stride=ps)

        out_mix = self.netG(feat_mix, mode='decoder')
        feat_mix_rec = self.netG(out_mix, mode='encoder')

        fake_feat = self.netG(tgt, mode='encoder')

        fake_feat_unfold = F.unfold(fake_feat, kernel_size=(ps, ps), padding=0, stride=ps)
        fake_feat_prm = fake_feat_unfold[:, :, rndperm]
        fake_feat_mix = lam * fake_feat_unfold + (1 - lam) * fake_feat_prm
        fake_feat_mix = F.fold(fake_feat_mix, output_size=(H, W), kernel_size=(ps, ps), padding=0, stride=ps)

        PM_loss = torch.mean(torch.abs(fake_feat_mix - feat_mix_rec))
        return 10 * PM_loss
