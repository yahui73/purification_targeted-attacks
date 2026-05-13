import torch
import sys

sys.path.append('/home/lyh/Codes/TA-DCH/stablediffpuri')
from stablediffpuri.utils1.transforms import raw_to_diff, diff_to_raw
import pytorch_ssim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import lpips


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Generate a 2D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    y = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = torch.meshgrid(x, y)
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / torch.sum(kernel)
    return kernel


def get_gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Generate a 4D Gaussian kernel suitable for conv2d."""
    kernel_2d = gaussian_kernel(kernel_size, sigma)
    kernel_3d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Make it 4D (1, 1, kernel_size, kernel_size)
    kernel_3d = kernel_3d.expand(channels, 1, kernel_size,
                                 kernel_size)  # Expand to (channels, 1, kernel_size, kernel_size)
    return kernel_3d


def gaussian_blur(input: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to a 4D tensor (batch of images)."""
    channels = input.shape[1]
    kernel = get_gaussian_kernel(kernel_size, sigma, channels).to(input.device)
    padding = kernel_size // 2
    return F.conv2d(input, kernel, padding=padding, groups=channels)

# # 预训练特征提取模型（例如 VGG16）
# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         self.features = models.vgg16(pretrained=True).features[:16]  # 提取前16层特征
#
#     def forward(self, x):
#         return self.features(x)


# # 计算特征相似度
# def feature_similarity(x_in, x_adv_t, model):
#     x_in_features = model(x_in)
#     x_adv_t_features = model(x_adv_t)
#     cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
#     return cos_sim(x_in_features, x_adv_t_features).mean()
#
#
# # 初始化特征提取器
# feature_extractor = FeatureExtractor().cuda()
# feature_extractor.eval()


# 简化后的特征提取模型，只提取前10层特征
class SimplifiedFeatureExtractor(nn.Module):
    def __init__(self):
        super(SimplifiedFeatureExtractor, self).__init__()
        self.features = models.vgg16(pretrained=True).features[:16]  # 提取前10层特征

    def forward(self, x):
        return self.features(x)


# 计算简化后的特征相似度
def simplified_feature_similarity(x_in, x_adv_t, model):
    x_in_features = model(x_in)
    x_adv_t_features = model(x_adv_t)
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6) # 解释清楚这一点
    return cos_sim(x_in_features, x_adv_t_features).mean()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化简化后的特征提取器
simplified_feature_extractor = SimplifiedFeatureExtractor().to(device)
simplified_feature_extractor.eval()

class LPIPSFeatureExtractor(nn.Module):
    def __init__(self):
        super(LPIPSFeatureExtractor, self).__init__()
        self.lpips_model = lpips.LPIPS(net='vgg')  # leave as trainable for autograd

    def forward(self, x, y):
        return self.lpips_model(x, y)

def lpips_similarity(x_in, x_adv_t, lpips_model):
    dist = lpips_model(x_in, x_adv_t)  # keep computation graph
    return dist.mean()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

lpips_extractor = LPIPSFeatureExtractor().to(device)
lpips_extractor.train()   # important for autograd

def pixel_cosine_similarity(x_in, x_adv_t):
    x_in_flat = x_in.view(x_in.size(0), -1)
    x_adv_flat = x_adv_t.view(x_adv_t.size(0), -1)
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos_sim(x_in_flat, x_adv_flat).mean()


# class PerceptualLoss(nn.Module):
#     def __init__(self, device):
#         super(PerceptualLoss, self).__init__()
#         vgg = models.vgg16(pretrained=True).features.to(device)
#         self.layers = ['0', '5', '10', '19', '28']
#         self.vgg = nn.Sequential(*[vgg[i] for i in range(max([int(l) for l in self.layers])+1)])
#         for param in self.vgg.parameters():
#             param.requires_grad = False
#         self.device = device
#
#     def forward(self, img1, img2):
#         img1 = img1.to(self.device)
#         img2 = img2.to(self.device)
#         img1_features = self.get_features(img1)
#         img2_features = self.get_features(img2)
#         perceptual_loss = sum([torch.mean((f1 - f2) ** 2) for f1, f2 in zip(img1_features, img2_features)])
#         return perceptual_loss
#
#     def get_features(self, x):
#         features = []
#         for name, layer in self.vgg._modules.items():
#             x = layer(x)
#             if name in self.layers:
#                 features.append(x)
#         return features
#
# class CombinedLoss(nn.Module):
#     def __init__(self, device):
#         super(CombinedLoss, self).__init__()
#         self.perceptual_loss_fn = PerceptualLoss(device)
#         self.ssim_weight = 0.5
#         self.perceptual_weight = 0.5
#         self.device = device
#
# def forward(self, img1, img2): perceptual_loss_value = self.perceptual_loss_fn(img1, img2) ssim_loss_value = 1 -
# pytorch_ssim.ssim(img1.to(self.device), img2.to(self.device))  # SSIM 越高越好，所以取 1-SSIM 作为损失 return
# self.perceptual_weight * perceptual_loss_value + self.ssim_weight * ssim_loss_value


# def mean_displacement_3d(tensor):
#     """
#     对输入张量应用 3D 均值位移滤波器。
#     参数：
#     - tensor (torch.Tensor): 输入张量，形状为 (N, C, D, H, W)
#     返回：
#     - torch.Tensor: 经过滤波的张量，形状与输入相同
#     """
#     # 获取输入张量的通道数
#     channels = tensor.shape[1]
#     # 创建一个用于均值位移滤波的核
#     kernel = torch.ones((channels, 1, 3, 3, 3), device=tensor.device) / 27.0
#     # 进行填充以保持张量的维度不变
#     padded_tensor = F.pad(tensor, (1, 1, 1, 1, 1, 1), mode='reflect')
#     # 应用卷积
#     filtered_tensor = F.conv3d(padded_tensor, kernel, stride=1, padding=0, groups=channels)
#     return filtered_tensor

# def mean_displacement_3d(tensor, kernel_size=5):
#     """
#     对输入张量应用 3D 均值位移滤波器。
#     参数：
#     - tensor (torch.Tensor): 输入张量，形状为 (N, C, D, H, W)
#     - kernel_size (int): 滤波核的大小，默认为 5
#     返回：
#     - torch.Tensor: 经过滤波的张量，形状与输入相同
#     """
#     N, C, D, H, W = tensor.shape
#
#     adjusted_kernel_size = min(kernel_size, D, H, W)
#     if adjusted_kernel_size < 1:
#         return tensor
#
#     kernel = torch.ones((C, 1, adjusted_kernel_size, adjusted_kernel_size, adjusted_kernel_size),
#                         device=tensor.device) / (adjusted_kernel_size ** 3)
#
#     padding = [adjusted_kernel_size // 2] * 3
#
#     padding[0] = min(padding[0], D - 1)
#     padding[1] = min(padding[1], H - 1)
#     padding[2] = min(padding[2], W - 1)
#
#     padded_tensor = F.pad(tensor, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]),
#                           mode='reflect')
#
#     filtered_tensor = F.conv3d(padded_tensor, kernel, stride=1, padding=0, groups=C)
#
#     return filtered_tensor

def purify_imagenet_first(x, diffusion, model, max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)
    x_adv = torch.nn.functional.interpolate(x_adv, size=[256, 256], mode="bilinear")  # transfrom size 224 -> 256
    x_adv = gaussian_blur(x_adv, kernel_size=3, sigma=0.5)

    t_steps = torch.ones(x_adv.shape[0], device=config.device.diff_device).long()
    t_steps = t_steps * (config.purification.purify_step - 1)
    shape = list(x_adv.shape)
    model_kwargs = {}

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)

            # x_adv_t = diffusion.q_sample(x_adv, t)
            x_adv_t = x_adv.requires_grad_(True)
            # scale = exp(config.purification.guide_exp_a * t /
            # config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base

            if config.purification.guide_mode == 'MSE':
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in, t,
                                                config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':
                # lyh-cond_fn(ILVR)
                # import sys
                # sys.path.append('..')
                # from ilvr_adm.resizer import Resizer as Resizer
                # batch_size = 1
                # down_N = 32
                # image_size = 256
                # shape = (batch_size, 3, image_size, image_size)
                # shape_d = (batch_size, 3, int(image_size / down_N), int(image_size / down_N))
                # down = Resizer(shape, 1 / down_N).to(next(model.parameters()).device)
                # up = Resizer(shape_d, down_N).to(next(model.parameters()).device)
                # resizers = (down, up)
                # if resizers is not None:
                #     x_in = up(down(x_in))
                #     x_adv_t = up(down(x_adv_t))

                # 3dmd 3x3x3
                # x_in_filtered = mean_displacement_3d(x_in.unsqueeze(0)).squeeze(0)
                # x_adv_t_filtered = mean_displacement_3d(x_adv_t.unsqueeze(0)).squeeze(0)

                # 3dmd 5x5x5
                # x_adv_t_filtered = mean_displacement_3d(x_adv_t.unsqueeze(2), kernel_size=5).squeeze(2)
                # x_in_filtered = mean_displacement_3d(x_in.unsqueeze(2), kernel_size=5).squeeze(2)

                # combined_loss
                # combined_loss_fn = CombinedLoss(config.device.diff_device)
                # selected = combined_loss_fn(x_in, x_adv_t)

                # gaussian
                # x_in = gaussian_blur(x_in, kernel_size=3, sigma=0.3)
                # x_adv_t = gaussian_blur(x_adv_t, kernel_size=3, sigma=0.3)

                # gaussian with guidedDDPM
                # x_1 = gaussian_blur(x_in, kernel_size=7, sigma=1.5)
                # x_2 = gaussian_blur(x_adv_t, kernel_size=7, sigma=1.5)
                # x_in = 0.5*x_in + 0.5*x_1
                # x_adv_t = 0.5*x_adv_t + 0.5*x_2
                # selected = pytorch_ssim.ssim(x_in, x_adv_t)

                # 计算基于SSIM的指导条件
                ssim_value = pytorch_ssim.ssim(x_in, x_adv_t)

                # 计算基于特征相似度的指导条件
                # feature_sim = feature_similarity(x_in, x_adv_t, feature_extractor)
                feature_sim = simplified_feature_similarity(x_in, x_adv_t, simplified_feature_extractor)

                # 基于lpips相似性的指导条件
                # lpips_sim = lpips_similarity(x_in, x_adv_t, lpips_extractor)

                # 结合SSIM和特征相似度
                selected = ssim_value * 0.5 + feature_sim * 0.5

                # selected = feature_sim # 仅使用vgg

                # selected = lpips_sim

                scale = diffusion.compute_scale(x_in, t,
                                                config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT':
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = config.purification.guide_scale
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):
            adv_sample = diffusion.q_sample(xt_reverse, t_steps)
            sample_fn = diffusion.p_sample_loop if not config.net.use_ddim else diffusion.ddim_sample_loop
            xt_reverse = sample_fn(
                model,
                shape,
                num_purifysteps=config.purification.purify_step,
                noise=adv_sample,
                clip_denoised=config.net.clip_denoised,
                cond_fn=cond_fn if config.purification.cond else None,
                model_kwargs=model_kwargs,
            )
            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            x_pur = torch.nn.functional.interpolate(x_pur, size=[224, 224],
                                                    mode="bilinear")  # transfrom size 256 -> 224
            images.append(x_pur)

    return images

def purify_imagenet_second(x, diffusion, model, max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)
    x_adv = torch.nn.functional.interpolate(x_adv, size=[256, 256], mode="bilinear")  # transfrom size 224 -> 256
    x_adv = gaussian_blur(x_adv, kernel_size=5, sigma=0.8)

    t_steps = torch.ones(x_adv.shape[0], device=config.device.diff_device).long()
    t_steps = t_steps * (config.purification.purify_step - 1)
    shape = list(x_adv.shape)
    model_kwargs = {}

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)

            # x_adv_t = diffusion.q_sample(x_adv, t)
            x_adv_t = x_adv.requires_grad_(True)
            # scale = exp(config.purification.guide_exp_a * t /
            # config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base

            if config.purification.guide_mode == 'MSE':
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in, t,
                                                config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':

                # 计算基于SSIM的指导条件
                ssim_value = pytorch_ssim.ssim(x_in, x_adv_t)

                # 计算基于特征相似度的指导条件
                # feature_sim = feature_similarity(x_in, x_adv_t, feature_extractor)
                feature_sim = simplified_feature_similarity(x_in, x_adv_t, simplified_feature_extractor)

                # 基于lpips相似性的指导条件
                # lpips_sim = lpips_similarity(x_in, x_adv_t, lpips_extractor)

                # 结合SSIM和特征相似度
                selected = ssim_value * 0.5 + feature_sim * 0.5

                # selected = feature_sim # 仅使用vgg

                # selected = lpips_sim

                scale = diffusion.compute_scale(x_in, t,
                                                config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT':
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = config.purification.guide_scale
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):
            adv_sample = diffusion.q_sample(xt_reverse, t_steps)
            sample_fn = diffusion.p_sample_loop if not config.net.use_ddim else diffusion.ddim_sample_loop
            xt_reverse = sample_fn(
                model,
                shape,
                num_purifysteps=config.purification.purify_step,
                noise=adv_sample,
                clip_denoised=config.net.clip_denoised,
                cond_fn=cond_fn if config.purification.cond else None,
                model_kwargs=model_kwargs,
            )
            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            x_pur = torch.nn.functional.interpolate(x_pur, size=[224, 224],
                                                    mode="bilinear")  # transfrom size 256 -> 224
            images.append(x_pur)

    return images

def purify_imagenet_third(x, diffusion, model, max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)
    x_adv = torch.nn.functional.interpolate(x_adv, size=[256, 256], mode="bilinear")  # transfrom size 224 -> 256
    # x_adv = gaussian_blur(x_adv, kernel_size=3, sigma=0.5)

    t_steps = torch.ones(x_adv.shape[0], device=config.device.diff_device).long()
    t_steps = t_steps * (config.purification.purify_step - 1)
    shape = list(x_adv.shape)
    model_kwargs = {}

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):
            adv_sample = diffusion.q_sample(xt_reverse, t_steps)
            sample_fn = diffusion.p_sample_loop if not config.net.use_ddim else diffusion.ddim_sample_loop
            xt_reverse = sample_fn(
                model,
                shape,
                num_purifysteps=config.purification.purify_step,
                noise=adv_sample,
                clip_denoised=config.net.clip_denoised,
                cond_fn=None,
                model_kwargs=model_kwargs,
            )
            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            x_pur = torch.nn.functional.interpolate(x_pur, size=[224, 224],
                                                    mode="bilinear")  # transfrom size 256 -> 224
            images.append(x_pur)

    return images

def purify_imagenet_forth(x, diffusion, model, max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)
    x_adv = torch.nn.functional.interpolate(x_adv, size=[256, 256], mode="bilinear")  # transfrom size 224 -> 256

    t_steps = torch.ones(x_adv.shape[0], device=config.device.diff_device).long()
    t_steps = t_steps * (config.purification.purify_step - 1)
    shape = list(x_adv.shape)
    model_kwargs = {}

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)

            # x_adv_t = diffusion.q_sample(x_adv, t)
            x_adv_t = x_adv.requires_grad_(True)
            # scale = exp(config.purification.guide_exp_a * t /
            # config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base

            if config.purification.guide_mode == 'MSE':
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in, t,
                                                config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':

                # 计算基于SSIM的指导条件
                ssim_value = pytorch_ssim.ssim(x_in, x_adv_t)

                selected = ssim_value # 仅使用ssim


                scale = diffusion.compute_scale(x_in, t,
                                                config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT':
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = config.purification.guide_scale
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):
            adv_sample = diffusion.q_sample(xt_reverse, t_steps)
            sample_fn = diffusion.p_sample_loop if not config.net.use_ddim else diffusion.ddim_sample_loop
            xt_reverse = sample_fn(
                model,
                shape,
                num_purifysteps=config.purification.purify_step,
                noise=adv_sample,
                clip_denoised=config.net.clip_denoised,
                cond_fn=cond_fn if config.purification.cond else None,
                model_kwargs=model_kwargs,
            )
            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            x_pur = torch.nn.functional.interpolate(x_pur, size=[224, 224],
                                                    mode="bilinear")  # transfrom size 256 -> 224
            images.append(x_pur)

    return images

def purify_imagenet(x, diffusion, model, max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)
    x_adv = torch.nn.functional.interpolate(x_adv, size=[256, 256], mode="bilinear")  # transfrom size 224 -> 256
    x_adv = gaussian_blur(x_adv, kernel_size=3, sigma=0.5)

    t_steps = torch.ones(x_adv.shape[0], device=config.device.diff_device).long()
    t_steps = t_steps * (config.purification.purify_step - 1)
    shape = list(x_adv.shape)
    model_kwargs = {}

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)

            # x_adv_t = diffusion.q_sample(x_adv, t)
            x_adv_t = x_adv.requires_grad_(True)
            # scale = exp(config.purification.guide_exp_a * t /
            # config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base

            if config.purification.guide_mode == 'MSE':
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in, t,
                                                config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':
                # lyh-cond_fn(ILVR)
                # import sys
                # sys.path.append('..')
                # from ilvr_adm.resizer import Resizer as Resizer
                # batch_size = 1
                # down_N = 32
                # image_size = 256
                # shape = (batch_size, 3, image_size, image_size)
                # shape_d = (batch_size, 3, int(image_size / down_N), int(image_size / down_N))
                # down = Resizer(shape, 1 / down_N).to(next(model.parameters()).device)
                # up = Resizer(shape_d, down_N).to(next(model.parameters()).device)
                # resizers = (down, up)
                # if resizers is not None:
                #     x_in = up(down(x_in))
                #     x_adv_t = up(down(x_adv_t))

                # 3dmd 3x3x3
                # x_in_filtered = mean_displacement_3d(x_in.unsqueeze(0)).squeeze(0)
                # x_adv_t_filtered = mean_displacement_3d(x_adv_t.unsqueeze(0)).squeeze(0)

                # 3dmd 5x5x5
                # x_adv_t_filtered = mean_displacement_3d(x_adv_t.unsqueeze(2), kernel_size=5).squeeze(2)
                # x_in_filtered = mean_displacement_3d(x_in.unsqueeze(2), kernel_size=5).squeeze(2)

                # combined_loss
                # combined_loss_fn = CombinedLoss(config.device.diff_device)
                # selected = combined_loss_fn(x_in, x_adv_t)

                # gaussian
                # x_in = gaussian_blur(x_in, kernel_size=3, sigma=0.3)
                # x_adv_t = gaussian_blur(x_adv_t, kernel_size=3, sigma=0.3)

                # gaussian with guidedDDPM
                # x_1 = gaussian_blur(x_in, kernel_size=7, sigma=1.5)
                # x_2 = gaussian_blur(x_adv_t, kernel_size=7, sigma=1.5)
                # x_in = 0.5*x_in + 0.5*x_1
                # x_adv_t = 0.5*x_adv_t + 0.5*x_2
                # selected = pytorch_ssim.ssim(x_in, x_adv_t)

                # 计算基于SSIM的指导条件
                # ssim_value = pytorch_ssim.ssim(x_in, x_adv_t)

                # 计算基于特征相似度的指导条件
                # feature_sim = feature_similarity(x_in, x_adv_t, feature_extractor)
                feature_sim = simplified_feature_similarity(x_in, x_adv_t, simplified_feature_extractor)

                # 基于lpips相似性的指导条件
                # lpips_sim = lpips_similarity(x_in, x_adv_t, lpips_extractor)

                # 结合SSIM和特征相似度
                # selected = ssim_value * 0.5 + feature_sim * 0.5

                selected = feature_sim # 仅使用vgg

                # selected = lpips_sim

                scale = diffusion.compute_scale(x_in, t,
                                                config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT':
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = config.purification.guide_scale
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):
            adv_sample = diffusion.q_sample(xt_reverse, t_steps)
            sample_fn = diffusion.p_sample_loop if not config.net.use_ddim else diffusion.ddim_sample_loop
            xt_reverse = sample_fn(
                model,
                shape,
                num_purifysteps=config.purification.purify_step,
                noise=adv_sample,
                clip_denoised=config.net.clip_denoised,
                cond_fn=cond_fn if config.purification.cond else None,
                model_kwargs=model_kwargs,
            )
            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            x_pur = torch.nn.functional.interpolate(x_pur, size=[224, 224],
                                                    mode="bilinear")  # transfrom size 256 -> 224
            images.append(x_pur)

    return images


def purify_imagenet_copy(x, diffusion, model, max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)
    x_adv = torch.nn.functional.interpolate(x_adv, size=[256, 256], mode="bilinear")  # transfrom size 224 -> 256
    x_adv = gaussian_blur(x_adv, kernel_size=3, sigma=0.5)

    t_steps = torch.ones(x_adv.shape[0], device=config.device.diff_device).long()
    t_steps = t_steps * (config.purification.purify_step - 1)
    shape = list(x_adv.shape)
    model_kwargs = {}

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)

            # x_adv_t = diffusion.q_sample(x_adv, t)
            x_adv_t = x_adv.requires_grad_(True)
            # scale = exp(config.purification.guide_exp_a * t /
            # config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base

            if config.purification.guide_mode == 'MSE':
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in, t,
                                                config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':
                # lyh-cond_fn(ILVR)
                # import sys
                # sys.path.append('..')
                # from ilvr_adm.resizer import Resizer as Resizer
                # batch_size = 1
                # down_N = 32
                # image_size = 256
                # shape = (batch_size, 3, image_size, image_size)
                # shape_d = (batch_size, 3, int(image_size / down_N), int(image_size / down_N))
                # down = Resizer(shape, 1 / down_N).to(next(model.parameters()).device)
                # up = Resizer(shape_d, down_N).to(next(model.parameters()).device)
                # resizers = (down, up)
                # if resizers is not None:
                #     x_in = up(down(x_in))
                #     x_adv_t = up(down(x_adv_t))

                # 3dmd 3x3x3
                # x_in_filtered = mean_displacement_3d(x_in.unsqueeze(0)).squeeze(0)
                # x_adv_t_filtered = mean_displacement_3d(x_adv_t.unsqueeze(0)).squeeze(0)

                # 3dmd 5x5x5
                # x_adv_t_filtered = mean_displacement_3d(x_adv_t.unsqueeze(2), kernel_size=5).squeeze(2)
                # x_in_filtered = mean_displacement_3d(x_in.unsqueeze(2), kernel_size=5).squeeze(2)

                # combined_loss
                # combined_loss_fn = CombinedLoss(config.device.diff_device)
                # selected = combined_loss_fn(x_in, x_adv_t)

                # gaussian
                # x_in = gaussian_blur(x_in, kernel_size=3, sigma=0.3)
                # x_adv_t = gaussian_blur(x_adv_t, kernel_size=3, sigma=0.3)

                # gaussian with guidedDDPM
                # x_1 = gaussian_blur(x_in, kernel_size=7, sigma=1.5)
                # x_2 = gaussian_blur(x_adv_t, kernel_size=7, sigma=1.5)
                # x_in = 0.5*x_in + 0.5*x_1
                # x_adv_t = 0.5*x_adv_t + 0.5*x_2
                # selected = pytorch_ssim.ssim(x_in, x_adv_t)

                # 计算基于SSIM的指导条件
                ssim_value = pytorch_ssim.ssim(x_in, x_adv_t)

                # 计算基于特征相似度的指导条件
                # feature_sim = feature_similarity(x_in, x_adv_t, feature_extractor)
                # feature_sim = simplified_feature_similarity(x_in, x_adv_t, simplified_feature_extractor)

                # 像素级别的cosine计算
                # pix_cos_sim = pixel_cosine_similarity(x_in, x_adv_t)

                # 结合SSIM和特征相似度
                # selected = ssim_value * 0.5 + feature_sim * 0.5

                # selected = feature_sim # 仅使用vgg

                selected = ssim_value # 仅使用ssim

                # selected = pix_cos_sim # 仅使用像素级别cosine

                scale = diffusion.compute_scale(x_in, t,
                                                config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT':
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = config.purification.guide_scale
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):
            adv_sample = diffusion.q_sample(xt_reverse, t_steps)
            sample_fn = diffusion.p_sample_loop if not config.net.use_ddim else diffusion.ddim_sample_loop
            xt_reverse = sample_fn(
                model,
                shape,
                num_purifysteps=config.purification.purify_step,
                noise=adv_sample,
                clip_denoised=config.net.clip_denoised,
                cond_fn=cond_fn if config.purification.cond else None,
                model_kwargs=model_kwargs,
            )
            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            x_pur = torch.nn.functional.interpolate(x_pur, size=[224, 224],
                                                    mode="bilinear")  # transfrom size 256 -> 224
            images.append(x_pur)

    return images
