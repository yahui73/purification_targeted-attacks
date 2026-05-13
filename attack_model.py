import os
import sys
import importlib
import yaml
import numpy as np
import scipy.io as scio
from torchvision import transforms
import torchvision.models as tv_models
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

import stablediffpuri.main
from model import PrototypeNet, Generator, Discriminator, GANLoss, get_scheduler
from utils import set_input_images, CalcSim, log_trick, CalcMap, mkdir_p
from attacked_model import Attacked_Model


class DCHTA(nn.Module):
    def __init__(self, args, Dcfg):
        super(DCHTA, self).__init__()
        self.bit = args.bit
        self.num_classes = Dcfg.num_label
        self.dim_text = Dcfg.tag_dim
        self.batch_size = args.batch_size
        self.model_name = '{}_{}_{}'.format(args.dataset, args.attacked_method, args.bit)
        self.args = args
        self._build_model(args, Dcfg)
        self._save_setting(args)
        if self.args.transfer_attack:
            self.transfer_bit = args.transfer_bit
            self.transfer_model = Attacked_Model(args.transfer_attacked_method, args.dataset, args.transfer_bit,
                                                 args.attacked_models_path, args.dataset_path)
            self.transfer_model.eval()

    def _ensure_transfer_model(self):
        if not hasattr(self, 'transfer_model'):
            self.transfer_bit = self.args.transfer_bit
            self.transfer_model = Attacked_Model(self.args.transfer_attacked_method, self.args.dataset,
                                                 self.args.transfer_bit, self.args.attacked_models_path,
                                                 self.args.dataset_path)
            self.transfer_model.eval()

    def _build_model(self, args, Dcfg):
        pretrain_model = scio.loadmat(Dcfg.vgg_path)
        self.prototypenet = nn.DataParallel(PrototypeNet(self.dim_text, self.bit, self.num_classes)).cuda()
        self.generator = nn.DataParallel(Generator()).cuda()
        self.discriminator = nn.DataParallel(Discriminator(self.num_classes)).cuda()
        self.criterionGAN = GANLoss('lsgan').cuda()
        self.attacked_model = Attacked_Model(args.attacked_method, args.dataset, args.bit, args.attacked_models_path,
                                             args.dataset_path)
        self.attacked_model.eval()

    def _save_setting(self, args):
        self.output_dir = os.path.join(args.output_path, args.output_dir)
        self.model_dir = os.path.join(self.output_dir, 'Model')
        self.image_dir = os.path.join(self.output_dir, 'Image')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)

    def sample(self, image, sample_dir, name):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        image = image.cpu().detach()[0]
        image = transforms.ToPILImage()(image)
        image.convert(mode='RGB').save(os.path.join(sample_dir, name + '.png'), quality=100)

    def _load_impure_model(self):
        if hasattr(self, 'impure_model'):
            return self.impure_model

        project_root = getattr(self.args, 'project_root', '/home/lyh/Desktop/TA-DCH')
        impure_root = getattr(self.args, 'impure_root', os.path.join(project_root, 'IMPure-main'))
        impure_root = os.path.abspath(impure_root)
        if not os.path.isdir(impure_root):
            raise FileNotFoundError('IMPure root does not exist: {}'.format(impure_root))

        old_path = list(sys.path)
        sys.path.insert(0, impure_root)
        try:
            import recon_model
            model_name = getattr(self.args, 'impure_model', 'rcmmff_idmae_vit_base_patch16_dec512d8b')
            model = getattr(recon_model, model_name)(
                use_conv=True,
                pre_norm=True,
                chunks=getattr(self.args, 'impure_chunks', 4),
                feature_mask=getattr(self.args, 'impure_feature_mask', 'noise'),
                img_mask=getattr(self.args, 'impure_img_mask', 'noise'),
            ).cuda()
        finally:
            sys.path = old_path

        ckpt_path = getattr(self.args, 'impure_ckpt', '')
        if not ckpt_path:
            ckpt_path = os.path.join(impure_root, 'checkpoints', 'impure.pth')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(impure_root, 'checkpoints', 'mipure.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError('IMPure checkpoint does not exist: {}'.format(ckpt_path))

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        del_layers = ('pos_embed', 'decoder_pos_embed', 'hgds.')
        state_dict = {
            key: value for key, value in state_dict.items()
            if not any(key.startswith(layer) for layer in del_layers)
        }
        model.load_state_dict(state_dict, strict=False)
        model.eval().requires_grad_(False)
        self.impure_model = model
        print('IMPure checkpoint:', ckpt_path)
        return self.impure_model

    def _impure_purify(self, images):
        impure_model = self._load_impure_model()
        with torch.no_grad():
            purified = impure_model(images)[1]
            purified = impure_model.unnormalize(purified)
        return torch.clamp(purified, 0, 1)

    def _impure_purify_grad(self, images):
        impure_model = self._load_impure_model()
        purified = impure_model(images)[1]
        purified = impure_model.unnormalize(purified)
        return torch.clamp(purified, 0, 1)

    def _gaussian_kernel(self, kernel_size, sigma, channels, device):
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    def _gaussian_blur(self, images, kernel_size=3, sigma=0.5):
        padding = kernel_size // 2
        kernel = self._gaussian_kernel(kernel_size, sigma, images.size(1), images.device)
        return F.conv2d(images, kernel, padding=padding, groups=images.size(1))

    def _ssim_value(self, x, y, window_size=11):
        channels = x.size(1)
        window = self._gaussian_kernel(window_size, 1.5, channels, x.device)
        mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
        mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x_sq
        sigma_y = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2))
        return ssim_map.mean()

    def _dadh_image_code_for_guidance(self, images):
        if self.args.attacked_method != 'DADH':
            raise NotImplementedError('IMPure hash guidance currently supports DADH only')
        generator = self.attacked_model.generator
        feature = generator.cnn_f(images * 255)
        feature = feature.squeeze()
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)
        image_feature = generator.image_module(feature)
        return generator.hash_module['image'](image_feature).reshape(-1, generator.output_dim)

    def _impure_guided_input(self, images, clean_images):
        steps = getattr(self.args, 'impure_guidance_steps', 1)
        step_size = getattr(self.args, 'impure_guidance_step_size', 0.25 / 255.0)
        kernel_size = getattr(self.args, 'impure_gaussian_kernel', 3)
        sigma = getattr(self.args, 'impure_gaussian_sigma', 0.3)

        reference = images.detach()
        clean_reference = clean_images.detach()
        denoised = torch.clamp(self._gaussian_blur(reference, kernel_size=kernel_size, sigma=sigma), 0, 1).detach()
        with torch.no_grad():
            clean_code = self._dadh_image_code_for_guidance(clean_reference)

        guided = denoised.clone().detach().requires_grad_(True)
        for _ in range(steps):
            purified = self._impure_purify_grad(guided)
            guided_code = self._dadh_image_code_for_guidance(purified)
            loss = F.l1_loss(guided_code, clean_code)
            grad = torch.autograd.grad(loss, guided)[0]
            guided = torch.clamp(guided - step_size * grad.sign(), 0, 1).detach().requires_grad_(True)
        return guided.detach()

    def _impure_test_common(self, database_images, database_texts, database_labels,
                            test_images, test_texts, test_labels, guided=False):
        self.load_prototypenet()
        self.load_generator()
        self._load_impure_model()

        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        method_name = 'impure_guided' if guided else 'impure'
        print('start generate target images and {} purification...'.format(method_name))

        for i in range(num_test):
            print('{}-test-count:'.format(method_name), i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2

            if guided:
                fake_image = self._impure_guided_input(fake_image, original_image.unsqueeze(0))
            purified_image = self._impure_purify(fake_image)
            target_image = 255 * purified_image

            target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, purified_image[0]).data

        print('generate {} images end!'.format(method_name))
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def Impure_test(self, database_images, database_texts, database_labels,
                    test_images, test_texts, test_labels):
        print('This is IMPure original purification test!')
        self._impure_test_common(database_images, database_texts, database_labels,
                                 test_images, test_texts, test_labels, guided=False)

    def Impure_guided_test(self, database_images, database_texts, database_labels,
                           test_images, test_texts, test_labels):
        print('This is IMPure purification test with Gaussian + SSIM/VGG guidance!')
        self._impure_test_common(database_images, database_texts, database_labels,
                                 test_images, test_texts, test_labels, guided=True)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()
        self.args.lr = self.optimizers[0].param_groups[0]['lr']

    # lyh
    def test_attacked_model(self, Te_I, Te_T, Te_L, Db_I, Db_T, Db_L):
        print('This is test_attacked_model function by using clean data!')
        IqB = self.attacked_model.generate_image_hashcode(Te_I)
        TqB = self.attacked_model.generate_text_hashcode(Te_T)
        IdB = self.attacked_model.generate_image_hashcode(Db_I)
        TdB = self.attacked_model.generate_text_hashcode(Db_T)
        I2T_map = CalcMap(IqB, TdB, Te_L, Db_L, 50)
        T2I_map = CalcMap(TqB, IdB, Te_L, Db_L, 50)
        I2I_map = CalcMap(IqB, IdB, Te_L, Db_L, 50)
        T2T_map = CalcMap(TqB, TdB, Te_L, Db_L, 50)
        print('I2T@50: {:.4f}'.format(I2T_map))
        print('T2I@50: {:.4f}'.format(T2I_map))
        print('I2I@50: {:.4f}'.format(I2I_map))
        print('T2T@50: {:.4f}'.format(T2T_map))
        print('Test of clean data is ending!')

    # lyh
    def tmap_attacked_model(self, Te_I, Te_T, Te_L, Db_I, Db_T, Db_L):
        print('This is test_attacked_model function by using clean data!')
        IqB = self.attacked_model.generate_image_hashcode(Te_I)
        IdB = self.attacked_model.generate_image_hashcode(Db_I)
        TdB = self.attacked_model.generate_text_hashcode(Db_T)
        select_index = np.random.choice(range(Db_L.size(0)), size=Te_L.size(0))
        target_labels = Db_L.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        I2T_map = CalcMap(IqB, TdB, target_labels.cpu(), Db_L.float(), 50)
        I2I_map = CalcMap(IqB, IdB, target_labels.cpu(), Db_L.float(), 50)
        print('T-Map of I2T@50: {:.4f}'.format(I2T_map))
        print('T-Map of I2I@50: {:.4f}'.format(I2I_map))
        print('T-Map of clean data is ending!')

    def save_prototypenet(self):
        torch.save(self.prototypenet.module.state_dict(),
                   os.path.join(self.model_dir, 'prototypenet_{}.pth'.format(self.model_name)))

    def save_generator(self):
        torch.save(self.generator.module.state_dict(),
                   os.path.join(self.model_dir, 'generator_{}.pth'.format(self.model_name)))

    def load_generator(self):
        self.generator.module.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'generator_{}.pth'.format(self.model_name))))
        self.generator.eval()

    def load_prototypenet(self):
        self.prototypenet.module.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'prototypenet_{}.pth'.format(self.model_name))))
        self.prototypenet.eval()

    def train_prototypenet(self, train_images, train_texts, train_labels, epochs=50, logloss_weight=5):
        num_train = train_labels.size(0)
        optimizer_a = torch.optim.Adam(self.prototypenet.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        batch_size = 64
        steps = num_train // batch_size + 1
        lr_steps = epochs * steps
        scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_a, milestones=[lr_steps / 2, lr_steps * 3 / 4],
                                                           gamma=0.1)
        criterion_l2 = torch.nn.MSELoss()
        # Depends on the attacked model
        B = self.attacked_model.generate_image_hashcode(train_images).cuda()
        # B = self.attacked_model.generate_text_hashcode(train_texts).cuda()
        for epoch in range(epochs):
            index = np.random.permutation(num_train)
            for i in range(steps):
                end_index = min((i + 1) * batch_size, num_train)
                num_index = end_index - i * batch_size
                ind = index[i * batch_size: end_index]
                batch_text = Variable(train_texts[ind]).type(torch.float).cuda()
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                optimizer_a.zero_grad()
                _, mixed_h, mixed_l = self.prototypenet(batch_label, batch_text)
                S = CalcSim(batch_label.cpu(), train_labels.type(torch.float))
                theta_m = mixed_h.mm(Variable(B).t()) / 2
                logloss_m = - ((Variable(S.cuda()) * theta_m - log_trick(theta_m)).sum() / (num_train * num_index))
                regterm_m = (torch.sign(mixed_h) - mixed_h).pow(2).sum() / num_index
                classifer_m = criterion_l2(mixed_l, batch_label)
                loss = classifer_m + logloss_weight * logloss_m + 1e-3 * regterm_m
                loss.backward()
                optimizer_a.step()
                if i % self.args.print_freq == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, l_m:{:.5f}, r_m: {:.5f}, c_m: {:.7f}'
                          .format(epoch, i, scheduler_a.get_last_lr()[0], logloss_m, regterm_m, classifer_m))
                scheduler_a.step()
        self.save_prototypenet()

    def train_prototypenet_16(self, train_images, train_texts, train_labels):
        print('This is train_prototypenet_16 function!')
        self.train_prototypenet(train_images, train_texts, train_labels, epochs=100, logloss_weight=10)

    def test_prototypenet(self, test_texts, test_labels, database_images, database_texts, database_labels):
        self.load_prototypenet()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        for i in range(num_test):
            _, mixed_h, __ = self.prototypenet(test_labels[i].cuda().float().unsqueeze(0),
                                               test_texts[i].cuda().float().unsqueeze(0))
            qB[i, :] = torch.sign(mixed_h.cpu().data)[0]
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        c2i_map = CalcMap(qB, IdB, test_labels, database_labels, 50)
        c2t_map = CalcMap(qB, TdB, test_labels, database_labels, 50)
        print('C2I_MAP: %3.5f, C2T_MAP: %3.5f' % (c2i_map, c2t_map))

    def train(self, train_images, train_texts, train_labels, database_images, database_texts, database_labels,
              test_texts, test_labels):
        print('This is train function!')
        prototypenet_path = os.path.join(self.model_dir, 'prototypenet_{}.pth'.format(self.model_name))
        if os.path.exists(prototypenet_path):
            print('Stage I prototypenet exists, skip Prototype Learning:', prototypenet_path)
            self.load_prototypenet()
        else:
            # Stage I: Prototype Learning
            self.train_prototypenet(train_images, train_texts, train_labels)
            self.test_prototypenet(test_texts, test_labels, database_images, database_texts, database_labels)
        # Stage II: Adversarial Generation
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizers = [optimizer_g, optimizer_d]
        self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]
        num_train = train_labels.size(0)
        batch_size = self.batch_size
        total_epochs = self.args.n_epochs + self.args.n_epochs_decay + 1
        print('The total epochs is:', total_epochs)
        criterion_l2 = torch.nn.MSELoss()
        B = self.attacked_model.generate_text_feature(train_texts)
        B = B.cuda()
        for epoch in range(self.args.epoch_count, total_epochs):
            print('\nTrain epoch: {}, learning rate: {:.7f}'.format(epoch, self.args.lr))
            index = np.random.permutation(num_train)
            for i in range(num_train // batch_size + 1):
                end_index = min((i + 1) * batch_size, num_train)
                num_index = end_index - i * batch_size
                ind = index[i * batch_size: end_index]
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                batch_text = Variable(train_texts[ind]).type(torch.float).cuda()
                batch_image = Variable(train_images[ind]).type(torch.float).cuda()
                batch_image = set_input_images(batch_image / 255)
                select_index = np.random.choice(range(train_labels.size(0)), size=num_index)
                batch_target_label = train_labels.index_select(0, torch.from_numpy(select_index)).type(
                    torch.float).cuda()
                batch_target_text = train_texts.index_select(0, torch.from_numpy(select_index)).type(torch.float).cuda()
                label_feature, target_hashcode, _ = self.prototypenet(batch_target_label, batch_target_text)
                batch_fake_image = self.generator(batch_image, label_feature.detach())
                # update D
                if i % 3 == 0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_d.zero_grad()
                    batch_image_d = self.discriminator(batch_image)
                    batch_fake_image_d = self.discriminator(batch_fake_image.detach())
                    real_d_loss = self.criterionGAN(batch_image_d, batch_label, True)
                    fake_d_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, False)
                    d_loss = (real_d_loss + fake_d_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()
                # update G
                self.set_requires_grad(self.discriminator, False)
                optimizer_g.zero_grad()
                batch_fake_image_m = (batch_fake_image + 1) / 2 * 255
                predicted_target_hash = self.attacked_model.image_model(batch_fake_image_m)
                logloss = - torch.mean(predicted_target_hash * target_hashcode) + 1
                batch_fake_image_d = self.discriminator(batch_fake_image)
                fake_g_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, True)
                reconstruction_loss_l = criterion_l2(batch_fake_image, batch_image)
                # backpropagation
                # Original setting:
                # g_loss = 5 * logloss + 1 * fake_g_loss + 150 * reconstruction_loss_l
                if self.bit == 16:
                    g_loss = 5 * logloss + 1 * fake_g_loss + 100 * reconstruction_loss_l
                elif self.bit == 64:
                    g_loss = 5 * logloss + 1 * fake_g_loss + 50 * reconstruction_loss_l # previous 5 1 150
                elif self.bit == 128:
                    g_loss = 5 * logloss + 1 * fake_g_loss + 30 * reconstruction_loss_l
                else:
                    g_loss = 5 * logloss + 1 * fake_g_loss + 1 * reconstruction_loss_l
                g_loss.backward()
                optimizer_g.step()
                if i % self.args.sample_freq == 0:
                    self.sample((batch_fake_image + 1) / 2, '{}/'.format(self.image_dir),
                                str(epoch) + '_' + str(i) + '_fake')
                    self.sample((batch_image + 1) / 2, '{}/'.format(self.image_dir),
                                str(epoch) + '_' + str(i) + '_real')
                if i % self.args.print_freq == 0:
                    print(
                        'step: {:3d} d_loss: {:.3f} g_loss: {:.3f} fake_g_loss: {:.3f} logloss: {:.3f} r_loss_l: {:.7f}'
                        .format(i, d_loss, g_loss, fake_g_loss, logloss, reconstruction_loss_l))
            self.update_learning_rate()
        self.save_generator()
        print('The process of train is end!')

    def train_16(self, train_images, train_texts, train_labels, database_images, database_texts, database_labels,
                 test_texts, test_labels):
        print('This is train_16 function!')
        if self.bit != 16:
            raise ValueError('train_16 is only for 16-bit, but current bit is {}'.format(self.bit))
        self.train_prototypenet_16(train_images, train_texts, train_labels)
        self.test_prototypenet(test_texts, test_labels, database_images, database_texts, database_labels)

        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizers = [optimizer_g, optimizer_d]
        self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]
        num_train = train_labels.size(0)
        batch_size = self.batch_size
        total_epochs = self.args.n_epochs + self.args.n_epochs_decay + 1
        print('The total epochs is:', total_epochs)
        criterion_l2 = torch.nn.MSELoss()
        B = self.attacked_model.generate_text_feature(train_texts)
        B = B.cuda()
        for epoch in range(self.args.epoch_count, total_epochs):
            print('\nTrain epoch: {}, learning rate: {:.7f}'.format(epoch, self.args.lr))
            index = np.random.permutation(num_train)
            for i in range(num_train // batch_size + 1):
                end_index = min((i + 1) * batch_size, num_train)
                num_index = end_index - i * batch_size
                ind = index[i * batch_size: end_index]
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                batch_text = Variable(train_texts[ind]).type(torch.float).cuda()
                batch_image = Variable(train_images[ind]).type(torch.float).cuda()
                batch_image = set_input_images(batch_image / 255)
                select_index = np.random.choice(range(train_labels.size(0)), size=num_index)
                batch_target_label = train_labels.index_select(0, torch.from_numpy(select_index)).type(
                    torch.float).cuda()
                batch_target_text = train_texts.index_select(0, torch.from_numpy(select_index)).type(torch.float).cuda()
                label_feature, target_hashcode, _ = self.prototypenet(batch_target_label, batch_target_text)
                batch_fake_image = self.generator(batch_image, label_feature.detach())
                if i % 3 == 0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_d.zero_grad()
                    batch_image_d = self.discriminator(batch_image)
                    batch_fake_image_d = self.discriminator(batch_fake_image.detach())
                    real_d_loss = self.criterionGAN(batch_image_d, batch_label, True)
                    fake_d_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, False)
                    d_loss = (real_d_loss + fake_d_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()
                self.set_requires_grad(self.discriminator, False)
                optimizer_g.zero_grad()
                batch_fake_image_m = (batch_fake_image + 1) / 2 * 255
                predicted_target_hash = self.attacked_model.image_model(batch_fake_image_m)
                logloss = - torch.mean(predicted_target_hash * target_hashcode) + 1
                batch_fake_image_d = self.discriminator(batch_fake_image)
                fake_g_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, True)
                reconstruction_loss_l = criterion_l2(batch_fake_image, batch_image)
                g_loss = 5 * logloss + 1 * fake_g_loss + 100 * reconstruction_loss_l
                g_loss.backward()
                optimizer_g.step()
                if i % self.args.sample_freq == 0:
                    self.sample((batch_fake_image + 1) / 2, '{}/'.format(self.image_dir),
                                str(epoch) + '_' + str(i) + '_fake')
                    self.sample((batch_image + 1) / 2, '{}/'.format(self.image_dir),
                                str(epoch) + '_' + str(i) + '_real')
                if i % self.args.print_freq == 0:
                    print(
                        'step: {:3d} d_loss: {:.3f} g_loss: {:.3f} fake_g_loss: {:.3f} logloss: {:.3f} r_loss_l: {:.7f}'
                        .format(i, d_loss, g_loss, fake_g_loss, logloss, reconstruction_loss_l))
            self.update_learning_rate()
        self.save_generator()
        print('The process of train_16 is end!')

    def test(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        print('This is test function!')
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            print('test-count:', i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            target_image = 255 * fake_image
            target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, fake_image[0]).data
        print('generate target images end!')
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    # lyh
    def puri_test(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        print("This is puri-test function!")
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            print('puri-test-count:', i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            # target_image = 255 * fake_image  # before lyh
            target_image = fake_image  # lyh
            # print('target_image size:', target_image.shape)
            # target_image = original_image.unsqueeze(0)  # lyh

            # begin purification
            from stablediffpuri.purification.purify_imagenet import purify_imagenet_first
            from stablediffpuri.guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                classifier_defaults,
                create_model_and_diffusion,
                create_classifier,
                add_dict_to_argparser,
                args_to_dict,
            )
            args, config = stablediffpuri.main.parse_args_and_config()

            model, diffusion = create_model_and_diffusion(
                **args_to_dict(config.net, model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load(config.net.model_path, map_location="cpu")
            )
            model.to(config.device.clf_device)
            if config.net.use_fp16:
                model.convert_to_fp16()
            model.eval()
            # print(target_image.shape)
            target_image = purify_imagenet_first(target_image, diffusion, model, max_iter=1, mode="purification",
                                           config=config)
            target_image = target_image[0]
            target_image = 255 * target_image
            # end purification

            target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, target_image / 255).data
        print('generate target images end!')
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def _load_purification_master_forward(self):
        purification_root = getattr(self.args, 'purification_master_root',
                                    '/home/lyh/Desktop/TA-DCH/Purification-master')
        purification_root = os.path.abspath(purification_root)
        if not os.path.isdir(purification_root):
            raise FileNotFoundError('Purification-master root does not exist: {}'.format(purification_root))

        old_path = list(sys.path)
        old_utils = sys.modules.get('utils')
        old_purification = sys.modules.get('purification')
        sys.path.insert(0, purification_root)
        sys.modules.pop('utils', None)
        sys.modules.pop('purification', None)
        try:
            from purification import PurificationForward
            from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
            import torchvision.models as tv_models
            try:
                from torchvision.models import ResNet50_Weights
                resnet_weights = ResNet50_Weights.DEFAULT
            except Exception:
                resnet_weights = None
        finally:
            sys.path = old_path
            if old_utils is not None:
                sys.modules['utils'] = old_utils
            if old_purification is not None:
                sys.modules['purification'] = old_purification
            else:
                sys.modules.pop('purification', None)

        class ImageOnlyPurificationForward(PurificationForward):
            def purify(self, x):
                if self.is_imagenet:
                    x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
                x_diff = torch.clamp((x - 0.5) * 2, -1, 1)
                x_diff = self.denoise(x_diff)
                if self.is_imagenet:
                    x_diff = F.interpolate(x_diff, size=(224, 224), mode='bilinear', align_corners=False)
                return torch.clamp((x_diff / 2) + 0.5, 0, 1)

        config_path = os.path.join(purification_root, 'diffusion_configs', 'imagenet.yml')
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        model_config = model_and_diffusion_defaults()
        model_config.update(config['model'])
        diffusion, _ = create_model_and_diffusion(**model_config)

        ckpt_path = getattr(self.args, 'purification_master_ckpt', '')
        if not ckpt_path:
            ckpt_path = os.path.join(purification_root, 'pretrained', 'guided_diffusion',
                                     '256x256_diffusion_uncond.pt')
        if not os.path.exists(ckpt_path):
            ckpt_path = '/home/lyh/Desktop/TA-DCH/stablediffpuri/models/256x256_diffusion_uncond.pt'
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError('Cannot find 256x256_diffusion_uncond.pt for Purification-master')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        diffusion.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        diffusion.eval().to(device)

        clf = tv_models.resnet50(weights=resnet_weights).to(device).eval()

        purifier = ImageOnlyPurificationForward(
            clf=clf,
            diffusion=diffusion,
            strength_a=getattr(self.args, 'purification_strength_l', 0.4),
            strength_b=getattr(self.args, 'purification_strength_s', 0.2),
            classifier_name='ResNet50',
            is_imagenet=True,
            threshold=0.9,
            threshold_percent=getattr(self.args, 'purification_threshold_percent', 0.15),
            ddim_steps=getattr(self.args, 'purification_defense_ddim_steps', 100),
            forward_noise_steps=getattr(self.args, 'purification_forward_noise_steps', 3),
            device=device,
        )
        purifier.eval().requires_grad_(False)
        print('Purification-master checkpoint:', ckpt_path)
        return purifier

    def purification_test(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        print("This is purification_test function by using Purification-master!")
        self.load_prototypenet()
        self.load_generator()
        purifier = self._load_purification_master_forward()

        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images and Purification-master purification...')
        for i in range(num_test):
            print('purification-test-count:', i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2

            with torch.no_grad():
                target_image = purifier.purify(fake_image)
            target_image = 255 * target_image

            target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, target_image[0] / 255).data
        print('generate target images end!')
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def puri_test_second(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        print("This is puri-test function!")
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            print('puri-test-count:', i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            # target_image = 255 * fake_image  # before lyh
            target_image = fake_image  # lyh
            # target_image = original_image.unsqueeze(0)  # lyh

            # begin purification
            from stablediffpuri.purification.purify_imagenet import purify_imagenet_second
            from stablediffpuri.guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                classifier_defaults,
                create_model_and_diffusion,
                create_classifier,
                add_dict_to_argparser,
                args_to_dict,
            )
            args, config = stablediffpuri.main.parse_args_and_config()

            model, diffusion = create_model_and_diffusion(
                **args_to_dict(config.net, model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load(config.net.model_path, map_location="cpu")
            )
            model.to(config.device.clf_device)
            if config.net.use_fp16:
                model.convert_to_fp16()
            model.eval()
            target_image = purify_imagenet_second(target_image, diffusion, model, max_iter=1, mode="purification",
                                           config=config)
            target_image = target_image[0]
            target_image = 255 * target_image
            # end purification

            target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, target_image / 255).data
        print('generate target images end!')
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def puri_test_third(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        print("This is puri-test function!")
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            print('puri-test-count:', i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            # target_image = 255 * fake_image  # before lyh
            target_image = fake_image  # lyh
            # target_image = original_image.unsqueeze(0)  # lyh

            # begin purification
            from stablediffpuri.purification.purify_imagenet import purify_imagenet_third
            from stablediffpuri.guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                classifier_defaults,
                create_model_and_diffusion,
                create_classifier,
                add_dict_to_argparser,
                args_to_dict,
            )
            args, config = stablediffpuri.main.parse_args_and_config()

            model, diffusion = create_model_and_diffusion(
                **args_to_dict(config.net, model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load(config.net.model_path, map_location="cpu")
            )
            model.to(config.device.clf_device)
            if config.net.use_fp16:
                model.convert_to_fp16()
            model.eval()
            target_image = purify_imagenet_third(target_image, diffusion, model, max_iter=1, mode="purification",
                                           config=config)
            target_image = target_image[0]
            target_image = 255 * target_image
            # end purification

            target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, target_image / 255).data
        print('generate target images end!')
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def puri_test_forth(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        print("This is puri-test function!")
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            print('puri-test-count:', i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            # target_image = 255 * fake_image  # before lyh
            target_image = fake_image  # lyh
            # target_image = original_image.unsqueeze(0)  # lyh

            # begin purification
            from stablediffpuri.purification.purify_imagenet import purify_imagenet_forth
            from stablediffpuri.guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                classifier_defaults,
                create_model_and_diffusion,
                create_classifier,
                add_dict_to_argparser,
                args_to_dict,
            )
            args, config = stablediffpuri.main.parse_args_and_config()

            model, diffusion = create_model_and_diffusion(
                **args_to_dict(config.net, model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load(config.net.model_path, map_location="cpu")
            )
            model.to(config.device.clf_device)
            if config.net.use_fp16:
                model.convert_to_fp16()
            model.eval()
            target_image = purify_imagenet_forth(target_image, diffusion, model, max_iter=1, mode="purification",
                                           config=config)
            target_image = target_image[0]
            target_image = 255 * target_image
            # end purification

            target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, target_image / 255).data
        print('generate target images end!')
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))


    # puri-clean-data
    def puri_clean_data(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        print("This is puri-test function!")
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            print('puri-clean-count:', i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            original_image = (original_image + 1) / 2
            target_image = original_image.unsqueeze(0)  # lyh

            # begin purification
            from stablediffpuri.purification.purify_imagenet import purify_imagenet
            from stablediffpuri.guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                classifier_defaults,
                create_model_and_diffusion,
                create_classifier,
                add_dict_to_argparser,
                args_to_dict,
            )
            args, config = stablediffpuri.main.parse_args_and_config()

            model, diffusion = create_model_and_diffusion(
                **args_to_dict(config.net, model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load(config.net.model_path, map_location="cpu")
            )
            model.to(config.device.clf_device)
            if config.net.use_fp16:
                model.convert_to_fp16()
            model.eval()
            target_image = purify_imagenet(target_image, diffusion, model, max_iter=1, mode="purification",
                                           config=config)
            target_image = target_image[0]
            target_image = 255 * target_image
            # end purification

            target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
        print('generate target images end!')
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def transfer_attack(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        print('This is transfer_attack function!')
        self.load_prototypenet()
        self.load_generator()
        self._ensure_transfer_model()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.transfer_bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            target_image = 255 * fake_image
            target_hashcode = self.transfer_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, fake_image[0]).data
        print('generate target images end!')
        TdB = self.transfer_model.generate_text_hashcode(database_texts)
        IdB = self.transfer_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def transfer_puri_test(self, database_images, database_texts, database_labels,
                           test_images, test_texts, test_labels):
        print('This is transfer_puri_test function by using stablepuri!')
        self.load_prototypenet()
        self.load_generator()
        self._ensure_transfer_model()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.transfer_bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate transfer attack images and stablepuri purification...')
        for i in range(num_test):
            print('transfer-puri-test-count:', i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            target_image = fake_image

            from stablediffpuri.purification.purify_imagenet import purify_imagenet_first
            from stablediffpuri.guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                classifier_defaults,
                create_model_and_diffusion,
                create_classifier,
                add_dict_to_argparser,
                args_to_dict,
            )
            args, config = stablediffpuri.main.parse_args_and_config()

            model, diffusion = create_model_and_diffusion(
                **args_to_dict(config.net, model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load(config.net.model_path, map_location="cpu")
            )
            model.to(config.device.clf_device)
            if config.net.use_fp16:
                model.convert_to_fp16()
            model.eval()
            target_image = purify_imagenet_first(target_image, diffusion, model, max_iter=1, mode="purification",
                                                 config=config)
            target_image = target_image[0]
            target_image = 255 * target_image

            target_hashcode = self.transfer_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, target_image / 255).data
        print('generate transfer stablepuri images end!')
        TdB = self.transfer_model.generate_text_hashcode(database_texts)
        IdB = self.transfer_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def _transfer_puri_test_with_stablepuri(self, database_images, database_texts, database_labels,
                                            test_images, test_texts, test_labels, purify_func_name,
                                            display_name, configure=None):
        print('This is {} function by using stablepuri!'.format(display_name))
        self.load_prototypenet()
        self.load_generator()
        self._ensure_transfer_model()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.transfer_bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate transfer attack images and {} purification...'.format(display_name))
        for i in range(num_test):
            print('{}-count:'.format(display_name), i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            target_image = fake_image

            purify_module = importlib.import_module('stablediffpuri.purification.purify_imagenet')
            from stablediffpuri.guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                classifier_defaults,
                create_model_and_diffusion,
                create_classifier,
                add_dict_to_argparser,
                args_to_dict,
            )
            args, config = stablediffpuri.main.parse_args_and_config()
            if configure is not None:
                configure(config)

            model, diffusion = create_model_and_diffusion(
                **args_to_dict(config.net, model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load(config.net.model_path, map_location="cpu")
            )
            model.to(config.device.clf_device)
            if config.net.use_fp16:
                model.convert_to_fp16()
            model.eval()
            purify_func = getattr(purify_module, purify_func_name)
            target_image = purify_func(target_image, diffusion, model, max_iter=1, mode="purification",
                                       config=config)
            target_image = target_image[0]
            target_image = 255 * target_image

            target_hashcode = self.transfer_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, target_image / 255).data
        print('generate {} images end!'.format(display_name))
        TdB = self.transfer_model.generate_text_hashcode(database_texts)
        IdB = self.transfer_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def transfer_puri_test_third(self, database_images, database_texts, database_labels,
                                 test_images, test_texts, test_labels):
        def configure(config):
            config.net.use_ddim = False
            config.purification.cond = False

        self._transfer_puri_test_with_stablepuri(
            database_images, database_texts, database_labels, test_images, test_texts, test_labels,
            'purify_imagenet_third', 'transfer_puri_test_third', configure)

    def transfer_puri_test_forth(self, database_images, database_texts, database_labels,
                                 test_images, test_texts, test_labels):
        def configure(config):
            config.net.use_ddim = False
            config.purification.cond = True
            config.purification.guide_mode = 'SSIM'

        self._transfer_puri_test_with_stablepuri(
            database_images, database_texts, database_labels, test_images, test_texts, test_labels,
            'purify_imagenet_forth', 'transfer_puri_test_forth', configure)

    def transfer_purification_test(self, database_images, database_texts, database_labels,
                                   test_images, test_texts, test_labels):
        print('This is transfer_purification_test function by using Purification-master!')
        self.load_prototypenet()
        self.load_generator()
        self._ensure_transfer_model()
        purifier = self._load_purification_master_forward()

        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.transfer_bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate transfer attack images and Purification-master purification...')
        for i in range(num_test):
            print('transfer-purification-count:', i)
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2

            with torch.no_grad():
                purified_image = purifier.purify(fake_image)
            target_image = 255 * purified_image

            target_hashcode = self.transfer_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, purified_image[0]).data
        print('generate transfer purified images end!')
        TdB = self.transfer_model.generate_text_hashcode(database_texts)
        IdB = self.transfer_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))
