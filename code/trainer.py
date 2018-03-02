from __future__ import print_function

import os
import sys
import time
from copy import deepcopy
from random import randint

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
from PIL import Image
from six.moves import range
from tensorboard import FileWriter
from tensorboard import summary
from torch.autograd import Variable

from miscc.config import cfg
from miscc.utils import mkdir_p
from model import G_NET, D_NET64, D_NET128, D_NET256, D_NET512, D_NET1024
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_recall_curve

# ################## Shared functions ###################
def compute_mean_covariance(img):
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)
    # batch_size * channel_num * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
             (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def load_network(gpus):
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    netsD = []

    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_NET64())
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_NET128())
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_NET256())
    if cfg.TREE.BRANCH_NUM > 3:
        netsD.append(D_NET512())
    if cfg.TREE.BRANCH_NUM > 4:
        netsD.append(D_NET1024())
    # TODO: if cfg.TREE.BRANCH_NUM > 5:

    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
        # print(netsD[i])
    print('# of netsD', len(netsD))

    count = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)

    # inception_model = INCEPTION_V3()


    if cfg.CUDA:
        netG.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()
    # inception_model = inception_model.cuda()
    # inception_model.eval()

    return netG, netsD, len(netsD), count  # inception_model,


def define_optimizers(netG, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    # G_opt_paras = []
    # for p in netG.parameters():
    #     if p.requires_grad:
    #         G_opt_paras.append(p)
    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))
    return optimizerG, optimizersD


def save_model(netG, avg_param_G, netsD, epoch, model_dir):
    load_params(netG, avg_param_G)
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(
            netD.state_dict(),
            '%s/netD%d_%d.pth' % (model_dir, i, epoch))
    print('Save G/Ds models.')


def save_img_results(imgs_tcpu, fake_imgs, num_imgs,
                     count, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT

    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/real_samples.png' % (image_dir),
        normalize=True)
    real_img_set = vutils.make_grid(real_img).numpy()
    real_img_set = np.transpose(real_img_set, (1, 2, 0))
    real_img_set = real_img_set * 255
    real_img_set = real_img_set.astype(np.uint8)
    sup_real_img = summary.image('real_img', real_img_set)
    summary_writer.add_summary(sup_real_img, count)

    for i in range(num_imgs):
        fake_img = fake_imgs[i][0:num]
        # The range of fake_img.data (i.e., self.fake_imgs[i][0:num])
        # is still [-1. 1]...
        vutils.save_image(
            fake_img.data, '%s/count_%09d_fake_samples%d.png' %
                           (image_dir, count, i), normalize=True)

        fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

        fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
        fake_img_set = (fake_img_set + 1) * 255 / 2
        fake_img_set = fake_img_set.astype(np.uint8)

        sup_fake_img = summary.image('fake_img%d' % i, fake_img_set)
        summary_writer.add_summary(sup_fake_img, count)
        summary_writer.flush()


# ################## For uncondional tasks ######################### #
class GANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def prepare_data(self, data):
        imgs = data

        vimgs = []
        for i in range(self.num_Ds):
            if cfg.CUDA:
                vimgs.append(Variable(imgs[i]).cuda())
            else:
                vimgs.append(Variable(imgs[i]))

        return imgs, vimgs

    def train_Dnet(self, idx, count):
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion = self.criterion

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        fake_imgs = self.fake_imgs[idx]
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        #
        netD.zero_grad()
        #
        real_logits = netD(real_imgs)
        fake_logits = netD(fake_imgs.detach())
        #
        errD_real = criterion(real_logits[0], real_labels)
        errD_fake = criterion(fake_logits[0], fake_labels)
        #
        errD = errD_real + errD_fake
        errD.backward()
        # update parameters
        optD.step()
        # log
        if flag == 0:
            summary_D = summary.scalar('D_loss%d' % idx, errD.data[0])
            self.summary_writer.add_summary(summary_D, count)
        return errD

    def train_Gnet(self, count):
        self.netG.zero_grad()
        errG_total = 0
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion = self.criterion
        real_labels = self.real_labels[:batch_size]

        for i in range(self.num_Ds):
            netD = self.netsD[i]
            outputs = netD(self.fake_imgs[i])
            errG = criterion(outputs[0], real_labels)
            # errG = self.stage_coeff[i] * errG
            errG_total = errG_total + errG
            if flag == 0:
                summary_G = summary.scalar('G_loss%d' % i, errG.data[0])
                self.summary_writer.add_summary(summary_G, count)

        # Compute color preserve losses
        if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
            if self.num_Ds > 1:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-1])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-2].detach())
                like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                            nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu2 + like_cov2
            if self.num_Ds > 2:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-2])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-3].detach())
                like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                            nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu1 + like_cov1

            if flag == 0:
                sum_mu = summary.scalar('G_like_mu2', like_mu2.data[0])
                self.summary_writer.add_summary(sum_mu, count)
                sum_cov = summary.scalar('G_like_cov2', like_cov2.data[0])
                self.summary_writer.add_summary(sum_cov, count)
                if self.num_Ds > 2:
                    sum_mu = summary.scalar('G_like_mu1', like_mu1.data[0])
                    self.summary_writer.add_summary(sum_mu, count)
                    sum_cov = summary.scalar('G_like_cov1', like_cov1.data[0])
                    self.summary_writer.add_summary(sum_cov, count)

        errG_total.backward()
        self.optimizerG.step()
        return errG_total

    def train(self):
        self.netG, self.netsD, self.num_Ds, \
        self.inception_model, start_count = load_network(self.gpus)
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.BCELoss()

        self.real_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(0))
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))

        if cfg.CUDA:
            self.criterion.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()

        predictions = []
        count = start_count
        start_epoch = start_count // (self.num_batches)
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, data in enumerate(self.data_loader, 0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.imgs_tcpu, self.real_imgs, self.labels, self.label_vectos \
                    = self.prepare_data(data)

                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                self.fake_imgs, _, _ = self.netG(noise)

                #######################################################
                # (2) Update D network
                ######################################################
                errD_total = 0
                for i in range(self.num_Ds):  # num_DS: D network numbers
                    errD = self.train_Dnet(i, count)
                    errD_total += errD

                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                errG_total = self.train_Gnet(count)
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # for inception score
                pred = self.inception_model(self.fake_imgs[-1].detach())
                predictions.append(pred.data.cpu().numpy())

                if count % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD_total.data[0])
                    summary_G = summary.scalar('G_loss', errG_total.data[0])
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)
                if step == 0:
                    print('''[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f'''
                          % (epoch, self.max_epoch, step, self.num_batches,
                             errD_total.data[0], errG_total.data[0]))
                count = count + 1

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    # Save images
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    #
                    self.fake_imgs, _, _ = self.netG(fixed_noise)
                    save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds,
                                     count, self.image_dir, self.summary_writer)
                    #
                    load_params(self.netG, backup_para)

                    # Compute inception score
                    if len(predictions) > 500:
                        predictions = np.concatenate(predictions, 0)
                        mean, std = compute_inception_score(predictions, 10)
                        # print('mean:', mean, 'std', std)
                        m_incep = summary.scalar('Inception_mean', mean)
                        self.summary_writer.add_summary(m_incep, count)
                        #
                        mean_nlpp, std_nlpp = \
                            negative_log_posterior_probability(predictions, 10)
                        m_nlpp = summary.scalar('NLPP_mean', mean_nlpp)
                        self.summary_writer.add_summary(m_nlpp, count)
                        #
                        predictions = []

            end_t = time.time()
            print('Total Time: %.2fsec' % (end_t - start_t))

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)

        self.summary_writer.close()

    def save_superimages(self, images, folder, startID, imsize):
        fullpath = '%s/%d_%d.png' % (folder, startID, imsize)
        vutils.save_image(images.data, fullpath, normalize=True)

    def save_singleimages(self, images, folder, startID, imsize):
        for i in range(images.size(0)):
            fullpath = '%s/%d_%d.png' % (folder, startID + i, imsize)
            # range from [-1, 1] to [0, 1]
            img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def evaluate(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            netG = G_NET()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = \
                torch.load(cfg.TRAIN.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load ', cfg.TRAIN.NET_G)

            # the path to save generated images
            s_tmp = cfg.TRAIN.NET_G
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d/%s' % (s_tmp, iteration, split_dir)
            if cfg.TEST.B_EXAMPLE:
                folder = '%s/super' % (save_dir)
            else:
                folder = '%s/single' % (save_dir)
            print('Make a new folder: ', folder)
            mkdir_p(folder)

            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(self.batch_size, nz))
            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            # switch to evaluate mode
            netG.eval()
            num_batches = int(cfg.TEST.SAMPLE_NUM / self.batch_size)
            cnt = 0
            for step in xrange(num_batches):
                noise.data.normal_(0, 1)
                fake_imgs, _, _ = netG(noise)
                if cfg.TEST.B_EXAMPLE:
                    self.save_superimages(fake_imgs[-1], folder, cnt, 256)
                else:
                    self.save_singleimages(fake_imgs[-1], folder, cnt, 256)
                    # self.save_singleimages(fake_imgs[-2], folder, 128)
                    # self.save_singleimages(fake_imgs[-3], folder, 64)
                cnt += self.batch_size


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, label_loader, unlabel_loader):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            self.theta = 0.5
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.label_loader = label_loader
        self.unlabel_loader = unlabel_loader
        self.num_batches = len(self.unlabel_loader)

    def prepare_data(self, data):
        imgs, labels, label_vectors = data
        #print(labels)
        #error_label_vectors = self.get_error_label(labels)
        real_vimgs = []
        if cfg.CUDA:
            vembedding = Variable(label_vectors).cuda()
            labels = Variable(labels).cuda()
            #error_label_vectors = Variable(error_label_vectors).cuda()
        else:
            vembedding = Variable(label_vectors)
            labels = Variable(labels)
            #error_label_vectors = Variable(error_label_vectors)
        for i in range(self.num_Ds):
            if cfg.CUDA:
                real_vimgs.append(Variable(imgs[i]).cuda())
            else:
                real_vimgs.append(Variable(imgs[i]))
        return imgs, real_vimgs, labels, vembedding#, error_label_vectors

    def get_extend(self, labels):
        ll = torch.zeros(self.batch_size, 1)

        extend_label = torch.cat((labels, Variable(ll)), 1)
        return extend_label

    def piece_wise(self, logits,count):
        logits_1 = logits > (0.5 -self.theta)
        logits_1 = logits_1.type(torch.FloatTensor)
        if cfg.CUDA:
           logits_1 = logits_1.cuda()
        logits_2 = logits_1 * logits
        logits_3 = logits > (0.5 + self.theta)
        logits_3 = logits_3.type(torch.FloatTensor)
        if cfg.CUDA:
           logits_3 = logits_3.cuda()
        logits_4 = (logits_3 + logits_2) - (logits_3 * logits_2)
        #print(logits_4)
        return logits_4

    def log_sum_exp(logits, mask=None, inf=1e7):
        if mask is not None:
            logits = logits * mask - inf * (1.0 - mask)
            max_logits = logits.max(1)[0]
            return ((logits - max_logits.expand_as(logits)).exp() * mask).sum(1).log().squeeze() + max_logits.squeeze()
        else:
            max_logits = logits.max(1)[0]
            return ((logits - max_logits.expand_as(logits)).exp()).sum(1).log().squeeze() + max_logits.squeeze()


    def train_Dnet(self, idx, count):
        flag = count % 25
        batch_size = self.real_imgs[0].size(0)
        criterion = self.criterion

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        label_imgs = self.label_real_imgs[idx]
        unlabel_imgs = self.unlabel_real_imgs[idx]
        fake_imgs = self.fake_imgs[idx]
        fake_imgs_2 = self.fake_imgs_2[idx]

        netD.zero_grad()

        lab_labels = self.labels[:batch_size].type(torch.LongTensor)
        error_labels = self.error_labels[:batch_size]
        label_logits, label_softmax_out, label_hash_logits, _ = netD(label_imgs)
        unlabel_logits, unlabel_softmax_out, unlabel_hash_logits, _ = netD(unlabel_imgs)
        fake_logits, fake_softmax_out, fake_hash_logits, _ = netD(fake_imgs.detach())
        fake2_logits, fake2_softmax_out, fake2_hash_logits, _ = netD(fake_imgs_2.detach())


        # standard classfication loss
        lab_loss = criterion(label_logits, lab_labels)
        fake_lab_loss = criterion(fake_logits, lab_labels)
        fake2_lab_loss = criterion(fake_logits, error_labels)

        supvised_loss = lab_loss + fake_lab_loss + fake2_lab_loss

        # GAN true-fake loss   adversary stream
        unl_logsumexp = self.log_sum_exp(unlabel_logits)
        fake_logsumexp = self.log_sum_exp(fake_logits)
        fake2_logsumexp = self.log_sum_exp(fake2_logits)

        true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
        fake_loss = 0.5 * torch.mean(F.softplus(fake_logsumexp))
        fake2_loss = 0.5 * torch.mean(F.softplus(fake2_logsumexp))
        adversary_loss = true_loss + fake_loss + fake2_loss

        # loss for hash
        positive = torch.sum((label_hash_logits - fake_logits) ** 2, 1)
        negtive = torch.sum((label_hash_logits - fake2_logits) ** 2, 1)
        hash_loss = 1 + positive - negtive
        hash_loss[hash_loss < 0] = 0
        hash_loss = torch.mean(hash_loss)
        print("hash_loss:%f" % (hash_loss.data[0]))

        d_total_loss = supvised_loss + adversary_loss + hash_loss
        print("d_supervied_loss_{0}:{1}".format(idx, supvised_loss))
        print("d_adversary_loss_{0}:{1}".format(idx, adversary_loss))
        print("d_hash_loss_{0}:{1}".format(idx, hash_loss))
        print("d_total_loss_{0}:{1}".format(idx, d_total_loss.data[0]))


        # adversary stream
        # for true
        # errD_real = criterion(real_logits, real_labels)
        # #errD_wrong = criterion(wrong_logits[0], fake_labels)
        # errD_fake = criterion(fake_logits, fake_labels)
        # # if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
        # #     errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
        # #                        criterion(real_logits[1], real_labels)
        # #     # errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
        # #     #                     criterion(wrong_logits[1], real_labels)
        # #     errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
        # #                        criterion(fake_logits[1], fake_labels)
        # #     #
        # #     errD_real = errD_real + errD_real_uncond
        # #     # errD_wrong = errD_wrong + errD_wrong_uncond
        # #     errD_fake = errD_fake + errD_fake_uncond
        # #     #
        # #     errD = errD_real + errD_wrong + errD_fake
        # # else:
        # #     errD = errD_real + 0.5 * (errD_wrong + errD_fake)

        # errD = errD_real + errD_fake
        # # # backward
        # errD.backward()
        # update parameters
        d_total_loss.backward()
        optD.step()

        # log
        if flag == 0:
            summary_D = summary.scalar('D_supervised%d' % idx, supvised_loss.data[0])
            summary_D1 = summary.scalar('D_hash_loss_%d' % idx, hash_loss.data[0])
            summary_D2 = summary.scalar('D_total_loss_%d' % idx, d_total_loss.data[0])
            summary_D3 = summary.scalar('D_adversary_loss_%d' % idx, adversary_loss.data[0])

            self.summary_writer.add_summary(summary_D, count)
            self.summary_writer.add_summary(summary_D1, count)
            self.summary_writer.add_summary(summary_D2, count)
            self.summary_writer.add_summary(summary_D3, count)
        return d_total_loss

    def train_Gnet(self, count):
        self.netG.zero_grad()
        errG_total = 0
        flag = count % 25
        batch_size = self.real_imgs[0].size(0)
        criterion = self.criterion

        for i in range(self.num_Ds):

            netD = self.netsD[i]
            label_imgs = self.label_real_imgs[i]
            unlabel_imgs = self.unlabel_real_imgs[i]
            fake_imgs = self.fake_imgs[i]
            fake_imgs_2 = self.fake_imgs_2[i]


            lab_labels = self.labels[:batch_size].type(torch.LongTensor)
            error_labels = self.error_labels[:batch_size]
            label_logits, label_softmax_out, label_hash_logits, _ = netD(label_imgs)
            unlabel_logits, unlabel_softmax_out, unlabel_hash_logits, _ = netD(unlabel_imgs)
            fake_logits, fake_softmax_out, fake_hash_logits, _ = netD(fake_imgs)
            fake2_logits, fake2_softmax_out, fake2_hash_logits, _ = netD(fake_imgs_2)


            # standard classfication loss
            lab_loss = criterion(label_logits, lab_labels)
            fake_lab_loss = criterion(fake_logits, lab_labels)
            fake2_lab_loss = criterion(fake_logits, error_labels)

            supvised_loss = (lab_loss + fake_lab_loss + fake2_lab_loss) / 3

            # GAN true-fake loss   adversary stream
            unl_logsumexp = self.log_sum_exp(unlabel_logits)
            fake_logsumexp = self.log_sum_exp(fake_logits)
            fake2_logsumexp = self.log_sum_exp(fake2_logits)

            true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
            fake_loss = 0.5 * torch.mean(F.softplus(fake_logsumexp))
            fake2_loss = 0.5 * torch.mean(F.softplus(fake2_logsumexp))
            adversary_loss = (true_loss + fake_loss + fake2_loss) / 3

            # loss for hash
            positive = torch.sum((label_hash_logits - fake_logits) ** 2, 1)
            negtive = torch.sum((label_hash_logits - fake2_logits) ** 2, 1)
            hash_loss = 1 + positive - negtive
            hash_loss[hash_loss < 0] = 0
            hash_loss = torch.mean(hash_loss)
            print("hash_loss:%f" % (hash_loss.data[0]))

            g_total_loss = supvised_loss - adversary_loss + hash_loss
            print("g_supervied_loss_{0}:{1}".format(i, supvised_loss))
            print("g_adversary_loss_{0}:{1}".format(i, adversary_loss))
            print("g_hash_loss_{0}:{1}".format(i, hash_loss))
            print("g_total_loss_{0}:{1}".format(i, g_total_loss.data[0]))


            # if len(outputs) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
            #     errG_patch = cfg.TRAIN.COEFF.UNCOND_LOSS * \
            #                  criterion(outputs[1], real_labels)
            #     errG = errG + errG_patch
            errG_total = errG_total + g_total_loss # + hash_loss
            print("g_loss_%d: %f" % (i, errG_total.data[0]))

            if flag == 0:
                summary_D = summary.scalar('G_supervised%d' % i, supvised_loss.data[0])
                summary_D1 = summary.scalar('G_hash_loss_%d' % i, hash_loss.data[0])
                summary_D2 = summary.scalar('G_total_loss_%d' % i, g_total_loss.data[0])
                summary_D3 = summary.scalar('G_adversary_loss_%d' % i, adversary_loss.data[0])

                self.summary_writer.add_summary(summary_D, count)
                self.summary_writer.add_summary(summary_D1, count)
                self.summary_writer.add_summary(summary_D2, count)
                self.summary_writer.add_summary(summary_D3, count)

        #
        # # Compute color consistency losses
        # if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
        #     if self.num_Ds > 1:
        #         mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-1])
        #         mu2, covariance2 = \
        #             compute_mean_covariance(self.fake_imgs[-2].detach())
        #         like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
        #         like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
        #                     nn.MSELoss()(covariance1, covariance2)
        #         errG_total = errG_total + like_mu2 + like_cov2
        #         if flag == 0:
        #             sum_mu = summary.scalar('G_like_mu2', like_mu2.data[0])
        #             self.summary_writer.add_summary(sum_mu, count)
        #             sum_cov = summary.scalar('G_like_cov2', like_cov2.data[0])
        #             self.summary_writer.add_summary(sum_cov, count)
        #     if self.num_Ds > 2:
        #         mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-2])
        #         mu2, covariance2 = \
        #             compute_mean_covariance(self.fake_imgs[-3].detach())
        #         like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
        #         like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
        #                     nn.MSELoss()(covariance1, covariance2)
        #         errG_total = errG_total + like_mu1 + like_cov1
        #         if flag == 0:
        #             sum_mu = summary.scalar('G_like_mu1', like_mu1.data[0])
        #             self.summary_writer.add_summary(sum_mu, count)
        #             sum_cov = summary.scalar('G_like_cov1', like_cov1.data[0])
        #             self.summary_writer.add_summary(sum_cov, count)

        # not use kl_loss
        # kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.COEFF.KL
        # errG_total = errG_total + kl_loss
        errG_total.backward()
        self.optimizerG.step()
        return errG_total

    def get_error_label(self, labels):

        class_num = cfg.GAN.CLASS_NUM
        batch_size = cfg.TRAIN.BATCH_SIZE

        error_labels_vector = torch.FloatTensor(batch_size, class_num).zero_()
        error_labels = []
        for i in range(batch_size):
            m = randint(0, class_num - 1)
            while m == labels[i]:
                m = randint(0, 9)

            error_labels_vector[i][m] = 1
            error_labels.append(m)

        error_labels = torch.LongTensor(error_labels)
        if cfg.CUDA:
            error_labels_vector = Variable(error_labels_vector).cuda()
            error_labels = Variable(error_labels).cuda()

        else:
            error_labels_vector = Variable(error_labels_vector)
            error_labels = Variable(error_labels)
        return error_labels_vector, error_labels

    def adjust_lr(self, optimizer, epoch):
        epoch_ratio = float(epoch) / float(cfg.TRAIN.MAX_EPOCH)
        lr = max(cfg.TRAIN.DISCRIMINATOR_LR * min(3. * (1 - epoch_ratio), 1.), 0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self):
        self.netG, self.netsD, self.num_Ds, start_count = load_network(self.gpus)
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.CrossEntropyLoss()

        # self.real_labels = \
        #     Variable(torch.FloatTensor(self.batch_size).fill_(1))
        # self.fake_labels = \
        #     Variable(torch.FloatTensor(self.batch_size).fill_(0))

        self.gradient_one = torch.FloatTensor([1.0])
        self.gradient_half = torch.FloatTensor([0.5])

        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        noise_2 = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(self.batch_size, nz))#.normal_(0, 1))

        if cfg.CUDA:
            self.criterion.cuda()
            # self.real_labels = self.real_labels.cuda()
            # self.fake_labels = self.fake_labels.cuda()
            self.gradient_one = self.gradient_one.cuda()
            self.gradient_half = self.gradient_half.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        predictions = []
        count = start_count
        start_epoch = start_count // (self.num_batches)
        label_iter = iter(self.label_loader)
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, unlabel_data in enumerate(self.unlabel_loader, 0):

                # print("data size:{0}".format(data))
                print("epoch:%d, step:%d" % (epoch, step))
                #######################################################
                # (0) Prepare training data
                ######################################################
                label_data = label_iter.next()
                self.label_imgs_tcpu, self.label_real_imgs, \
                self.labels, self.label_vectors = self.prepare_data(label_data)

                self.unlabel_imgs_tcpu, self.unlabel_real_imgs, \
                _, self.unlabel_vectors = self.prepare_data(unlabel_data)

                self.error_label_vector, self.error_labels = self.get_error_label(label_data[1])
                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                noise_2.data.normal_(0, 1)

                #
                self.fake_imgs = \
                    self.netG(noise, self.label_vectors)

                self.fake_imgs_2 = self.netG(noise_2, self.error_label_vectors)

                #######################################################
                # (2) Update D network
                ######################################################
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i, count)
                    errD_total += errD

                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                self.train_Gnet(count)
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                errG_total = self.train_Gnet(count)
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)
                """
                # for inception score
                pred = self.inception_model(self.fake_imgs[-1].detach())
                predictions.append(pred.data.cpu().numpy())
                """

                if count % 25 == 0:
                    summary_D = summary.scalar('D_loss', errD_total.data[0])
                    summary_G = summary.scalar('G_loss', errG_total.data[0])
                    # summary_KL = summary.scalar('KL_loss', kl_loss.data[0])
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)
                    # self.summary_writer.add_summary(summary_KL, count)

                count = count + 1
                
                if count % 3000 == 0:
                    self.theta = self.theta * 0.8
                    print("theta:%f, count:%d"% (self.theta, count))
                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    # Save images
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    #
                    fixed_noise.data.normal_(0, 1)
                    self.fake_imgs = \
                        self.netG(fixed_noise, self.label_vectors)
                    save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds,
                                     count, self.image_dir, self.summary_writer)
                    #
                    load_params(self.netG, backup_para)

                    # # Compute inception score
                    # if len(predictions) > 500:
                    #     predictions = np.concatenate(predictions, 0)
                    #     mean, std = compute_inception_score(predictions, 10)
                    #     # print('mean:', mean, 'std', std)
                    #     m_incep = summary.scalar('Inception_mean', mean)
                    #     self.summary_writer.add_summary(m_incep, count)
                    #     #
                    #     mean_nlpp, std_nlpp = \
                    #         negative_log_posterior_probability(predictions, 10)
                    #     m_nlpp = summary.scalar('NLPP_mean', mean_nlpp)
                    #     self.summary_writer.add_summary(m_nlpp, count)
                    #     #
                    #     predictions = []

                print("____________________________________________________________")
                end_t = time.time()


            self.adjust_lr(self.optimizerG,epoch)
            for i in range(cfg.TREE.BRANCH_NUM):
                self.adjust_lr(self.optimizersD[i], epoch)

            print('''[%d/%d][%d]
                         Loss_D: %.2f Loss_G: %.2f Time: %.2fs
                      '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.data[0], errG_total.data[0],
                     end_t - start_t))

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
        self.summary_writer.close()

    def save_superimages(self, images_list, filenames,
                         save_dir, split_dir, imsize):
        batch_size = images_list[0].size(0)
        num_sentences = len(images_list)
        for i in range(batch_size):
            s_tmp = '%s/super/%s/%s' % \
                    (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            #
            savename = '%s_%d.png' % (s_tmp, imsize)
            super_img = []
            for j in range(num_sentences):
                img = images_list[j][i]
                # print(img.size())
                img = img.view(1, 3, imsize, imsize)
                # print(img.size())
                super_img.append(img)
                # break
            super_img = torch.cat(super_img, 0)
            vutils.save_image(super_img, savename, nrow=10, normalize=True)

    def save_singleimages(self, images, filenames,
                          save_dir, split_dir, sentenceID, imsize):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' % \
                    (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            # range from [-1, 1] to [0, 255]
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def evaluate(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            if split_dir == 'image_test':
                split_dir = 'valid'
            netG = G_NET()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = \
                torch.load(cfg.TRAIN.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load ', cfg.TRAIN.NET_G)

            # the path to save generated images
            s_tmp = cfg.TRAIN.NET_G
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d' % (s_tmp, iteration)

            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(self.batch_size, nz))
            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            # switch to evaluate mode
            netG.eval()
            for step, data in enumerate(self.data_loader, 0):
                imgs, t_embeddings, filenames = data
                if cfg.CUDA:
                    t_embeddings = Variable(t_embeddings).cuda()
                else:
                    t_embeddings = Variable(t_embeddings)
                # print(t_embeddings[:, 0, :], t_embeddings.size(1))

                embedding_dim = t_embeddings.size(1)
                batch_size = imgs[0].size(0)
                noise.data.resize_(batch_size, nz)
                noise.data.normal_(0, 1)

                fake_img_list = []
                for i in range(embedding_dim):
                    fake_imgs, _, _ = netG(noise, t_embeddings[:, i, :])
                    if cfg.TEST.B_EXAMPLE:
                        # fake_img_list.append(fake_imgs[0].data.cpu())
                        # fake_img_list.append(fake_imgs[1].data.cpu())
                        fake_img_list.append(fake_imgs[2].data.cpu())
                    else:
                        self.save_singleimages(fake_imgs[-1], filenames,
                                               save_dir, split_dir, i, 256)
                        # self.save_singleimages(fake_imgs[-2], filenames,
                        #                        save_dir, split_dir, i, 128)
                        # self.save_singleimages(fake_imgs[-3], filenames,
                        #                        save_dir, split_dir, i, 64)
                        # break
                if cfg.TEST.B_EXAMPLE:
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 64)
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 128)
                    self.save_superimages(fake_img_list, filenames,
                                          save_dir, split_dir, 256)

    def get_hash(self, dataloader, netsD, dataset_name):
        hash_dict = {}
        imgs_total = None
        label_total = None
        output_dir = os.path.join("eval", dataset_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for step, data in enumerate(dataloader, 0):
            imgs_tcpu, real_imgs, \
            labels, label_vectors = self.prepare_data(data)

            for i in range(cfg.TREE.BRANCH_NUM):
                net = netsD[i]
                real_img = real_imgs[i]
                real_logits, softmax_out, hash_logits, _ = net(real_img)
                #hash_logits = hash_logits > 0.5
                if i in hash_dict:
                    hash_dict[i] = torch.cat((hash_dict[i], hash_logits), 0)
                else:
                    hash_dict[i] = hash_logits

            if label_total is None:
                label_total = labels
                imgs_total = real_imgs[0]
            else:
                label_total = torch.cat((label_total, labels), 0)
                imgs_total = torch.cat((imgs_total, real_imgs[0]), 0)

            print("step %d done!" % step)
        print(hash_dict[0].size())
        print(label_total.size())

        output_img = os.path.join(output_dir, dataset_name+"_images.npy")
        np.save(output_img, imgs_total.data.numpy())

        output_label = os.path.join(output_dir, dataset_name + "_label.npy")
        np.save(output_label, label_total.data.numpy())

        if label_total is not None:
            for i in range(cfg.TREE.BRANCH_NUM):
                output_hash = os.path.join(output_dir, "branch_%d_hash_%s.npy" % (i, dataset_name))
                np.save(output_hash, hash_dict[i].data.numpy())

            # print("total images is %d" % (step * cfg.TRAIN.BATCH_SIZE))
            # return hash_dict, img_dict, label_total

    def load_Dnet(self, gpus):
        if cfg.TRAIN.NET_D == '':
            print('Error: the path for morels is not found!')
            sys.exit(-1)
        else:
            netsD = []
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            if cfg.TREE.BRANCH_NUM > 3:
                netsD.append(D_NET512())
            if cfg.TREE.BRANCH_NUM > 4:
                netsD.append(D_NET1024())

            self.num_Ds = len(netsD)
            for i in range(len(netsD)):
                netsD[i].apply(weights_init)
                netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)

            for i in range(cfg.TREE.BRANCH_NUM):
                print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
                state_dict = torch.load('%snetD%d_16000.pth' % (cfg.TRAIN.NET_D, i),
                                        map_location=lambda storage, loc: storage)
                netsD[i].load_state_dict(state_dict)

            if cfg.CUDA:
                for i in range(len(netsD)):
                    netsD[i].cuda()
            return netsD

    def get_numpy(self, root, files):
        print(files)
        arr = None
        for f in files:
            path = os.path.join(root, f)
            a = np.load(path)
            print("a:{0}".format(a.shape))
            if arr is not None:
                arr = np.concatenate((arr, a), axis=0)
            else:
                arr = a
        return arr

    def compute_MAP_sklearn(self, test_features, db_features, test_label, db_label, metric='euclidean'):

        Y = cdist(test_features, db_features, metric)
        ind = np.argsort(Y, axis=1)
        prec_total = 0.0
        recall_total = None
        precision_total = None
        for k in range(np.shape(test_features)[0]):
            class_values = db_label[ind[k,:]]
            y_true = (test_label[k] == class_values)
            y_scores = np.arange(y_true.shape[0],0,-1)
            ap = average_precision_score(y_true, y_scores)
            prec_total += ap

            if recall_total is None:
                precision_total, recall_total, _ = precision_recall_curve(y_true, y_scores)
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                precision_total = precision_total + precision
                recall_total = recall_total + recall

        test_num = test_features.shape[0]
        MAP = prec_total / test_num
        recall_total = [i/test_num for i in recall_total]
        precision_total = [i / test_num for i in precision_total]

        print("MAP: %f" % MAP)
        print("recall:{0}".format(recall_total[:30]))
        print("precision:{0}".format(precision_total[:30]))

        with open(metric + "_result.txt", 'w') as f:
            f.write("MAP:%f\n" % MAP)
            f.write("recall\n:{0}".format(recall_total))
            f.write("precision\n:{0}".format(precision_total))

        np.save("recall.npy", recall_total)
        np.save("precision.npy", precision_total)

    def compute_MAP(self, root, branch, query_labels, db_labels):

        query_hash_path = os.path.join(root, "test", "branch_%d_hash_%s.npy" % (branch, "test"))
        db_hash_path = os.path.join(root, "db", "branch_%d_hash_%s.npy" % (branch, "db"))

        query_hashs = np.load(query_hash_path)
        db_hashs = np.load(db_hash_path)

        assert db_labels.shape[0] == db_hashs.shape[0], "db labels num must be equal to db hash num"
        assert query_labels.shape[0] == query_hashs.shape[0], "db labels num must be equal to db hash num"

        print("-----------------------------------use features----------------------")
        self.compute_MAP_sklearn(query_hashs, db_hashs, query_labels, db_labels)
        print("-----------------------------------use hash--------------------------")
        query_h = query_hashs > 0.5
        db_h= db_hashs > 0.5
        self.compute_MAP_sklearn(query_hashs, db_hashs, query_labels, db_labels)

        return db_hashs, query_hashs

       # query_labels = query_labels.tolist()
        # recall_num = 5900
        # MAP = 0
        # cnt = 0
        # precision_topk = [0 for i in range(len(query_labels) / 10 + 1)]
        # recall_curve = [0 for i in range(len(query_labels) / 25 + 1)]
        # precision_curve = [0 for i in range(len(query_labels) / 25 + 1)]

        # for i in range(query_labels.shape[0]):
        #     hamming_dis = 0.0
        #     for h in query_hashs:
        #         q = h[i]
        #         hamming_dis += np.sum((h - q) ** 2, axis=1)
        #     hamming_dis = hamming_dis.tolist()
        #     hamming_label = zip(query_labels, hamming_dis)
        #     sorted_by_hamming = sorted(hamming_label, key=lambda m: m[1])
        #
        #     smi = 0.0
        #     count = 0
        #     ap = 0
        #     for label, hamming in sorted_by_hamming:
        #         count += 1
        #         if label == query_labels[i]:
        #             smi += 1
        #             ap += smi / count
        #         if count % 10 == 0:
        #             precision_topk[count / 10] += smi / count
        #
        #         if count % 25 == 0:
        #             recall_curve[count / 25] += smi / recall_num
        #             precision_curve[count / 25] += smi / count
        #     #print("ap:%f"%(ap / smi))
        #     MAP += ap / smi
        #     cnt += 1
        #     if cnt % 100 == 0:
        #         print("has process %d queries" % cnt)
        #
        # MAP = MAP / len(query_labels)
        # precision_topk = [i / len(query_labels) for i in precision_topk]
        # recall_curve = [i / len(query_labels) for i in recall_curve]
        # precision_curve = [i / len(query_labels) for i in precision_curve]
        #
        # print("MAP_BRANCH_%d: %f" % (branch, MAP))
        # print("precision_top:{0}".format(precision_topk[0]))
        # print("recall curve:{0}".format(recall_curve[0]))
        # print("precision curve:{0}".format(precision_curve[0]))
        #
        # f = open("branch_%d_eval" % branch, 'w')
        # f.write("MAP: %f\n" % (MAP))
        # f.write("precision_top:\n")
        # self.write(f, precision_topk, "precision_top")
        # self.write(f, recall_curve, "recall_curve")
        # self.write(f, precision_curve, "precision_curve")
        # f.close()

    def compute_MAP_hash(self, root, paths, branch):
        query_label_path, query_hash_path = paths

        query_labels = self.get_numpy(root, query_label_path)
        query_hash = self.get_numpy(root, query_hash_path)
        print("total test number:%d" % query_hash.shape[0])
        assert query_hash.shape[0] == query_hash.shape[0], "query hash size not equal to query label size"
        query_hash = (query_hash > 0.5) + 0
        query_labels = query_labels.tolist()
        recall_num = 100

        r_1 = np.matmul(query_hash, query_hash.T)
        r_2 = np.matmul(1 - query_hash, (1 - query_hash).T)
        r = r_1 + r_2

        hamming_distance = cfg.GAN.HASH_DIM - r
        hamming_distance_list = hamming_distance.tolist()
        query_hamming = zip(query_labels, hamming_distance_list)

        MAP = 0
        cnt = 0
        precision_topk = [0 for i in range(len(query_labels) / 10 + 1)]
        recall_curve = [0 for i in range(len(query_labels) / 25 + 1)]
        precision_curve = [0 for i in range(len(query_labels) / 25 + 1)]

        for q_label, hamming_dis in query_hamming:
            hamming_label = zip(query_labels, hamming_dis)
            sorted_by_hamming = sorted(hamming_label, key=lambda m: m[1])

            smi = 0.0
            count = 0
            ap = 0
            for label, hamming in sorted_by_hamming:
                count += 1
                if label == q_label:
                    smi += 1
                    ap += smi / count
                if count % 10 == 0:
                    precision_topk[count / 10] += smi / count

                if count % 25 == 0:
                    recall_curve[count / 25] += smi / recall_num
                    precision_curve[count / 25] += smi / count
            # print("ap:%f"%(ap / smi))
            MAP += ap / smi
            cnt += 1
            if cnt % 100 == 0:
                print("has process %d queries" % cnt)

        MAP = MAP / len(query_labels)
        precision_topk = [i / len(query_labels) for i in precision_topk]
        recall_curve = [i / len(query_labels) for i in recall_curve]
        precision_curve = [i / len(query_labels) for i in precision_curve]

        print("MAP_BRANCH_%d: %f" % (branch, MAP))
        print("precision_top:{0}".format(precision_topk))
        print("recall curve:{0}".format(recall_curve))
        print("precision curve:{0}".format(precision_curve))

        f = open("branch_%d_eval" % branch, 'w')
        f.write("MAP: %f\n" % (MAP))
        f.write("precision_top:\n")
        self.write(f, precision_topk, "precision_top")
        self.write(f, recall_curve, "recall_curve")
        self.write(f, precision_curve, "precision_curve")
        f.close()

    def evaluate_MAP(self, db_dataloader, query_dataloader, root):

        if len(os.listdir(os.path.join(root, "test"))) == 0:
            netsD = self.load_Dnet(self.gpus)
            self.get_hash(query_dataloader, netsD, "test")
            print("get query image hash")
            self.get_hash(db_dataloader, netsD, "db")
            print("get db image hash!")

        eval_path = root
        db_hashs = None
        query_hashs = None
        query_label_path = os.path.join(root, "test", "test_label.npy")
        db_label_path = os.path.join(root, "db", "db_label.npy")
        query_labels = np.load(query_label_path)
        db_labels = np.load(db_label_path)

        for i in range(cfg.TREE.BRANCH_NUM):
            print("--------------------------branch %d-------------------------------------------" % i)
            db_hash, query_hash = self.compute_MAP(eval_path, i, query_labels, db_labels)

            if db_hashs is None:
                db_hashs = db_hash
                query_hashs = query_hash
            else:
                db_hashs += db_hash
                query_hashs += query_hash

        db_hashs /= cfg.TREE.BRANCH_NUM
        query_hashs /= cfg.TREE.BRANCH_NUM

        print("--------------------------------total--------------------------------------------")
        print("-----------------------------------use features----------------------")
        self.compute_MAP_sklearn(query_hashs, db_hashs, query_labels, db_labels)
        print("-----------------------------------use hash--------------------------")

        db_hashs = db_hashs > 0.5
        query_hashs = query_hashs > 0.5
        self.compute_MAP_sklearn(query_hashs, db_hashs, query_labels, db_labels, metric="hamming")
