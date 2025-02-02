import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats
from skimage.segmentation import slic


def tvloss(y_hat, TV_WEIGHT):
    """
    Calculates the total variation (TV) loss for a given input tensor.

    Parameters:
    y_hat (torch.Tensor): The input tensor of shape (batch_size, channels, height, width)
    TV_WEIGHT (float): The weight for the TV loss term

    Returns:
    torch.Tensor: The TV loss value of shape (1)

    Note:
    The TV loss is calculated as the sum of absolute differences between neighboring pixels in the input tensor.
    """
    diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
    tv_loss = TV_WEIGHT*(diff_i + diff_j)
    return tv_loss


def gradient_img(img, device):
    """
    Calculates the gradient image of a given input image using convolutions.

    Parameters:
    img (torch.Tensor): The input image of shape (batch_size, channels, height, width)
    device (torch.device): The device on which the tensor is located

    Returns:
    torch.Tensor: The gradient image of shape (batch_size, channels*4, height, width)

    Note:
    The gradient image is calculated by convolving the input image with four different kernels
    representing the gradients in the x, y, xy, and yx directions.
    """
    img = img.to(device)
    img = img.squeeze(0)  # Remove the batch dimension if present
    # Define the convolution kernels for the gradients
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    c = np.array([[-2, -1, -0], [-1, 0, 1], [0, 1, 2]])
    d = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    # Create convolution layers for each kernel
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1,
                      padding=1, bias=False).to(device)
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1,
                      padding=1, bias=False).to(device)
    conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1,
                      padding=1, bias=False).to(device)
    conv4 = nn.Conv2d(1, 1, kernel_size=3, stride=1,
                      padding=1, bias=False).to(device)
    # Set the weights of the convolution layers to the defined kernels
    conv1.weight = nn.Parameter(torch.from_numpy(
        a).float().unsqueeze(0).unsqueeze(0).to(device))
    conv2.weight = nn.Parameter(torch.from_numpy(
        b).float().unsqueeze(0).unsqueeze(0).to(device))
    conv3.weight = nn.Parameter(torch.from_numpy(
        c).float().unsqueeze(0).unsqueeze(0).to(device))
    conv4.weight = nn.Parameter(torch.from_numpy(
        d).float().unsqueeze(0).unsqueeze(0).to(device))
    # Calculate the gradients using the convolution layers
    G_x = conv1(img)
    G_y = conv2(img)
    G_xy = conv3(img)
    G_yx = conv4(img)
    # Concatenate the gradients to form the gradient image
    return torch.cat([G_x, G_y, G_xy, G_yx], dim=1)


def power_loss(images, device, E=100):
    """
       Calculates the power loss of an image using Fourier Transform.

       Parameters:
       images (torch.Tensor): The input images of shape (batch_size, channels, height, width)
       device (torch.device): The device on which the tensor is located
       E (int, optional): The number of elements to keep in the sorted power spectrum. Default is 100.

       Returns:
       torch.Tensor: The indices of the sorted power spectrum of shape (E)

       Note:
       The power loss is calculated by taking the Fourier Transform of the input images,
       squaring the magnitudes, and then binning the power spectrum into frequency bins.
       The function returns the indices of the sorted power spectrum.
       """
    # Convert the images to numpy array and detach from the computational graph
    image = images.detach().cpu().numpy().astype(int)
    npix = images.shape[1]
    # Compute the Fourier Transform of the image
    fourier_image = np.fft.fftn(image)
    # Compute the magnitudes of the Fourier Transform
    fourier_amplitudes = np.abs(fourier_image)**2
    # Compute the frequencies corresponding to each pixel
    kfreq = np.fft.fftfreq(npix) * npix
    # Create a 2D grid of frequencies
    kfreq2D = np.meshgrid(kfreq, kfreq)
    # Compute the norm of the frequencies
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    # Flatten the norm of the frequencies
    knrm = knrm.flatten()
    # Reshape the fourier amplitudes to match the batch size
    fourier_amplitudes = fourier_amplitudes.reshape(images.shape[0], -1)
    # Define the bins for the power spectrum
    kbins = np.arange(0.5, npix//2+1, 1.)
    # Compute the midpoints of the bins
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    # Compute the mean power spectrum within each bin
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)
    # Weight the mean power spectrum by the area of each bin
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    # Get the indices of the sorted power spectrum
    ind = np.argpartition(Abins, E)
    # Convert the indices to a torch tensor and move it to the specified device
    return torch.FloatTensor(ind).to(device)


class MS_SSIM_L1_LOSS(nn.Module):

    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros(
            (3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :,
                    :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :,
                    :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :,
                    :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=1, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=1, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=1,
                           padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=1,
                           padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=1,
                           padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=1, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + \
            (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()


class loss_custom():

    def __init__(self, phase, batchsize, c, n_segments, TV_WEIGHT, device):
        """
        Initializes the loss class.

        Parameters:
        phase (str): The phase of the training process ('first' or'second').
        batchsize (int): The batch size.
        c (int): The number of channels.
        n_segments (int): The number of segments for the SLIC algorithm.
        device (torch.device): The device on which the tensors are located.
        """
        super(loss_custom, self).__init__()
        self.mseLoss = nn.MSELoss()
        self.cosLoss = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.msSSIMLoss = MS_SSIM_L1_LOSS()
        self.l1Loss = nn.L1Loss()
        self.phase = phase
        self.batchsize = batchsize
        self.c = c
        self.n_segments = n_segments
        self.TV_WEIGHT = TV_WEIGHT
        self.device = device

    def get_loss(self, fake, gts, inps):
        """
        Calculates the loss for the model.

        Parameters:
        fake (torch.Tensor): The generated estimatated image of shape (batch_size, channels, height, width).
        gts (torch.Tensor): The ground truth image of shape (batch_size, channels, height, width).
        inps (torch.Tensor): The input images of shape (batch_size, channels, height, width).

        Returns:
        torch.Tensor: The calculated loss.
        """
        fake = fake.to(self.device)
        gts = gts.to(self.device)
        inps = inps.to(self.device)
        up_sampled = inps[:, 3, :, :].unsqueeze(1)  # shape: [15,1,256,256]
        if self.phase == 'first':
            lossM = self.mseLoss(fake, gts)
            g_fake = gradient_img(fake, self.device)
            # ,nan=0,posinf=100,neginf=100)
            g_fake = torch.nan_to_num(g_fake)
            g_up_sampled = gradient_img(up_sampled, self.device)
            # ,nan=0,posinf=100,neginf=100)
            g_up_sampled = torch.nan_to_num(g_up_sampled)
            lossGS = self.cosLoss(g_up_sampled, g_fake)
            lossGS = 1 - torch.mean(lossGS)
            lossTV = tvloss(fake, self.TV_WEIGHT)
            losses = lossM + 10*lossTV + 10*lossGS

        else:
            lossSSIM = self.msSSIMLoss(fake, gts)
            lossL1 = self.mseLoss(fake, gts)

            lossE = self.l1Loss(power_loss(up_sampled.squeeze(
                1), self.device), power_loss(fake.squeeze(1), self.device))
            lossTV = tvloss(fake)
            segmentsi = torch.zeros(self.batchsize, self.c)
            segmentf = torch.zeros(self.batchsize, self.c)
            for i in range(self.batchsize):
                inps = inps.detach().cpu()
                si = slic(
                    inps[i, 2, :, :], n_segments=self.n_segments, sigma=5, channel_axis=None)
                for n, j in enumerate(np.unique(si)):
                    x, y = np.where(si == j)
                    xm, ym = np.unravel_index(
                        np.argmax(inps[i, 2, :, :][x, y], axis=None), inps[i, 2, :, :].shape)
                    segmentsi[i][n] = inps[i, 2, :, :][xm, ym]
                    segmentf[i][n] = fake[i, 0, xm, ym]
            lossC = self.l1Loss(segmentsi.to(self.device),
                                segmentf.to(self.device))
            losses = 100*lossL1 + 0.001*lossTV + 84 * lossSSIM + lossE + lossC
            # + lossC #10*lossL1#+  lossE +  lossTV + lossC
        return losses

    def getMseLoss(self, fake, gts):
        return self.mseLoss(fake, gts)


class fl_loss_custom():
    def __init__(self, phase, batchsize, c, n_segments, TV_WEIGHT, device, agent_size=(85, 85), full_size=(256, 256)):
        super(fl_loss_custom, self).__init__()
        self.mseLoss = nn.MSELoss()
        self.cosLoss = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.msSSIMLoss = MS_SSIM_L1_LOSS()
        self.l1Loss = nn.L1Loss()
        self.phase = phase
        self.batchsize = batchsize
        self.c = c
        self.n_segments = n_segments
        self.TV_WEIGHT = TV_WEIGHT
        self.device = device
        self.agent_size = agent_size
        self.full_size = full_size

    def get_loss(self, fake, gts, inps, agent_position):
        """
        Calculates the loss for a single agent in the federated learning setup.

        Parameters:
        fake (torch.Tensor): The generated estimated image of shape (batch_size, channels, 256, 256).
        gts (torch.Tensor): The ground truth image of shape (batch_size, channels, 85, 85).
        inps (torch.Tensor): The input images of shape (batch_size, channels, 85, 85).
        agent_position (tuple): The position of the agent's portion in the full image (top, left).

        Returns:
        torch.Tensor: The calculated loss.
        """
        # Extract the relevant portion from the fake image
        top, left = agent_position
        fake_portion = fake[:, :, top:top +
                            self.agent_size[0], left:left+self.agent_size[1]]

        up_sampled = inps[:, 3, :, :].unsqueeze(1)  # [batch_size, 1, 85, 85]
        up_sampled = inps
        if self.phase == 'first':
            lossM = self.mseLoss(fake_portion, gts)
            g_fake = gradient_img(fake_portion, self.device)
            g_fake = torch.nan_to_num(g_fake)
            g_up_sampled = gradient_img(up_sampled, self.device)
            g_up_sampled = torch.nan_to_num(g_up_sampled)
            lossGS = self.cosLoss(g_up_sampled, g_fake)
            lossGS = 1 - torch.mean(lossGS)
            lossTV = tvloss(fake_portion, self.TV_WEIGHT)
            losses = lossM + 10*lossTV + 10*lossGS

        else:
            lossSSIM = self.msSSIMLoss(fake_portion, gts)
            lossL1 = self.mseLoss(fake_portion, gts)

            lossE = self.l1Loss(power_loss(up_sampled.squeeze(1), self.device),
                                power_loss(fake_portion.squeeze(1), self.device))
            lossTV = tvloss(fake_portion)

            # Adapt SLIC segmentation for the agent's portion
            segmentsi = torch.zeros(self.batchsize, self.n_segments)
            segmentf = torch.zeros(self.batchsize, self.n_segments)
            for i in range(self.batchsize):
                inps_cpu = inps[i, 2, :, :].detach().cpu().numpy()
                si = slic(inps_cpu, n_segments=self.n_segments,
                          sigma=5, channel_axis=None)
                for n, j in enumerate(np.unique(si)):
                    x, y = np.where(si == j)
                    xm, ym = np.unravel_index(
                        np.argmax(inps_cpu[x, y]), inps_cpu.shape)
                    segmentsi[i][n] = inps[i, 2, xm, ym]
                    segmentf[i][n] = fake_portion[i, 0, xm, ym]

            lossC = self.l1Loss(segmentsi.to(self.device),
                                segmentf.to(self.device))
            losses = 100*lossL1 + 0.001*lossTV + 84 * lossSSIM + lossE + lossC

        return losses

    def getMseLoss(self, fake, gts):
        # top, left = agent_position
        # fake_portion = fake[:, :, top:top +
        #                     self.agent_size[0], left:left+self.agent_size[1]]
        return self.mseLoss(fake, gts)
