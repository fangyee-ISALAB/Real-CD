import torch
import torch.nn as nn
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from sklearn.mixture import GaussianMixture

class FixedMeanGMM(GaussianMixture):
    def __init__(self, n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1,
                 init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None,
                 fixed_mean=None, warm_start = False):
        super().__init__(n_components=n_components, covariance_type=covariance_type, tol=tol,
                         reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                         weights_init=weights_init, means_init=means_init, precisions_init=precisions_init,
                         random_state=random_state, warm_start = warm_start)
        self.fixed_mean = fixed_mean

    def _initialize_parameters(self, X, random_state):
        super()._initialize_parameters(X, random_state)
        self.means_[-1] = self.fixed_mean

    def _m_step(self, X, log_resp):
        super()._m_step(X, log_resp)

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1-self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


class AugCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(AugCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_aug, x_ema):
        return -(1-self.alpha) * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
                  - self.alpha * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)


class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return - (probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(1)


class GeneralizedCrossEntropy(nn.Module):
    """ Paper: https://arxiv.org/abs/1805.07836 """
    def __init__(self, q=0.8):
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q

    def __call__(self, logits, targets=None):
        probs = logits.softmax(1)
        if targets is None:
            targets = probs.argmax(dim=1)
        loss = 0.0
        for i in range(logits.shape[0]):
            t_ = targets[i].flatten(0)
            p_ = probs[i].flatten(1)
            probs_with_correct_idx = p_.index_select(-1, t_).diag()
            loss += (1.0 - probs_with_correct_idx ** self.q) / self.q
        return 1.0 * loss / logits.shape[0]


class ASTLoss:
    def __init__(self, param_dict):
        self.tau1 = param_dict.get('tau1', 0.0)
        self.tau2 = param_dict.get('tau2', 1.0)
        self.sample_ratio = param_dict.get('sample_ratio', 1)

    def __call__(self, anomaly_score, logit=None, eps = 1e-8):

        # Calibrate the OOD scores into the outlier probability.
        ax, bx = self.get_calibration_params(anomaly_score)
        anomaly_prob = torch.sigmoid((anomaly_score - ax) / bx)

        # Select a subset of pixels with high confidence.
        pseudo_labels = (anomaly_prob > 0.5).detach().long()
        pseudo_labels[(anomaly_prob > self.tau1)&(anomaly_prob < self.tau2)] = -1

        # Introduce the class weights to cope with the class imbalance between inliers and outliers.
        if (pseudo_labels==1).sum() == 0 or (pseudo_labels==0).sum() == 0:
            weight_ratio = 1
        else:
            weight_ratio =  (pseudo_labels==0).sum() / (pseudo_labels==1).sum()

        # The final loss. Correspond to Eq. (10) in the paper.
        g = anomaly_prob.unsqueeze(1)
        f = torch.softmax(logit, dim=1)
        t = pseudo_labels.float().unsqueeze(1)
        loss = -((((f*(1-t)*torch.log(f*(1-g)+eps)).sum(1) + weight_ratio * t*torch.log(g+eps)))[t!=-1]).mean()

        if np.isnan(loss.item()):
            loss = anomaly_score.sum() * 0 # invalid loss, no grad.

        return loss

    def get_calibration_params(self, score, MAX_NUM = 1000, BIN_NUM = 200):
        if isinstance(score, torch.Tensor):
            all_data = score.detach().cpu().numpy()
        else:
            all_data = score

        #  We sample 1% of the total data in clustering to reduce computation overhead.
        if len(all_data.flatten()) > MAX_NUM and self.sample_ratio < 1:
            selected_data = np.random.choice(all_data.flatten(), int(len(all_data.flatten()) * self.sample_ratio))
        else:
            selected_data = all_data.flatten()

        # We employ a peak-finding algorithm to identify the right-most peak in the distribution of OOD score.
        # The goal is to mitigate potential issues arising from the presence of multiple peaks in the inlier distribution.
        h = np.histogram(all_data, bins=BIN_NUM)
        h_smooth = gaussian_filter1d(h[0], sigma=1)
        peaks = argrelextrema(h_smooth, np.greater, mode = 'wrap')
        right_peak = h[1][peaks[0][-1]]

        model = FixedMeanGMM(fixed_mean=right_peak, n_components=2)
        model.fit(selected_data.reshape(-1, 1))

        # Set the calibration parameter a(x) as the value achieving equal probability under two Gaussian distributions
        uniform_points = np.linspace(selected_data.min(), selected_data.max(), num=1000).reshape(-1, 1)
        probs = model.predict_proba(uniform_points)
        diffs = np.abs(probs[:, 0] - probs[:, 1])
        equal_prob_index = np.where(diffs == diffs.min())[-1][-1]
        ax = uniform_points[equal_prob_index][0]

        # Set b(x) as the standard derivation.
        bx = all_data.std()

        return ax, bx