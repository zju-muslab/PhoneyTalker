import torch
import torch.nn as nn
import numpy as np


class GPLDA(nn.Module):

    def __init__(self, phi, sigma, W, miu):
        super(GPLDA, self).__init__()
        n_phi = phi.shape[0]
        sigma_ac = np.dot(phi, np.transpose(phi))
        sigma_tot = sigma_ac + sigma
        inv_sigma_tot = np.linalg.pinv(sigma_tot)
        inv_sigma = np.linalg.pinv(sigma_tot - np.dot(np.dot(sigma_ac, inv_sigma_tot), sigma_ac))
        Q = inv_sigma_tot - inv_sigma
        P = np.dot(np.dot(inv_sigma_tot, sigma_ac), inv_sigma)
        U, S, _ = np.linalg.svd(P)

        self.W = torch.from_numpy(W)
        self.miu = torch.from_numpy(miu)
        self.lambda_v = torch.from_numpy(np.diag(S[0:n_phi]))
        self.U_k = torch.from_numpy(U[:, 0:n_phi])
        self.Q_hat = torch.from_numpy(np.dot(np.dot(np.transpose(U[:, 0:n_phi]), Q), U[:, 0:n_phi]))

    def to_device(self, device):
        self.W = self.W.to(device)
        self.miu = self.miu.to(device)
        self.lambda_v = self.lambda_v.to(device)
        self.U_k = self.U_k.to(device)
        self.Q_hat = self.Q_hat.to(device)

    def forward(self, x_enroll, x_test):
        x_enroll = x_enroll.T
        x_test = x_test.T
        # prepare enroll x vector
        x_enroll = x_enroll.double()
        x_enroll = torch.stack([x_i - self.miu[i] for i, x_i in enumerate(torch.unbind(x_enroll, dim=0))], dim=0)
        norm = torch.sqrt(torch.sum(x_enroll ** 2, dim=0))
        x_enroll = torch.stack([x_i / (norm[i] + 1e-8) for i, x_i in enumerate(torch.unbind(x_enroll, dim=1))], dim=1)
        x_enroll = self.W.T.mm(x_enroll)

        # prepare test x vector
        x_test = x_test.double()
        x_test = torch.stack([x_i - self.miu[i] for i, x_i in enumerate(torch.unbind(x_test, dim=0))], dim=0)
        norm = torch.sqrt(torch.sum(x_test ** 2, dim=0))
        x_test = torch.stack([x_i / (norm[i] + 1e-8) for i, x_i in enumerate(torch.unbind(x_test, dim=1))], dim=1)
        x_test = self.W.T.mm(x_test)

        # score data
        x_enroll = self.U_k.T.mm(x_enroll)
        x_test = self.U_k.T.mm(x_test)
        score_h1 = torch.diag(x_enroll.T.mm(self.Q_hat).mm(x_enroll))
        score_h2 = torch.diag(x_test.T.mm(self.Q_hat).mm(x_test))
        score_h1h2 = 2 * x_enroll.T.mm(self.lambda_v).mm(x_test)
        scores = torch.stack([x_i + score_h1[i] for i, x_i in enumerate(torch.unbind(score_h1h2, dim=0))], dim=0)
        scores = torch.stack([x_i + score_h2[i] for i, x_i in enumerate(torch.unbind(scores, dim=1))], dim=1)
        return scores