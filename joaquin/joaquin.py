"""
TODO: change API so Joaquin takes "stars"
"""

import numpy as np
from scipy.optimize import minimize

from .design_matrix import DesignMatrix
from .logger import logger


class Joaquin:

    def __init__(self, stars, terms=['lsf', 'phot', 'spec'],
                 frozen=None, DM_kwargs=None):
        # TODO
        if DM_kwargs is None:
            DM_kwargs = dict()
        self.dm = DesignMatrix(stars, **DM_kwargs)
        self.X, self.y, self.y_ivar, self.idx_map = self.dm.get_Xy(terms)

        # Currently, stores parameter names and shapes
        self._param_info = {}

        # duh
        self._param_info['parallax_zpt'] = 1

        # the inv-var of the prior on the spectral components in beta
        self._param_info['L2_ivar'] = 1

        # linear coefficients (in the exp argument)
        self._param_info['beta'] = self.X.shape[1]

        if 'spec' in terms:
            L2_slice = self.idx_map['spec']
        else:
            L2_slice = np.ones(self.X.shape[1], dtype=bool)
        self.L2_slice = L2_slice

        if frozen is None:
            frozen = {}
        self.frozen = frozen

    def unpack_pars(self, par_list):
        i = 0
        par_dict = {}
        for key, par_len in self._param_info.items():
            if key in self.frozen:
                par_dict[key] = self.frozen[key]
            else:
                par_dict[key] = np.array(par_list[i:i+par_len])
                if len(par_dict[key]) == 1:  # HORRIBLE
                    par_dict[key] = par_dict[key][0]

                i += par_len

        return par_dict

    def pack_pars(self, par_dict):
        parvec = []
        for i, k in enumerate(self._param_info):
            if k not in self.frozen:
                parvec.append(par_dict[k])
        return np.concatenate(parvec)

    def init_beta(self, parallax_zpt=None, L2_ivar=None):
        parallax_zpt = self.frozen.get('parallax_zpt', parallax_zpt)
        L2_ivar = self.frozen.get('L2_ivar', L2_ivar)

        if parallax_zpt is None or L2_ivar is None:
            raise ValueError('todo')

        y = self.y + parallax_zpt
        plx_mask = y > (3 / np.sqrt(self.y_ivar))  # 3 sigma

        X = self.X[plx_mask]
        y = y[plx_mask]
        y_ivar = self.y_ivar[plx_mask]

        ln_plx_ivar = y**2 * y_ivar
        ln_y = np.log(y)

        XT_Cinv = X.T * ln_plx_ivar
        XT_Cinv_X = np.dot(XT_Cinv, X)
        XT_Cinv_X[np.diag_indices(X.shape[1])] += L2_ivar

        beta = np.linalg.solve(XT_Cinv_X, np.dot(XT_Cinv, ln_y))
        return beta

    def chi(self, parallax_zpt, L2_ivar, beta):
        y = self.y + parallax_zpt
        model_ln_plx = np.dot(self.X, beta)
        model_y = np.exp(model_ln_plx)
        resid = y - model_y
        return resid * np.sqrt(self.y_ivar)

    def ln_likelihood(self, parallax_zpt, L2_ivar, beta):
        y = self.y + parallax_zpt
        model_ln_plx = np.dot(self.X, beta)
        model_y = np.exp(model_ln_plx)
        resid = y - model_y

        ll = -0.5 * np.sum(resid**2 * self.y_ivar)
        ll_grad = np.dot(self.X.T * model_y,  # broadcasting trickery
                         self.y_ivar * resid)

        return ll, ll_grad

    def ln_prior(self, parallax_zpt, L2_ivar, beta):
        lp = - 0.5 * L2_ivar * np.sum(beta[self.L2_slice] ** 2)
        lp_grad = np.zeros_like(beta)
        lp_grad[self.L2_slice] = - L2_ivar * beta[self.L2_slice]
        return lp, lp_grad

    def neg_ln_posterior(self, parallax_zpt, L2_ivar, beta):
        ll, ll_grad = self.ln_likelihood(parallax_zpt, L2_ivar, beta)
        lp, lp_grad = self.ln_prior(parallax_zpt, L2_ivar, beta)
        logger.log(0, f'objective function evaluation: ll={ll}, lp={lp}')
        return - (ll + lp), - (ll_grad + lp_grad)

    def __call__(self, p):
        par_dict = self.unpack_pars(p)
        return self.neg_ln_posterior(**par_dict)

    def optimize(self, init=None, **minimize_kwargs):
        """
        To set the maximum number of function evaluations, pass:

            options={'maxfun': ...}

        """
        if init is None:
            init = {}

        init.setdefault('parallax_zpt', 0.)
        init.setdefault('L2_ivar', 1.)

        if 'beta' not in init:
            init['beta'] = self.init_beta(**init)

        x0 = self.pack_pars(init)

        minimize_kwargs.setdefault('method', 'L-BFGS-B')
        if minimize_kwargs['method'] == 'L-BFGS-B':
            minimize_kwargs.setdefault('options', {'maxfun': 1024})

        res = minimize(
            self,
            x0=x0,
            jac=True,
            **minimize_kwargs)

        return res
