"""
TODO: change API so Joaquin takes "stars"
"""

import numpy as np
from scipy.optimize import minimize

from .logger import logger


class Joaquin:

    def __init__(self, X, y, y_ivar, idx_map, frozen=None):
        self.X = X
        self.y = y
        self.y_ivar = y_ivar

        # Currently, stores parameter names and shapes
        self._param_info = {}

        # duh
        self._param_info['parallax_zpt'] = 1

        # the inv-var of the prior on the spectral components in beta
        self._param_info['L2_ivar'] = 1

        # linear coefficients (in the exp argument)
        self._param_info['beta'] = self.X.shape[1]

        self.idx_map = idx_map
        if 'spec' in idx_map:
            L2_slice = self.idx_map['spec']
        else:
            L2_slice = np.ones(self.X.shape[1], dtype=bool)
        self.L2_slice = L2_slice

        if frozen is None:
            frozen = {}
        self.frozen = frozen

    @classmethod
    def from_data(self, config, data, **kwargs):
        X, idx_map = data.get_X(phot_names=config.phot_names)
        y = data.stars['parallax']
        y_ivar = 1 / data.stars['parallax_error'] ** 2
        return Joaquin(X, y, y_ivar, idx_map, **kwargs)

    def unpack_pars(self, par_list):
        i = 0
        par_dict = {}
        for key, par_len in self._param_info.items():
            if key in self.frozen:
                par_dict[key] = self.frozen[key]
            else:
                par_dict[key] = np.squeeze(par_list[i:i+par_len])
                i += par_len

        return par_dict

    def pack_pars(self, par_dict):
        parvec = []
        for i, k in enumerate(self._param_info):
            if k not in self.frozen:
                parvec.append(np.atleast_1d(par_dict[k]))
        return np.concatenate(parvec)

    def init_beta(self, parallax_zpt=None, L2_ivar=None, **_):
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

    def init(self, pack=True, **kwargs):
        init_dict = kwargs.copy()

        if 'beta' not in self.frozen:
            init_dict['beta'] = self.init_beta(**kwargs)

        if 'parallaz_zpt' not in self.frozen:
            init_dict['parallax_zpt'] = kwargs.get(
                'parallax_zpt', -0.03)  # MAGIC NUMBER (Gaia value)

        if pack:
            return self.pack_pars(init_dict)
        else:
            return init_dict

    @classmethod
    def model_y(cls, X, *, beta, **_):
        model_ln_plx = np.dot(X, beta)
        model_y = np.exp(model_ln_plx)
        return model_y

    def chi(self, *, parallax_zpt, beta, **_):
        y = self.y + parallax_zpt
        # model_ln_plx = np.dot(self.X, beta)
        # model_y = np.exp(model_ln_plx)
        model_y = self.model_y(self.X, parallax_zpt=parallax_zpt, beta=beta)
        resid = y - model_y
        return resid * np.sqrt(self.y_ivar)

    def ln_likelihood(self, *, parallax_zpt, beta, **_):
        ll_grad = {}

        y = self.y + parallax_zpt

        # model_ln_plx = np.dot(self.X, beta)
        # model_y = np.exp(model_ln_plx)
        model_y = self.model_y(self.X, parallax_zpt=parallax_zpt, beta=beta)
        resid = y - model_y

        ll = -0.5 * np.sum(resid**2 * self.y_ivar)

        if 'beta' not in self.frozen:
            ll_grad['beta'] = np.dot(self.X.T,
                                     model_y * self.y_ivar * resid)

        if 'parallax_zpt' not in self.frozen:
            ll_grad['parallax_zpt'] = \
                - np.sum((y + parallax_zpt - model_y) * self.y_ivar)

        return ll, ll_grad

    def ln_prior(self, *, parallax_zpt, L2_ivar, beta, **_):
        lp = 0.
        lp_grad = {}

        if 'beta' not in self.frozen:
            lp += - 0.5 * L2_ivar * np.sum(beta[self.L2_slice] ** 2)
            lp_grad['beta'] = np.zeros_like(beta)
            lp_grad['beta'][self.L2_slice] = - L2_ivar * beta[self.L2_slice]

        if 'parallax_zpt' not in self.frozen:
            plx_zpt_var = 0.1 ** 2  # MAGIC NUMBER
            lp += - 0.5 * parallax_zpt ** 2 / plx_zpt_var
            lp_grad['parallax_zpt'] = -1 / plx_zpt_var * parallax_zpt

        return lp, lp_grad

    def neg_ln_posterior(self, parallax_zpt, L2_ivar, beta):
        ll, ll_grad = self.ln_likelihood(
            parallax_zpt=parallax_zpt,
            beta=beta)

        lp, lp_grad = self.ln_prior(
            L2_ivar=L2_ivar,
            parallax_zpt=parallax_zpt,
            beta=beta)

        lp_grad = np.concatenate([np.atleast_1d(lp_grad[k])
                                  for k in self._param_info
                                  if k not in self.frozen])
        ll_grad = np.concatenate([np.atleast_1d(ll_grad[k])
                                  for k in self._param_info
                                  if k not in self.frozen])

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
