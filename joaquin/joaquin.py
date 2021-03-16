import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def ln_likelihood(X, y, ivar, parallax_zpt, L2_ivar, beta):
    y = y + parallax_zpt
    model_ln_plx = jnp.dot(X, beta)
    resid = y - jnp.exp(model_ln_plx)
    return -0.5 * jnp.sum(resid**2 * ivar)


def neg_ln_posterior(X, y, ivar, parallax_zpt, L2_ivar, beta, L2_slice):
    ll = ln_likelihood(X, y, ivar, parallax_zpt, L2_ivar, beta)
    return - (ll - 0.5 * L2_ivar * jnp.sum(beta[L2_slice] ** 2)) / len(y)


class Joaquin:

    def __init__(self, X, y, y_ivar, frozen=None, L2_slice=None):
        assert X.shape[0] == len(y)
        assert len(y) == len(y_ivar)

        self._params = {}

        # duh
        self._params['parallax_zpt'] = 1

        # either the inv-var of the prior on the spectral components in beta
        self._params['L2_ivar'] = 1

        # linear coefficients
        self._params['beta'] = X.shape[1]

        self.X = X
        self.y = y
        self.y_ivar = y_ivar

        if L2_slice is None:
            L2_slice = np.ones_like(y, dtype=int)
        self.L2_slice = L2_slice

        if frozen is None:
            frozen = {}
        self.frozen = frozen

    def unpack_pars(self, par_list):
        i = 0
        par_dict = {}
        for key, par_len in self._params.items():
            if key in self.frozen:
                par_dict[key] = self.frozen[key]
            else:
                par_dict[key] = jnp.array(par_list[i:i+par_len])
                if len(par_dict[key]) == 1:  # HORRIBLE
                    par_dict[key] = par_dict[key][0]

                i += par_len

        return par_dict

#     def pack_pars(self, par_dict):
#         pass

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
        ln_y = jnp.log(y)

        XT_Cinv = X.T * ln_plx_ivar
        XT_Cinv_X = np.dot(XT_Cinv, X)
        XT_Cinv_X[np.diag_indices(X.shape[1])] += L2_ivar

        beta = np.linalg.solve(XT_Cinv_X, np.dot(XT_Cinv, ln_y))
        return beta

    def __call__(self, p):
        """Computes the negative ln posterior"""
        par_dict = self.unpack_pars(p)
        return neg_ln_posterior(
            self.X, self.y, self.y_ivar,
            par_dict['parallax_zpt'],
            par_dict['L2_ivar'],
            par_dict['beta'],
            self.L2_slice)

    def optimize(self, x0=None, **kwargs):
        obj = jax.value_and_grad(neg_ln_posterior, argnums=[3, 4, 5])

        def wrapper(p):  # noqa
            par_dict = self.unpack_pars(p)
            val, grads = obj(self.X, self.y, self.y_ivar,
                             par_dict['parallax_zpt'],
                             par_dict['L2_ivar'],
                             par_dict['beta'],
                             self.L2_slice)
            return val, jnp.concatenate([g.reshape(-1) for g in grads])

        if x0 is None:
            x0 = [0, 1.]
            beta0 = self.init_beta(*x0)
            x0 = x0 + list(beta0)

        elif len(x0) == 2:
            beta0 = self.init_beta(*x0)
            x0 = list(x0) + list(beta0)

        res = minimize(wrapper, x0=x0, method='BFGS', jac=True)
        return res
