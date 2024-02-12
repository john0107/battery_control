import numpy as np
from scipy import stats, optimize
import warnings

######## util functions for config preprocessing #######


def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping:
                if isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                    updated_mapping[k] = deep_update(updated_mapping[k], v)
                else:
                    updated_mapping[k] = v
            else:
                raise KeyError(f'Invalid key: {k}')
    return updated_mapping


######## convert generalized hyperbolic distribution parameters #######

def compute_CARMA_parameters(phi1, phi2, theta):
    lambd1 = np.log(phi1 / 2 + np.sqrt(np.square(phi1 / 2) + phi2))
    lambd2 = np.log(phi1 / 2 - np.sqrt(np.square(phi1 / 2) + phi2))

    a1, a2 = - (lambd1 + lambd2), lambd1 * lambd2

    def compute_alpha1(b0): return (b0 + lambd1) / (lambd1 - lambd2)
    def compute_alpha2(b0): return (b0 + lambd2) / (lambd1 - lambd2)

    def compute_w1(b0):
        alpha1, alpha2 = compute_alpha1(b0), compute_alpha2(b0)
        numerator = alpha1**2 * lambd1 * lambd2 + alpha1**2 * \
            lambd2**2 + 2 * lambd1 * lambd2 * alpha1 * alpha2
        denominator = 2 * lambd1 * lambd2 * (lambd1 + lambd2)
        return numerator / denominator

    def compute_w2(b0):
        alpha1, alpha2 = compute_alpha1(b0), compute_alpha2(b0)
        numerator = alpha2**2 * lambd1 * lambd2 + alpha2**2 * \
            lambd1**2 + 2 * lambd1 * lambd2 * alpha1 * alpha2
        denominator = 2 * lambd1 * lambd2 * (lambd1 + lambd2)
        return numerator / denominator

    def compute_gamma_Y(b0, s):
        w1, w2 = compute_w1(b0), compute_w2(b0)
        return w1 * np.exp(lambd1 * s) + w2 * np.exp(lambd2 * s)

    def compute_gamma_U(b0, s):
        gammy_Y_0 = compute_gamma_Y(b0, 0)
        gammy_Y_1 = compute_gamma_Y(b0, 1)
        gammy_Y_2 = compute_gamma_Y(b0, 2)
        if s == 0:
            return (1 + phi1**2 + phi2**2) * gammy_Y_0 + (2*phi2*phi2 - 2*phi1) * gammy_Y_1 - (2*phi2) * gammy_Y_2
        elif s == 1:
            gammy_Y_3 = compute_gamma_Y(b0, 3)
            return (-phi2) * gammy_Y_3 + (phi1 * (phi2 - 1)) * gammy_Y_2 + (1 + phi1**2 + phi2**2 - phi2) * gammy_Y_1 + (phi1 * (phi2 - 1)) * gammy_Y_0
        else:
            raise

    def f(b0): return compute_gamma_U(b0, 1) / \
        compute_gamma_U(b0, 0) - (theta / (1 + theta**2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b0 = optimize.fsolve(func=f, x0=0.0)[0]
    alpha1, alpha2 = compute_alpha1(b0), compute_alpha2(b0)

    result = {
        'alpha1': alpha1,
        'alpha2': alpha2,
        'lambd1': -lambd1,
        'lambd2': -lambd2,
    }
    return result


def convert_ghyd_params(from_param, to_param, n_dim=1, **kwargs):
    if from_param == to_param:
        return kwargs
    elif from_param == 'alpha_bar':
        alpha_bar, mu, Sigma, gamma = kwargs['alpha_bar'], kwargs['mu'], kwargs['Sigma'], kwargs['gamma']
        # reparameterize
        chi, phi, mu, Sigma, gamma = alpha_bar, alpha_bar, mu, Sigma, gamma
        return convert_ghyd_params(chi=chi, phi=phi, mu=mu, Sigma=Sigma, gamma=gamma,
                                   from_param='chi', to_param=to_param)
    elif from_param == 'chi':
        # get parameters
        chi, phi, mu, Sigma, gamma = kwargs['chi'], kwargs['phi'], kwargs['mu'], kwargs['Sigma'], kwargs['gamma']
        # reparameterize
        alpha = np.sqrt(np.power(Sigma, -1/n_dim) *
                        (phi+gamma*np.power(Sigma, -1)*gamma))
        mu = mu
        delta = np.sqrt(chi*np.power(Sigma, 1/n_dim))
        beta = np.power(Sigma, -1)*gamma
        return convert_ghyd_params(alpha=alpha, mu=mu, delta=delta, beta=beta,
                                   from_param='alpha', to_param=to_param)
    elif from_param == 'alpha':
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid parameterization: {from_param}')

def convert_ghyd_to_nig_params(parameterization='alpha_bar', n_dim=1, **kwargs):
    # get parameters
    params = convert_ghyd_params(
        from_param=parameterization, to_param='alpha', n_dim=n_dim, **kwargs)
    alpha, mu, delta, beta = params['alpha'], params['mu'], params['delta'], params['beta']
    # reparameterize
    a, b, loc, scale = alpha*delta, beta*delta, mu, delta
    return {
        'a': a,
        'b': b,
        'loc': loc,
        'scale': scale,
    }

def convert_nig_to_ghyd(a, b, loc, scale, to_param='alpha_bar'):
    lambd, alpha_bar, mu, Sigma, gamma = - \
        0.5, (a**2-b**2)**0.5, loc, scale**2 / \
        (a**2-b**2)**0.5, (b*scale)/(a**2-b**2)**0.5
    return convert_ghyd_params(lambd=lambd, alpha_bar=alpha_bar, mu=mu, Sigma=Sigma, gamma=gamma, from_param='alpha_bar', to_param=to_param)

######## kernel functions ########

def gamma_kernel(x, shape=1, rate=1, div=1, func=None):
    pdf = np.where(x > 0, stats.gamma.pdf(x=x, a=shape, scale=1/rate) / div, 0)
    if func:
        return func(pdf)
    else:
        return pdf
