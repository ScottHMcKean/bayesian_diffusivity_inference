import cloudpickle
import numpy as np
import pandas as pd
import arviz as az
import aesara.tensor as at 
from aesara.tensor.random.op import RandomVariable
from pymc.aesaraf import floatX
from aesara.raise_op import Assert
from pymc.distributions import Continuous
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple
import pymc as pm

def quantile_loss(quantile, y, y_pred):
    "Quantile Loss Function for calculating diffusivity"
    error = y - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))


def linear_diffusivity(t, C=2.45, Deff=1):
    """
    Predict Linear Diffusivity, setup to use in scipy optimize
    x: list of parameters, x[0] = C, x[1] = Deff
    t: numpy array of times in seconds
    """
    if not isinstance(t, np.ndarray):
        t = np.array(t)

    return C * np.sqrt(Deff * t)


def linear_diff_quant_loss(Deff, t, y, quantile=0.95, C=2.45):
    """
    scipy function for minimization
    """
    y_pred = linear_diffusivity(t, C, Deff)
    return quantile_loss(quantile, y, y_pred)


def fit_ql_diffusivity(t, y, quantile=0.95, C=2.45, deff_bounds=[0.1, 20]):
    """
    Function factory for gp_minimize
    """
    assert isinstance(t, np.ndarray)
    assert isinstance(y, np.ndarray)
    res_sp = minimize_scalar(
        linear_diff_quant_loss,
        bounds=deff_bounds,
        args=(t, y, quantile, C),
        method="bounded",
    )
    mql = linear_diff_quant_loss(res_sp.x, t, y, quantile, C)
    return (res_sp.x, mql)


def calculate_lsq_diffusivities(
    dist_grp: pd.DataFrame, quantile: float = 0.95
) -> pd.DataFrame:
    """Calculate vertical, horizontal, and radial diffusivity
    Requires at least 10 microseismic events for diffusivity calculations
    Have individual values for C because we are likely going to use a linear value
    for linear diffusivity and a spherical value for radial...

    Parameters
    ----------
    dist_grp : pd.DataFrame
        Distance group  from 'calculate_distances'

    Returns
    -------
    pd.DataFrame
        Diffusivity metrics, rolled up per well/stage
    """
    if dist_grp.shape[0] < 5:
        return None
    
    # fit lots of diffusivities using a consistent quantile
    uwi = dist_grp.WellID.iloc[0]
    stage = dist_grp.Stage.iloc[0]
    t = dist_grp['t_start_s'].values
    pos_t = t>0
    t = t[pos_t]
    y_x = dist_grp['dx_strike_m'].abs().values
    y_x = y_x[pos_t]
    y_y = dist_grp['dy_strike_m'].abs().values
    y_y = y_y[pos_t]
    y_z = dist_grp['dz_strike_m'].abs().values
    y_z = y_z[pos_t]

    strike_x_diff = fit_ql_diffusivity(
        t, dist_grp.dx_strike_m.values, quantile=0.95, C=2.45
    )
    strike_y_diff = fit_ql_diffusivity(
        t, dist_grp.dy_strike_m.values, quantile=0.95, C=2.45
    )
    strike_z_diff = fit_ql_diffusivity(
        t, dist_grp.dz_strike_m.values, quantile=0.95, C=2.45
    )

    # output series
    return pd.Series(
        {
            "WellID": dist_grp.iloc[0].WellID,
            "Stage": dist_grp.iloc[0].Stage,
            "n_events": dist_grp.shape[0],
            "mean_Do_x": strike_x_diff[0],
            "mean_Do_y": strike_y_diff[0],
            "mean_Do_z": strike_z_diff[0],
            "loss_Do_x": strike_x_diff[1],
            "loss_Do_y": strike_y_diff[1],
            "loss_Do_z": strike_z_diff[1],
        }
    )

def within_ellipsoid(
    points: np.ndarray,
    ellipsoid_rxryrz=[500, 100, 100000]
):
    """Check if an array of points in within an axis oriented ellipsoid
    Points must be rotated and centered to the axis first
    Applies a whitening matrix to filter points
    See: https://math.stackexchange.com/q/1793612

    Args:
        points (np.ndarray): Array of points, shape=(nx3)
        ellipsoid_xyz (list, optional): Half-length of ellipsoid length, width, height (xyz). Defaults to [1000,500,1000].
    """
    assert isinstance(points, np.ndarray)
    # setup ellipsoid
    ell_dim = np.diag(ellipsoid_rxryrz) ** 2
    # axis oriented principal axes
    ell_vec = np.diag([1,1,1])
    # make whitening matrix
    whitening = np.dot(np.diag(np.diag(ell_dim) ** (-0.5)), np.transpose(ell_vec))
    # check points within ellipsoid
    in_ellipsoid = np.apply_along_axis(
        lambda x: np.linalg.norm(np.dot(whitening, x)) < 1, 1, points
    )
    return in_ellipsoid

class AsymmetricLaplaceQuantileRV(RandomVariable):
    name = "asymmetriclaplace"
    ndim_supp = 0
    ndims_params: List[int] = [0, 0, 0]
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("AsymmetricLaplaceQuantile", "\\operatorname{AsymmetricLaplaceQuantile}")

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        mu: np.ndarray,
        sigma: np.ndarray,
        p: np.ndarray,
        size: Tuple[int, ...] = None,
    ) -> np.ndarray:
        u = rng.uniform(size=size)
        non_positive_x = mu + sigma/(1-p) * np.log(u/p)
        positive_x = mu - sigma/p * np.log((1-u)/(1-p))
        draws = non_positive_x * (u <= p) + positive_x * (u > p)
        return np.asarray(draws)

def assert_negative_support(var, label, distname):
    msg = f"The variable specified for {label} has negative support for {distname}, "
    msg += "likely making it unsuitable for this parameter."
    return Assert(msg)(var, at.all(at.ge(var, 0.0)))

# Module for ML and Bayesian Diffusivity Estimate
class AsymmetricLaplaceQuantile(Continuous):
    r"""
    Asymmetric-Laplace log-likelihood intended for quantile regression.

    The pdf of this distribution is

    .. math::
        {f(x;m,\lambda,p)  = \frac{p(1-p)}{\lambda}
            \begin{cases}
            \exp \left( -((p-1)/\lambda)(x-m) \right) & \text{if }x \leq m
            \\[4pt]
            \exp ( -(p/\lambda)(x-m) )  & \text{if }x > m
            \end{cases}
        }

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \frac{1-2p}{p(1-p)}b`
    Variance  :math:`\frac{1-2p+2p^2}{p(1-p)}b^2`
    ========  ========================

    Parameters
    ----------
    sigma: float
        Scale parameter (b > 0)
    mu: float
        Location parameter (b E R)
    p: float
        Percentile (i.e. quantile) parameter (0 < p < 1)


    See Also:
    --------
    `Reference <https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution>`_
    """
    rv_op = AsymmetricLaplaceQuantileRV()
    
    @classmethod
    def dist(cls, sigma, p, mu, *args, **kwargs):
        sigma = at.as_tensor_variable(floatX(sigma))
        p = at.as_tensor_variable(floatX(p))
        mu = mu = at.as_tensor_variable(floatX(mu))

        assert_negative_support(sigma, "sigma", "AsymmetricLaplace")
        assert_negative_support(p, "p", "AsymmetricLaplace")

        return super().dist([sigma, p, mu], *args, **kwargs)

    def moment(rv, size, sigma, p, mu):
        mean = mu + (1-2*p)/(p*(1-p))*sigma

        if not rv_size_is_none(size):
            mean = at.full(size, mean)
        return mean

    def variance(rv, size, sigma, p):
        var = (1-2*p+2*p**2)/(p**2*(1-p)**2)*sigma**2

        if not rv_size_is_none(size):
            var = at.full(size, var)
        return var

    def logp(value, sigma, p, mu):
        """
        Calculate log-probability of Asymmetric-Laplace distribution at specified value.
        Parameters
        ----------
        value : tensor_like of float
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.
        Returns
        -------
        TensorVariable
        """
        res = at.log(at.where(
            value <= mu, 
            at.exp(-((p-1)/sigma)*(value-mu)), 
            at.exp(-(p/sigma)*(value-mu))
            ))

        return check_parameters(res, 0 < sigma, 0 < p, msg="sigma > 0, p > 0")

def calc_tilda_c(perm_compliance:float, pressure: float) -> float:
    """Hummel and Muller (2012) Tilda C calculation for diffusivity
    estimation. We normalize the pressure pulse and permeability 
    compliance to 1 Pa to provide a consistent dimensionless analysis.

    Args:
        perm_compliance (float): permeability compliance in GPa
        pressure (float): average pressure pulse in MPa

    Returns:
        float: tilda C calculation
    """
    k = perm_compliance*pressure/1000
    upper = np.sqrt(np.exp(k)*(2*k**-1 - k**-2) - k**-1 + k**-2)
    lower = 1 - (np.exp(k)*k**-1 - 1 - k**-1)*(np.exp(k)-1)**-1
    return upper / lower


def calc_k_quantile(Po: np.ndarray, k: np.ndarray, max_quantile=0.95):
    """Calculate the quantile of k based on the Po and k values
    The quantile is independent of diffusivity
    The result is scaled to a maximum quantile, which represents the triggering front

    Args:
        Po (np.ndarray):Permeability Compliance (GPa)
        k (np.ndarray): Net Injection Pressure (MPa)
        max_quantile (float): Maximum quantile for the regression. Defaults to 0.95.

    Returns:
        np.ndarrray: Quantiles
    """
    Do = 1
    k_tilda = k * Po / 1000
    d = Do * np.exp(k_tilda)
    r_tf = np.sqrt(6*d*10)
    tilda_c = calc_tilda_c(k, Po)
    r_fd = tilda_c * np.sqrt(Do * 10)
    quantile = r_fd/r_tf*max_quantile
    np.minimum(quantile, 1)
    return quantile


def within_diffusivity_envelope(dist_grp, diffusivities, model='nonlinear'):
    """Check for events within the diffusivity ellipsoid, using
    three on-strike dimensions (lateral, perpendicular, and vertical)
    This assumes the dimensions are independent.

    Args:
        dist_grp (pd.DataFrame): Single stage slice of distances dataframe
        diffusivities (pd.DataFrame): Bayes diffusivity 
        fracturing_domain (bool, optional). Use fracturing_domain instead of triggering front? Defaults to False. 

    Returns:
        pd.DataFrame: dist_grp with classifications
    """
    
    assert model in ['linear','nonlinear','maximum_likelihood']

    stg_diff = (diffusivities
            .query("WellID == @dist_grp.WellID.iloc[0]")
            .query("Stage == @dist_grp.Stage.iloc[0]")
            .query("model == @model")
            )

    if stg_diff.shape[0] == 0:
        return None

    t = dist_grp.t_start_s.values

    if model == 'maximum_likelihood':
        x_diff_boundary = linear_diffusivity(t, Deff=stg_diff['mean_Do_x'].iloc[0])
        y_diff_boundary = linear_diffusivity(t, Deff=stg_diff['mean_Do_y'].iloc[0])
        z_diff_boundary = linear_diffusivity(t, Deff=stg_diff['mean_Do_z'].iloc[0])
    else:
        x_diff_boundary = linear_diffusivity(t, Deff=stg_diff['hdi_97%_Do_x'].iloc[0])
        y_diff_boundary = linear_diffusivity(t, Deff=stg_diff['hdi_97%_Do_y'].iloc[0])
        z_diff_boundary = linear_diffusivity(t, Deff=stg_diff['hdi_97%_Do_z'].iloc[0])

    dist_grp['in_diff_ellipsoid'] = (
        (dist_grp.dx_strike_m.abs() <= x_diff_boundary) &
        (dist_grp.dy_strike_m.abs() <= y_diff_boundary) &
        (dist_grp.dz_strike_m.abs() <= z_diff_boundary)
    )
    return dist_grp

def make_failed_linear_summary(dist_grp):
    uwi = dist_grp.WellID.iloc[0]
    stage = dist_grp.Stage.iloc[0]
    n = dist_grp.shape[0]

    return pd.DataFrame({
        'mean': {'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'sd': {'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'hdi_3%': {'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'hdi_97%': {'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'mcse_mean': {'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'mcse_sd': {'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'ess_bulk': {'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'ess_tail': {'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'r_hat': {'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'uwi': {'Do_x': uwi, 'Do_y': uwi, 'Do_z': uwi},
        'stage': {'Do_x': stage, 'Do_y': stage, 'Do_z': stage},
        'n': {'Do_x': n, 'Do_y': n, 'Do_z': n},
        'model': {'Do_x': 'linear', 'Do_y': 'linear', 'Do_z': 'linear'},
        'converged': {'Do_x': False, 'Do_y': False, 'Do_z': False}}
        )

def make_failed_nonlinear_summary(dist_grp):
    uwi = dist_grp.WellID.iloc[0]
    stage = dist_grp.Stage.iloc[0]
    n = dist_grp.shape[0]

    return pd.DataFrame({
        'mean': {'Po': np.nan, 'k': np.nan, 'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'sd': {'Po': np.nan, 'k': np.nan, 'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'hdi_3%': {'Po': np.nan, 'k': np.nan, 'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'hdi_97%': {'Po': np.nan, 'k': np.nan, 'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'mcse_mean': {'Po': np.nan, 'k': np.nan, 'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'mcse_sd': {'Po': np.nan, 'k': np.nan, 'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'ess_bulk':{'Po': np.nan, 'k': np.nan, 'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'ess_tail':{'Po': np.nan, 'k': np.nan, 'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'r_hat':{'Po': np.nan, 'k': np.nan, 'Do_x': np.nan, 'Do_y': np.nan, 'Do_z': np.nan},
        'uwi': {'Po': uwi, 'k': uwi, 'Do_x': uwi, 'Do_y': uwi, 'Do_z': uwi},
        'stage': {'Po': stage, 'k': stage, 'Do_x': stage, 'Do_y': stage, 'Do_z': stage},
        'n': {'Po': n, 'k': n, 'Do_x': n, 'Do_y': n, 'Do_z': n},
        'model': {'Po': 'nonlinear', 'k': 'nonlinear', 'Do_x': 'nonlinear', 'Do_y': 'nonlinear', 'Do_z': 'nonlinear'},
        'converged': {'Po': False, 'k': False, 'Do_x': False, 'Do_y': False, 'Do_z': False}}
        )


def calculate_linear_bayesian_diffusivities(dist_grp: pd.DataFrame, params: Dict) -> bool:
    """Calculate onstrike lateral, vertical, and perpendicular diffusivity
    Requires at least n microseismic events for diffusivity calculations


    Args:
        dist_grp (pd.DataFrame): Distance group (one stage in a groupby apply)
        params (Dict): Parameters including prior specifications, container, and quantile GAM model

    Returns:
        bool: Whether the run was succesfully executed or not
    """
    if dist_grp.shape[0] < params['min_pts']:
        return None

    uwi = dist_grp.WellID.iloc[0]
    stage = dist_grp.Stage.iloc[0]
    t = dist_grp['t_start_s'].values
    pos_t = t>0
    t = t[pos_t]
    y_x = dist_grp['dx_strike_m'].abs().values
    y_x = y_x[pos_t]
    y_y = dist_grp['dy_strike_m'].abs().values
    y_y = y_y[pos_t]
    y_z = dist_grp['dz_strike_m'].abs().values
    y_z = y_z[pos_t]

    file_prefix = str(dist_grp.WellID.iloc[0]) + "_" +str(dist_grp.Stage.iloc[0])

    try:
        with pm.Model() as model:
            do_x_prior = pm.Gamma('Do_x', mu=params['do_x_mu'], sigma=params['do_x_sigma'])
            do_y_prior = pm.Gamma('Do_y', mu=params['do_y_mu'], sigma=params['do_y_sigma'])
            do_z_prior = pm.Gamma('Do_z', mu=params['do_z_mu'], sigma=params['do_z_sigma'])
            triggering_front_x = AsymmetricLaplaceQuantile('TF_x', sigma=1, p=0.95, mu=np.sqrt(6*do_x_prior*t), observed=y_x)
            triggering_front_y = AsymmetricLaplaceQuantile('TF_y', sigma=1, p=0.95, mu=np.sqrt(6*do_y_prior*t), observed=y_y)
            triggering_front_z = AsymmetricLaplaceQuantile('TF_z', sigma=1, p=0.95, mu=np.sqrt(6*do_z_prior*t), observed=y_z)
        
        with model:
            step = pm.NUTS(target_accept=params['nuts_target_accept'])
            trace = pm.sample(int(params['total_draws']/params['n_chains']), step=step, return_inferencedata=True, cores=1, chains=params['n_chains'], tune=params['burn_in'])

        summary = (az.summary(trace)
            .assign(uwi=uwi)
            .assign(stage=stage)
            .assign(n=dist_grp.shape[0])
            .assign(model='linear')
            .assign(converged=True)
            )
    except Exception as e:
        print(e)
        summary = make_failed_linear_summary(dist_grp)
        summary.to_parquet("outputs/" + file_prefix+"_linear.parquet")
        return False
    
    summary = (az.summary(trace)
        .assign(uwi=uwi)
        .assign(stage=stage)
        .assign(n=dist_grp.shape[0])
        .assign(model='linear')
        .assign(converged=True)
        )

    summary.to_parquet("outputs/" + file_prefix+"_linear.parquet")
    
    with open("outputs/" + file_prefix+"_lineartrace.pkl", 'wb') as f:
        cloudpickle.dump(trace, f)
    
    with open("outputs/" + file_prefix+"_linearmodel.pkl", 'wb') as f:
        cloudpickle.dump(model, f)

    return True

def calculate_non_linear_bayesian_diffusivities(dist_grp: pd.DataFrame, params: Dict) -> bool:
    """Calculate onstrike lateral, vertical, and perpendicular diffusivity
    Requires at least n microseismic events for diffusivity calculations
    Enforces that time must be positive
    Saves files locally to outputs

    Args:
        dist_grp (pd.DataFrame): Distance group (one stage in a groupby apply)
        params (Dict): Parameters including prior specifications

    Returns:
        bool: Whether the run was succesfully executed or not
    """
    if dist_grp.shape[0] < params['min_pts']:
        return None

    uwi = dist_grp.WellID.iloc[0]
    stage = dist_grp.Stage.iloc[0]
    t = dist_grp['t_start_s'].values
    pos_t = t>0
    t = t[pos_t]
    y_x = dist_grp['dx_strike_m'].abs().values
    y_x = y_x[pos_t]
    y_y = dist_grp['dy_strike_m'].abs().values
    y_y = y_y[pos_t]
    y_z = dist_grp['dz_strike_m'].abs().values
    y_z = y_z[pos_t]
    
    file_prefix = str(dist_grp.WellID.iloc[0]) + "_" +str(dist_grp.Stage.iloc[0])

    try:
        with pm.Model() as model:
            k_prior = pm.Gamma('k', mu=params['k_mu'], sigma=params['k_sigma'])
            p_prior = pm.Normal('Po', mu=params['po_mu'], sigma=params['po_sigma'])
            c_tilda = calc_tilda_c(k_prior, p_prior)
            quantile = calc_k_quantile(Po=p_prior.eval(), k=k_prior.eval())
            do_x_prior = pm.Gamma('Do_x', mu=params['do_x_mu'], sigma=params['do_x_sigma'])
            do_y_prior = pm.Gamma('Do_y', mu=params['do_y_mu'], sigma=params['do_y_sigma'])
            do_z_prior = pm.Gamma('Do_z', mu=params['do_z_mu'], sigma=params['do_z_sigma'])
            fracture_domain_x = AsymmetricLaplaceQuantile('FD_x', sigma=1, p=quantile, mu=c_tilda*np.sqrt(do_x_prior*t), observed=y_x)
            triggering_front_y = AsymmetricLaplaceQuantile('TF_y', sigma=1, p=0.95, mu=np.sqrt(6*do_y_prior*t), observed=y_y)
            fracture_domain_z = AsymmetricLaplaceQuantile('FD_z', sigma=1, p=quantile, mu=c_tilda*np.sqrt(do_z_prior*t), observed=y_z)
            
        with model:
            step = pm.NUTS(target_accept=params['nuts_target_accept'])
            trace = pm.sample(int(params['total_draws']/params['n_chains']), step=step, return_inferencedata=True, cores=1, chains=params['n_chains'], tune=params['burn_in'])
    except Exception as e:
        print(e)
        summary = make_failed_nonlinear_summary(dist_grp)
        summary.to_parquet("outputs/" + file_prefix+"_nonlinear.parquet")
        return False

    summary = (az.summary(trace)
        .assign(uwi=uwi)
        .assign(stage=stage)
        .assign(n=dist_grp.shape[0])
        .assign(model='nonlinear')
        .assign(converged=True)
        )
        
    summary.to_parquet("outputs/" + file_prefix+"_nonlinear.parquet")
    
    with open("outputs/" + file_prefix+"_nonlineartrace.pkl", 'wb') as f:
        cloudpickle.dump(trace, f)
    
    with open("outputs/" + file_prefix+"_nonlinearmodel.pkl", 'wb') as f:
        cloudpickle.dump(model, f)

    return True

