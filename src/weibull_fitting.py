from lifelines import WeibullFitter
import pandas as pd
import numpy as np
from scipy.optimize import minimize, basinhopping

rng = np.random.default_rng()


def generate_weibull(n, data_cutoff, scale, shape, const=0, gauss_noise=True):
    """Generate data frame of Time and Survival that follows
    c + (1-c) * Weibull(shape, scale).

    Parameters
    ----------
    n : int
        Number of patients.
    data_cutoff : float
        Data cutoff time in months.
    scale : float
        Weibull scale parameter.
    shape : float
        Weibull shape parameter.
    const : float
        const% have long lasting effect.
    gauss_noise : bool
        If true, add gaussian noise to time to event (defalut is True).

    Returns
    -------
    pandas.DataFrame
        Data frame of Time and Survival.

    """

    const_n = int(n * const / 100)
    s = np.linspace(const + 100 / n, 100, n - const_n)
    t = scale * (-np.log((s - const) / (100 - const)))**(1 / shape)
    if gauss_noise:
        t = t + rng.normal(0, 0.5, s.size)
    # scale time to fit [0, data_cutoff]
    # if t > data_cutoff, convert to data_cutoff
    t = np.where(t > data_cutoff, data_cutoff, t)
    t = np.where(t < 0, 0, t)  # if t < 0, convert it to 0
    t = np.hstack((t, np.repeat(data_cutoff, const_n)))
    df = pd.DataFrame(
        {'Time': np.sort(t)[::-1], 'Survival': np.linspace(0, 100 - 100 / n, n)})
    return df


def weibull_from_ipd(ipd, N):
    """ Get weibull survivual data from individual patient data.

    Parameters
    ----------
    ipd : pandas.DataFrame
        Dataframe consisting of time and event.
    N : type
        Number of patients.

    Returns
    -------
    pandas.DataFrame
        Smoothe Weibull-fitted data of time and survival.

    """
    wbf = WeibullFitter()
    wbf.fit(ipd['Time'], ipd['Event'])
    data = get_weibull_survival_dataframe(wbf.rho_, wbf.lambda_, N)
    return data


def weibull_param_from_ipd(ipd):
    wbf = WeibullFitter()
    wbf.fit(ipd['Time'], ipd['Event'])
    return (wbf.rho_, wbf.lambda_)


def sse(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred))


def fit_weibull3_survival(params: tuple, dat: pd.DataFrame) -> float:
    """Objective function for fitting to 3-parameter Weibull survival function.

    Args:
        params (tuple): tuple of (shape, scale, cure rate)
        dat (pd.DataFrame): survival data to fit to

    Returns:
        float: sum of squared errors
    """    
    # make sure input data survival is in ascending order
    # a: shape, b: scale, c: cure rate (%)
    a, b, c = params
    s = dat['Survival'].values
    t_true = dat['Time'].values
    t = b * (-np.log((s - c) / (100 - c)))**(1 / a)
    return sse(t_true, t)


def fit_weibull2_survival(params, args):
    # make sure input data survival is in ascending order
    # a: shape, b: scale
    a, b = params
    dat = args
    s = dat['Survival'].values
    t_true = dat['Time'].values
    t = b * (-np.log(s / 100))**(1 / a)
    return sse(t_true, t)


def get_weibull_survival_dataframe(a, b, N):
    s = np.linspace(100 / N, 100, N)
    t = b * (-np.log(s / 100))**(1 / a)
    data = pd.DataFrame({'Time': t, 'Survival': s - 100 / N}).round(5)
    return data


def weibull3_pdf(t, a: float, b: float, c: float):
    """Returns 3-parameter Weibull PDF.

    Args:
        t (iterable): time points
        a (float): Weibull shape parameter
        b (float): Weibull scale parameter
        c (float): cure rate (%)

    Returns:
        float or iterable: PDF value(s) of 3-parameter Weibull PDF
    """    
    c = c/100
    return (a * (1-c)/t) * ((t/b)**(a-1)) * np.exp(-((t/b)**a))


def weibull3_survival(t, a: float, b: float, c: float):
    """Returns 3-parameter Weibull survival.

    Args:
        t (iterable): time points
        a (float): Weibull shape parameter
        b (float): Weibull scale parameter
        c (float): cure rate (%)

    Returns:
        float or iterable: survival value(s) of 3-parameter Weibull PDF
    """
    c = c/100
    return c + (1-c) * np.exp(-((t/b)**a))


def weibull3_from_digitized(df: pd.DataFrame, N: int, tmax: float) -> pd.DataFrame:
    """Fit the survival data to a 3-parameter Weibull survival function
    and return the Weibull-fitted survival data

    Args:
        df (pd.DataFrame): survival data with Time and Survival columns
        N (int): number of patients
        tmax (float): max follow-up time to use

    Returns:
        pd.DataFrame: Weibull fitted survival data
    """    
    params0 = np.array([1.0, 5.0, 5.0])  # initial guess (a, b)
    bnds = [(0.01, 5), (0.01, 100), (0, 100)]  # parameter bounds
    res = minimize(fit_weibull3_survival, params0,
                   args=(df[df['Time'] < tmax],),
                   method='COBYLA', bounds=bnds, options={'disp': True})
    a, b, c = res.x[0], res.x[1], res.x[2]
    data = generate_weibull(N, tmax, b, a, const=c, gauss_noise=False)
    return data


def weibull3_params_from_digitized(df: pd.DataFrame, tmax: float, n=100) -> tuple:
    """Fit the survival data to a 3-parameter Weibull survival function
    and return the fitted Weibull parameters

    Args:
        df (pd.DataFrame): survival data with Time and Survival columns
        tmax (float): max follow-up time to use

    Returns:
        tuple: Weibull parameters (shape, scale, cure rate)
    """
    
    params0 = np.array([1.0, 5.0, 5.0])   # initial guess (a, b)
    bnds = [(0.01, 5), (0.01, 100), (0, 100)]  # parameter bounds
    res = minimize(fit_weibull3_survival, params0,
                   args=(df[df['Time'] < tmax],),
                   method='COBYLA', bounds=bnds)
    a, b, c = res.x[0], res.x[1], res.x[2]
    return (a, b, c)


def weibull3_params_from_digitized2(df, tmax, n=100):
    params0 =  np.array([1.0, 5.0, 5.0])
    bnds = [(0.01, 5), (0.01, 100), (0, 100)]
    minimizer_kwargs = {'method': 'COBYLA', "args": (df[df['Time'] < tmax],), "bounds": bnds}
    res = basinhopping(fit_weibull3_survival, params0, 
                       minimizer_kwargs=minimizer_kwargs, niter=200, disp=True)
    a, b, c = res.x[0], res.x[1], res.x[2]
    return (a, b, c)
