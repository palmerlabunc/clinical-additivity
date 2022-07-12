import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import spearmanr



def interpolate(df, x='Time', y='Survival', kind='zero'):
    return interp1d(df[x], df[y], kind=kind, fill_value='extrapolate')


def populate_N_patients(ori_df, N):
    """Scale to make N patients from survival 0 to 100.

    Parameters
    ----------
    ori_df : pandas.DataFrame
        Original survival data.

    Returns
    -------
    pandas.DataFrame
        Survival data with N data points.

    """
    df = ori_df.copy()
    # add starting point (survival=100, time=0)
    df = df.append(pd.DataFrame(
        {'Survival': 100, 'Time': 0}, index=[df.shape[0]]))
    min_survival = df['Survival'].min()
    step = 100 / N

    f = interpolate(df, x='Survival', y='Time')  # survival -> time
    if min_survival <= 0:
        points = np.linspace(0, 100 - step, N)
        new_df = pd.DataFrame({'Time': f(points), 'Survival': points})
    else:
        existing_n = int(np.round((100 - min_survival) / 100 * N, 0))
        existing_points = np.linspace(
            100 * (N - existing_n) / N + step, 100 - step, existing_n)

        # pad patients from [0, min_survival]
        new_df = pd.DataFrame({'Time': df['Time'].max(),
                               'Survival': np.linspace(0, 100 * (N - existing_n) / N, N - existing_n)})
        new_df = new_df.append(pd.DataFrame({'Survival': existing_points,
                                             'Time': f(existing_points)}))
        new_df = new_df.round(
            {'Time': 5, 'Survival': int(np.ceil(-np.log10(step)))})
    assert new_df.shape[0] == N
    return new_df[['Time', 'Survival']].sort_values('Survival').reset_index(drop=True)


def fit_rho3(a, b, rho, ori_rho=None, seed=0):
    """ Shuffle data of two sorted dataset to make two dataset to have a desired Spearman correlation.
    Note that a and b should be have the same length.
    Modified from: https://www.mathworks.com/help/stats/generate-correlated-data-using-rank-correlation.html

    Args:
        a (array_like): sorted dataset 1
        b (array_like): sorted dataset 2
        rho (float): desired spearman correlation coefficient
        ori_rho (float): internal argument for recursive part (default: None)
        seed (int): random generator seed

    Returns:
        tuple: tuple of shuffled datsets (np.ndarray)

    """
    if ori_rho is None:
        ori_rho = rho
    
    rng = np.random.default_rng(seed)
    
    n = len(a)
    pearson_r = 2 * np.sin(rho * np.pi / 6)
    rho_mat = np.array([[1, pearson_r], [pearson_r, 1]])
    size = rho_mat.shape[0]
    means = np.zeros(size)
    u = rng.multivariate_normal(means, rho_mat, size=n)
    i1 = np.argsort(u[:, 0])
    i2 = np.argsort(u[:, 1])
    x1, x2 = np.zeros(n), np.zeros(n)
    x1[i1] = a
    x2[i2] = b

    # check if desired rho is achieved
    result, _ = spearmanr(x1, x2)
    # recursive until reaches 2 decimal point accuracy
    if ori_rho - result > 0.01:  # aim for higher rho
        x1, x2 = fit_rho3(a, b, rho + 0.01, ori_rho=ori_rho, seed=seed)
    elif ori_rho - result < -0.01:  # aim for lower rho
        x1, x2 = fit_rho3(a, b, rho - 0.01, ori_rho=ori_rho, seed=seed)

    return (x1, x2)
