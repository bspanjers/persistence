# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize
from scipy.stats import chi2
import pandas as pd
from typing import List, Union, Dict

import warnings

warnings.filterwarnings("ignore")

def duration_test(
    violations: Union[List[int], np.ndarray, pd.Series, pd.DataFrame],
    conf_level: float = 0.95, a: float = 0.1,
) -> Dict:

    """Perform the Christoffersen and Pelletier Test (2004) called Duration Test.
        The main objective is to know if the VaR model responds quickly to market movements
         in order to do not form volatility clusters.
        Duration is time betwenn violations of VaR.
        This test verifies if violations has no memory i.e. should be independent.

        Parameters:
            violations (series): series of violations of VaR
            conf_level (float):  test confidence level
        Returns:
            answer (dict):       statistics and decision of the test
    """
    typeok = False
    if isinstance(violations, pd.core.series.Series) or isinstance(
        violations, pd.core.frame.DataFrame
    ):
        violations = violations.values.flatten()
        typeok = True
    elif isinstance(violations, np.ndarray):
        violations = violations.flatten()
        typeok = True
    elif isinstance(violations, list):
        typeok = True
    if not typeok:
        raise ValueError("Input must be list, array, series or dataframe.")

    N = int(sum(violations))
    first_hit = violations[0]
    last_hit = violations[-1]

    duration = [i + 1 for i, x in enumerate(violations) if x == 1]

    D = np.diff(duration)

    TN = len(violations)
    C = np.zeros(len(D))

    if not duration or (D.shape[0] == 0 and len(duration) == 0):
        duration = [0]
        D = [0]
        N = 1

    if first_hit == 0:
        C = np.append(1, C)
        D = np.append(duration[0], D)  # days until first violation

    if last_hit == 0:
        C = np.append(C, 1)
        D = np.append(D, TN - duration[-1])

    else:
        N = len(D)

    def likDurationW(x, a, D, C, N):
        b = x
        lik = (
            C[0] * np.log(pweibull(D[0], a, b, survival=True))
            + (1 - C[0]) * dweibull(D[0], a, b, log=True)
            + sum(dweibull(D[1 : (N - 1)], a, b, log=True))
            + C[N - 1] * np.log(pweibull(D[N - 1], a, b, survival=True))
            + (1 - C[N - 1]) * dweibull(D[N - 1], a, b, log=True)
        )

        if np.isnan(lik) or np.isinf(lik):
            lik = 1e10
        else:
            lik = -lik
        return lik

    # When b=1 we get the exponential
    def dweibull(D, a, b, log=False):
        # density of Weibull
        pdf = b * np.log(a) + np.log(b) + (b - 1) * np.log(D) - (a * D) ** b
        if not log:
            pdf = np.exp(pdf)
        return pdf

    def pweibull(D, a, b, survival=False):
        # distribution of Weibull
        cdf = 1 - np.exp(-((a * D) ** b))
        if survival:
            cdf = 1 - cdf
        return cdf

    optimizedBetas = optimize.minimize(
        likDurationW, x0=[2], args=(a, D, C, N), method="L-BFGS-B", bounds=[(0.001, 10)]
    )

    print(optimizedBetas.message)

    b = optimizedBetas.x
    uLL = -likDurationW(b, a, D, C, N)
    rLL = -likDurationW(np.array([1]), a, D, C, N)
    LR = 2 * (uLL - rLL)
    LRp = 1 - chi2.cdf(LR, 1)

    H0 = "Duration Between Exceedances have no memory (Weibull b=1 = Exponential)"
    # i.e. whether we fail to reject the alternative in the LR test that b=1 (hence correct model)
    if LRp < (1 - conf_level):
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"

    answer = {
        "weibull exponential": b,
        "unrestricted log-likelihood": uLL,
        "restricted log-likelihood": rLL,
        "log-likelihood": LR,
        "log-likelihood ratio test statistic": LRp,
        "null hypothesis": H0,
        "decision": decision,
    }

    return answer
