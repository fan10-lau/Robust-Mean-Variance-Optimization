import numpy as np
from services.estimators import *
from services.optimization import *
from services.uncertainty import *

# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class


def equal_weight(periodReturns):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return:x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x


class HistoricalMeanVarianceOptimization:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:
        :return: x
        """
        factorReturns = None  # we are not using the factor returns
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        print(len(returns))
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values
        x = MVO(mu, Q)

        return x


class OLS_MVO:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q)
        return x


class OLS_RobustBox:
    """
    Phase 1 strategy: Robust MVO with a Box uncertainty set.
    """
    def __init__(self,
                 target_return: float,
                 lookback: int = 60,
                 z_or_conf: float = 1.96,       # 95% by default
                 allow_short: bool = False,
                 conf_is_level: bool = False):
        """
        z_or_conf: pass a z-score (1.96) or a confidence level (0.95 with conf_is_level=True)
        """
        if conf_is_level:
            z = Z_FROM_CL.get(z_or_conf, 1.96)
        else:
            z = z_or_conf

        self.R = target_return
        self.NumObs = lookback
        self.z = float(z)
        self.allow_short = allow_short

    def execute_strategy(self, periodReturns, factorReturns):
        # estimation window
        returns = periodReturns.iloc[-self.NumObs:, :]
        factRet = factorReturns.iloc[-self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        T = returns.shape[0]

        # Robust min-variance at target return using box penalty μ^T x − δ^T|x| ≥ R
        x = robust_mvo_box(mu, Q,
                           target_return=self.R,
                           T=T,
                           epsilon=self.z,
                           allow_short=self.allow_short)
        return x