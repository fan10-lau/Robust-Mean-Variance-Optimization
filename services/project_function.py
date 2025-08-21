from services.strategies import *


def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """

    TARGET_R = 0.005        # 0.5% per month, for example
    LOOKBACK = 60           # use the past 60 monthly obs
    Z = 1.96                # 95% confidence for box radius

    Strategy = OLS_RobustBox(target_return=TARGET_R, lookback=LOOKBACK, z_or_conf=Z, allow_short=False)
    x = Strategy.execute_strategy(periodReturns, periodFactRet)
    return x
