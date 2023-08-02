import numpy as np
import pandas as pd
from scipy.stats import norm

def RW_Calc(Input_String: str) -> str:

    """ This function calculates a risk weight for a given set of parameters

    Parameters:
        Segment (str): Possible values are 'Bank', 'Corporate', and 'Retail'
        PD (float): Probability of Default
        LGD (float): Loss Given Default
        m (float): Remaining maturity of the loan in years
        Large_Fin (str): If 'Y' the client is a Flag for Large Financial Institution, otherwise 'N'
        size (float): size of the client in MEUR, usually this is the client's turnover
        mortgage (str): If 'Y' the exposure is a mortgage loan, otherwise 'N'
        revolving (str): If 'Y' the exposure is a revolving loan, otherwise 'N'

    Returns:
        RW_Value (float): the calculated risk weight for the loan
   """
    
    Segment_s, PD_s, LGD_s, m_s, Large_Fin_s, size_s, mortgage_s, revolving_s = Input_String.split(",")

    Segment = (Segment_s.split("=")[1].strip()).replace("'", "")
    PD = float(eval(PD_s.split("=")[1].strip()))
    LGD = float(eval(LGD_s.split("=")[1].strip()))
    m = float(eval(m_s.split("=")[1].strip()))
    Large_Fin = (Large_Fin_s.split("=")[1].strip()).replace("'", "")
    size = float(eval(size_s.split("=")[1].strip()))
    mortgage = (mortgage_s.split("=")[1].strip()).replace("'", "")
    revolving = (revolving_s.split("=")[1].strip()).replace("'", "")
    
    
    pd_final = np.maximum(0.0003, PD)

    RW_Value = np.select(
        [
            (Segment == 'Bank') & (PD < 1),
            (Segment == 'Retail') & (PD < 1),
            (Segment == 'Corporate') & (PD < 1)
        ],
        [
            12.5 * 1.06 * RW_Bank(PD, LGD, m, Large_Fin, size),
            12.5 * 1.06 * RW_Retail(PD, LGD, mortgage, revolving),
            12.5 * 1.06 * RW_Corporate(PD, LGD, m, Large_Fin, size)
        ],
        default=np.nan
    )
    return RW_Value

def RW_Bank(PD, LGD, m, Large_Fin, size):
    k = np.where(Large_Fin == 'Y', 1.25, 1)
    size_final = 50
    m_final = np.minimum(5, m)
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    R = (0.12 * (1.0 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0))) + \
        (0.24 * (1.0 - (1 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0)))) - \
        (0.04 * (1.0 - (size_final - 5.0) / 45.0))
    MA = ((1 - 1.5 * b) ** -1) * (1 + (m_final - 2.5) * b)
    return (LGD * norm.cdf((1 - R) ** -0.5 * norm.ppf(PD) + (R / (1 - R)) ** 0.5 * norm.ppf(0.999)) - PD * LGD) * MA

def RW_Retail(PD, LGD, mortgage, revolving):
    R = np.where(mortgage == 'Y', 0.15, np.where(revolving == 'Y', 0.04,
                                                0.03 * (1 - np.exp(-35.0 * PD)) / (1 - np.exp(-35.0)) +
                                                0.16 * (1.0 - (1.0 - np.exp(-35.0 * PD)) / (1.0 - np.exp(-35.0)))))
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    MA = 1.0
    return (LGD * norm.cdf((1 - R) ** -0.5 * norm.ppf(PD) + (R / (1 - R)) ** 0.5 * norm.ppf(0.999)) - PD * LGD) * MA

def RW_Corporate(PD, LGD, m, Large_Fin, size):
    k = np.where(Large_Fin == 'Y', 1.25, 1)
    size_final = np.maximum(5.0, np.minimum(size, 50))
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    R = (0.12 * (1.0 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0))) + \
        (0.24 * (1.0 - (1 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0)))) - \
        (0.04 * (1.0 - (size_final - 5.0) / 45.0))
    MA = ((1 - 1.5 * b) ** -1) * (1 + (m - 2.5) * b)
    return (LGD * norm.cdf((1 - R) ** -0.5 * norm.ppf(PD) + (R / (1 - R)) ** 0.5 * norm.ppf(0.999)) - PD * LGD) * MA


# version 2

def RW_Calc2(Input_String: str) -> str:

    """ This function calculates a risk weight for a given set of parameters

    Parameters:
        Segment (str): Possible values are 'Bank', 'Corporate', and 'Retail'
        PD (float): Probability of Default
        LGD (float): Loss Given Default
        m (float): Remaining maturity of the loan in years
        Large_Fin (str): If 'Y' the client is a Flag for Large Financial Institution, otherwise 'N'
        size (float): size of the client in MEUR, usually this is the client's turnover
        mortgage (str): If 'Y' the exposure is a mortgage loan, otherwise 'N'
        revolving (str): If 'Y' the exposure is a revolving loan, otherwise 'N'

    Returns:
        RW_Value (float): the calculated risk weight for the loan
   """
    
    Segment_s, PD_s, LGD_s, m_s, Large_Fin_s, size_s, mortgage_s, revolving_s = Input_String.split(",")

    Segment = (Segment_s.split(":")[1].strip())
    PD = float(eval(PD_s.split(":")[1].strip()))
    LGD = float(eval(LGD_s.split(":")[1].strip()))
    m = float(eval(m_s.split(":")[1].strip()))
    Large_Fin = (Large_Fin_s.split(":")[1].strip())
    size = float(eval(size_s.split(":")[1].strip()))
    mortgage = (mortgage_s.split(":")[1].strip())
    revolving = (revolving_s.split(":")[1].strip())
    
    
    pd_final = np.maximum(0.0003, PD)

    RW_Value = np.select(
        [
            (Segment == 'Bank') & (PD < 1),
            (Segment == 'Retail') & (PD < 1),
            (Segment == 'Corporate') & (PD < 1)
        ],
        [
            12.5 * 1.06 * RW_Bank(PD, LGD, m, Large_Fin, size),
            12.5 * 1.06 * RW_Retail(PD, LGD, mortgage, revolving),
            12.5 * 1.06 * RW_Corporate(PD, LGD, m, Large_Fin, size)
        ],
        default=np.nan
    )
    return RW_Value

def RW_Bank(PD, LGD, m, Large_Fin, size):
    k = np.where(Large_Fin == 'Y', 1.25, 1)
    size_final = 50
    m_final = np.minimum(5, m)
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    R = (0.12 * (1.0 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0))) + \
        (0.24 * (1.0 - (1 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0)))) - \
        (0.04 * (1.0 - (size_final - 5.0) / 45.0))
    MA = ((1 - 1.5 * b) ** -1) * (1 + (m_final - 2.5) * b)
    return (LGD * norm.cdf((1 - R) ** -0.5 * norm.ppf(PD) + (R / (1 - R)) ** 0.5 * norm.ppf(0.999)) - PD * LGD) * MA

def RW_Retail(PD, LGD, mortgage, revolving):
    R = np.where(mortgage == 'Y', 0.15, np.where(revolving == 'Y', 0.04,
                                                0.03 * (1 - np.exp(-35.0 * PD)) / (1 - np.exp(-35.0)) +
                                                0.16 * (1.0 - (1.0 - np.exp(-35.0 * PD)) / (1.0 - np.exp(-35.0)))))
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    MA = 1.0
    return (LGD * norm.cdf((1 - R) ** -0.5 * norm.ppf(PD) + (R / (1 - R)) ** 0.5 * norm.ppf(0.999)) - PD * LGD) * MA

def RW_Corporate(PD, LGD, m, Large_Fin, size):
    k = np.where(Large_Fin == 'Y', 1.25, 1)
    size_final = np.maximum(5.0, np.minimum(size, 50))
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    R = (0.12 * (1.0 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0))) + \
        (0.24 * (1.0 - (1 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0)))) - \
        (0.04 * (1.0 - (size_final - 5.0) / 45.0))
    MA = ((1 - 1.5 * b) ** -1) * (1 + (m - 2.5) * b)
    return (LGD * norm.cdf((1 - R) ** -0.5 * norm.ppf(PD) + (R / (1 - R)) ** 0.5 * norm.ppf(0.999)) - PD * LGD) * MA



# version 3


def RW_Calc3(Segment, PD, LGD, m, Large_Fin, size, mortgage, revolving):
    pd_final = np.maximum(0.0003, PD)

    RW_Value = np.select(
        [
            (Segment == 'Bank') & (PD < 1),
            (Segment == 'Retail') & (PD < 1),
            (Segment == 'Corporate') & (PD < 1)
        ],
        [
            12.5 * 1.06 * RW_Bank(PD, LGD, m, Large_Fin, size),
            12.5 * 1.06 * RW_Retail(PD, LGD, mortgage, revolving),
            12.5 * 1.06 * RW_Corporate(PD, LGD, m, Large_Fin, size)
        ],
        default=np.nan
    )
    return RW_Value

def RW_Bank(PD, LGD, m, Large_Fin, size):
    k = np.where(Large_Fin == 'Y', 1.25, 1)
    size_final = 50
    m_final = np.minimum(5, m)
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    R = (0.12 * (1.0 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0))) + \
        (0.24 * (1.0 - (1 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0)))) - \
        (0.04 * (1.0 - (size_final - 5.0) / 45.0))
    MA = ((1 - 1.5 * b) ** -1) * (1 + (m_final - 2.5) * b)
    return (LGD * norm.cdf((1 - R) ** -0.5 * norm.ppf(PD) + (R / (1 - R)) ** 0.5 * norm.ppf(0.999)) - PD * LGD) * MA

def RW_Retail(PD, LGD, mortgage, revolving):
    R = np.where(mortgage == 'Y', 0.15, np.where(revolving == 'Y', 0.04,
                                                0.03 * (1 - np.exp(-35.0 * PD)) / (1 - np.exp(-35.0)) +
                                                0.16 * (1.0 - (1.0 - np.exp(-35.0 * PD)) / (1.0 - np.exp(-35.0)))))
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    MA = 1.0
    return (LGD * norm.cdf((1 - R) ** -0.5 * norm.ppf(PD) + (R / (1 - R)) ** 0.5 * norm.ppf(0.999)) - PD * LGD) * MA

def RW_Corporate(PD, LGD, m, Large_Fin, size):
    k = np.where(Large_Fin == 'Y', 1.25, 1)
    size_final = np.maximum(5.0, np.minimum(size, 50))
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    R = (0.12 * (1.0 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0))) + \
        (0.24 * (1.0 - (1 - np.exp(-50.0 * PD)) / (1.0 - np.exp(-50.0)))) - \
        (0.04 * (1.0 - (size_final - 5.0) / 45.0))
    MA = ((1 - 1.5 * b) ** -1) * (1 + (m - 2.5) * b)
    return (LGD * norm.cdf((1 - R) ** -0.5 * norm.ppf(PD) + (R / (1 - R)) ** 0.5 * norm.ppf(0.999)) - PD * LGD) * MA