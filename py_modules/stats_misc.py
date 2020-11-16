
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats.distributions import  t
import pandas as pd
import csv


def curve_fit_ci(popt,pcov,n):
    
    alpha = 0.05
    p = len(popt)
    dof = max(0,n-p)
    tval = t.ppf(1.0-alpha/2., dof)

    lower = []
    upper = []
    for i,coef,var in zip(range(p),popt, np.diag(pcov)):
        sigma = var**0.5
        print('p{0}: {1} [{2}  {3}]'.format(i, coef,
                                  coef - sigma/np.sqrt(n)*tval,
                                  coef + sigma/np.sqrt(n)*tval))
        lower.append(coef - sigma/np.sqrt(n)*tval)
        upper.append(coef + sigma/np.sqrt(n)*tval)

    return lower,upper

def save_summary(fs,filename="stats_summary.txt",output=1):
    if output:
        print(fs)
    
    lines = fs.as_text()

    with open(filename, 'w') as writeFile:
        writeFile.writelines([lines])
    
    writeFile.close()
