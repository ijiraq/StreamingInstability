import os

import numpy
import pandas as pd
import seaborn as sns
from astropy.table import Table
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats, optimize

sns.set_theme()


DETECTIONS_FILE = "DES.cds"
DEEP_SURVEYS_FILE = "deep_surveys.csv"
B04_DETECTIONS_FILE = "B04-free-cla_m.detections-full"

i_c = 5.0
gmb = 1.0+1.0/6023600.0+1.0/408523.71+1.0/328900.56+1.0/3098708.0+1.0/1047.3486+1.0/3497.898+1.0/22902.98+1.0/19412.24+1.0/1.35e8

F14 = {'lower': [1.5-0.2, 0.38+0.05, 7.36+0.04, 6.9+0.1],
       'upper': [1.5+0.4, 0.38-0.09, 7.36-0.18, 6.9-0.2],
       'best': [1.5, 0.38, 7.36, 6.9]}
publfs = {'F14': [1.5, 0.38, 7.36, 6.9], 'A14': [1.32, 0.42, 7.36, 7.2+0.2]}


def fraser(H, a1, a2, Ho, Hb):
    """
    LF from Fraser et al. 2014
    :param H: h magnitude
    :param a1: log slope bright of break)
    :param a2: log slope faint of break)
    :param Ho: normalization of bright component
    :param Hb: normalization of faint component
    :return: N
    """
    #print(f"alpha1: {a1}, alpha2: {a2}, Ho: {Ho}, Hb:{Hb}")
    N = 10**(a1*(H-Ho))
    N[H > Hb] = 10**(a2*(H[H > Hb]-Ho)+(a1-a2)*(Hb-Ho))
    return N


def double_plaw(H, simga_23=0.68, a1=1.5, a2=0.55, R_eq=22.8):
    """
    Double powerlaw form from B14
    :param H:
    :param simga_23:
    :param a1:
    :param a2:
    :return: simga(H)
    """
    r = H + 10*numpy.log10(43.3)
    C = 10**((a2-a2)*(R_eq - 23))
    return (1+C)*simga_23/(10**(-a1*(r-23)) + C*10**(-a2*(r-23)))


def rolling_plaw(H, sigma_23, a1, a2):
    """
    Rolling power law form form Bernstein et al. 2004
    :param H: numpy array of H mags.
    :param sigma_23: normalization at R=23
    :param a1: bright slope.
    :param a2: faint slope.
    :return: N
    """
    r = H + 10*numpy.log10(44)
    return sigma_23 * 10**(a1*(r-23)+a2*(r-23)**2)



ln10 = numpy.log(10.)

rho = 1.0e12
col = {0.4: 'b', 0.5: 'g', 0.45: 'c'}
nu = 0.15


def variably_tapered(h, Dp, E, alpha_SI, beta_SI):
    """
    Exponentially tappered powerlaw size distribution re-expressed in H magnitude space.
    """
    return Dp*10.**(alpha_SI*3.*h/5.)*numpy.exp(-E*10.**(-beta_SI*3.*h/5.))


def variably_tapered_diff(h, Dp, E, alpha_SI, beta_SI):
    return (3./5.*ln10)*Dp*10.**(alpha_SI*3.*h/5.)*(alpha_SI+beta_SI*E*10.**(-beta_SI*3.*h/5.))*numpy.exp(-E*10.**(-beta_SI*3.*h/5.))


def log_variably_tapered(h, Dp, E, alpha_SI, beta_SI):
    return numpy.log10(variably_tapered(h, Dp, E, alpha_SI, beta_SI))


def log_variably_tapered_diff(h, Dp, E, alpha_SI, beta_SI):
    return numpy.log10(variably_tapered_diff(h, Dp, E, alpha_SI, beta_SI))


def poisson_range(k, prob):
    """
    Given an measured count rate, k, and an expectation of a poisson process return estimates of 'mu' that
    are consistent with 'k' being measured within probability prob.

    i.e. if we measure k objects compute a bunch of mu and return the range of mu that are consistent within prob.
    """
    # mu holds the  range of plausible estimates for actual mu
    mu = numpy.arange(max(0, k.min()-20*(k.min())**0.5), k.max()+20*(k.max())**0.5+10, 10+20*k.max()**0.5/100.)
    upper = stats.poisson.ppf(0.5-prob/2., mu)
    lower = stats.poisson.ppf(0.5+prob/2., mu)
    return numpy.interp(k, lower, mu), numpy.interp(k, upper, mu)


def plot():
    plt.style.use('PaperDoubleFig.mplstyle')
    #plt.clf()

    rep = './'
    out = './'

    # Load detection list with bias column from disk
    detection_file = os.path.join(rep, DETECTIONS_FILE)

    alld = Table.read(detection_file, format='ascii.cds')

    alld.sort('Hx')
    # weight data by the inverse of the bias.
    alld['weight'] = 1/alld['bias']

    # set some conditions to select sub-sets of the detection list
    cold = ((alld['ifree'] <= 4.) & (alld['a'] >= 42.4) & (alld['a'] <= 47.7) & (alld['cl'] == 'CLASSICAL'))
    alld['Hx'] += 0.25
    hlim = 7.0 + 0.25
    small = alld['Hx'] > hlim
    faint = small & cold
    bright = (~small) & cold
    print(numpy.median(alld['dist'][bright]))

    # plot the raw CDF without correcting for biases.
    yraw = numpy.cumsum(numpy.ones(len(alld[cold])))
    # Make a debiased CDF of the bright cold classical objects detections
    yd = numpy.cumsum(alld['weight'][bright])
    xd = alld['Hx'][bright]

    # now do the CDF of the full cold classical set but only
    # keep the faint ones (to plot this group differently.
    yf = numpy.cumsum(alld['weight'][cold])[alld['Hx'][cold] > hlim]
    xf = alld['Hx'][faint]

    # compute the poisson interval consistent with the number of
    # objects at each point in the cumulative distribution.
    y = numpy.cumsum(numpy.ones(len(alld[bright])))
    yl, yu = poisson_range(y, 0.95)
    # Turn the CDF determine poisson upper/lower bounds into
    # a differential
    yl[1:] -= yl[:-1]
    yu[1:] -= yu[:-1]
    # use the weights to scale the Differential poison bounded counts
    yu *= alld[bright]['weight']
    yl *= alld[bright]['weight']
    # Take a cumulative sum of the weighted differential
    yu = numpy.cumsum(yu)
    yl = numpy.cumsum(yl)
    # Insert a couple point at the start of the CDF for the '0'
    # detected sources brighter than the brightest object.
    # here we arbitrarily put this at H=5
    yu = numpy.insert(yu, 0, 3.65)
    yl = numpy.insert(yl, 0, 0)
    # And get a vector of 'H' values, same as the xd vector made above
    # but with that bright 'non-detection' point added at the start.
    Hx = numpy.insert(xd, 0, 5)
    df = pd.DataFrame({'Hx': Hx, 'yl': yl, 'yu': yu})

    # setup the plot using seaborn functions
    sns.set_theme(style='ticks')
    sns.set_color_codes(palette='colorblind')

    # plot the bright and faint CDFs with dashed line for the faint part.
    plt.gca().plot(xd, yd, '-', color='c', linewidth=2)
    plt.gca().plot(xf, yf, '--', color='c', linewidth=2)
    plt.gca().plot(alld['Hx'][cold], yraw, '--', color='c', alpha=0.5)
    #sns.lineplot(data=bright_cdf, x='Hx', y='yd', color='g', linewidth=2, zorder=10, ax=plt.gca())
    #sns.lineplot(data=bright_cdf, x='Hx', y='raw', color='k', linewidth=2, zorder=10, linestyle="--")
    #sns.lineplot(data=faint_cdf, x='Hx', y='yd', color='g', linewidth=4, zorder=10, alpha=0.5, linestyle=":")

    # Now shade the area between the upper and lower limits as determined using our poisson approximation
    plt.gca().fill_between(df['Hx'], df['yl'], df['yu'], color='c', alpha=0.4)

    # set plot configuration.
    plt.xlim(4., 13.)
    plt.ylim(1., 2.e6)
    plt.yscale('log')
    plt.xlabel(r'$H_r$')
    plt.ylabel('Cumulative number of cold classical KBOs')
    plt.gca().set_xticks([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    for i, tick in enumerate(plt.gca().xaxis.get_ticklabels()):
        if i % 2 != 0:
            tick.set_visible(False)

    # save to file
    plt.savefig(out+'des.pdf'.format(nu))
