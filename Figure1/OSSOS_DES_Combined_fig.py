import os

import numpy
import pandas as pd
import seaborn as sns
from astropy.table import Table, vstack
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
publfs = {'F14': [1.5, 0.38, 7.36, 6.9], 'A14': [1.32, 0.42, 7.36, 7.2]}


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


def double_plaw(H, simga_23=0.68, a1=1.36, a2=0.38, R_eq=22.8):
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
    detection_file = os.path.join(rep, "All_Surveys_v11-free-cla_m.detections-full")

    ossos = Table.read(detection_file, format='ascii')

    ossos.sort('Hx')
    # weight data by the inverse of the bias.
    ossos['weight'] = 1/ossos['bias']

    # set some conditions to select sub-sets of the detection list
    ossos_hlim = 8.3
    print(f"OSSOS H_r limit: {ossos_hlim:3.1f}")
    cold = ((ossos['ifree'] <= 4.) & (ossos['a'] >= 42.4) & (ossos['a'] <= 47.7) & (ossos['Hx'] < ossos_hlim))
    ossos = ossos[cold]
    # print(ossos['mag'].max())

    # Load detection list with bias column from disk
    detection_file = os.path.join(rep, DETECTIONS_FILE)

    des = Table.read(detection_file, format='ascii.cds')
    des = des[des['VRmag'] < 23.0]

    des.sort('Hx')

    # set some conditions to select sub-sets of the detection list
    des_hlim = 6.9
    des_hlim = ossos_hlim - 1.85 + 0.5

    print(f"DES H_VR limit: {des_hlim:3.1f}")
    # print(des[des['Hx'] < 5])
    cold = ((des['ifree'] <= 4.) & (des['a'] >= 42.4) & (des['a'] <= 47.7) & (des['cl'] == 'CLASSICAL') & (des['Hx'] < des_hlim))
    fcold = ((des['ifree'] <= 5.) & (des['dist'] >= 39) & (des['dist'] <= 48) & (des['cl'] == 'CLASSICAL') & (des['Hx'] < des_hlim))
    print(fcold.sum(), cold.sum())
    print(des[(fcold) & (~cold)])
    des = des[cold]
    # correct to go from VR calibrated against USNO-B red to r
    # Comparing the DES H_VR value to MBOSS H_R values for objects in common gives <H_R - H_VR> = -0.01
    #
    # and Jordi et al. (2005) [https://www.sdss.org/dr12/algorithms/sdssubvritransform/] give
    #
    # r-R   =     (0.267 ± 0.005)*(V-R)  + (0.088 ± 0.003) if V-R <= 0.93
    # r-R   =     (0.77 ± 0.04)*(V-R)    - (0.37 ± 0.04)   if V-R >  0.93
    #
    # Looking at Peixinho <V-R> = 0.65 or so [Figure 3 from Wes resent ColOSSOS paper]
    # Using the MBOSS VR colors for the DES objects in MBOSS gives <V-R> = 0.59 and thus r-R = 0.267*0.59 + 0.088 = 0.25
    # and then subtract one to go from H_DES to
    des['Hx'] += 0.25
    # des['bias'] *= 0.5
    # in the zone where ossos and des overlap we have 2 CDFs.  Give each weight based on their relative bias factor
    # total_bias = ossos['bias'][ossos['Hx'] < des_hlim].mean()+des['bias'].mean()
    # ossos_weight = 4*ossos['bias'][ossos['Hx'] < des_hlim].mean()/total_bias
    # des_weight = 1/des['bias']
    #des_weight = 4*des['bias'].mean()/total_bias
    # print(f"OSSOS and DES combined with OSSOS weighted as {ossos_weight/2:3.1f} and DES weighted as {des_weight/2:3.1f} for H_VR < {des_hlim:3.1f}")
    # ossos['bias'][ossos['Hx'] < des_hlim] *= ossos_weight
    # des['bias'] *= des_weight
    # weight data by the inverse of the bias.
    des['weight'] = 1/des['bias']
    ossos['weight'] = 1/ossos['bias']
    print(ossos['weight'][ossos['Hx']<des_hlim].sum(), des['weight'].sum())
    # alld = vstack([ossos, des])
    alld = des
    alld.sort('Hx')
    print(numpy.median(alld['dist']))

    # plot the raw CDF without correcting for biases.
    yraw = numpy.cumsum(numpy.ones(len(alld)))
    # Make a debiased CDF of the bright cold classical objects detections
    yd = numpy.cumsum(alld['weight'])
    xd = alld['Hx']

    # compute the poisson interval consistent with the number of
    # objects at each point in the cumulative distribution.
    y = numpy.cumsum(numpy.ones(len(alld)))
    yl, yu = poisson_range(y, 0.95)
    # Turn the CDF determine poisson upper/lower bounds into
    # a differential
    yl[1:] -= yl[:-1]
    yu[1:] -= yu[:-1]
    # use the weights to scale the Differential poison bounded counts
    yu *= alld['weight']
    yl *= alld['weight']
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
    # plt.gca().plot(alld['Hx'], yraw, '--', color='c', alpha=0.5)
    # Now shade the area between the upper and lower limits as determined using our poisson approximation
    plt.gca().fill_between(df['Hx'], df['yl'], df['yu'], color='c', alpha=0.4)

    # set plot configuration.
    # CFEPS
    # plt.plot([8-0.9], [3900], 'sr')
    # Table 4 from Adam's et al.
    # plt.plot([5.15, 5.25, 6.25, 7.45, 7.75], [4, 19, 200, 3400, 4500], 'sk')
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
    plt.savefig(out+'combined.pdf'.format(nu))
