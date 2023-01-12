import os

import numpy
import pandas as pd
import seaborn as sns
from astropy import units
from astropy.coordinates import SkyCoord, HeliocentricTrueEcliptic
from astropy.table import Table
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats, optimize

sns.set_theme()


DETECTIONS_FILE = "All_Surveys_v11-free-cla_m.detections-full"
DEEP_SURVEYS_FILE = "deep_surveys.csv"
B04_DETECTIONS_FILE = "B04-free-cla_m.detections-full"

i_c = 5.0
gmb = 1.0+1.0/6023600.0+1.0/408523.71+1.0/328900.56+1.0/3098708.0+1.0/1047.3486+1.0/3497.898+1.0/22902.98+1.0/19412.24+1.0/1.35e8

F14 = {'lower': [1.5-0.2, 0.38+0.05, 7.36+0.04, 6.9+0.1],
       'upper': [1.5+0.4, 0.38-0.09, 7.36-0.18, 6.9-0.2],
       'best': [1.5, 0.38, 7.36, 6.9]}
publfs = {'F14': [1.5, 0.38, 7.36, 6.9], 'A14': [1.32, 0.42, 7.36, 7.2+0.25]} #, 'A14o': [1.32, 0.42, 7.36, 7.2]}

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
    plt.clf()

    rep = './'
    out = './'

    # Load detection list with bias column from disk
    detection_file = os.path.join(rep, DETECTIONS_FILE)

    alld = Table.read(detection_file, format='ascii')
    alld['coord'] = SkyCoord(alld['RAdeg'], alld['DEdeg'], distance=alld['dist'], unit=('degree', 'degree', 'au'))
    alld['ecl'] = alld['coord'].transform_to(HeliocentricTrueEcliptic)

    alld.sort('Hx')
    # weight data by the inverse of the bias.
    alld['weight'] = 1/alld['bias']

    # set some conditions to select sub-sets of the detection list
    cold = ((alld['ifree'] <= 4.) & (alld['a'] >= 42.4) & (alld['a'] <= 47.7))
    small = alld['Hx'] > 8.3
    faint = small & cold
    bright = (~small) & cold
    fcold = ((alld['i'] <= 5.) & (alld['dist'] >= 38) & (alld['dist'] <= 48))
    on_ecliptic = (alld['ecl'].lat < 3*units.degree) & (alld['ecl'].lat > -3*units.degree) & fcold & ~small
    f14_to_ossos_ratio = alld['weight'][on_ecliptic].sum()/alld['weight'][bright].sum()
    print(f"Ratio of OSSOS-CCKB on on-eclitic i<5: {1/f14_to_ossos_ratio}")
    on_ecliptic = (alld['ecl'].lat < 0*units.degree) & (alld['ecl'].lat > -1.5*units.degree) & fcold & ~small
    FH08_to_ossos_ratio = alld['weight'][on_ecliptic].sum()/alld['weight'][bright].sum()
    print(f"Ratio of OSSOS-CCKB on FH08 sample i<5: {1/FH08_to_ossos_ratio}")
    on_ecliptic = (alld['ecl'].lat < 3*units.degree) & (alld['ecl'].lat > -3*units.degree) & fcold & ~small
    B04_to_ossos_ratio = 6*360*alld['weight'][bright].sum()/(alld['weight'][on_ecliptic].sum())
    print(f"Ratio of OSSOS-CCKB on B04 sample i<5: {B04_to_ossos_ratio}")

    dmin = numpy.percentile(alld['dist'][cold], 5)
    dmax = numpy.percentile(alld['dist'][cold], 95)

    # plot the raw CDF without correcting for biases.
    yraw = numpy.cumsum(numpy.ones(len(alld[cold])))
    # plt.plot(xd, yraw, '--k', alpha=0.4)

    # Make a CDF of the bright cold classical objects detections
    # and turn into a Pandas data frame
    yd = numpy.cumsum(alld['weight'][bright])
    xd = alld['Hx'][bright]
    bright_cdf = pd.DataFrame({"Hx": xd, 'yd': yd})

    # now do the CDF of the full cold classical set but only
    # keep the faint ones (to plot this group differently.
    yf = numpy.cumsum(alld['weight'][cold])[alld['Hx'][cold] > 8.3]
    xf = alld['Hx'][faint]
    faint_cdf = pd.DataFrame({"Hx": xf, 'yd': yf})

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
    Hx = numpy.insert(xd, 0, 4.7)
    df = pd.DataFrame({'Hx': Hx, 'yl': yl, 'yu': yu})

    # setup the plot using seaborn functions
    sns.set_theme(style='ticks')
    sns.set_color_codes(palette='colorblind')

    # plot the bright and faint CDFs with dashed line for the faint part.
    # ax = sns.lineplot(data=bright_cdf, x='Hx', y='yd', color='r', linewidth=2, zorder=10)
    # sns.lineplot(data=faint_cdf, x='Hx', y='yd', color='r', linewidth=4, zorder=10, alpha=0.5, linestyle=":")
    # plt.plot(alld['Hx'][cold], yraw, color='k', alpha=0.3, linestyle="--")
    # Now shade the area between the upper and lower limits as determined using our poisson approximation
    plt.gca().fill_between(df['Hx'], df['yl'], df['yu'], color='r', alpha=0.5, zorder=0)

    # Now add the functional lines for variably tapered CDF.
    h = numpy.arange(4, 13, .1)
    A = 3.*13.94/5.
    for alpha in [0.4, 0.5]:
        alpha_SI = 5.*alpha/3.
        # fit a variable tapered functional log form.
        popt, pcov = optimize.curve_fit(log_variably_tapered, xd[(xd < 8.2) & (xd > 5.2)], numpy.log10(yd[(xd < 8.2) & (xd > 5.2)]),
                                        p0=[20., 200., alpha_SI, 0.5],
                                        bounds=([-numpy.inf, -numpy.inf, alpha_SI-0.001, -numpy.inf],
                                                [numpy.inf, numpy.inf, alpha_SI+0.001, numpy.inf]),
                                        maxfev=5000)
        y = variably_tapered(h, popt[0], popt[1], popt[2], popt[3])
        plt.plot(h, y, col[alpha], alpha=0.8, lw=2, label=r"$this\ work$")
        # bsi = popt[3]*3./5
        bsi = popt[3]
        mexp = 4.*numpy.pi/3.*rho/nu**(3./2.)*10.**A/popt[1]**(1./bsi)
        hexp = 5./3./bsi*numpy.log10(popt[1])
        Nexp = variably_tapered(hexp, popt[0], popt[1], popt[2], popt[3])
        print('alpha_SI = {:4.2f}, inferred beta_SI = {:4.2f}, inferred M_exp = {:8.2e}kg'.format(alpha_SI, bsi, mexp))
        print(f'A: {popt[0]:5.2g} B: {popt[1]:5.2g}')
        print('H_exp = {:5.2f}, N(<H_exp) = {:5.1f}'.format(hexp, Nexp))

    # plt.text(11.1, 3.8e5, r'$\alpha = 0.4$', fontsize=10, horizontalalignment='right', color=col[0.4])

    # overlay F14 and Adams best fit values normalized to match OSSOS populations at
    hexp = 8.2
    h = numpy.arange(4, 13, 0.1)
    dh = 0.1
    Nexp = variably_tapered(hexp, popt[0], popt[1], popt[2], popt[3])
    plcol = {'F14': 'k', 'A14': col[0.5], 'A14o': col[0.5]}
    hlims = {'F14': (5., 8.0), 'A14': (5., 7.3), 'A14o': (5., 7.3)}
    scales = {'F14': 1.8*6*360/f14_to_ossos_ratio, 'A14': 3.64*1.5*360.0, 'A14o': 1/0.00055}
    for group in publfs:
        def func(H, A):
            return A*fraser(H, *publfs[group]).sum()
        xcond = (xd < hlims[group][1]) & (xd > hlims[group][0])
        hcond = (h < hlims[group][1]) * (h > hlims[group][0])
        # params, covar = optimize.curve_fit(func, xd[hcond], yd[hcond])
        # print(group, params)
        fn = scales[group]*fraser(h, *publfs[group], dh=dh)
        print(h[hcond][-1], xd[xcond][-1])
        print(group, publfs[group], hlims[group], yd[xcond][-1], fn[hcond].sum(), fn[hcond].sum()/yd[xcond][-1])
        fn = numpy.cumsum(fn) # * yd[xcond][-1]/fn[hcond].sum()
        plt.plot(h, fn, f':{plcol[group]}', label=group, alpha=1)

    def func(H, A):
        return A*double_plaw(H)
    # params, covar = optimize.curve_fit(func, xd[xd <8], yd[xd < 8])
    # N = double_plaw([hexp, ])
    # fn = Nexp*double_plaw(h)/N
    # fn = double_plaw(h, dh=1)*B04_to_ossos_ratio
    # plt.plot(h, fn, ':k', label='B04', alpha=1)
    h = h - 0.25
    fn = double_plaw(h, R_eq=22.8, dh=1)*6*360/0.9
    h = h + 0.25
    plt.plot(h, fn, '-k', label='B04', alpha=1)

    plt.legend()
    # plt.plot([7.2,], [4500,], 'x')
    # Plot the deep survey rectangles.
    deep = Table.read(DEEP_SURVEYS_FILE, format='ascii')
    rlow, rhigh = poisson_range(deep['N_cold'], 0.95)
    # deep['H_min'] = deep['mag_lim'] - 10*numpy.log10(dmax)
    # deep['H_max'] = deep['mag_lim'] - 10*numpy.log10(dmin)
    deep['n_low'] = rlow*deep['sky_fraction']/deep['eta']
    deep['n_high'] = rhigh*deep['sky_fraction']/deep['eta']

    for row in deep:
        ec = 'm'
        if 'Bern' in row['survey']:
            ec = 'c'
        rect = Rectangle((row['H_min'], row['n_low']),
                         width=row['H_max'] - row['H_min'],
                         height=row['n_high'] - row['n_low'],
                         fc='none',
                         ec=ec)
        plt.gca().add_artist(rect)

    # plot the specific point for detections in B04
    B04 = Table.read(B04_DETECTIONS_FILE, format='ascii')
    cold = B04['ifree'] <= 4.
    n = cold.sum()
    yb = (1./B04['bias'][cold]).sum()
    xb = B04['Hx'][cold].max()
    r1, r2 = poisson_range(n, 0.95)
    r1 = (n - r1)*yb/n
    r2 = (r2 - n)*yb/n
    err = numpy.array(((r1, r2),)).T
    plt.errorbar(xb, yb, yerr=err, fmt='cd', ecolor='c', capsize=5)

    # plot the locations of the Bright MPC counts
    # plt.plot([5.13, 5.51], [3, 11], 'ok', marker='o', mec='k', mew=1, mfc='w', ms=10, zorder=0)
    # plt.plot([7.6], [4*360.0], 'ok')
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
    plt.savefig(out+'Fig-2-SI-paper-no-run-{:4.2f}_4.pdf'.format(nu))
