"""
emcee likelihood estimation of variably tapered exponential for H distribution
"""
import os
import emcee
import numpy
import pandas as pd
import seaborn as sns
from astropy.table import Table
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from funcs import poisson_range, likelihood2, ll, variably_tapered2

import corner

sns.set_theme()


DETECTIONS_FILE = "All_Surveys_v11-free-cla_m.detections-full"
DEEP_SURVEYS_FILE = "deep_surveys.csv"
B04_DETECTIONS_FILE = "B04-free-cla_m.detections-full"

i_c = 5.0
gmb = 1.0+1.0/6023600.0+1.0/408523.71+1.0/328900.56+1.0/3098708.0+1.0/1047.3486+1.0/3497.898+1.0/22902.98+1.0/19412.24+1.0/1.35e8

F14 = {'lower': [1.5-0.2, 0.38+0.05, 7.36+0.04, 6.9+0.1],
       'upper': [1.5+0.4, 0.38-0.09, 7.36-0.18, 6.9-0.2],
       'best': [1.5, 0.38, 7.36, 6.9]}
publfs = {'fraser': [1.5, 0.38, 7.36, 6.9], 'adams': [1.32, 0.42, 7.36, 7.2]}


rho = 1.0e12
nu = 0.15


def plot(do_fit=True, debug=False):
    """
    Do the emcee likelihood estimate and plot some figures.
    :return: None
    """

    rep = './'
    out = './'

    # Load detection list with bias column from disk
    detection_file = os.path.join(rep, DETECTIONS_FILE)
    alld = Table.read(detection_file, format='ascii')
    hlim = 8.3
    # weight data by the inverse of the bias.
    alld['weight'] = 1/alld['bias']
    alld.sort(['Hx', 'bias'])

    # set some conditions to select sub-sets of the detection list
    cold = ((alld['ifree'] <= 4.) & (alld['a'] >= 42.4) & (alld['a'] <= 47.7))
    small = alld['Hx'] > hlim
    faint = small & cold
    bright = (~small) & cold

    # CDF without correcting for biases.
    yraw = numpy.cumsum(numpy.ones(len(alld[cold])))
    xraw = alld['Hx'][cold]
    raw_cdf = pd.DataFrame({"Hx": xraw, 'raw': yraw})
    # Make a CDF of the bright cold classical objects detections weighted by the bias
    # add on GB data point.
    B04 = Table.read(B04_DETECTIONS_FILE, format='ascii')
    bcold = B04['ifree'] <= 4.
    B04.sort('Hx')

    # yd_fit = numpy.append(alld['bias'][bright], B04['bias'][bcold])
    yd_fit = alld['bias'][bright]
    # xd_fit = numpy.append(alld['Hx'][bright], B04['Hx'][bcold])
    xd_fit = alld['Hx'][bright]
    if debug:
        print(numpy.array((xd_fit, yd_fit)).T)

    # yd = numpy.cumsum(alld['weight'][bright])
    yd = numpy.cumsum(1/yd_fit)
    # xd = alld['Hx'][bright]
    xd = xd_fit
    bright_cdf = pd.DataFrame({"Hx": xd, 'yd': yd})

    # now do the CDF of the full cold classical set but only
    # keep the faint ones (to plot this group differently)
    yf = numpy.cumsum(alld['weight'][cold])[alld['Hx'][cold] > hlim]
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
    Hx = numpy.insert(alld['Hx'][bright], 0, 4.8)
    df = pd.DataFrame({'Hx': Hx, 'yl': yl, 'yu': yu})

    # Now add the functional lines for variably tapered CDF.
    models = []
    # pfit = ((8.742, 172.151, 0.4*5./3., 0.484), (0.894, 434.304, 5*0.5/2.0, 0.638))
    # pfit = ((8.742, 172.151, 0.4*5./3., 0.484), (0.006, 384.955, 5*0.5/2.0, 0.875))
    # below in comment is the best from eemc approach, which looks very good.
    # pfit = ((8.742, 172.151, 0.4*5./3., 0.484), (0.83, 450, 5*0.5/3.0, 0.65))
    # pfit = ((8.742, 172.151, 0.4*5./3., 0.484),)
    # pfit = ((-3, 7.7, 0.4*5./3., 0.5),)
    # H1: -2.468 - 2.797 - -2.144
    # H2: 7.887 7.425 - 8.434
    # alpha_SI: 0.667 0.662 - 0.671
    # beta_SI: 0.438 0.356 - 0.549
    pfit = ({'Ho': -2.5, 'Hb': 7.9, 'alpha_SI': 0.4*5/3., 'beta_SI': 0.44},)
    labels = list(pfit[0].keys())
    col = 'k'

    for p0 in pfit:
        alpha_SI = p0['alpha_SI']

        if do_fit:
            ivar = numpy.array((p0['Ho'], p0['Hb'], p0['alpha_SI'], p0['beta_SI']))
            ndim, nwalkers = 4, 100
            # spread the initial values around a small amount
            ivar = ivar + (numpy.random.rand(nwalkers, ndim)-0.5)*ivar*0.025
            nsteps = 10000
            # sampler = emcee.EnsembleSampler(nwalkers, ndim, ll, args=(xd, alld['bias'][bright]))
            kwargs = {'H': xd_fit, 'bias': yd_fit}
            sampler = emcee.EnsembleSampler(nwalkers, ndim, ll, kwargs=kwargs)
            state = sampler.run_mcmc(ivar, 2000, progress=True)
            sampler.run_mcmc(state, nsteps, progress=True)
            flat_samples = sampler.get_chain(flat=True)
            fig = plt.figure('corner')
            corner.corner(flat_samples, labels=labels, truths=[p0[label] for label in labels], fig=fig)
            plt.show()
            # fig.savefig(f'corner_{alpha_SI*10:2.0f}.pdf')
            # fit a variable tapered functional log form.
            for i in range(ndim):
                mcmc = numpy.percentile(flat_samples[:, i], [16, 50, 84])
                p0[labels[i]] = mcmc[1]
                print(f"\t{labels[i]}: {mcmc[1]:.3f} {mcmc[0]:.3f}-{mcmc[2]:.3f}")
        else:
            h = numpy.arange(5, 13, .1)
            y = variably_tapered2(h, **p0)
            plt.plot(h, y, col, alpha=0.3)

    # plt.text(10.9, 8.8e4, r'$\alpha = 0.4$', fontsize=10, horizontalalignment='left', color=cols['shallow'])
    # plt.text(11.1, 3.8e5, r'$\alpha = 0.5$', fontsize=10, horizontalalignment='right', color=cols['steep'])

    # Plot the deep survey rectangles.
    deep = Table.read(DEEP_SURVEYS_FILE, format='ascii')
    rlow, rhigh = poisson_range(deep['N_cold'], 0.95)
    deep['n_low'] = rlow*deep['sky_fraction']/deep['eta']
    deep['n_high'] = rhigh*deep['sky_fraction']/deep['eta']

    plt.clf()
    plt.style.use('PaperDoubleFig.mplstyle')
    # setup the plot using seaborn functions
    sns.set_theme(style='ticks')
    sns.set_color_codes(palette='colorblind')

    h = numpy.arange(5, 13, .1)
    if do_fit:
        inds = numpy.random.randint(len(flat_samples), size=200)
        for ind in inds:
            sample = flat_samples[ind]
            kwargs = {'Ho': sample[0], 'Hb': sample[1], 'alpha_SI': sample[2], 'beta_SI': sample[3] }
            y = variably_tapered2(h, **kwargs)
            plt.plot(h, y, col, alpha=0.01)

    # plot the bright and faint CDFs with dashed line for the faint part.
    ax = sns.lineplot(data=bright_cdf, x='Hx', y='yd', color='r', linewidth=2, zorder=10)
    sns.lineplot(data=raw_cdf, x='Hx', y='raw', color='k', alpha=0.3, linestyle="--", ax=ax)
    sns.lineplot(data=faint_cdf, x='Hx', y='yd', color='r', linewidth=4, zorder=10, alpha=0.5, linestyle=":", ax=ax)

    if debug:
        dy, N, l = likelihood2(xd_fit, yd_fit, **pfit[0])
        print(f"Predicted total: {N}")
        print(numpy.array((xd_fit, yd_fit, dy)).T)
        x = numpy.linspace(xd_fit.min(), xd_fit.max())
        models.append((x[1] - x[0]) * numpy.interp(x, xd_fit, dy).cumsum())
        plt.plot(x, models[-1], col, markeredgewidth=0.3, alpha=0.7)

    # Now shade the area between the upper and lower limits as determined using our poisson approximation
    ax.fill_between(df['Hx'], df['yl'], df['yu'], color='r', alpha=0.2)


    for row in deep:
        ec = 'm'
        if 'Bern' in row['survey']:
            ec = 'c'
        rect = Rectangle((row['H_min'], row['n_low']),
                         width=row['H_max'] - row['H_min'],
                         height=row['n_high'] - row['n_low'],
                         fc='none',
                         ec=ec)
        ax.add_artist(rect)

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
    plt.plot([5.13, 5.51], [3, 11], 'ok', marker='o', mec='k', mew=1, mfc='w', ms=10, zorder=10)

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
    plt.show()
    # plt.savefig(out+'Fig-1-SI-paper-no-run-{:4.2f}_4.pdf'.format(nu))
