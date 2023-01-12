#! /usr/bin/env python3
from matplotlib import pyplot as plt
import gzip
from scipy import stats, optimize
import numpy


def trim(string):
    """
    Removes spaces at end and beginning of string 'string'.
    """
    trim_str = string
    return trim_str.strip(' \n\t')


def open_file(fichier):
    """
    Reads in file 'fichier' and returns a list of lines.
    No processing is done on any line.
    """
    try:
        f = open(fichier, 'r')
    except IOError:
        try:
            f = gzip.open(fichier, 'rt')
        except IOError:
            try:
                f = gzip.open(fichier+'.gz', 'rt')
            except IOError:
                print ('Cannot open', fichier, ' nor ', fichier+'.gz')
                raise FileNotFoundError("Aucun fichier ou dossier de ce type: '"+fichier+"' or '"+fichier+".gz'")
    return f


def read_headed(fichier, convert=True, nl_max=numpy.nan):
    """
    Reads in file 'fichier' which is a fixed number of field file (not
    necessarily fixed column), with the last comment line defining the columns,
    and returns a dictionary, keys being the headers of the columns and values
    sequences of values of the given column.
    """
    M, keys = {}, []
    f = open_file(fichier)
    if (f):
        l0 = '---'
        while True:
            line = f.readline()
            if line[0] != "#":
                break
            l0 = line
        keys = l0[1:-1].split()
        while True:
            try:
                keys.remove('')
            except ValueError:
                break
        n = len(keys)
        for j in range(n):
            keys[j] = trim(keys[j])
            M[keys[j]] = []
        nl = 0
        while True:
            if (not line) or (nl >= nl_max):
                break
            if (line[0] != "#"):
                vals = line[:-1].split()
                while True:
                    try:
                        vals.remove('')
                    except ValueError:
                        break
                for j in range(n):
                    M[keys[j]].append(trim(vals[j]))
                nl += 1
            line = f.readline()
        f.close()
        for k in keys:
            M[k] = numpy.array(M[k])
        if convert:
            for k in keys:
                try:
                    M[k] = numpy.array(list(map(int, M[k])))
                except ValueError:
                    try:
                        M[k] = numpy.array(list(map(float, M[k])))
                    except ValueError:
                        pass
    return M, keys

i_c = 5.0
gmb = 1.0+1.0/6023600.0+1.0/408523.71+1.0/328900.56+1.0/3098708.0+1.0/1047.3486+1.0/3497.898+1.0/22902.98+1.0/19412.24+1.0/1.35e8
class Detections:
    """
Class to hold informations on real or simulated detections.
Remember that all angles are in degree
    """

    def __init__(self, fich):
        self.data, ks = read_headed(fich)
        if ('Omega' in ks) and ('node' not in ks):
            self.data['node'] = self.data['Omega']
        if ('Omega' not in ks) and ('node' in ks):
            self.data['Omega'] = self.data['node']
        if ('omega' in ks) and ('peri' not in ks):
            self.data['peri'] = self.data['omega']
        if ('omega' not in ks) and ('peri' in ks):
            self.data['omega'] = self.data['peri']
        if ('a' in ks) and ('e' in ks) and ('q' not in ks):
            self.data['q'] = self.data['a']*(1.-self.data['e'])
        if ('a' in ks) and ('e' not in ks) and ('q' in ks):
            self.data['e'] = 1. - self.data['q']/self.data['a']
        if ('M' not in ks) and ('tperi' in ks) and ('JD' in ks) and ('a' in ks):
            self.data['M'] = 0. + numpy.sqrt(gmb)*360./(self.data['a']**1.5*365.25)*(self.data['JD']-self.data['tperi']-2400000.5)
        if ('Hsur' not in ks) and ('H_rand' in ks):
            self.data['Hsur'] = self.data['H_rand']
        if ('Hsur' in ks) and ('H_rand' not in ks):
            self.data['H_rand'] = self.data['Hsur']
        if ('Filt' not in ks) and ('Surv.' in ks):
            self.data['Filt'] = numpy.array(list(map(lambda x: Filtre[x], list(map(lambda x: 'HL9m-s' if x[0:3] == 'HL9' else x, self.data['Surv.'])))))
        if ('Hx' not in ks):
            if ('H' in ks):
                self.data['Hx'] = self.data['H']
            else:
                self.data['Hx'] = numpy.where(self.data['Filt']=='r',self.data['Hsur'],numpy.where(self.data['Filt']=='R',self.data['Hsur']+0.1,numpy.where(self.data['ifree']<i_c,self.data['Hsur']-0.9,self.data['Hsur']-0.6)))
        if ('object' in ks) and ('Obj' not in ks):
            self.data['Obj'] = self.data['object']
        if ('object' not in ks) and ('Obj' in ks):
            self.data['object'] = self.data['Obj']
        self.ks = self.data.keys()
        if ('object' in ks):
            self.OSSOS = numpy.array(list(map(lambda x: x[0] == 'o', self.data['object'])))
        if ('Surv.' in ks):
            self.OSSOS = numpy.array(list(map(lambda x: x[0:3] == '201', self.data['Surv.'])))
        if ('cl' in ks):
            self.res = self.data['cl'] == 'res'
            self.sca = self.data['cl'] == 'sca'
            self.det = self.data['cl'] == 'det'
            self.xxx = self.data['cl'] == 'xxx'
            self.cen = self.data['cl'] == 'cen'
            self.cla = self.data['cl'] == 'cla'
        if ('p' in ks):
            self.clam = self.data['p'] == 'm'
            self.clai = self.data['p'] == 'i'
            self.clao = self.data['p'] == 'o'

def CDF(data, vals=None, Norm=True, Up=True):
    """
    Compute the Cumulative Distribution Function from a 1D array of values.
    Can assign specific weigh to each value, instead of 1.
    """
    d=data.argsort()
    if (not isinstance(vals,numpy.ndarray)) and (not (vals)):
        vals=numpy.ones(len(data))*1.
    if Up:
        ds = d
    else:
        ds = d[numpy.arange(len(d),0,-1)-1]
    nd=vals[ds].cumsum()
    if Norm:
        nd=nd/nd[-1]
    return data[ds], nd

def CDFplot(data, vals=None, xl="Data", yl=None, ls="k-", xp=None, Norm=True, Up=True, lw=1, size=16):
    """
    Plots the Cumulative Distribution Function from a 1D array of values.
    Can assign specific weigh to each value, instead of 1.
    If 'xp' is not 'None', then restrict the plot to the [xp[0], xp[1]] range of
    values from 'data', while the CDF is still computed from the full range.
    """
    datas,nd=CDF(data,vals=vals,Norm=Norm, Up=Up)
    if (xp):
        m = (datas >= xp[0])*(datas <= xp[1])
        datas = datas[m]
        nd = nd[m]
    plt.plot(datas, nd, ls, linewidth=lw)
    plt.xlabel(xl,size=size)
    if (yl==None):
        if (not isinstance(vals,numpy.ndarray)) and (not (vals)):
            yl = 'Cumulative number density'
        else:
            yl = 'Weighed cumulative number density'
    plt.ylabel(yl,size=size)

ln10 = numpy.log(10.)
def VariablyTapered(h, Dp, E, alpha_SI, beta_SI):
    return Dp*10.**(alpha_SI*3.*h/5.)*numpy.exp(-E*10.**(-beta_SI*3.*h/5.))

def VariablyTaperedDiff(h, Dp, E, alpha_SI, beta_SI):
    return (3./5.*ln10)*Dp*10.**(alpha_SI*3.*h/5.)*(alpha_SI+beta_SI*E*10.**(-beta_SI*3.*h/5.))*numpy.exp(-E*10.**(-beta_SI*3.*h/5.))

def LogVariablyTapered(h, Dp, E, alpha_SI, beta_SI):
    return numpy.log10(VariablyTapered(h, Dp, E, alpha_SI, beta_SI))

def LogVariablyTaperedDiff(h, Dp, E, alpha_SI, beta_SI):
    return numpy.log10(VariablyTaperedDiff(h, Dp, E, alpha_SI, beta_SI))

def dDPluto(D):
    return 0.08*D**1.151

def dDCharon(D):
    return 0.07*D**1.151

def Hr(r, nu):
    return 13.94 - 2.5*numpy.log10(nu*r**2)

def MH(h):
    return 4.*numpy.pi/3.*rho/(nu**1.5)*10.**(3.*13.94/5.)*10.**(-0.6*h)

def HM(m):
    return 13.94 - 5./3.*numpy.log10(m/4.*3/numpy.pi*nu**1.5/rho)

def rM(m):
    return (m/4.*3./numpy.pi/rho)**(1./3.)

def Mr(r):
    return r**3*rho*4./3.*numpy.pi

def Poisson_cum(k, l):
    P = numpy.ones(k+1)
    el = numpy.exp(-l)
    P[0] -= el
    t = 1.
    sum = 1.
    for i in range(1,k+1):
        t *= (l/i)
        sum += t
        P[i] -= el*sum
    return P

def Dichotomy(a, x1, x2, eps, k, func):
    lo, hi = x1, x2
    f1 = func(k,lo)[k]
    f2 = func(k,hi)[k]
    if (f1-a)*(f2-a) > 0.:
        exit
    while abs(hi - lo) > eps:
        r = (lo + hi)/2.
        f = func(k,r)[k]
        if f == a:
            return r
        if (f-a)*(f1-a) > 0.:
            f1 = f
            lo = r
        else:
            f2 = f
            hi = r
    return r

def Confidence_range(ks, lev, eps=1.e-5):
    try:
        len(ks)
    except TypeError:
        ks = numpy.array(ks)
    if not isinstance(ks, numpy.ndarray):
        ks = numpy.array(ks)
    a1 = (1.-lev)/2.
    a2 = lev + a1
    r1, r2 = [], []
    for k in ks:
        if k == 0:
            r1.append(0.)
            r2.append(Dichotomy(lev, 0, 10, eps, k, Poisson_cum))
        else:
            r1.append(Dichotomy(a1, 0., k, eps, k, Poisson_cum))
            r2.append(Dichotomy(a2, k, 8*k, eps, k, Poisson_cum))
    return numpy.array(r1), numpy.array(r2)

rho = 1.0e12
col = {0.4: 'b', 0.5: 'g'}
nu = 0.15
tbl = [['Fuentes \\& Holman (2008)', 2.8, 0.88, 25.0, 25.1, 8.1, 9.1, 44, 30, 8.7, 930], ['Fraser et al (2008) CFHT', 1.49, 0.97, 25.1, 25.2, 8.2, 9.2, 19, 11, 8.8, 1726], ['Gladman et al (2001) CFHT', 0.27, 1, 25.7, 25.8, 8.8, 9.8, 9, 3, 9.4, 10733], ['Fraser \\& Kavelaars (2009)', 0.255, 0.95, 26.4, 26.5, 9.5, 10.5, 22, 10, 10.1, 11594], ['Bernstein et al (2004)', 0.019, 1.0, 28.4, 28.5, 11.5, 12.5, 3, 3, 12.1, 115325]]

if __name__ == "__main__":
    plt.style.use('PaperDoubleFig.mplstyle')

    rep = './'
    out = './'
    alld = Detections(rep+'All_Surveys_v11-free-cla_m.detections-full')
    cold = (alld.data['ifree'] <= 4.)*(alld.data['a'] >= 42.4)*(alld.data['a'] <= 47.7)
    x = numpy.arange(4., 18., 0.1)
    dsc = alld.data['Hx'][cold].argsort()
    Hxc = alld.data['Hx'][cold][dsc]
    wpc = 1./alld.data['bias'][cold][dsc]
    npc = wpc.cumsum()
    xd = list(numpy.arange(5.6, 9.21, 0.3))
    yd = []
    for hc in xd[:-1]:
        hmin = Hxc[Hxc<=hc].max()
        hmax = Hxc[Hxc>=hc].min()
        yd.append(npc[(Hxc>=hmin)*(Hxc<=hmax)].mean())
    yd.append(npc[-1])
    # keep faint end for dotted line plotting, but remove from fit
    xd_faint = numpy.array(xd)[numpy.array(xd)>8.4]
    yd_faint = numpy.array(yd)[numpy.array(xd)>8.4]
    yd = list(numpy.array(yd)[numpy.array(xd)<=8.4])
    xd = list(numpy.array(xd)[numpy.array(xd)<=8.4])
    xd_deep, yd_deep, xl_deep, xh_deep, yl_deep, yh_deep = [], [], [], [], [], []
    x_lim, y_lim = [], []
    print('#+begin_example')
    for s in tbl:
        r1, r2 = Confidence_range([s[8]], 0.95)
        print('{:26s}  {:2d} {:4.1f} {:4.1f}  {:9.1f} {:9.1f} {:9.1f}'.format(s[0], s[8], r1[0], r2[0], s[8]/s[2]*s[10], r1[0]/s[2]*s[10], r2[0]/s[2]*s[10]))
        if s[10] > 0:
            xl_deep.append(s[5])
            xh_deep.append(s[6])
            nd = s[8]/s[2]*s[10]
            yl_deep.append(r1/s[2]*s[10])
            yh_deep.append(r2/s[2]*s[10])
        else:
            x_lim.append((s[5]+s[6])/2)
            y_lim.append(r2/s[2]*s[10])
    print('#+end_example')
    xd = numpy.array(xd)
    yd = numpy.array(numpy.log10(yd))
    # ion()
    # fig1 = plt.figure(1)
    # plt.clf()
    print('\\newpage')
    CDFplot(alld.data['Hx'][(alld.data['Hx']<8.4)*cold], vals=1./alld.data['bias'][(alld.data['Hx']<8.4)*cold], xl=r'Cummulative $H_r$', yl='Number of cold classical KBOs', ls='r-', lw=4., Norm=False, size=13)
    plt.yscale('log')
    #labels = ['OSSOS cold']
    kwargs = {}
    kwargs['maxfev'] = 5000
    print('#+begin_example')
    hb = numpy.linspace(4.5, 5.5, 11)
    A = 3.*13.94/5.
    nb = {}
    for alpha in [0.4, 0.5]:
        alpha_SI = 5.*alpha/3.
        popt, pcov = optimize.curve_fit(LogVariablyTapered, xd, yd, p0=[20., 200., alpha_SI, 0.5], bounds=([-numpy.inf, -numpy.inf, alpha_SI-0.001, -numpy.inf], [numpy.inf, numpy.inf, alpha_SI+0.001, numpy.inf]), **kwargs)
        y = VariablyTapered(x, popt[0], popt[1], popt[2], popt[3])
        plt.plot(x, y, col[alpha]+'--', lw=1.5, zorder=10)
        bsi = popt[3]*3./5
        mexp = 4.*numpy.pi/3.*rho/nu**(3./2.)*10.**A/popt[1]**(1./bsi)
        hexp = 5./3./bsi*numpy.log10(popt[1])
        print('alpha = {:4.2f}, inferred beta_SI = {:4.2f}, inferred M_exp = {:8.2e}kg'.format(alpha, bsi, mexp))
        print('H_exp = {:5.2f}, N(<H_exp) = {:5.1f}'.format(hexp, VariablyTapered(hexp, popt[0], popt[1], popt[2], popt[3])))
        nb[alpha] = VariablyTapered(hb, popt[0], popt[1], popt[2], popt[3])
    hs, ns = CDF(alld.data['Hx'][(alld.data['Hx']<9.2)*cold], vals=1./alld.data['bias'][(alld.data['Hx']<9.2)*cold], Norm=False)
    h10 = (hs[ns<=10].max()+hs[ns>=10].min())/2.
    h100 = (hs[ns<=100].max()+hs[ns>=100].min())/2.
    h1000 = (hs[ns<=1000].max()+hs[ns>=1000].min())/2.
    print('H(N=10) = {:4.2f}; H(N=100) = {:4.2f}; H(N=1000) = {:4.2f}'.format(h10, h100, h1000))
    print('#+end_example')
    plt.text(12.8, 6.e5, r'$\alpha = 0.4$', fontsize=10, horizontalalignment='left', color=col[0.4])
    plt.text(11.2, 4.e5, r'$\alpha = 0.5$', fontsize=10, horizontalalignment='right', color=col[0.5])
    plt.xlim(4., 14.)
    plt.ylim(1., 2.e6)
    CDFplot(alld.data['Hx'][(alld.data['Hx']<9.2)*cold], vals=1./alld.data['bias'][(alld.data['Hx']<9.2)*cold], xl=r'Cummulative $H_r$', yl='Number of cold classical KBOs', ls='r:', lw=2., Norm=False, size=13)
    for xl, xh, yl, yh in zip(xl_deep, xh_deep, yl_deep, yh_deep):
        plt.plot([xl, xh, xh, xl, xl], [yl, yl, yh, yh, yl], 'm-', lw=0.5)
    for xl, xh, yl, yh in zip(xl_deep[-1:], xh_deep[-1:], yl_deep[-1:], yh_deep[-1:]):
        plt.plot([xl, xh, xh, xl, xl], [yl, yl, yh, yh, yl], 'c-', lw=0.5)
    repB04 = './'
    B04 = Detections(repB04+'B04-free-cla_m.detections-full')
    coldB04 = B04.data['ifree'] <= 4.
    HB04, nB04 = CDF(B04.data['Hx'][coldB04], vals=1./B04.data['bias'][coldB04], Norm=False)
    xb = HB04[-1]
    yb = nB04[-1]
    r1, r2 = Confidence_range([len(nB04)], 0.95)
    r1 = (len(nB04) - r1)*yb/len(nB04)
    r2 = (r2 - len(nB04))*yb/len(nB04)
    plt.errorbar(xb, yb, yerr=numpy.array([r1,r2]), fmt='cd', ecolor='c', capsize=5)
    plt.axes().set_xticks([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    for i, tick in enumerate(plt.axes().xaxis.get_ticklabels()):
        if i % 2 != 0:
            tick.set_visible(False)
    plt.plot([5.13, 5.51], [3, 11], 'ok', marker='o', mec='k', mew=1, mfc='w', ms=10)
    plt.savefig(out+'Fig-1-SI-paper-no-run-{:4.2f}_4.pdf'.format(nu))
