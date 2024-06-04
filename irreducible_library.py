import numpy as np
import os, sys
import scipy
import math
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.optimize import brentq

me = 0.511
mmu=105.658
alpha_em = 1/137.
e = np.sqrt(alpha_em*4*np.pi)
Mpl = 2.435e21 #reduced planck mass (i.e. factor of 8 pi)
cm2_conv = 2.5e21
h = 0.67
Omegab = 0.05
eta = 5.5 * 10**(-10)*h**2*Omegab/0.02
BH = 13.6 * 10 ** (-6)
me = 0.511
RZetaThree = 1.202056903

def nxrelic(mx, T):
    """
    This computes the number density of dark matter at a certain temperature. Assuming an mx.
    """
    return 4.35e-7 /mx  * 2 *np.pi**2 *T**3* (2 + 7/8 * 6* 4/11)/45

def rho_gamma(T):
    """
    energy density of plasmons. 
    """
    return np.pi**2 * T**4/15.
def p_gamma(T):
    """
    pressure density of plasmons.
    """
    return rho_gamma(T)/3.
def s_gamma(T):
    """
    entropy density of plasmons.
    """
    return 4 * rho_gamma(T)/(3.* T)

# The temperature for the following function is the temperature of the neutrinos.

def rho_nu(T):
    """
    energy density of neutrinos.
    """
    return 6*7/8*np.pi**2*T**4/(30.)

def mu_e(T):
    if T < (0.0018):
        # the expression diverges to 1 at lower temperatures because of issues with fsolve. I scale the chemical potential with expansion after this
        return me + (0.5030347810001615-me)*(T/0.0018)
    else:
        return fsolve(lambda mu: -2*eta*RZetaThree/np.pi**2+(me*T/(2*np.pi))**(3/2)*np.exp(-me/T)*np.sinh(mu/T),1)[0]
    
Tlist = np.logspace(2.5,-5,num=1000)
mu_elist = np.array([mu_e(i) for i in Tlist])
#mu_elist = np.array([0 for i in Tlist])
ue = interp1d(Tlist, mu_elist, fill_value='extrapolate')

def n_e(T):
    """
    number density of electrons
    """
    integral= scipy.integrate.quad(lambda y: \
    2/np.pi**2*(y*T)*T * np.sqrt((y*T)**2-me**2)/(np.exp(y-ue(T)/T)+1), me/T, np.inf)
    return integral[0]

def p_e(T):
    """
    pressure density of electron
    """
    integral= scipy.integrate.quad(lambda y: \
    2*T/(3*np.pi**2)*np.sqrt((y*T)**2-me**2)**3/(np.exp(y-ue(T)/T)+1), me/T, np.inf)
    return integral[0]

def rho_e(T):
    """
    energy density of electrons
    """

    integral= scipy.integrate.quad(lambda y: \
    2*T/np.pi**2*(y*T)**2 * np.sqrt((y*T)**2-me**2)/(np.exp(y-ue(T)/T)+1), me/T, np.inf)
    return integral[0]
    

def s_e(T):
    """
    entropy density of electron
    """
    return (p_e(T)+rho_e(T))/T

s_const = (s_e(10)+s_gamma(10))/(10.0591)**3 #define a=1 at T= 1 MeV

def T_full(a):
    """
    Tracking entropy to convert between a and T. 
    """
    return fsolve(lambda T:\
    (s_e(T)+s_gamma(T))*a**3 - s_const, 1/a)


alist = np.logspace(5.1439, -2.5026, num=1000)
Tfull_list = np.array([T_full(i)[0] for i in alist])
T = interp1d(alist, Tfull_list,fill_value='extrapolate')

def a_full(T):
    return fsolve(lambda a:\
    (s_e(T)+s_gamma(T))*a**3 - s_const, 1/T)

Tlist = np.logspace(2.5, -5, num=1000)
afull_list = np.array([a_full(i)[0] for i in Tlist])
a = interp1d(Tlist, afull_list,fill_value='extrapolate')

def H_full(aH):
    """
    hubble parameter. We can see the neutrino energy density doesn't really matter.
    """
    TH = T(aH)
    return np.sqrt((rho_gamma(TH)+rho_e(TH)+rho_nu(10)*(a(10)/aH)**4)/(3*Mpl**2))

alist = np.logspace(5.1439, -2.5026, num=5000)
Hfull_list = np.array([H_full(i) for i in alist])
H = interp1d(alist, Hfull_list,fill_value='extrapolate')

def wp_full(T): #flag to make draft more clear about factor of 2
    # Add chemical potential
    """
    plasma frequency. Added in chemical potential of electrons to the occupancy function.
    """
    integral = integrate.quad(lambda p: 8*alpha_em/np.pi\
                *p**2/np.sqrt(p**2+me**2)*(1 - p**2/(3*(p**2+me**2)))\
                *1/(1+np.exp((np.sqrt(p**2+me**2)-ue(T))/T))
                            , 0, np.inf)
    return np.sqrt(integral[0])
    


Tlist = np.logspace(2.5, -5, num=500)
wp_list = np.array([wp_full(i) for i in Tlist])
wp = interp1d(Tlist, wp_list,fill_value='extrapolate')

def wp_high(T):
    """
    High temp plasma frequency behaviour.
    """
    return np.sqrt(4*np.pi*alpha_em* T**2/9)

def wp_low(T):
    """ 
    Low temp behaviour.
    """
    return np.sqrt(4*np.pi*alpha_em*n_e(T)/me)

def w1_full(T):
    # chemical potential
    
    integral = integrate.quad(lambda p: 8*alpha_em/np.pi\
                *p**2/np.sqrt(p**2+me**2)*\
                (5*p**2/(3*(p**2+me**2))-p**4/((p**2+me**2)**2))\
                *1/(1+np.exp((np.sqrt(p**2+me**2)-ue(T))/T))
                            , 0, np.inf)
    return np.sqrt(integral[0])


Tlist = np.logspace(2.5, -5, num=500)
w1_list = np.array([w1_full(i) for i in Tlist])
w1 = interp1d(Tlist, w1_list,fill_value='extrapolate')

def vstar(T):
    """ 
    Typical velocity of electron in the plasma.
    """
    return w1(T)/wp(T)

# Bunch of stuff from Braten and Siegel

def Pi_ell(T, k, omega):
    wpP = wp(T)
    vs = vstar(T)
    return 3*wpP**2/vs**2 *(omega/(2*vs*k)*\
            np.log((omega + vs*k)/(omega- vs*k))-1)

def Pi_t(T, k, omega):
    wpP = wp(T)
    vs = vstar(T)
    return 3*wpP**2/(2*vs**2)*(omega**2/k**2 \
    -omega*(omega**2-vs**2*k**2)/(2*vs*k**3)\
    *np.log((omega + vs*k)/(omega- vs*k)))

def kmax(T):
    wpP = wp(T)
    vs = vstar(T)
    return np.sqrt(3/vs**2* (1/(2*vs)*np.log((1+vs)/(1-vs))-1))*wpP

def omega_ell(T, k):
    if type(k)==int or type(k)== float or type(k)== np.float64:
        if k>kmax(T):
            return 0
        else:
            return fsolve(lambda w: Pi_ell(T, k, w)-k**2, k)
    else:
        km = kmax(T)
        kudu = np.zeros_like(k)
        for i in range(len(k)):
            if k[i] < km:
                kudu[i] = fsolve(lambda w: Pi_ell(T, k[i], w)-k[i]**2, k[i])
        return kudu
    
def omega_t(T, k):
    return fsolve(lambda w: Pi_t(T, k, w)+k**2-w**2, k)

def m_ell(T, k):
    if type(k) == int or type(k) == float or type(k)== np.float64:
        if k>kmax(T):
            return 0
        else:
            return np.sqrt(omega_ell(T, k)**2 - k**2)
        
    else:
        km = kmax(T)
        kudu = np.zeros_like(k)
        for i in range(len(k)):
            if k[i] < km: 
                kudu[i] = np.sqrt(omega_ell(T, k[i])**2 - k[i]**2)
        return kudu
    
def m_t(T, k):
    """ 
    This is m_t from equation 2 in the paper
    """
    return np.sqrt(omega_t(T, k)**2 - k**2)

def Z_ell(T, k):
    km = kmax(T)
    if type(k) == int or type(k) == float or type(k)== np.float64:
        if k>km:
            return 0
        else: 
            wpP = wp(T)
            vs = vstar(T)
            wl = omega_ell(T, k)
            return 2*(wl**2- vs**2*k**2)/(3 *wpP**2 -(wl**2-vs**2*k**2))
    else:
        wpP = wp(T)
        vs = vstar(T)
        kudu = np.zeros_like(k)
        for i in range(len(k)):
            wl = omega_ell(T, k[i])
            if k[i]<km:
                kudu[i] = 2*(wl**2- vs**2*k[i]**2)/(3 *wpP**2 -(wl**2-vs**2*k[i]**2))
        #print(kudu)
        return kudu
    
def Z_t(T, k):
    wt = omega_t(T, k)
    wpP = wp(T)
    vs = vstar(T)
    return 2*wt**2 *(wt**2 - vs**2*k**2)/\
    (3*wpP**2 *wt**2 + (wt**2 +k**2)*(wt**2 - vs**2 * k**2)- 2*wt**2*(wt**2-k**2))

def gamma_ann(Q, mx, T,noQ = False,muon=True):
    """ 
    Reaction rate of e+e- 
    Focus on case of no muons
    """
    if noQ:
        if muon==False:
            if np.size(T)==1:
                integrand = lambda s: e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                *(s**2 +1/3*(s-4*me**2)*(s-4*mx**2)+4*s*(mx**2+me**2))\
                *1/(8*np.pi)*np.sqrt(1-4*me**2/s)*np.sqrt(s)\
                *scipy.special.kn(1, np.sqrt(s)/T) *T/(2*np.pi)**3
                return integrate.quad(integrand, max(4*me**2, 4*mx**2), np.inf)[0]
            else:
                kudu = np.zeros_like(T)
                for i in range(len(T)):
                    t = T[i]
                    integrand = lambda s: e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                    *(s**2 +1/3*(s-4*me**2)*(s-4*mx**2)+4*s*(mx**2+me**2))\
                    *1/(8*np.pi)*np.sqrt(1-4*me**2/s)*np.sqrt(s)\
                    *scipy.special.kn(1, np.sqrt(s)/t) *t/(2*np.pi)**3
                    kudu[i] = integrate.quad(integrand, max(4*me**2, 4*mx**2), np.inf)[0]
                return kudu
        else:
            if np.size(T)==1:
                
                integrand = lambda s: e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                *(s**2 +1/3*(s-4*me**2)*(s-4*mx**2)+4*s*(mx**2+me**2))\
                *1/(8*np.pi)*np.sqrt(1-4*me**2/s)*np.sqrt(s)\
                *scipy.special.kn(1, np.sqrt(s)/T) *T/(2*np.pi)**3
                
                integrandmu = lambda s: e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                *(s**2 +1/3*(s-4*mmu**2)*(s-4*mx**2)+4*s*(mx**2+mmu**2))\
                *1/(8*np.pi)*np.sqrt(1-4*mmu**2/s)*np.sqrt(s)\
                *scipy.special.kn(1, np.sqrt(s)/T) *T/(2*np.pi)**3
                
                return integrate.quad(integrand, max(4*me**2, 4*mx**2), np.inf)[0]\
                +integrate.quad(integrandmu, 4*mmu**2, np.inf)[0]
            else:
                kudu = np.zeros_like(T)
                for i in range(len(T)):
                    t = T[i]

                    integrand = lambda s: e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                    *(s**2 +1/3*(s-4*me**2)*(s-4*mx**2)+4*s*(mx**2+me**2))\
                    *1/(8*np.pi)*np.sqrt(1-4*me**2/s)*np.sqrt(s)\
                    *scipy.special.kn(1, np.sqrt(s)/t) *t/(2*np.pi)**3
                    
                    integrandmu = lambda s: e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                    *(s**2 +1/3*(s-4*mmu**2)*(s-4*mx**2)+4*s*(mx**2+mmu**2))\
                    *1/(8*np.pi)*np.sqrt(1-4*mmu**2/s)*np.sqrt(s)\
                    *scipy.special.kn(1, np.sqrt(s)/t) *t/(2*np.pi)**3
                    
                    kudu[i] = integrate.quad(integrand, max(4*me**2, 4*mx**2), np.inf)[0]+\
                    +integrate.quad(integrandmu, 4*mmu**2, np.inf)[0]
                
                return kudu
    else:
        if muon==False:
            if np.size(T)==1:
                integrand = lambda s: Q**2*e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                *(s**2 +1/3*(s-4*me**2)*(s-4*mx**2)+4*s*(mx**2+me**2))\
                *1/(8*np.pi)*np.sqrt(1-4*me**2/s)*np.sqrt(s)\
                *scipy.special.kn(1, np.sqrt(s)/T) *T/(2*np.pi)**3
                return integrate.quad(integrand, max(4*me**2, 4*mx**2), np.inf)[0]
            else:
                kudu = np.zeros_like(T)
                for i in range(len(T)):
                    t = T[i]
                    integrand = lambda s: Q**2*e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                    *(s**2 +1/3*(s-4*me**2)*(s-4*mx**2)+4*s*(mx**2+me**2))\
                    *1/(8*np.pi)*np.sqrt(1-4*me**2/s)*np.sqrt(s)\
                    *scipy.special.kn(1, np.sqrt(s)/t) *t/(2*np.pi)**3
                    kudu[i] = integrate.quad(integrand, max(4*me**2, 4*mx**2), np.inf)[0]
                return kudu
        else:
            if np.size(T)==1:
                
                integrand = lambda s: Q**2*e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                *(s**2 +1/3*(s-4*me**2)*(s-4*mx**2)+4*s*(mx**2+me**2))\
                *1/(8*np.pi)*np.sqrt(1-4*me**2/s)*np.sqrt(s)\
                *scipy.special.kn(1, np.sqrt(s)/T) *T/(2*np.pi)**3
                
                integrandmu = lambda s: Q**2*e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                *(s**2 +1/3*(s-4*mmu**2)*(s-4*mx**2)+4*s*(mx**2+mmu**2))\
                *1/(8*np.pi)*np.sqrt(1-4*mmu**2/s)*np.sqrt(s)\
                *scipy.special.kn(1, np.sqrt(s)/T) *T/(2*np.pi)**3
                
                return integrate.quad(integrand, max(4*me**2, 4*mx**2), np.inf)[0]\
                +integrate.quad(integrandmu, 4*mmu**2, np.inf)[0]
            else:
                kudu = np.zeros_like(T)
                for i in range(len(T)):
                    t = T[i]

                    integrand = lambda s: Q**2*e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                    *(s**2 +1/3*(s-4*me**2)*(s-4*mx**2)+4*s*(mx**2+me**2))\
                    *1/(8*np.pi)*np.sqrt(1-4*me**2/s)*np.sqrt(s)\
                    *scipy.special.kn(1, np.sqrt(s)/t) *t/(2*np.pi)**3
                    
                    integrandmu = lambda s: Q**2*e**4/(2*np.pi*s**2)*np.sqrt(1 - 4*mx**2/s)\
                    *(s**2 +1/3*(s-4*mmu**2)*(s-4*mx**2)+4*s*(mx**2+mmu**2))\
                    *1/(8*np.pi)*np.sqrt(1-4*mmu**2/s)*np.sqrt(s)\
                    *scipy.special.kn(1, np.sqrt(s)/t) *t/(2*np.pi)**3
                    
                    kudu[i] = integrate.quad(integrand, max(4*me**2, 4*mx**2), np.inf)[0]+\
                    +integrate.quad(integrandmu, 4*mmu**2, np.inf)[0]
                
                return kudu
        

def long_integrand(Q, mx, T, k,noQ = False):
    """
    longitudional case
    """
    if noQ:
        if type(k) == int or type(k) == float or type(k)== np.float64:
            ml = m_ell(T, k)
            if mx>ml/2:
                return 0
            elif k>kmax(T):
                return 0
            else:
                wl = np.sqrt(ml**2+k**2)#omega_ell(T, k)
                return e**2/(2*np.pi)**3*k**2\
                *Z_ell(T, k)*wl*(ml**2+2*mx**2)\
                *np.sqrt(ml**2*(ml**2-4*mx**2))\
                /(3*ml**4*(np.exp(wl/T)-1))
        else:
            kudu = np.zeros_like(k)
            for i in range(len(k)):
                ml = m_ell(T, k[i])
                km=kmax(T)
                if mx<ml/2 and k[i]<km:
                    wl = np.sqrt(ml**2+k[i]**2)#omega_ell(T, k[i])
                    kudu[i] = e**2/(2*np.pi)**3*k[i]**2\
                    *Z_ell(T, k[i])*wl*(ml**2+2*mx**2)\
                    *np.sqrt(ml**2*(ml**2-4*mx**2))\
                    /(3*ml**4*(np.exp(wl/T)-1))
            return kudu  
    else:
        if type(k) == int or type(k) == float or type(k)== np.float64:
            ml = m_ell(T, k)
            if mx>ml/2:
                return 0
            elif k>kmax(T):
                return 0
            else:
                wl = np.sqrt(ml**2+k**2)#omega_ell(T, k)
                return Q**2*e**2/(2*np.pi)**3*k**2\
                *Z_ell(T, k)*wl*(ml**2+2*mx**2)\
                *np.sqrt(ml**2*(ml**2-4*mx**2))\
                /(3*ml**4*(np.exp(wl/T)-1))
        else:
            kudu = np.zeros_like(k)
            for i in range(len(k)):
                ml = m_ell(T, k[i])
                km=kmax(T)
                if mx<ml/2 and k[i]<km:
                    wl = np.sqrt(ml**2+k[i]**2)#omega_ell(T, k[i])
                    kudu[i] = Q**2*e**2/(2*np.pi)**3*k[i]**2\
                    *Z_ell(T, k[i])*wl*(ml**2+2*mx**2)\
                    *np.sqrt(ml**2*(ml**2-4*mx**2))\
                    /(3*ml**4*(np.exp(wl/T)-1))
            return kudu

def trans_integrand(Q, mx, T, k, noQ = False):
    """
    transverse case
    """
    mt = m_t(T, k)
    if noQ:
        if mx>mt/2:
            return 0
        else:
            wt = np.sqrt(mt**2+k**2)#omega_t(T, k)
            return 2*e**2/(2*np.pi)**3*k**2\
            *Z_t(T, k)*(mt**2+2*mx**2)*np.sqrt(mt**2*(mt**2 - 4*mx**2))\
            /(3*wt*mt**2 *(np.exp(wt/T)-1))
    else:
        if mx>mt/2:
            return 0
        else:
            wt = np.sqrt(mt**2+k**2)#omega_t(T, k)
            return 2*Q**2*e**2/(2*np.pi)**3*k**2\
            *Z_t(T, k)*(mt**2+2*mx**2)*np.sqrt(mt**2*(mt**2 - 4*mx**2))\
            /(3*wt*mt**2 *(np.exp(wt/T)-1))
        
def kkin_trans(mx, T):
    return brentq(lambda k: m_t(T, k)-2*mx, 1e-3,10*T)#fsolve(lambda k: m_t(T, k)-2*mx, mx)

def kkin_long(mx, T):
    return fsolve(lambda k: m_ell(T, k)-2*mx, mx)
    
def gamma_long(Q, mx, T,noQ = False):
    """
    longitudional reaction rate of plasmon reactions
    """
    if mx>wp(T)/2:
        return 0
    #kkin = kkin_long(mx, T)
    else:
        #return integrate.quadrature(lambda k: long_integrand(Q,mx,T,k),0, kmax(T), tol=1e-9)
        return integrate.quad(lambda k: long_integrand(Q,mx,T,k,noQ = noQ),0, kmax(T))[0]#,min(kmax(T), kkin))[0]
    
def gamma_trans(Q, mx, T,noQ = False):
    """
    transverse reaction rate of plasmon reactions
    """
    if mx>np.sqrt(3/2.)*wp(T)/2:
        return 0
    else:
    #kkin = kkin_trans(mx, T)
        return integrate.quad(lambda k: trans_integrand(Q,mx,T,k, noQ = noQ),0, np.inf)[0]#,kkin)[0]
    #, np.inf)[0]#
    
def gamma_tot(Q, mx, T, noQ = False,muon=True):
    """ 
    Given a value of Q and mx returns the reaction rate.
    """
    return gamma_ann(Q, mx, T, noQ = noQ,muon=muon)+gamma_long(Q, mx, T,noQ = noQ)+gamma_trans(Q, mx, T, noQ = noQ)

def relic_ann(Q, mx, whole_shebang=False, muon=True, noQ = False):
    """ 
    Given a value of Q and mx returns the number density over relic abundance due to e+e- annihilations.
    """
    Tlist = np.logspace(2.5, -5,num=500)
    gamma_list = [gamma_ann(Q, mx, i,muon=muon, noQ = noQ) for i in Tlist]
    gamma=scipy.interpolate.interp1d(Tlist, gamma_list, fill_value = 'extrapolate')
    alist = np.logspace(-1.9, 4.5,num=500)
    if noQ:
        # This section is for plotting lines of constant abundance by solving a modified boltzmann
        # equation and not using scaling arguments.
        if mx < 0.001:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(6e-05/10**(-30))], alist)
        else:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(6e-05/10**(-22))], alist)
        #print("This function is returning (nDM(modified no Q) final, a_max)")
        return np.array([np.exp(gulu[-1][-1]),alist[-1]])
    else:
        if mx < 0.001:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(6e-05)], alist)
        else:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(6e-05)], alist)
        if whole_shebang:
            print(np.exp(gulu[-1])/nxrelic(mx, T(100))*alist[-1]**3/100**3)
            return np.exp(gulu).flatten()
        else:
            return np.exp(np.squeeze(gulu[-1]))/nxrelic(mx, T(100))*alist[-1]**3/100**3


def relic(Q, mx,whole_shebang=False, muon=True, noQ = False):
    """ 
    Returns number density over relic density due to plasmon decay and e+e- annihilations.
    whole_shebang is the progression of this.
    """
    Tlist = np.logspace(2.5, -5,num=500)
    gamma_list = [gamma_tot(Q, mx, i, muon=muon, noQ = noQ) for i in Tlist]
    gamma=scipy.interpolate.interp1d(Tlist, gamma_list, fill_value = 'extrapolate')
    alist = np.logspace(-1.9, 4.5, num = 500)
    
    if noQ:
        # This section is for plotting lines of constant abundance by solving a modified boltzmann
        # equation and not using scaling arguments.
        if mx < 0.001:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(0.004653/10**(-30))], alist)
        else:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(0.004653/10**(-22))], alist)
        #print("This function is returning (nDM(modified no Q) final, a_max)")
        return np.array([np.exp(gulu[-1][-1]),alist[-1]])
    else:
        if mx < 0.001:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(0.004653)], alist)
        else:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(0.004653)], alist)
        if whole_shebang:
            print(np.exp(gulu[-1])/nxrelic(mx, T(100))*alist[-1]**3/100**3)
            return np.exp(gulu).flatten()
        else:
            return np.exp(np.squeeze(gulu[-1]))/nxrelic(mx, T(100))*alist[-1]**3/100**3
    

def relic_plas(Q, mx, whole_shebang=False, noQ = False):
    
    Tlist = np.logspace(2.5, -5, num = 500)
    gamma_list = [gamma_long(Q, mx, i, noQ = noQ)+gamma_trans(Q, mx, i,noQ = noQ) for i in Tlist]
    gamma=scipy.interpolate.interp1d(Tlist, gamma_list, fill_value = 'extrapolate')
    alist = np.logspace(-1.9, 4.5, num = 500)
    
    if noQ:
        # This section is for plotting lines of constant abundance by solving a modified boltzmann
        # equation and not using scaling arguments.
        if mx < 0.001:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(0.004591/10**(-30))], alist)
        else:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(0.004591/10**(-22))], alist)
        #print("This function is returning (nDM(modified no Q) final, a_max)")
        return np.array([np.exp(gulu[-1][-1]),alist[-1]])
    else:
        if mx < 0.001:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(0.004591)], alist)
        else:
            gulu = scipy.integrate.odeint(lambda logn, a: \
                - 3/a +2/(np.exp(logn)*a*H(a))*gamma(T(a)), [np.log(0.004591)], alist)
        if whole_shebang:
            print(np.exp(gulu[-1])/nxrelic(mx, T(100))*alist[-1]**3/100**3)
            return np.exp(gulu).flatten()
        else:
            return np.exp(np.squeeze(gulu[-1]))/nxrelic(mx, T(100))*alist[-1]**3/100**3
        
    
def s_nu(T):
    return 4*rho_nu(10)/30 *(a(10)/a(T))**3

def freezeout(Q, mx, whole_shebang=False):
    Tlist = np.logspace(2, -2.1,num=200)
    gamma_list = [gamma_ann(Q, mx, i,muon=False) for i in Tlist]
    gamma=interp1d(Tlist, gamma_list)
    neq_list = [integrate.quad(lambda E: \
    2/np.pi**2*E * np.sqrt(E**2-mx**2)/(np.exp(E/i)+1), mx, np.inf)[0] for i in Tlist]
    neq = interp1d(Tlist, neq_list)
    slist = [s_nu(i)+ s_e(i)+s_gamma(i) for i in Tlist]
    s = interp1d(Tlist, slist)
    #f, ax = plt.subplots()
    #ax.loglog(Tlist, neq_list)
    alist = np.logspace(-1.8, 2,num=200)
    #gulu = integrate.odeint(lambda n, a: \
            #- 3*n/a +1/(2*a*H(a))*gamma(T(a))*(1-n**2/neq(T(a))**2), [neq(T(alist[0]))], alist)
    #print(neq(T(alist[0]))/s(T(alist[0])))
    gulu = integrate.odeint(lambda Y, a: -a *gamma(T(a))*s(T(a))/(H(1)*neq(T(a))**2)*(Y**2 - neq(T(a))**2/s(T(a))**2),\
                          [neq(T(alist[0]))/s(T(alist[0]))], alist) 
    if whole_shebang:
        print(gulu[-1]*s(T(alist[-1]))/nxrelic(mx, T(100))*alist[-1]**3/100**3)
        return gulu.flatten()
    else:
        return gulu[-1]/nxrelic(mx, T(100))*alist[-1]**3/100**3