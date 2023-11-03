#!/bin/env/python
#! -*- coding: utf-8 -*-
import numpy as _np
from pylab import subplots as _subplots
from matplotlib import cm as _cm

from Quadratures import n_theorie

def plot_C4_correction(Cdc,C4dc_init,Cac,C4ac_init,Labels,label_slice=slice(None)):
    fig,ax = _subplots(2,1)
    color_idx = _np.linspace(0, 1.0, len(Labels))
    for c2,c4,c2ac,c4ac,label,c_idx in zip( Cdc[label_slice,...,2], C4dc_init[label_slice],Cac[label_slice,...,2], C4ac_init[label_slice], Labels[label_slice],color_idx[label_slice] ):
        l, = ax[0].plot(c2,c4,marker='None',ls='-',label=label,color=_cm.cool(c_idx))
        ax[0].plot(c2ac,c4ac,marker='None',ls='-',color=l.get_color())
    for c2,c4,c2ac,c4ac,c_idx in zip( Cdc[label_slice,...,2], Cdc[label_slice,...,4], Cac[label_slice,...,2], Cac[label_slice,...,4],color_idx[label_slice]  ) :
        l, = ax[1].plot(c2,c4,marker='None',ls='-',color=_cm.cool(c_idx))
        ax[1].plot(c2ac,c4ac,marker='None',ls='-',color=l.get_color())
    ax[1].set_xlabel('C2')
    ax[0].set_ylabel('C4 initial')
    ax[1].set_ylabel('C4 corrected')
    fig.tight_layout()
    fig.legend(loc='lower right',ncol=1, bbox_to_anchor=(0.95, 0.55),title = 'Mode',fontsize=12)
    return ax

def plot_betas(freq,betas,Labels):
    fig , ax = _subplots(1,1)
    color_idx = _np.linspace(0, 1.0, len(Labels))
    for beta , l,c_idx in zip (betas,Labels,color_idx[:]) :
        ax.plot(freq*1e-9,abs(beta)*1e-9,label=l,color=_cm.cool(c_idx))
    ax.set_xlabel(r'$f$[GHz]')
    ax.set_ylabel(r'$\beta(f)$[Hz$^{-1/2}$]')
    ax.set_xlim(0,12)
    fig.tight_layout()
    #fig.legend(loc='lower right',ncol=1, bbox_to_anchor=(0.965, 0.525),title = 'Mode',fontsize=12)
    return ax

def plot_n_vs_I_DC(I,ns,P,freq_mins,freq_maxs,Labels,R,Te,theorie=False):
    label_slice=slice(None)
    fig, ax = _subplots(1,1)

    color_idx = _np.linspace(0, 1.0, len(Labels[label_slice]))

    A = P[label_slice,0]
    B = -P[label_slice,1]

    for var,a,b,l,c_idx,f_min,f_max in zip( ns[...,0] , A  , B ,Labels[label_slice] ,color_idx, freq_mins,freq_maxs) :
        if theorie :
            ax.plot(I*1e6, (var )  ,marker='.',ls='None',markersize=10.,color=_cm.cool(c_idx))
            ax.plot(I*1e6, (a*I-b)  ,ls='--',color=_cm.cool(c_idx))
            ax.plot(I*1e6, n_theorie(I,f_min,f_max,R,Te),color=_cm.cool(c_idx),ls='-',label=l)
        else :
            ax.plot(I*1e6, (var )  ,marker='.',ls='-',markersize=10.,color=_cm.cool(c_idx),label=l)
            ax.plot(I*1e6, (a*I-b)  ,ls='--',color=_cm.cool(c_idx))
            
    #ax.set_ylim(0)
    ax.set_ylabel(r'$\langle n \rangle $ [~]')
    ax.set_xlabel(r'$I_{dc}[\mu A]$')
    fig.tight_layout()
    #fig.legend(loc='best', bbox_to_anchor=(1.15, 0.9))
    #fig.legend(loc='lower right',ncol=1, bbox_to_anchor=(0.95, 0.55),title = 'Mode',fontsize=12)
    return ax
        
def plot_n_vs_I_AC(I,ns,Labels,label_slice=slice(None)):
    fig, ax = _subplots(1,1)
    color_idx = _np.linspace(0, 1.0, len(Labels[label_slice]))
    for var,l,c_idx in zip( ns[...,0] ,Labels[label_slice] ,color_idx) :
        ax.plot(I*1e6, (var )  ,marker='.',ls='-',markersize=10.,color=_cm.cool(c_idx),label=l)   
    ax.set_ylabel(r'$\langle n \rangle $ [~]')
    ax.set_xlabel(r'$I_{ac}$[$\mu A$ rms]')
    fig.tight_layout()
    #fig.legend(loc='lower right',ncol=1, bbox_to_anchor=(0.95, 0.55),title = 'Mode',fontsize=12)
    return ax
    
def plot_dn2_vs_I_DC(I,ns,Labels,label_slice=slice(None)):
    fig, ax = _subplots(1,1)
    color_idx = _np.linspace(0, 1.0, len(Labels[label_slice]))
    for var,l,c_idx in zip( ns[...,1] ,Labels[label_slice] ,color_idx) :
        ax.plot(I*1e6, (var )  ,marker='.',ls='-',markersize=10.,color=_cm.cool(c_idx),label=l)
    ax.set_ylabel(r'$\langle \delta n^2 \rangle $ [~]')
    ax.set_xlabel(r'$I_{dc}[\mu A]$')
    fig.tight_layout()
    #fig.legend(loc='lower right',ncol=1, bbox_to_anchor=(0.95, 0.55),title = 'Mode',fontsize=12)
    #ax.grid()
    return ax
    
def plot_dn2_vs_I_AC(I,ns,Labels,label_slice=slice(None)):
    fig, ax = _subplots(1,1)
    color_idx = _np.linspace(0, 1.0, len(Labels[label_slice]))
    for var,l,c_idx in zip( ns[...,1] ,Labels[label_slice] ,color_idx) :
        ax.plot(I*1e6, (var )  ,marker='.',ls='-',markersize=10.,color=_cm.cool(c_idx),label=l)
    ax.set_ylabel(r'$\langle \delta n^2 \rangle $ [~]')
    ax.set_xlabel(r'$I_{ac}$[$\mu A$ rms]')
    fig.tight_layout()
    #fig.legend(loc='lower right',ncol=1, bbox_to_anchor=(0.95, 0.55),title = 'Mode',fontsize=12)
    #ax.grid()
    return ax
    
def plot_fano(ns,Labels,label_slice=slice(None),NNplusUn=True):
    fig, ax = _subplots(1,1)
    color_idx = _np.linspace(0, 1.0, len(Labels[label_slice]))
    for n,dn2,l,c_idx in zip( ns[label_slice,...,0],ns[label_slice,...,1] , Labels[label_slice] ,color_idx) :
        ax.plot(n, (dn2)/(n)  ,marker='.',ls='-',markersize=15.,color=_cm.cool(c_idx),label=l) 
        if NNplusUn :
            ax.plot(n,(n+1),color='k')
    ax.set_ylim(1,3.0)
    ax.set_xlim(-0.1,2.0)
    ax.set_ylabel(r'$\langle \delta n^2 \rangle / \langle n \rangle$ [~]')
    ax.set_xlabel(r'$\langle n\rangle $ [~]')
    fig.tight_layout()
    #fig.legend(loc='lower right',ncol=1, bbox_to_anchor=(0.95, 0.55),title = 'Mode',fontsize=12)
    return ax
