#!/bin/env/python
#! -*- coding: utf-8 -*-

import numpy as _np
from numpy.fft import rfft, ifftshift, rfftfreq
from numpy import pi,convolve
from scipy import constants as C
from scipy.ndimage import convolve1d
from scipy import polyfit
from matplotlib.pyplot import subplots
from numba import njit,vectorize,float64


from SBB.Data_analysis.window import window_after
from SBB.Math_extra.Math_extra import central_derivative_3points
from SBB.Numpy_extra.numpy_extra import symetrize,find_nearest_A_to_a
from SBB.Data_analysis.fit import polyfit_multi_between,polyfit_multi_check,lstsq,lstsq_2D
from SBB.Phys.Tunnel_Junction import Sdc_of_f,Sdc_asym_of_f
from SBB.Data_Manager.Data_Manager import remove_nan_subarrays

from Methods import build_imin_imax,reshape_reorder_all,reshape_reorder_all_1,reshape_reorder_dSII,centered_refed_dSII,dSII_reshape_reorder, centered_ref_X,V_jct_unreshape_reorder
from SBB.Pyhegel_extra.Experiment import get_all_with_key

_dt = 31.25e-12


def ROUTINE_SCOPE_0(Vyoko,Vjct,Vpol,SII,Rpol,V_per_bin,flip_Ipol=False) :
    """
       
    """
    
    Vjct = remove_nan_subarrays(Vjct)
    Vpol = remove_nan_subarrays(Vpol)
    SII  = remove_nan_subarrays(SII)
        
    SII   = SII*(V_per_bin)**2 / 50.0**2 ## A**2
    dSII = centered_refed_dSII(SII)
    Vyoko,Vjct,Vpol,Sii,dSii = reshape_reorder_all(Vyoko,Vjct,Vpol,SII,dSII)
    
    
    
    if flip_Ipol :
        Ipol = (-1)*_np.nanmean( Vpol/Rpol, axis = 0 )
    else: 
        Ipol = _np.nanmean( Vpol/Rpol, axis = 0 )
    
    return Vjct,Ipol,Sii,dSii
    
def ROUTINE_SCOPE_1(Vyoko,SII,Rtot,V_per_bin) :
    """
    no dmms   
    """
    SII  = remove_nan_subarrays(SII)
    SII   = SII*(V_per_bin)**2 / 50.0**2 ## A**2
    dSII = centered_refed_dSII(SII)
    Vyoko,Sii,dSii = reshape_reorder_all_1(Vyoko,SII,dSII)
    Ipol = Vyoko/Rtot
    return Ipol,Sii,dSii

def ROUTINE_LOAD_0(file,R_pol = 100.0185e3,V_per_bin=0.00021957308303365236,flip_Ipol=False) :
    """
        Loads already combiened and re-ordered data
        Returns 
        Vdc, I_pol, SII, dSII
    """
    data  = _np.load(file)
    Vdc   = data['Vdc']
    if flip_Ipol :
        I_pol = (-1)*_np.nanmean( data['V_pol']/R_pol, axis = 0 )
    else:
        I_pol = _np.nanmean( data['V_pol']/R_pol, axis = 0 )
    V_jct = data['V_jct']
    SII   = data['SII']*(V_per_bin)**2 / 50.0**2 ## A**2
    dSII  = data['dSII']*(V_per_bin)**2 / 50.0**2 ## A**2
    return Vdc, I_pol, V_jct, SII, dSII
    
def ROUTINE_LOAD_1(file,Rtot = 110204,V_per_bin=0.00021957308303365236,flip_Ipol=False) :
    """
        Loads already combiened and re-ordered data
        Returns 
        Vdc, I_pol, SII, dSII
    """
    data  = _np.load(file)
    Vdc   = data['Vdc']
    
    Ipol = _np.full((2,2,len(Vdc)),_np.nan)
    if flip_Ipol :
        Ipol = (-1)*Vdc/Rtot
    else: 
        Ipol = Vdc/Rtot
    SII   = data['SII']*(V_per_bin)**2 / 50.0**2 ## A**2
    dSII  = data['dSII']*(V_per_bin)**2 / 50.0**2 ## A**2
    return Vdc, Ipol, SII, dSII
    
def ROUTINE_COMBINE_LOAD_0(files) :
    """
        Returns 
        Vdc, SII, dSII
    """
    Vdc   = get_all_with_key(files,'Vdc',)
    try :
        V_pol = get_all_with_key(files,'V_1M',)
    except KeyError :
        V_pol = get_all_with_key(files,'V_pol',)
    V_jct = get_all_with_key(files,'V_jct',)
    SII   = get_all_with_key(files,'SII',)
    dSII  = [centered_refed_dSII(sii) for sii in SII]  
    return Vdc,V_pol,V_jct,SII, dSII

def ROUTINE_COMBINE_LOAD_1(files) :
    """
        Returns 
        Vdc, SII, dSII
    """
    Vdc   = get_all_with_key(files,'Vdc',)
    SII   = get_all_with_key(files,'SII',)
    dSII  = [centered_refed_dSII(sii) for sii in SII]  
    return Vdc,SII, dSII
    
def ROUTINE_COMBINE_LOAD_2(files) :
    """
        Returns 
        Vdc,Vac, SII, dSII
    """
    Vdc   = get_all_with_key(files,'Vdc',)
    Vac   = get_all_with_key(files,'Vac',)
    Labels  = get_all_with_key(files,'_meta_info',EVAL="[()]['filter_info']['labels']")
    
    SII_dc   = get_all_with_key(files,'S2_vdc',)
    dSII_dc  = [centered_refed_dSII(sii) for sii in SII_dc]
    SII_ac   = get_all_with_key(files,'S2_vac',)
    dSII_ac  = [centered_refed_dSII(sii) for sii in SII_ac]
    
    ks    = get_all_with_key(files,'ks',)
    betas    = get_all_with_key(files,'betas',)
    filters    = get_all_with_key(files,'filters',)
    
    G_avg    = get_all_with_key(files,'G_avg',)
    data_gz    = get_all_with_key(files,'data_gz',)
    quads    = get_all_with_key(files,'quads',)
    hs_vdc    = get_all_with_key(files,'hs_vdc',)
    hs_vac    = get_all_with_key(files,'hs_vac',)
    
    moments_dc    = get_all_with_key(files,'moments_dc',)
    moments_ac    = get_all_with_key(files,'moments_ac',)
    
    return Vdc,Vac,Labels,SII_dc, dSII_dc,SII_ac, dSII_ac,ks,betas,filters,G_avg,data_gz,quads,hs_vdc,hs_vac,moments_dc,moments_ac
    
def ROUTINE_COMBINE_LOAD_3(files) :
    """
        Returns 
        Vdc,Vac, SII, dSII
    """
    Labels  = get_all_with_key(files,'_meta_info',EVAL="[()]['filter_info']['labels']")
    
    R_jct      = get_all_with_key(files,'_meta_info',EVAL="[()]['R_jct']")
    F          = get_all_with_key(files,'_meta_info',EVAL="[()]['F']")
    filter_info = get_all_with_key(files,'_meta_info',EVAL="[()]['filter_info']")
    V_per_bin  = get_all_with_key(files,'_meta_info',EVAL="[()]['V_per_bin']")
    gain_fit_params = get_all_with_key(files,'_meta_info',EVAL="[()]['gain_fit_params']")
    l_kernel = get_all_with_key(files,'_meta_info',EVAL="[()]['l_kernel']")
    
    return R_jct,F,filter_info,V_per_bin,gain_fit_params,l_kernel
    
def ROUTINE_FIT_T (ipol,freq,dSIIx,i_slice,f_slice,Rjct=70.0,T_xpctd=0.055,tol=1e-15):
    """
    VARIATION ON ROUTINE_FIT_R
    2D (Idc,f) fit on excess noise to find only Te simultaneously
    """
    def fit_func_2D(I,f,p):
        Te = p[0]
        tmp = Sdc_of_f(2*pi*f[None,:],(C.e*(I*Rjct)/C.hbar)[:,None],Te,Rjct)
        return tmp.flatten()

    p0 = [T_xpctd,]

    data = _np.nanmean(dSIIx[:,i_slice,f_slice],axis=0)
    F = lstsq_2D(ipol[i_slice],freq[f_slice],data.flatten(),p0,fit_func_2D,tol=tol)
    Te_fit = F[0][0]
    return Te_fit

def ROUTINE_Vjct_0(V_jct):
    neg = centered_ref_X(V_jct[:,0,1],V_jct[:,0,0])    
    pos = centered_ref_X(V_jct[:,1,1],V_jct[:,1,0])    
    return neg,pos
    
def ROUTINE_Vjct_1(ipol,Vjct,mov_avg=None):
    L = len(ipol)
    cnd,ref = V_jct_unreshape_reorder(Vjct) 
    Vjct = centered_ref_X(cnd,ref)
    di = _np.mean(ipol[1:] - ipol[0:-1])
    ipol = _np.r_[ (-1)*ipol[::-1], ipol ] 
    ipol = _np.delete(ipol,L,axis=-1)  # removing the duplicate at 0 current
    cntr = _np.nanmean( Vjct[...,[L-1,L]], axis=1 ) # avg center bin
    Vjct = _np.delete(Vjct,L,axis=-1)  # removing the duplicate at 0 current
    Vjct[...,L-1] = cntr # avg center bin
    dVdI = central_derivative_3points(di,Vjct)
    if not(mov_avg is None) :
        k = _np.ones((mov_avg,))/float(mov_avg)
        dVdI = convolve1d(dVdI,k,mode='constant',cval=0.0,axis=-1)
        if mov_avg >1 :
            dVdI[...,:mov_avg]    = _np.nan # keeping valif part only
            dVdI[...,-mov_avg+1:] = _np.nan
    return ipol,Vjct,dVdI
    
def ROUTINE_Vjct_2(ipol,Vjct):
    L = len(ipol)
    cnd,ref = V_jct_unreshape_reorder(Vjct) 
    Vjct = centered_ref_X(cnd,ref)
    ipol = _np.r_[ (-1)*ipol[::-1], ipol ] 
    ipol = _np.delete(ipol,L,axis=-1)  # removing the duplicate at 0 current
    cntr = _np.nanmean( Vjct[...,[L-1,L]], axis=1 ) # avg center bin
    Vjct = _np.delete(Vjct,L,axis=-1)  # removing the duplicate at 0 current
    Vjct[...,L-1] = cntr # avg center bin
    return ipol,Vjct
 
def ROUTINE_FIT_COUL_BLOCK(ipol,Vjct,i_below=-1.e-6,i_above=1.e-6,i_mask=None,Dcb=1.0e-6,tol=1e-15):
    vjct = _np.nanmean(Vjct,axis=0)        
    @vectorize([float64(float64, float64,float64)])
    def func(i,R,Vo):
        if i == 0 :
            return 0.0
        else :
            return i*R+_np.sign(i)*Vo
    
    def fit_func(I,p):
        R,Vo = p[0],p[1]
        return func(I,R,Vo)
    
    below=(ipol<i_below)
    above=(ipol>i_above)
    i_slice = _np.where( (below|above)&i_mask )[0]
    
    x    = ipol[i_slice]
    data = vjct[i_slice]
    p0 = [50,1e-6]
    R,Vo = lstsq(x,data,p0,fit_func,tol=tol)[0] 
    
    def fit_func(I,p):
        Dcb = p[0]
        return (Vo*_np.tanh(I/abs(Dcb)))
    
    x    = ipol[i_mask]
    data = vjct[i_mask] - R*x
    
    p0 = [Dcb,]
    
    Dcb, = lstsq(x,data,p0,fit_func,tol=tol)[0] 
    
    return R,Vo,Dcb,x,data,fit_func(ipol,[Dcb])/ipol+R
    
def ROUTINE_SII_0(I_pol,SII,dSII,fast=True,windowing=True,i=65,l_kernel=257) :
    """
    Computes different noise_of_f metrics.
    Returns 
        freq,ipol,SII_of_f,SII_antisym_of_f,dSII_of_f
    """
    
    freq = rfftfreq(l_kernel,_dt)
    
    if fast :
        SII   = _np.nanmean(SII  ,axis = 0)[None,...]
        dSII  = _np.nanmean(dSII ,axis = 0)[None,...]

    if I_pol.ndim >=3 :
        I_mean = _np.mean( [I_pol[0,0],I_pol[1,0]])
        I_pol -= I_mean ## removing offset
        ipol = I_pol[1,1]
    else : #1 dim
        ipol = I_pol

    if windowing :
        SII     = window_after(SII , i=i, t_demi=1)
        dSII    = window_after(dSII , i=i, t_demi=1)

    SII_sym     = SII.mean(axis=1)
    dSII_sym    = dSII.mean(axis=1)
    SII_antisym = ( SII[:,1,...] - SII[:,0,...] )/2.0

    SII_of_f         =  ( rfft(ifftshift(symetrize(SII_sym)    ,axes=-1))*_dt ).real
    SII_antisym_of_f =  ( rfft(ifftshift(symetrize(SII_antisym),axes=-1))*_dt ).real
    dSII_of_f        =  ( rfft(ifftshift(symetrize(dSII_sym)   ,axes=-1))*_dt ).real
    
    return freq,ipol,SII_of_f,SII_antisym_of_f,dSII_of_f
    
def ROUTINE_SII_1(dSII,fast=True,windowing=True,i=65) :
    """
    Computes different noise_of_f metrics.
    Returns 
        freq,ipol,SII_of_f,SII_antisym_of_f,dSII_of_f
    """
    if fast :
        dSII  = _np.nanmean(dSII ,axis = 0)[None,...]
    if windowing :
        dSII    = window_after(dSII , i=i, t_demi=1)
    dSII_sym    = dSII.mean(axis=1)
    dSII_of_f        =  ( rfft(ifftshift(symetrize(dSII_sym)   ,axes=-1))*_dt ).real
    
    return dSII_of_f

def ROUTINE_SII_2(SII,fast=True,windowing=True,i=65) :
    """
    Computes different noise_of_f metrics.
    Returns 
        SII_of_f,
    """
    if fast :
        SII  = _np.nanmean(SII ,axis = 0)[None,...]
    if windowing :
        SII    = window_after(SII , i=i, t_demi=1)
    SII_of_f        =  ( rfft(ifftshift(symetrize(SII_sym)   ,axes=-1))*_dt ).real  
    return SII_of_f
        
def ROUTINE_GAIN_0 (freq,ipol,SII_of_f,dSII_of_f,degree = 1,R=70.00,T_xpctd=0.055,fmax =10.e9,imax=2.0e-6,epsilon=0.0001):
    """
        Gain, Noise Temperature, zero bias excess noise
    """
    shape = dSII_of_f.swapaxes(-2,-1).shape[:-1]
    imin,imax = build_imin_imax(freq,shape,R=R,T=T_xpctd,fmax=fmax,imax=imax,epsilon=epsilon)

    P  = polyfit_multi_between(ipol, SII_of_f[:,1,:,:].swapaxes(-2,-1),imin,imax,deg=degree)
    dP = polyfit_multi_between(ipol, dSII_of_f.swapaxes(-2,-1),imin,imax,deg=degree)

    G_of_f  = P[...,-2]/C.e
    B_of_f  = P[...,-1]
    dG_of_f  = dP[...,-2]/C.e
    dB_of_f = dP[...,-1]
    
    return G_of_f, B_of_f, P, dG_of_f, dB_of_f, dP
    
def ROUTINE_GAIN_1(freq,ipol,dSII_of_f,degree = 1,R=70.00,T_xpctd=0.055,fmax =10.e9,imax=2.0e-6,epsilon=0.0001):
    """
        Gain, Noise Temperature, zero bias excess noise
    """
    shape = dSII_of_f.swapaxes(-2,-1).shape[:-1]
    imin,imax = build_imin_imax(freq,shape,R=R,T=T_xpctd,fmax=fmax,imax=imax,epsilon=epsilon)
    dP = polyfit_multi_between(ipol, dSII_of_f.swapaxes(-2,-1),imin,imax,deg=degree)
     
    dG_of_f  = dP[...,-2]/C.e
    dB_of_f = dP[...,-1]
    
    return dG_of_f, dB_of_f, dP
    
def ROUTINE_GAIN_2 (freq,ipol,SII_of_f,dSII_of_f,degree = 1,R=70.00,T_xpctd=0.055,fmax =10.e9,imax=2.0e-6,epsilon=0.0001):
    """
        Returns Noise Temperature(B_of_f), gain (dG_of_f), zero bias excess noise (dB_of_f)
    """
    shape = dSII_of_f.swapaxes(-2,-1).shape[:-1]
    imin,imax = build_imin_imax(freq,shape,R=R,T=T_xpctd,fmax=fmax,imax=imax,epsilon=epsilon)

    P  = polyfit_multi_between(ipol, SII_of_f[:,1,:,:].swapaxes(-2,-1),imin,imax,deg=degree)
    dP = polyfit_multi_between(ipol, dSII_of_f.swapaxes(-2,-1),imin,imax,deg=degree)

    B_of_f  = P[...,-1]
    dG_of_f = dP[...,-2]/C.e
    dB_of_f = dP[...,-1]
    
    return B_of_f, dG_of_f, dB_of_f
    
def ROUTINE_GAIN_3 (freq,ipol,SII_of_f,dSII_of_f,gain_fit_params):
    """
        Returns gain (dG_of_f)
    """
    shape = dSII_of_f.swapaxes(-2,-1).shape[:-1]
    imin,imax = build_imin_imax(freq,shape,**gain_fit_params)

    P  = polyfit_multi_between(ipol, SII_of_f[:,1,:,:].swapaxes(-2,-1),imin,imax,deg=1)
    dP = polyfit_multi_between(ipol, dSII_of_f.swapaxes(-2,-1),imin,imax,deg=1)

    B_of_f  = P[...,-1]
    dG_of_f = dP[...,-2]/C.e
    dB_of_f = dP[...,-1]
    
    return B_of_f, dG_of_f, dB_of_f
    
def ROUTINE_AVG_GAIN(Vyoko,SII,Rtot,V_per_bin,l_kernel,gain_fit_params,windowing=True,i=65):
    """
    Intended for quadrature experiments
    Return dG_of_f
        No dmm ==> Ipol deduced from R_tot
    """ 
    SII  = _np.nanmean(SII,axis=0)[None,:]*(V_per_bin)**2 / 50.0**2 ## A**2
    dSII = centered_refed_dSII(SII)
    Vyoko,dSII = reshape_reorder_dSII(Vyoko,dSII)
    Ipol = Vyoko[1,1]/Rtot
    freq = rfftfreq(l_kernel,_dt)
    if windowing :
        dSII    = window_after(dSII , i=i, t_demi=1)
    dSII_sym    = dSII.mean(axis=1)
    dSII_of_f   = ( rfft(ifftshift(symetrize(dSII_sym)   ,axes=-1))*_dt ).real
    shape       = dSII_of_f.swapaxes(-2,-1).shape[:-1]
    imin,imax   = build_imin_imax(freq,shape,**gain_fit_params)
    dP          = polyfit_multi_between(Ipol, dSII_of_f.swapaxes(-2,-1),imin,imax,deg=1) 
    return dP[0,...,-2]/C.e


def ROUTINE_dSIIx(dSII_of_f,dB_of_f,dG_of_f):
    return (dSII_of_f-dB_of_f[:,None,:])/dG_of_f[:,None,:]
    
def ROUTINE_dSIIx_1(dSII_of_f,dB_of_f,dG_of_f,SII_antisym_of_f):
    dSIIx = (dSII_of_f-dB_of_f[:,None,:])/dG_of_f[:,None,:]
    dSIIx_asym = SII_antisym_of_f[:,1,...]/dG_of_f[:,None,:]
    return dSIIx,dSIIx_asym
    
def ROUTINE_FIT_T (ipol,freq,dSIIx,i_slice,f_slice,Rjct=70.0,T_xpctd=0.055,tol=1e-15):
    """
    VARIATION ON ROUTINE_FIT_R
    2D (Idc,f) fit on excess noise to find only Te simultaneously
    """
    def fit_func_2D(I,f,p):
        Te = p[0]
        tmp = Sdc_of_f(2*pi*f[None,:],(C.e*(I*Rjct)/C.hbar)[:,None],Te,Rjct)
        return tmp.flatten()

    p0 = [T_xpctd,]

    data = _np.nanmean(dSIIx[:,i_slice,f_slice],axis=0)
    F = lstsq_2D(ipol[i_slice],freq[f_slice],data.flatten(),p0,fit_func_2D,tol=tol)
    Te_fit = F[0][0]
    return Te_fit

def ROUTINE_FIT_RT(ipol,freq,dSIIx,i_slice,f_slice,R_xptd=70.0,T_xpctd=0.055,tol=1e-15):
    """
    2D (Idc,f) fit on excess noise to find Rjct and Te simultaneously
    """
    def fit_func_2D(I,f,p):
        Te = p[0]
        R  = p[1]
        tmp = Sdc_of_f(2*pi*f[None,:],(C.e*(I*R)/C.hbar)[:,None],Te,R)
        return tmp.flatten()

    p0 = [T_xpctd,R_xptd]

    data = _np.nanmean(dSIIx[:,i_slice,f_slice],axis=0)
    F = lstsq_2D(ipol[i_slice],freq[f_slice],data.flatten(),p0,fit_func_2D,tol=tol)
    Te_fit = F[0][0]
    R_fit = F[0][1]
    return R_fit,Te_fit
    
def ROUTINE_COULBLOCK_FIT_T (ipol,r,freq,dSIIx,i_slice,f_slice,T_xpctd=0.055,tol=1e-15):
    """
    VARIATION ON ROUTINE_FIT_R
    2D (Idc,f) fit on excess noise to find only Te simultaneously
    """
    def fit_func_2D(X,f,p):
        I,R = X[0],X[1]
        Te = p[0]
        tmp = Sdc_of_f(2*pi*f[None,:],(C.e*(I*R)/C.hbar)[:,None],Te,R[:,None])
        return tmp.flatten()

    p0 = [T_xpctd,]

    data = _np.nanmean(dSIIx[:,i_slice,f_slice],axis=0)
    X = [ipol[i_slice],r[i_slice]]
    F = lstsq_2D(X,freq[f_slice],data.flatten(),p0,fit_func_2D,tol=tol)
    Te_fit = F[0][0]
    return Te_fit
    
def ROUTINE_COULBLOCK_FIT_RT (ipol,dr,freq,dSIIx,i_slice,f_slice,R_xptd=70.0,T_xpctd=0.055,tol=1e-15):
    """
    VARIATION ON ROUTINE_FIT_R
    2D (Idc,f) fit on excess noise to find only Te simultaneously
    """
    def fit_func_2D(X,f,p):
        I,dR = X[0],X[1]
        Te = p[0]
        R  = p[1]+dR
        tmp = Sdc_of_f(2*pi*f[None,:],(C.e*(I*R)/C.hbar)[:,None],Te,R[:,None])
        return tmp.flatten()

    p0 = [T_xpctd,R_xptd]

    data = _np.nanmean(dSIIx[:,i_slice,f_slice],axis=0)
    X = [ipol[i_slice],dr[i_slice]]
    F = lstsq_2D(X,freq[f_slice],data.flatten(),p0,fit_func_2D,tol=tol)
    Te_fit = F[0][0]
    R_fit = F[0][1]
    return R_fit,Te_fit
    
def ROUTINE_FIT_TvsF(ipol,freq,dSIIx,Rjct=70.0,T_xpctd=0.055,tol=1e-15):
    """
    Temperature fit for each frequency
    """
    Temps = []
    for sii ,f  in zip ( _np.moveaxis( dSIIx[:,:,:],-1,0),freq[:] ):
        p0 = [T_xpctd,]
        def fit_func(I,p):
            return Sdc_of_f(2*pi*f,(C.e*(I*Rjct)/C.hbar),p[0],Rjct)
        Temps += [ lstsq(ipol,_np.nanmean(sii,axis=0),p0,fit_func,tol=tol)[0][0],]
    Temps = _np.r_[Temps]
    return Temps
  
def ROUTINE_COULBLOCK_FIT_TvsF(ipol,freq,dSIIx,RvsI,T_xpctd=0.055,tol=1e-15):
    """
    Temperature fit for each frequency
    """
    Temps = []
    for sii ,f  in zip ( _np.moveaxis( dSIIx[:,:,:],-1,0),freq[:] ):
        p0 = [T_xpctd,]
        def fit_func(I,p):
            return Sdc_of_f(2*pi*f,(C.e*(I*RvsI)/C.hbar),p[0],RvsI)
        Temps += [ lstsq(ipol,_np.nanmean(sii,axis=0),p0,fit_func,tol=tol)[0][0],]
    Temps = _np.r_[Temps]
    return Temps
    
def ROUTINE_TEMP_in_TIME(ipol,freq,dSII,R,imax=0.8e-6,fmin=3.1e9,fmax=10.0e9,fast=True,windowing=True,i_win=65):
    imax_idx = find_nearest_A_to_a(imax,ipol)[1][0]
    fmin_idx = find_nearest_A_to_a(fmin,freq)[1][0]
    fmax_idx = find_nearest_A_to_a(fmax,freq)[1][0]
    i_slice = slice(0,imax_idx)         # ipol < 0.8uA
    f_slice = slice(fmin_idx,fmax_idx)  # 3.1 à 9.8 GHz
    def find_T_between(n_begin,n_end,dSII):
        dSII = dSII[n_begin:n_end]
        dSII_of_f       = ROUTINE_SII_1(dSII,fast=fast,windowing=windowing,i=i_win) 
        dG_of_f, dB_of_f, _ = ROUTINE_GAIN_1(freq,ipol,dSII_of_f,degree=1)
        dSIIx = ROUTINE_dSIIx(dSII_of_f,dB_of_f,dG_of_f)
        Te_fit = ROUTINE_FIT_T(ipol,freq,dSIIx,i_slice,f_slice,Rjct=R)
        return Te_fit
    
    Te_fit = []
    L = len(dSII)
    for n_being,n_end in zip(_np.r_[:L],_np.r_[1:L+1]):
        tmp = find_T_between(n_being,n_end,dSII)
        Te_fit +=[ tmp ]
    return _np.r_[Te_fit]

def FIT_I0 (ipol,freq,dSII_asym,i_slice=slice(None),f_slice=slice(15,80),R_xptd=50.0,T_xpctd=0.016,tol=1e-15):
    def fit_func_2D(I,f,p):
        i0 = p[0]
        tmp = Sdc_asym_of_f(2*pi*f[None,:],(C.e*I*R_xptd/C.hbar)[:,None], C.e*i0*R_xptd/C.hbar,T_xpctd,R_xptd)
        return tmp.flatten()
    p0 = [5.0e-9,]
    data = _np.nanmean(dSII_asym[...,i_slice,f_slice],axis=0)
    F = lstsq_2D(ipol[i_slice],freq[f_slice],data.flatten(),p0,fit_func_2D,tol=tol)
    i0 = F[0][0]
    return i0

def FIT_I0_w_dR (ipol,dr,freq,dSII_asym,i_slice=slice(None),f_slice=slice(15,80),R_xptd=50.0,T_xpctd=0.016,tol=1e-15):
    def fit_func_2D(X,f,p):
        I,dR = X[0],X[1]
        i0 = p[0]
        R  = R_xptd+dR
        tmp = Sdc_asym_of_f(2*pi*f[None,:],(C.e*I*R/C.hbar)[:,None], (C.e*i0*R/C.hbar)[:,None],T_xpctd,R[:,None])
        return tmp.flatten()
    p0 = [5.0e-9,]
    data = _np.nanmean(dSII_asym[...,i_slice,f_slice],axis=0)
    X = [ipol[i_slice],dr[i_slice]]
    F = lstsq_2D(X,freq[f_slice],data.flatten(),p0,fit_func_2D,tol=tol)
    i0 = F[0][0]
    return i0
