#!/bin/env/python
#! -*- coding: utf-8 -*-

from Scopes import VvsI_scope_0,VvsI_scope_1,VvsI_scope_2,gain_scope,noise_Temps_scope,dSIIx_vs_I_scope,Tvsf_scope,dSIIx_origin_Vsf_scope,Temps_in_Time_scope
from Routines import ROUTINE_SCOPE_0,ROUTINE_SCOPE_1,ROUTINE_SII_0,ROUTINE_GAIN_0,ROUTINE_FIT_T,ROUTINE_FIT_TvsF,ROUTINE_Vjct_0,ROUTINE_Vjct_1,ROUTINE_Vjct_2,ROUTINE_dSIIx,ROUTINE_TEMP_in_TIME,ROUTINE_FIT_COUL_BLOCK

from matplotlib.pyplot import subplots
from SBB.Numpy_extra.numpy_extra import find_nearest_A_to_a

def Tunnel_jct_multi_scopes_0(Vyoko,Vjct,Vpol,SII,Rjct,Rpol=100.0185e3,V_per_bin=0.00021957308303365236,T_xpctd=0.055,imax_temp=0.8e-6,imax_gain=2.0e-6,fmin=3.1e9,fmax=10.0e9,fast=True,windowing=True,axes=None) :
    Vjct,Ipol,Sii,dSII,dSII_time            = ROUTINE_SCOPE_0(Vyoko,Vjct,Vpol,SII,Rpol,V_per_bin) 
    Vjct_neg,Vjct_pos                             = ROUTINE_Vjct_0(Vjct)
    freq,ipol,SII_of_f,SII_antisym_of_f,dSII_of_f = ROUTINE_SII_0(Ipol,Sii,dSII,fast=fast,windowing=windowing)
    G_of_f, B_of_f, _, dG_of_f, dB_of_f, _        = ROUTINE_GAIN_0(freq,ipol,SII_of_f,dSII_of_f,degree=1,R=Rjct,T_xpctd=T_xpctd,fmax=fmax,imax=imax_gain,epsilon=0.0001)
    dSIIx                                         = ROUTINE_dSIIx(dSII_of_f,dB_of_f,dG_of_f)
    
    imax_idx = find_nearest_A_to_a(imax_temp,ipol)[1][0]
    fmin_idx = find_nearest_A_to_a(fmin,freq)[1][0]
    fmax_idx = find_nearest_A_to_a(fmax,freq)[1][0]
    i_slice = slice(0,imax_idx)         # ipol < 0.8uA
    f_slice = slice(fmin_idx,fmax_idx)  # 3.1 à 9.8 GHz
    
    Te_fit  = ROUTINE_FIT_T(ipol,freq,dSIIx,i_slice,f_slice,Rjct,T_xpctd=T_xpctd,tol=1e-15)
    Temps   = ROUTINE_FIT_TvsF(ipol,freq,dSIIx,Rjct,T_xpctd=T_xpctd,tol=1e-15)
    
    T_in_time = ROUTINE_TEMP_in_TIME(ipol,freq,dSII_time,Rjct,imax=imax_temp,fmin=fmin,fmax=fmax,fast=True,windowing=windowing)
    
    if axes is None : #creating all axes
        axes =[]
        for i in range(7) :
            _, ax = subplots(1,1)
            axes += [ax,]
            
    _ = VvsI_scope_0      (ipol,Vjct_neg,Vjct_pos,ax=axes[0])
    _ = gain_scope        (freq,G_of_f,ax=axes[1])
    _ = noise_Temps_scope (freq,G_of_f,B_of_f,ax=axes[2])
    _ = dSIIx_vs_I_scope  (ipol,freq,dSIIx,f_slice=slice(30,75,5),ax=axes[3],R=Rjct,T=Te_fit)
    _ = Tvsf_scope        (freq,Temps,Te_fit,ax=axes[4])
    
    _ = dSIIx_origin_Vsf_scope  (freq,dB_of_f,dG_of_f,Te=Te_fit,R=Rjct,ax=axes[5])
    _ = Temps_in_Time_scope     (T_in_time,ax=axes[6])
    
    for ax in axes :
        fig = ax.get_figure()
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    return axes
    
def Tunnel_jct_multi_scopes_1(Vyoko,Vjct,Vpol,SII,Rjct,Rpol=100.0185e3,V_per_bin=0.00021957308303365236,T_xpctd=0.055,imax_temp=0.8e-6,imax_gain=2.0e-6,fmin=3.1e9,fmax=10.0e9,fast=True,windowing=True,axes=None) :
    Vjct,Ipol,Sii,dSII                            = ROUTINE_SCOPE_0(Vyoko,Vjct,Vpol,SII,Rpol,V_per_bin)
    dSII_time                                     = dSII.copy()    
    freq,ipol,SII_of_f,SII_antisym_of_f,dSII_of_f = ROUTINE_SII_0(Ipol,Sii,dSII,fast=fast,windowing=windowing)
    isym,Vjct,dVdI                                = ROUTINE_Vjct_1(ipol,V_jct,mov_avg=20)
    _, B_of_f, _, dG_of_f, dB_of_f, _             = ROUTINE_GAIN_0(freq,ipol,SII_of_f,dSII_of_f,degree=1,R=Rjct,T_xpctd=T_xpctd,fmax=fmax,imax=imax_gain,epsilon=0.0001)
    dSIIx                                         = ROUTINE_dSIIx(dSII_of_f,dB_of_f,dG_of_f)
    
    imax_idx = find_nearest_A_to_a(imax_temp,ipol)[1][0]
    fmin_idx = find_nearest_A_to_a(fmin,freq)[1][0]
    fmax_idx = find_nearest_A_to_a(fmax,freq)[1][0]
    i_slice = slice(0,imax_idx)         # ipol < 0.8uA
    f_slice = slice(fmin_idx,fmax_idx)  # 3.1 à 9.8 GHz
    
    Te_fit  = ROUTINE_FIT_T(ipol,freq,dSIIx,i_slice,f_slice,Rjct,T_xpctd=T_xpctd,tol=1e-15)
    Temps   = ROUTINE_FIT_TvsF(ipol,freq,dSIIx,Rjct,T_xpctd=T_xpctd,tol=1e-15)
    
    T_in_time = ROUTINE_TEMP_in_TIME(ipol,freq,dSII_time,Rjct,imax=imax_temp,fmin=fmin,fmax=fmax,fast=True,windowing=windowing)
    
    if axes is None : #creating all axes
        axes =[]
        for i in range(7) :
            _, ax = subplots(1,1)
            axes += [ax,]
            
                
    _ = VvsI_scope_1      (isym,Vjct,dVdI,imin=1e-6,imax=2.1e-6,plot_deviation=True,ax=None)
    _ = gain_scope        (freq,dG_of_f,ax=axes[1])
    _ = noise_Temps_scope (freq,dG_of_f,B_of_f,ax=axes[2])
    _ = dSIIx_vs_I_scope  (ipol,freq,dSIIx,f_slice=slice(30,75,5),ax=axes[3],R=Rjct,T=Te_fit)
    _ = Tvsf_scope        (freq,Temps,Te_fit,ax=axes[4])
    
    _ = dSIIx_origin_Vsf_scope  (freq,dB_of_f,dG_of_f,Te=Te_fit,R=Rjct,ax=axes[5])
    _ = Temps_in_Time_scope     (T_in_time,ax=axes[6])
    
    for ax in axes :
        fig = ax.get_figure()
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    return axes
    
def Tunnel_jct_multi_scopes_2(Vyoko,Vjct,Vpol,SII,Rjct,Rpol=100.0185e3,V_per_bin=0.00021957308303365236,T_xpctd=0.055,imax_temp=0.8e-6,imax_gain=2.0e-6,fmin=3.1e9,fmax=10.0e9,fast=True,windowing=True,axes=None) :
    Vjct,Ipol,Sii,dSII                            = ROUTINE_SCOPE_0(Vyoko,Vjct,Vpol,SII,Rpol,V_per_bin) 
    dSII_time                                     = dSII.copy()
    freq,ipol,SII_of_f,SII_antisym_of_f,dSII_of_f = ROUTINE_SII_0(Ipol,Sii,dSII,fast=fast,windowing=windowing)
    isym,Vjct                                     = ROUTINE_Vjct_2(ipol,Vjct)
    mask = isym != isym[0]
    R,Vo,Dcb,x,data,RvsI                          = ROUTINE_FIT_COUL_BLOCK(isym,Vjct,i_below=-1.e-6,i_above=1.e-6,i_mask=mask,tol=1e-15)
    rvsi = RvsI[len(RvsI)//2:]
    dRvsI = rvsi - R
    _, B_of_f, _, dG_of_f, dB_of_f, _        = ROUTINE_GAIN_0(freq,ipol,SII_of_f,dSII_of_f,degree=1,R=Rjct,T_xpctd=T_xpctd,fmax=fmax,imax=imax_gain,epsilon=0.001)
    dSIIx                                         = ROUTINE_dSIIx(dSII_of_f,dB_of_f,dG_of_f)
    
    imax_idx = find_nearest_A_to_a(imax_temp,ipol)[1][0]
    fmin_idx = find_nearest_A_to_a(fmin,freq)[1][0]
    fmax_idx = find_nearest_A_to_a(fmax,freq)[1][0]
    i_slice = slice(0,imax_idx)         # ipol < 0.8uA
    f_slice = slice(fmin_idx,fmax_idx)  # 3.1 à 9.8 GHz
    
    Te_fit  = ROUTINE_FIT_T(ipol,freq,dSIIx,i_slice,f_slice,Rjct,T_xpctd=T_xpctd,tol=1e-15)
    Temps   = ROUTINE_FIT_TvsF(ipol,freq,dSIIx,Rjct,T_xpctd=T_xpctd,tol=1e-15)
    
    T_in_time = ROUTINE_TEMP_in_TIME(ipol,freq,dSII_time,Rjct,imax=imax_temp,fmin=fmin,fmax=fmax,fast=True,windowing=windowing)
    
    if axes is None : #creating all axes
        axes =[]
        for i in range(7) :
            _, ax = subplots(1,1)
            axes += [ax,]
            
                          
    _ = VvsI_scope_2      (isym,Vjct,R,Vo,Dcb,x,data,RvsI,ax=axes[0])
    _ = gain_scope        (freq,dG_of_f,ax=axes[1])
    _ = noise_Temps_scope (freq,dG_of_f,B_of_f,ax=axes[2])
    _ = dSIIx_vs_I_scope  (ipol,freq,dSIIx,f_slice=slice(30,75,5),ax=axes[3],R=Rjct,T=Te_fit)
    _ = Tvsf_scope        (freq,Temps,Te_fit,ax=axes[4])
    
    _ = dSIIx_origin_Vsf_scope  (freq,dB_of_f,dG_of_f,Te=Te_fit,R=Rjct,ax=axes[5])
    _ = Temps_in_Time_scope     (T_in_time,ax=axes[6])
    
    for ax in axes :
        fig = ax.get_figure()
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    return axes
    
def Tunnel_jct_multi_scopes_3(Vyoko,SII,Rjct,Rtot=100.e3,V_per_bin=0.00021957308303365236,T_xpctd=0.055,imax_temp=0.8e-6,imax_gain=2.0e-6,fmin=3.1e9,fmax=10.0e9,fast=True,windowing=True,axes=None) :
    """
    No dmm
    """ 
    Ipol,Sii,dSII                                 = ROUTINE_SCOPE_1(Vyoko,SII,Rtot,V_per_bin) 
    dSII_time                                     = dSII.copy()
    freq,ipol,SII_of_f,SII_antisym_of_f,dSII_of_f = ROUTINE_SII_0(Ipol,Sii,dSII,fast=fast,windowing=windowing)
    _, B_of_f, _, dG_of_f, dB_of_f, _        = ROUTINE_GAIN_0(freq,ipol,SII_of_f,dSII_of_f,degree=1,R=Rjct,T_xpctd=T_xpctd,fmax=fmax,imax=imax_gain,epsilon=0.001)
    dSIIx                                         = ROUTINE_dSIIx(dSII_of_f,dB_of_f,dG_of_f)
    
    imax_idx = find_nearest_A_to_a(imax_temp,ipol)[1][0]
    fmin_idx = find_nearest_A_to_a(fmin,freq)[1][0]
    fmax_idx = find_nearest_A_to_a(fmax,freq)[1][0]
    i_slice = slice(0,imax_idx)         # ipol < 0.8uA
    f_slice = slice(fmin_idx,fmax_idx)  # 3.1 à 9.8 GHz
    
    Te_fit  = ROUTINE_FIT_T(ipol,freq,dSIIx,i_slice,f_slice,Rjct,T_xpctd=T_xpctd,tol=1e-15)
    Temps   = ROUTINE_FIT_TvsF(ipol,freq,dSIIx,Rjct,T_xpctd=T_xpctd,tol=1e-15)
    
    T_in_time = ROUTINE_TEMP_in_TIME(ipol,freq,dSII_time,Rjct,imax=imax_temp,fmin=fmin,fmax=fmax,fast=True,windowing=windowing)
    
    if axes is None : #creating all axes
        axes =[]
        for i in range(6) :
            _, ax = subplots(1,1)
            axes += [ax,]
            
    _ = gain_scope        (freq,dG_of_f,ax=axes[0])
    _ = noise_Temps_scope (freq,dG_of_f,B_of_f,ax=axes[1])
    _ = dSIIx_vs_I_scope  (ipol,freq,dSIIx,f_slice=slice(30,75,5),ax=axes[2],R=Rjct,T=Te_fit)
    _ = Tvsf_scope        (freq,Temps,Te_fit,ax=axes[3])
    _ = dSIIx_origin_Vsf_scope  (freq,dB_of_f,dG_of_f,Te=Te_fit,R=Rjct,ax=axes[4])
    _ = Temps_in_Time_scope     (T_in_time,ax=axes[5])
    
    for ax in axes :
        fig = ax.get_figure()
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    return axes