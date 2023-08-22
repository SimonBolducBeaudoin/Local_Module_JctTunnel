#!/bin/env/python
#! -*- coding: utf-8 -*-

import numpy as _np
from SBB.Matplotlib_extra.plot import plot_interval,scope_interval_update
from SBB.Phys.Tunnel_Junction import Sdc_of_f,V_th
from SBB.Data_analysis.fit import polyfit_multi_between,lstsq,lstsq_2D
from SBB.Numpy_extra.numpy_extra import reshape_axis,slice_axes

import pdb

def reshape_reorder_swap(Y,l,axis=-1,sym=True,ref='interlaced',copy=True):
    if copy : # if false the original array is modified
        Y = Y.copy() 
    ax = range(Y.ndim)[axis] # conveting to positive only axis
    # Reshape
    if sym and (ref is 'interlaced') :
        x_shape = (2,l,2)
    elif sym and (ref is 'first') :
        x_shape = (2,l)
    elif not(sym) and (ref is 'interlaced') :
        x_shape = (l,2)
    else : # not(sym) and (ref is 'first') :
        x_shape = (l,)
    Y = reshape_axis(Y,x_shape,axis=ax) # Reshaping
    
    # Reordering
    if sym and (ref is 'interlaced') :
        # Y[...,0,:,:,...] = Y[...,0,::-1,:,...] 
        Y_view =  slice_axes(Y, ((ax,0),(ax+1,':')) )    # Getting a view of Y
        Y_view[...] =  slice_axes(Y, ((ax,0),(ax+1,'::-1')) ) # Coyping data at the right place
        return Y.swapaxes(ax+1,ax+2) # ... (2,2,l) ... == ... (down/up,ref/cdn,,vdc) ...
    elif sym and (ref is 'first') :
        # Y[...,0,:,:,...] = Y[...,0,::-1,:,...] 
        Y_view =  slice_axes(Y, ((ax,0),(ax+1,':')) )    # Getting a view of Y
        Y_view[...] =  slice_axes(Y, ((ax,0),(ax+1,'::-1')) ) # Coyping data at the right place
        return Y # ... (2,l) ...
    elif not(sym) and (ref is 'interlaced') :
        # Y[...,0,:,:,...] = Y[...,0,::-1,:,...] 
        return Y.swapaxes(ax,ax+1) # ... (2,l) ...
    elif not(sym) and (ref is 'first') :
        return Y
    else :
        return None
    
def Vdc_reshape_reorder(Vdc,Vdc_shape,l,sym=True,ref='interlaced'):
    """Deprecated"""
    return reshape_reorder_swap(Vdc,l,axis=-1,sym=sym,ref=ref)
def V_jct_reshape_reorder(V_jct,Vjct_shape,l,sym=True,ref='interlaced'):
    """Deprecated"""
    return reshape_reorder_swap(V_jct,l,axis=-1,sym=sym,ref=ref)
def SII_reshape_reorder(SII,SII_shape,l,sym=True,ref='interlaced'):
    """Deprecated"""
    return reshape_reorder_swap(SII,l,axis=-2,sym=sym,ref=ref)
def dSII_reshape_reorder(dSII,dSII_shape,l,sym=True):
    """Deprecated"""
    #tmp = _np.full(dSII_shape,_np.nan)
    #tmp[...,0,:,:] = dSII[...,: l ,:][...,::-1,:]
    #tmp[...,1,:,:] = dSII[...,  l: ,:]    
    return reshape_reorder_swap(dSII,l,axis=-2,sym=sym,ref='first')

def V_jct_unreshape_reorder(Vjct):
    ref = _np.concatenate( (Vjct[...,0,0,:][...,::-1], Vjct[...,1,0,:]),axis=-1 )
    cnd = _np.concatenate( (Vjct[...,0,1,:][...,::-1], Vjct[...,1,1,:]),axis=-1 )
    return cnd,ref

def reshape_reorder_all(Vdc,V_jct,V_pol,SII,dSII):
    l = len(Vdc)//4
    Vdc_shape  = (2,2,l)
    Vjct_shape = V_jct.shape[:-1] + Vdc_shape
    SII_shape  = SII.shape[:-2]   + Vdc_shape + SII.shape[-1:]
    dSII_shape =  SII.shape[:1]   + (2,l)     + SII.shape[-1:]
    
    Vdc   = Vdc_reshape_reorder(Vdc,Vdc_shape,l) 
    V_jct = V_jct_reshape_reorder(V_jct,Vjct_shape,l)
    V_pol = V_jct_reshape_reorder(V_pol,Vjct_shape,l)
    SII   = SII_reshape_reorder(SII,SII_shape,l)
    dSII  = dSII_reshape_reorder(dSII,dSII_shape,l)
    
    return Vdc,V_jct,V_pol,SII,dSII
    
def reshape_reorder_all_1(Vdc,SII,dSII):
    l = len(Vdc)//4
    Vdc_shape  = (2,2,l)
    SII_shape  = SII.shape[:-2]   + Vdc_shape + SII.shape[-1:]
    dSII_shape =  SII.shape[:1]   + (2,l)     + SII.shape[-1:]
    
    Vdc   = Vdc_reshape_reorder(Vdc,Vdc_shape,l) 
    SII   = SII_reshape_reorder(SII,SII_shape,l)
    dSII  = dSII_reshape_reorder(dSII,dSII_shape,l)
    
    return Vdc,SII,dSII

def reshape_reorder_all_2(Vdc,Vac,SII_dc,dSII_dc,SII_ac,dSII_ac,hs_vdc,hs_vac,moments_dc,moments_ac,sym=True,ref='interlaced'):
    ldc = len(Vdc)//4 # sym and interlacing
    lac = len(Vac)//2 # sym and interlacing
    
    Vdc   = reshape_reorder_swap(Vdc,ldc,axis=-1,sym=sym,ref=ref)
    Vac   = reshape_reorder_swap(Vac,lac,axis=-1,sym=False,ref=ref)
    
    SII_dc   = reshape_reorder_swap(SII_dc,ldc,axis=-2,sym=sym,ref=ref)
    dSII_dc  = reshape_reorder_swap(dSII_dc,ldc,axis=-2,sym=sym,ref='first')
    SII_ac   = reshape_reorder_swap(SII_ac,lac,axis=-2,sym=False,ref=ref)
    dSII_ac  = reshape_reorder_swap(dSII_ac,lac,axis=-2,sym=False,ref='first')
    
    hs_vdc   = reshape_reorder_swap(hs_vdc,ldc,axis=-2,sym=sym,ref=ref)
    hs_vac   = reshape_reorder_swap(hs_vac,lac,axis=-2,sym=False,ref=ref)
    
    moments_dc   = reshape_reorder_swap(moments_dc,ldc,axis=-2,sym=sym,ref=ref)
    moments_ac   = reshape_reorder_swap(moments_ac,lac,axis=-2,sym=False,ref=ref)
    
    return Vdc,Vac,SII_dc,dSII_dc,SII_ac,dSII_ac,hs_vdc[None,...],hs_vac[None,...],moments_dc,moments_ac

def reshape_reorder_dSII(Vdc,dSII):
    l = len(Vdc)//4
    Vdc_shape  = (2,2,l)
    dSII_shape =  dSII.shape[:1]   + (2,l)     + dSII.shape[-1:]
    
    Vdc   = Vdc_reshape_reorder(Vdc,Vdc_shape,l) 
    dSII  = dSII_reshape_reorder(dSII,dSII_shape,l)
    
    return Vdc,dSII
       
def centered_ref_X(X,ref=None):
    """
    Un-biased X calculation (uses centered reference)
    X.shape = (...,L)
    
    X = [cdn0,ref0,cdn1,ref1,...]
    """
    if ref is None : 
        cnd = X[...,1::2] # suppose odd positions are the conditions
        ref = X[..., ::2] # suppose even positions are the references
    else :
        cnd = X
    Y   = _np.full(cnd.shape,_np.nan)
    Y[...,:-1] = cnd[...,:-1] - (ref[...,0:-1] + ref[...,1:] )/2.0 # other points are centered 
    Y[...,-1 ] = cnd[...,-1 ] -  ref[...,-1] # last point is biased 
    return Y
    
def centered_refed_dSII(SII,axis=-2) :
    """
    Un-biased dSII calculation (uses centered reference)
    """
    tmp  = SII.swapaxes(axis,-1)
    return centered_ref_X(tmp,ref=None).swapaxes(axis,-1)
    
def build_imin_imax(freq,shape,R=50.00,T=0.055,fmax =10.e9,imax = 2.1e-6,epsilon=0.0001):
    imin = _np.full(shape,_np.nan)
    for i,f in enumerate(freq) :
        if f < fmax :
            imin[:,i] = V_th(f,T,epsilon)/R
        else :
            imin[:,i] = V_th(fmax,T,epsilon)/R
    return imin,imax
    
def build_ipol_ipolsym(V_yoko,Cpol):
    ### Usefull!!!????
    l = len(V_yoko)//4
    Vdc_shape  = (2,2,l)
    ipol = V_yoko*Cpol
    Ipol = Vdc_reshape_reorder(V_yoko*Cpol,Vdc_shape,l) 
    ipol_sym = Ipol[1,1]
    return ipol,ipol_sym