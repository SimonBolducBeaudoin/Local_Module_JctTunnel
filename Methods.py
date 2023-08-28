#!/bin/env/python
#! -*- coding: utf-8 -*-

import numpy as _np
from SBB.Matplotlib_extra.plot import plot_interval,scope_interval_update
from SBB.Phys.Tunnel_Junction import Sdc_of_f,V_th
from SBB.Data_analysis.fit import polyfit_multi_between,lstsq,lstsq_2D
from SBB.Numpy_extra.numpy_extra import reshape_axis,slice_axes,fuse_axes

def reshape_reorder_swap(Y,axis=-1,sym=True,ref='interlaced',copy=True):
    if copy : # if false the original array is modified
        Y = Y.copy() 
    ax = range(Y.ndim)[axis] # conveting to positive only axis
    # Reshape
    if sym and (ref is 'interlaced') :
        l = Y.shape[ax]//4
        x_shape = (2,l,2)
    elif sym and (ref in ['first','None']) :
        l = Y.shape[ax]//2
        x_shape = (2,l)
    elif not(sym) and (ref is 'interlaced') :
        l = Y.shape[ax]//2
        x_shape = (l,2)
    else : # not(sym) and (ref in ['first','None','none']) :
        l = Y.shape[ax]
        x_shape = (l,)
    Y = reshape_axis(Y,x_shape,axis=ax) # Reshaping
    
    # Reordering
    if sym and (ref is 'interlaced') :
        # Y[...,0,:,:,...] = Y[...,0,::-1,:,...] 
        Y_view      =  slice_axes(Y, ((ax,0),(ax+1,':')) )    # Getting a view of Y
        Y_view[...] =  slice_axes(Y, ((ax,0),(ax+1,'::-1')) ) # Coyping data at the right place
        return Y.swapaxes(ax+1,ax+2) # ... (2,2,l) ... == ... (down/up,ref/cdn,,vdc) ...
    elif sym and (ref in ['first','None']) :
        # Y[...,0,:,:,...] = Y[...,0,::-1,:,...] 
        Y_view      =  slice_axes(Y, ((ax,0),(ax+1,':')) )    # Getting a view of Y
        Y_view[...] =  slice_axes(Y, ((ax,0),(ax+1,'::-1')) ) # Coyping data at the right place
        return Y # ... (2,l) ...
    elif not(sym) and (ref is 'interlaced') :
        # Y[...,0,:,:,...] = Y[...,0,::-1,:,...] 
        return Y.swapaxes(ax,ax+1) # ... (2,l) ...
    elif not(sym) and (ref in ['first','None']) :
        return Y
    else :
        return None
        
def unreshape_reorder_swap(Y,axis=-1,sym=True,ref='interlaced',copy=True):
    if copy : # if false the original array is modified
        Y = Y.copy() 
    # Reordering
    if sym and (ref is 'interlaced') :
        ax = range(Y.ndim-2)[axis] 
        Y = Y.swapaxes(ax+1,ax+2)
        Y_view      =  slice_axes(Y, ((ax,0),(ax+1,':')) )   
        Y_view[...] =  slice_axes(Y, ((ax,0),(ax+1,'::-1')) ) 
    elif sym and (ref in ['first','None']) :
        ax = range(Y.ndim-1)[axis] 
        Y_view      =  slice_axes(Y, ((ax,0),(ax+1,':')) )    
        Y_view[...] =  slice_axes(Y, ((ax,0),(ax+1,'::-1')) ) 
    elif not(sym) and (ref is 'interlaced') :
        ax = range(Y.ndim-1)[axis] 
        Y = Y.swapaxes(ax+1,ax+2)
    else : # not(sym) and (ref in ['first','None']) :
        ax = range(Y.ndim)[axis] 
    # Reshape
    if sym and (ref is 'interlaced') :
        axes = (ax,ax+1,ax+2)
    elif sym and (ref in ['first','None']) :
        axes = (ax,ax+1)
    elif not(sym) and (ref is 'interlaced') :
        axes = (ax,ax+1)
    else : # not(sym) and (ref in ['first','None']) :
        axes = (ax,)
    return fuse_axes(Y,axes)
           
def centered_ref_X(X,ref=None,axis=-1,copy=True):
    """
    Un-biased X calculation (uses centered reference)
    X.shape = (...,L)
    
    X = [cdn0,ref0,cdn1,ref1,...]
    """
    if copy :
         X.copy()
    X  = X.swapaxes(axis,-1)
    if ref is None : 
        cnd = X[...,1::2] # suppose odd  positions are the conditions
        ref = X[..., ::2] # suppose even positions are the references
    else :
        cnd = X
    Y   = _np.full(cnd.shape,_np.nan)
    Y[...,:-1] = cnd[...,:-1] - (ref[...,0:-1] + ref[...,1:] )/2.0 # other points are centered 
    Y[...,-1 ] = cnd[...,-1 ] -  ref[...,-1] # last point is biased 
    return Y.swapaxes(axis,-1)
    
def build_imin_imax(freq,shape,R=50.00,T=0.055,fmax =10.e9,imax = 2.1e-6,epsilon=0.0001):
    imin = _np.full(shape,_np.nan)
    for i,f in enumerate(freq) :
        if f < fmax :
            imin[:,i] = V_th(f,T,epsilon)/R
        else :
            imin[:,i] = V_th(fmax,T,epsilon)/R
    return imin,imax
    
def V_jct_unreshape_reorder(Vjct):
    """Deprecated"""
    ref = _np.concatenate( (Vjct[...,0,0,:][...,::-1], Vjct[...,1,0,:]),axis=-1 )
    cnd = _np.concatenate( (Vjct[...,0,1,:][...,::-1], Vjct[...,1,1,:]),axis=-1 )
    return cnd,ref