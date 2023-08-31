#!/bin/env/python
#! -*- coding: utf-8 -*-

import numpy as _np
from matplotlib.pyplot import subplots
from scipy import constants as _C

from SBB.Phys.Tunnel_Junction           import C4_th, dC2ac_th , n_th, dn2_th , dnac_th , ddn2ac_th
from SBB.Numpy_extra.numpy_extra        import sub_flat,slice_axis,sub_flatten_no_copy
from SBB.Data_analysis.fit              import lstsq,polyfit_multi_between
from SBB.Histograms.histograms_helper   import std_moments_to_moments

from Methods import build_imin_imax

def extract_numbers(input_list):
    result = []
    for item in input_list:
        numbers = item.split('&')
        extracted_numbers = []
        for num in numbers:
            try:
                extracted_numbers.append(float(num))
            except ValueError:
                pass  # Ignore non-numeric values
        result.append(extracted_numbers)
    return result

def gen_fmins_fmaxs(Labels):
    labels = extract_numbers(Labels)
    freq_maxs = _np.r_[ [max(L) for L in labels]]*1e9
    freq_mins = _np.r_[ [min(L) for L in labels]]*1e9
    return freq_mins,freq_maxs

def Cmpt_cumulants(std_moments,fold=2.0):
    """
    1D moments ==> 1D cumulants
    """
    moments = std_moments_to_moments(std_moments)
    cumulants           =  _np.full( moments.shape  ,0.0) #_np.full( shape ,_np.nan)
    cumulants[...,2]    =   moments[...,2] / fold
    cumulants[...,4]    = ( moments[...,4]     - 3.0*(moments[...,2] )**2  ) /(fold**2)
    return cumulants

def Cmpt_std_cumulants(std_moments,fold=2.0):
    """
    1D moments ==> 1D cumulants
    """
    cumulants           =  _np.full( std_moments.shape  ,0.0) #_np.full( shape ,_np.nan)
    cumulants[...,2]    =  1.0 / fold
    cumulants[...,4]    = ( std_moments[...,4]  - 3.0  ) /(fold**2)
    return cumulants

def C_to_n(C):
    ns =  _np.full( C.shape[:-1] + (2,) ,_np.nan )
    ns[...,0] = (C[...,2]  ) - 1./2
    ns[...,1] = (2./3)*C[...,4] + C[...,2]**2   - 1./4
    return ns

def Add_Vac_to_Cdc(Ipol,Cdc,f_maxs,R,Te,fmax=10000000000.0, imax=2.1e-06,epsilon=0.001):
    imin,imax = build_imin_imax(f_maxs,Cdc[...,2].shape[:-1], R=R,T=Te,fmax=fmax, imax=imax,epsilon=epsilon)
    Pdc = polyfit_multi_between(Ipol,Cdc[...,2],imin,imax)
    Vacuum = - Pdc[...,1]
    Cdc[...,2] += Vacuum[...,None] # Add origin
    return Cdc,Pdc

def n_theorie(I,f1,f2,R,Te):
    n1 = n_th(f1*2*_np.pi,_C.e*I*R/_C.hbar,Te=Te,R=R)
    n2 = n_th(f2*2*_np.pi,_C.e*I*R/_C.hbar,Te=Te,R=R)
    return (n1+n2)/2.0
    

def Fit_nonlin(C4,C2,cdn_th_i=slice(None),p0=(1.0,),show_fit=True,copy=False):
    """
    Computes non lineare coefficient affecting C4
    
    Inputs 
        C (array) of cumulants (Total cumulants not sample cumulants)
    """
    if copy :
        C4 = C4.copy()
        C2 = C2.copy()
    C4_shape = sub_flatten_no_copy(C4    ,axis=-1) # No copy saves initial shape
    C2_shape = sub_flatten_no_copy(C2 ,axis=-1) # No copy saves initial shape
    A        = _np.full( C4.shape[:-1] + (len(p0),), _np.nan )  
    
    # fit m
    def non_lin_model(c2,p):
        """
        C4 = K*C2**3
        """
        return p[0]*c2**3
    for i,(c4,c2) in enumerate(zip( C4[...,cdn_th_i], C2[...,cdn_th_i] )):
        f    = lstsq(c2,c4,p0,non_lin_model)
        A[i] = f[0]
    # Restoring shapes
    A.shape = C4_shape[:-1] + (len(p0),)
    C4.shape = C4_shape
    C2.shape = C2_shape
    ax = None
    if show_fit :
        fig,ax = subplots(2,1)
        for c4,a,c2 in zip(sub_flat(_np.nanmean(C4,axis=0)[None,:]),sub_flat(_np.nanmean(A,axis=0)[None,:]),sub_flat(_np.nanmean(C2,axis=0)[None,:])):
            l,= ax[0].plot(c2,c4,ls='None',marker='.',markersize=10)
            ax[0].plot(c2,non_lin_model(c2,a),color=l.get_color())
            ax[1].plot(c2,c4-non_lin_model(c2,a),ls='None',marker='.',markersize=10)
        ax[0].set_xlabel('C2')
        ax[0].set_ylabel('C4')
        ax[1].set_xlabel('C2')
        ax[1].set_ylabel('C4')
        ax[0].set_xlim(0)
        ax[1].set_xlim(0)
    return A , non_lin_model , ax

def Cmpt_C4_corrected(C4,C2,A,model) :
    tmp = _np.expand_dims(_np.moveaxis(A,-1,0),-1)
    return C4 - model(C2,tmp)

def C4_correction(Cdc,Cac,fuse_last_two_axis=True,show_fit=False):
    """
    Computes Non linear coefficient for the DC case and then applies corrections
    on C4 for the DC and AC case.
    """
    if fuse_last_two_axis :
        dc_shape = Cdc.shape
        Cdc = Cdc.reshape( (dc_shape[:-3] + (dc_shape[-3]*dc_shape[-2],) + dc_shape[-1:]) )
        ac_shape = Cac.shape
        Cac = Cac.reshape( (ac_shape[:-3] + (ac_shape[-3]*ac_shape[-2],) + ac_shape[-1:]) )
    
    K, model,ax = Fit_nonlin(Cdc[...,4],Cdc[...,2],show_fit=show_fit,copy=False)
    
    C4dc = Cmpt_C4_corrected(Cdc[...,4],Cdc[...,2],K,model).reshape(dc_shape[:-1])
    C4ac = Cmpt_C4_corrected(Cac[...,4],Cac[...,2],K,model).reshape(ac_shape[:-1])
        
    if fuse_last_two_axis : #restoring shapes
        Cdc = Cdc.reshape(dc_shape)
        Cac = Cac.reshape(ac_shape)
    return C4dc,C4ac