#!/bin/env/python
#! -*- coding: utf-8 -*-

from Methods import reshape_reorder_swap

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
    

def reshape_reorder_all(Vdc,V_jct,V_pol,SII,dSII):
    """Deprecated"""
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
    """Deprecated"""
    l = len(Vdc)//4
    Vdc_shape  = (2,2,l)
    SII_shape  = SII.shape[:-2]   + Vdc_shape + SII.shape[-1:]
    dSII_shape =  SII.shape[:1]   + (2,l)     + SII.shape[-1:]
    
    Vdc   = Vdc_reshape_reorder(Vdc,Vdc_shape,l) 
    SII   = SII_reshape_reorder(SII,SII_shape,l)
    dSII  = dSII_reshape_reorder(dSII,dSII_shape,l)
    
    return Vdc,SII,dSII

def reshape_reorder_all_2(Vdc,Vac,SII_dc,dSII_dc,SII_ac,dSII_ac,hs_vdc,hs_vac,moments_dc,moments_ac,sym=True,ref='interlaced'):
    """Deprecated"""
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
    """Deprecated"""
    l = len(Vdc)//4
    Vdc_shape  = (2,2,l)
    dSII_shape =  dSII.shape[:1]   + (2,l)     + dSII.shape[-1:]
    Vdc   = Vdc_reshape_reorder(Vdc,Vdc_shape,l) 
    dSII  = dSII_reshape_reorder(dSII,dSII_shape,l)
    return Vdc,dSII

def centered_refed_dSII(SII,axis=-2) :
    """
    Un-biased dSII calculation (uses centered reference)
    """
    tmp  = SII.swapaxes(axis,-1)
    return centered_ref_X(tmp,ref=None).swapaxes(axis,-1)

def build_ipol_ipolsym(V_yoko,Cpol):
    ### Usefull!!!????
    l = len(V_yoko)//4
    Vdc_shape  = (2,2,l)
    ipol = V_yoko*Cpol
    Ipol = Vdc_reshape_reorder(V_yoko*Cpol,Vdc_shape,l) 
    ipol_sym = Ipol[1,1]
    return ipol,ipol_sym