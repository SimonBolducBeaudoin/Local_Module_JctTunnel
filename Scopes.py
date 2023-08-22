#!/bin/env/python
#! -*- coding: utf-8 -*-
import numpy as _np
from numpy import pi
from scipy import constants as C
from scipy import polyfit
from pylab import subplots
from matplotlib import cm
from SBB.Matplotlib_extra.logic import get_twin,get_twins, get_next_ax
from SBB.Matplotlib_extra.plot import plot_interval,scope_interval_update
from SBB.Phys.Phys_Stat import BoseEinstein
from SBB.Phys.Tunnel_Junction import Sdc_of_f

def VvsI_scope_0(ipol,Vjct_neg,Vjct_pos,imin=None,imax=None,plot_deviation=True,ax=None):
    if ax is None:
        fig,ax =subplots(1,1)
    if (imin is None) and (imax is None):
        w = _np.where(ipol==ipol)
    else :
        logic_min = True if imin is None else ipol>=imin
        logic_max = True if imax is None else ipol<=imax
        w = _np.where( logic_min & logic_max )
    ypos = _np.nanmean(Vjct_pos,axis=0)
    yneg = _np.nanmean(abs(Vjct_neg),axis=0)
    Rpos,Vo_pos = polyfit(ipol[w],ypos[w],1)
    Rneg,Vo_neg = polyfit(ipol[w],yneg[w],1)
    R = (Rpos+Rneg)/2
    if len(ax.lines)==0 : 
        plot_interval(ax,ipol*1e6,abs(Vjct_neg)*1e6,label='neg',marker='.',markersize=10)
        plot_interval(ax,ipol*1e6,Vjct_pos*1e6,label='pos',marker='.',markersize=10)
        ax.plot(ipol*1e6,ipol*R*1e6,label='R={:0.2f}'.format(R))
        ax.set_xlabel('Ijct[uA]')
        ax.set_ylabel('Vjct[uV]')
        ax.legend(loc='lower right')
        textstr = "Rpos:{:0.3f}[Ohm] \nVo_pos:{:0.3f}[uV]\n Rneg:{:0.3f}[Ohm]\nVo_neg:{:0.3f}[uV]".format(Rpos,Vo_pos*1e6,Rneg,Vo_neg*1e6)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')
    else : 
        scope_interval_update(ax,ipol*1e6,abs(Vjct_neg)*1e6,std=True, n_sigma=3, min_max=True,line_idx=0)
        scope_interval_update(ax,ipol*1e6,Vjct_pos*1e6,std=True, n_sigma=3, min_max=True,line_idx=1)
    if plot_deviation:
        x = ipol
        twin = get_twin(ax, axis='x')
        if twin is None :
            twin = ax.twinx()
        if len(twin.lines) == 0 :
            twin.plot(ipol*1e6,(ypos-(Rpos*ipol+Vo_pos))*1e6,ls='None',marker='.',markersize=10)
            twin.plot(ipol*1e6,(yneg-(Rneg*ipol+Vo_neg))*1e6,ls='None',marker='.',markersize=10)
            twin.set_ylabel('Vjct- RI[uV]')
        else :
            twin.lines[0].set_data(ipol*1e6,(ypos-(Rpos*ipol+Vo_pos))*1e6)
            twin.lines[1].set_data(ipol*1e6,(yneg-(Rneg*ipol+Vo_neg))*1e6)
    return ax
    
def VvsI_scope_1(ipol,Vjct,dVdI=None,imin=None,imax=None,plot_deviation=True,ax=None):
    plot_dVdI = not(dVdI is None)
    
    if ax is None:
        fig,ax =subplots(1,1)
    if (imin is None) and (imax is None):
        w = _np.where(ipol=ipol)
    else :
        logic_min = True if imin is None else ipol>=imin
        logic_max = True if imax is None else ipol<=imax
        w = _np.where( logic_min & logic_max )
    y  = _np.nanmean(Vjct,axis=0)
    R,Vo = polyfit(ipol[w],y[w],1)
    if len(ax.lines)==0 : 
        plot_interval(ax,ipol*1e6,Vjct*1e6,label='data',marker='.',markersize=10)
        ax.plot(ipol*1e6,ipol*R*1e6,label='R={:0.2f}'.format(R))
        ax.set_xlabel('Ijct[uA]')
        ax.set_ylabel('Vjct[uV]')
        ax.legend(loc='lower right')
        textstr = "R:{:0.3f}[Ohm] \nVo:{:0.3f}[uV]".format(R,Vo*1e6)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')
        prop = ax._get_lines.prop_cycler
    else : 
        scope_interval_update(ax,ipol*1e6,Vjct*1e6,std=True, n_sigma=3, min_max=True,line_idx=0)
    
    if not(plot_deviation) and not(plot_dVdI) :
        return ax
    else :
        twins = get_twins(ax, axis='x')
        if plot_deviation and plot_dVdI :
            if twins is None :
                twins = [ ax.twinx(),ax.twinx() ]
            twins[1].spines["right"].set_position(("axes", 1.1))
        elif plot_deviation and not plot_dVdI :
            if twins is None :
                twins = [ ax.twinx(), None ]
            else :
                twins = [ twins[0]  , None ]
        else :
            if twins is None :
                twins = [ None , ax.twinx() ]
            else :
                twins = [ None , twins[0] ]
                
    if plot_deviation:
        twin = twins[0]
        if len(twin.lines) == 0 :
            color = next(prop)['color']
            twin.plot(ipol*1e6,(y-(R*ipol+Vo))*1e6,ls='None',marker='.',markersize=10,color=color)
            twin.set_ylabel('Vjct- RI[uV]') 
            twin.yaxis.label.set_color(color)
        else :
            twin.lines[0].set_data(ipol*1e6,(y-(R*ipol+Vo))*1e6)
    if plot_dVdI:
        twin = twins[1]
        if len(twin.lines) == 0 :
            color = next(prop)['color']
            twin.plot(ipol*1e6,_np.nanmean(dVdI,axis=0),ls='None',marker='.',markersize=10,color=color)
            twin.set_ylabel('dV/dI [Ohm]')
            twin.yaxis.label.set_color(color)
        else :
            twin.lines[0].set_data(ipol*1e6,_np.nanmean(dVdI,axis=0))
    return ax
    
def VvsI_scope_2(ipol,Vjct,R,Vo,Dcb,x,data,RvsI,ax=None):    
    if ax is None:
        fig,ax =subplots(1,1)
    twin = get_twin(ax, axis='x')
    if twin is None :
        twin = ax.twinx()  
    
    textstr = "R:{:0.2f}[Ohm] \nVo:{:0.2f}[uV]\n".format(R,Vo*1e6)+r"$\Delta$I:{:0.2f}[uA]".format(Dcb*1e6)
    
    if len(ax.lines)==0 : 
        plot_interval(ax,ipol*1e6,Vjct*1e6,label='data',marker='.',markersize=10)
        ax.plot(ipol*1e6,ipol*R*1e6)
        ax.set_xlabel('Ijct[uA]')
        ax.set_ylabel('Vjct[uV]')
        prop = ax._get_lines.prop_cycler
        twin.plot(x*1e6,data/x+R,color=next(prop)['color'])
        twin.plot(ipol*1e6,RvsI,color=next(prop)['color'])
        twin.set_ylim(R,R+1)
        twin.set_ylabel('R[Ohm]')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')
    else:    
        scope_interval_update(ax,ipol*1e6,Vjct*1e6,std=True, n_sigma=3, min_max=True,line_idx=0)
        ax.lines[1].set_data(ipol*1e6,ipol*R*1e6)
        twin.lines[0].set_data(x*1e6,data/x+R)
        twin.lines[1].set_data(ipol*1e6,RvsI)
        ax.texts[0].set_text(textstr)
    return ax

def dSII_vs_I_scope(ipol,freq,dSII,f_slice=slice(30,75,5),ax=None):
    if ax is None :
        fig,ax = subplots(1,1)
    ax2 = get_next_ax(ax)
    if ax2 is None :
        left, bottom, width, height = [0.15, 0.45, 0.35, 0.4]
        ax2 = ax.figure.add_axes([left, bottom, width, height])      
    color_idx = _np.linspace(0, 1, len(freq))
    if len(ax.lines)==0 : 
        for sii ,f ,c_idx in zip ( _np.moveaxis( dSII[...,f_slice],-1,0),freq[f_slice],color_idx[f_slice]):
            plot_interval(ax,ipol*1e6,sii,marker='.',markersize=10,linestyle='none',color=cm.cool(c_idx),label='{:0.2f}'.format(f*1e-9),std=True,min_max=True,n_sigma=1,alpha=[None, 0.25, 0.05])
            plot_interval(ax2,ipol*1e6,sii,marker='.',markersize=10,linestyle='-',color=cm.cool(c_idx),std=True,min_max=True,n_sigma=1,alpha=[None,0.25,0.05])
        ax.plot(ipol*1e6,C.e*ipol,color='k')        
        ax.set_xlabel('Ijct [uA]')
        ax.set_ylabel('SII [A**2/Hz]')
        ax.legend(fontsize=12,loc='upper right',title='f[GHz]')
        ax2.set_ylim(-1e-20,)
        ax.set_xlim(0,0.75)
        ax.set_ylim(-1e-20,1.0e-19)
        ax.grid()
        ax2.grid()
    else:
        for i,sii in enumerate( _np.moveaxis( dSII[...,f_slice],-1,0) ) :
            scope_interval_update(ax,ipol*1e6,sii,std=True, n_sigma=1, min_max=True,line_idx=i)
            scope_interval_update(ax2,ipol*1e6,sii,std=True, n_sigma=1, min_max=True,line_idx=i)
    ax.set_title('Differential noise at the detector')
    return ax

def gain_scope(freq,G_of_f,ax=None):
    if ax is None :
        fig,ax = subplots(1,1)
    x = freq*1e-9
    y = 10*_np.log10(abs(G_of_f)) # abs to avoid warning
    if len(ax.lines)==0 :
        plot_interval(ax,x,y,std=True, n_sigma=3, min_max=True, label=None, linestyle=None, marker=None, alpha=[None, 0.25, 0.01])
        ax.set_ylim(0,75)
        ax.grid(True)
        ax.set_xlabel('f[GHz]')
        ax.set_ylabel('G [dB]')
    else :    
        scope_interval_update(ax,x,y,ax)
    return ax

def noise_Temps_scope(freq,G_of_f,B_of_f,ax=None):
    if ax is None :
        fig,ax = subplots(1,1)
    x = freq*1e-9
    y = 50.0/(2*C.k)*(B_of_f/G_of_f)
    if len(ax.lines)==0 : 
        plot_interval(ax,x,y,std=True, n_sigma=1, min_max=True, label=None, linestyle=None, marker=None, alpha=[None, 0.25, 0.01])
        ax.set_ylim(-1,25)
        ax.grid(True)
        ax.set_xlabel('f[GHz]')
        ax.set_ylabel('Excess noise origin [K]')
    else :
        scope_interval_update(ax,x,y,ax)
    return ax
    
def dSIIx_vs_I_scope(ipol,freq,dSIIx,f_slice=slice(30,75,5),R=70.00,T=0.035,ax=None):
    if ax is None :
        fig,ax = subplots(1,1)
    ax2 = get_next_ax(ax)
    if ax2 is None :
        left, bottom, width, height = [0.535, 0.15, 0.35, 0.4]
        ax2 = ax.figure.add_axes([left, bottom, width, height])
        
    color_idx = _np.linspace(0, 1, len(freq))
    if len(ax.lines)==0 : 
        for sii ,f ,c_idx in zip ( _np.moveaxis( dSIIx[...,f_slice],-1,0),freq[f_slice],color_idx[f_slice]):
            plot_interval(ax,ipol*1e6,sii,marker='.',markersize=10,linestyle='none',color=cm.cool(c_idx),label='{:0.2f}'.format(f*1e-9),std=True,min_max=True,n_sigma=1,alpha=[None, 0.25, 0.05])
            ax.plot(ipol*1e6,Sdc_of_f(2*pi*f,(C.e*(ipol*R)/C.hbar),T,R),color=cm.cool(c_idx))
            plot_interval(ax2,ipol*1e6,sii,marker='.',markersize=10,linestyle='-',color=cm.cool(c_idx),std=True,min_max=True,n_sigma=1,alpha=[None,0.25,0.05])
            ax2.plot(ipol*1e6,Sdc_of_f(2*pi*f,(C.e*(ipol*R)/C.hbar),T,R),color=cm.cool(c_idx))
        ax.plot(ipol*1e6,C.e*ipol,color='k')
        ax2.plot(ipol*1e6,C.e*ipol,color='k')         
        ax.set_xlabel('Ijct [uA]')
        ax.set_ylabel('SII [A**2/Hz]')
        ax.legend(fontsize=12,loc='upper right',title='f[GHz]')
        #ax2.set_xlim(0,max())
        ax2.set_ylim(-1e-27,)
        ax.set_xlim(0,0.75)
        ax.set_ylim(-1e-27,1.25e-25)
    else:
        for i,(sii,f) in enumerate( zip(_np.moveaxis( dSIIx[...,f_slice],-1,0),freq[f_slice]) ) :
            scope_interval_update(ax,ipol*1e6,sii,std=True, n_sigma=1, min_max=True,line_idx=i)
            ax.lines[i].set_data(ipol*1e6,Sdc_of_f(2*pi*f,(C.e*(ipol*R)/C.hbar),T,R))
            scope_interval_update(ax2,ipol*1e6,sii,std=True, n_sigma=1, min_max=True,line_idx=i)
    ax.set_title('Rjct={:0.2f}[Ohm];Te={:0.2f}[mK]'.format(R,T*1000))
    return ax
    
def Tvsf_scope(freq,Temps,Te_fit,ax=None):
    if ax is None :
        fig,ax = subplots(1,1)
    if len(ax.lines)==0 : 
        line, = ax.plot(freq*1e-9,Temps*1000)
        ax.plot(freq*1e-9,_np.ones(freq.shape)*Te_fit*1000,color=line.get_color())
        ax.set_ylim(0,60)
        ax.set_xlabel('f[GHz]')
        ax.set_ylabel('Te[mK]')
    else :
        ax.lines[0].set_data(freq*1e-9,Temps*1000)    
        ax.lines[1].set_data(freq*1e-9,_np.ones(freq.shape)*Te_fit*1000)
    ax.set_title('Te_avg={:0.2f}[mK]'.format(Te_fit*1000))
    return ax
    
def dSIIx_origin_Vsf_scope(freq,dB_of_f,dG_of_f,Te=0.055,R=70.0,ax=None):
    if ax is None:
        fig,ax =subplots(1,1) 
    ax2 = get_twin(ax, axis='x')
    if ax2 is None :
        ax2 = ax.twinx()
    if len(ax.lines)==0 : 
        line, = plot_interval(ax,freq*1e-9,R/(2*C.k)*(-dB_of_f/(dG_of_f)))
        ax.plot(freq*1e-9,C.h*freq/(2*C.k),ls=':')
        line, = plot_interval(ax2,freq[1:]*1e-9,0.5*(-dB_of_f/dG_of_f)[...,1:]/(C.h*freq[1:]/(R)),color='b',std=True,n_sigma=3,min_max=True,alpha=[None,0.25,0.05])
        ax2.plot(freq[1:]*1e-9,BoseEinstein(freq[1:],Te)+0.5,ls=':',color=line.get_color())
        ax2.plot(_np.r_[0,16],_np.r_[0.5,0.5],ls='--',color=line.get_color())
        ax.set_ylim(-0.00,0.3)
        ax.grid(True)
        ax.set_xlabel('f[GHz]')
        ax.set_ylabel('Origin excess noise : dB[K]')
        ax2.set_ylabel('Origin excess noise : dB[ph]')
        ax2.set_ylim(-0.0,1.)
    else :
        scope_interval_update(ax,freq*1e-9,R/(2*C.k)*(-dB_of_f/(dG_of_f)),std=True,n_sigma=3,min_max=True,alpha=[None,0.25,0.05],line_idx=0) 
        ax.lines[1].set_data(freq*1e-9,C.h*freq/(2*C.k))
        scope_interval_update(ax2,freq[1:]*1e-9,0.5*(-dB_of_f/dG_of_f)[...,1:]/(C.h*freq[1:]/(R)),std=True,n_sigma=3,min_max=True,alpha=[None,0.25,0.05],line_idx=0)
        ax2.lines[1].set_data(freq[1:]*1e-9,BoseEinstein(freq[1:],Te)+0.5)
    ax.set_title('Rjct={:0.2f}[Ohm];Te={:0.2f}[mK]'.format(R,Te*1000))
    return ax
            
def Temps_in_Time_scope(Temps,ax=None):
    if ax is None :
        fig,ax = subplots(1,1)
    ax2 = get_next_ax(ax)
    if ax2 is None :
        left, bottom, width, height = [0.18, 0.18, 0.65, 0.2]
        ax2 = ax.figure.add_axes([left, bottom, width, height])   
    ax.clear()    
    ax.plot(Temps,marker='.',markersize=10)
    ax.set_ylim(0)
    ax.set_ylabel('T[mK]')
    ax.set_xlabel('Repetitions[~]')
    ax2.clear()
    ax2.hist(Temps)
    ax2.set_xlabel('T[mK]')
    return ax
    
    
    
    
    