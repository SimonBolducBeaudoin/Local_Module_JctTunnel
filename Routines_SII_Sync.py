#!/bin/env/python
#! -*- coding: utf-8 -*-
from __future__ import division
from past.utils import old_div
import numpy as _np

from SBB.Pyhegel_extra.Experiment                   import logger,Info, Cross_Patern_Lagging_computation, Experiment
# from SBB.Pyhegel_extra.Pyhegel_wrappers             import Yoko_wrapper, Guzik_wrapper , PSG_wrapper,DelayLine_wrapper
from SBB.AutoCorr.aCorrsOTF.acorrs_otf              import ACorrUpTo
from SBB.Numpy_extra.numpy_extra                    import find_nearest_A_to_a,build_array_of_objects
from SBB.FFT.DFT.utils                              import singleDFTterm

from SBB.Data_analysis.window import window_after

_dt = 31.25e-12

class SIISyncInfo(Info):
    """
        Last update
        -----------
        Computes and save symtetric and antisymetric part of SII    
            
        Conditions
        -----------
            Vdc [V]     = self.conditions[0]
            Vac [dBm]   = self.conditions[1]
        
        Condition logic
        ---------------
        self._conditions_options
            antisym
        self._ref_options
            interlacing : a (Vdc,Vac) = (0,-135) conditions is added in between each conditions
            no_ref      : no_referencing is done
    """
    @staticmethod
    def gen_meta_info(R_jct,R_tot,n_threads,l_data,l_kernel,F,period,ref_idxs,yo_wait,gz_phase_mes_params):
        return {
            'R_jct':R_jct,'R_tot':R_tot,
            'n_threads':n_threads,
            'l_data':l_data,
            'l_kernel':l_kernel,'F':F,
            'ref_idxs':ref_idxs,
            'yo_wait':yo_wait,
            'period':period,
            'gz_phase_mes_params':gz_phase_mes_params
            }
    def _set_options(self,options):
        super(SIISyncInfo,self)._set_options(options)
        self._conditions_options    =   {'antisym':options.get('Vdc_antisym') }                                      # Sweeping on positive and negative DC current
        self._ref_options           =   {'interlacing': options.get('interlacing') , 'no_ref':options.get('no_ref')} # Referencing patern
    def _set_conditions(self,conditions):
        super(SIISyncInfo,self)._set_conditions(conditions)
    @staticmethod
    def compute_interlacing(Vdc,ref=0):
        Vdc_interlaced = _np.ones(2*len(Vdc))*ref
        Vdc_interlaced[1::2] = Vdc
        return Vdc_interlaced
    @staticmethod
    def compute_default_ref(Vdc,ref=0):
        return _np.concatenate(([ref],Vdc))
    @staticmethod
    def add_antisym(Vdc,**sym_options):
        """
        Return Vdc_antisym conditional to sym_options
        """
        return _np.concatenate(([(-1.0)*Vdc[::-1],Vdc])) if sym_options.get('antisym') else Vdc
    @staticmethod
    def add_ref_conditions(Vdc,ref=0,**ref_options):
        """
            Add the right referencing partern to Vdc conditionnal to ref_options
        """
        if    ref_options.get('no_ref'): 
            return Vdc
        elif  ref_options.get('interlacing'):
            return SIISyncInfo.compute_interlacing(Vdc,ref)
        else :
            return SIISyncInfo.compute_default_ref(Vdc,ref)
    @staticmethod
    def ref_idxs_update(ref_idxs,len_vdc,**ref_options):
        """
            Only modifies reference for Vdc
        """
        tmp = ref_idxs
        if    ref_options.get('no_ref'): 
            pass
        elif  ref_options.get('interlacing'):
            tmp[0] *=2
            tmp[0] +=1            
            len_vdc *=2
        else :
            tmp[0] +=1
            len_vdc +=1
        
        if ref_options.get('antisym') :
            tmp[0]+= len_vdc  
        return tmp  
        
    def _build_attributes(self):
        super(SIISyncInfo,self)._build_attributes()
        Vdc_antisym                       = self.add_antisym            (self._conditions_core_loop_raw[0]   ,**self._conditions_options )
        Vdc_exp                           = self.add_ref_conditions     (Vdc_antisym ,**self._ref_options )
        tmp_dict = self._conditions_options.copy()
        tmp_dict.update(self._ref_options)
        self._ref_idxs                    = self.ref_idxs_update(self._ref_idxs,len(self._conditions_core_loop_raw[0]),**tmp_dict)
        Vac                               = self._conditions_core_loop_raw[1]
        Vac_exp                           = self.add_ref_conditions     (Vac ,ref=-135,**self._ref_options )
        self._conditions                  = self._conditions[0],Vdc_exp,Vac_exp ## overwriting condition tuple !!!
        self._conditions_core_loop_raw    = Vdc_exp,Vac_exp                     ## overwriting condition tuple !!!
        #self.conditions                   = self.get_conditions() 
        self.meta                         = self._meta_info
        
        self.Idc                          = old_div(self._conditions_core_loop_raw[0],(self.meta['R_tot']))
        self.yo_wait                      = self.meta['yo_wait']
        
        ## Converting only int to make sure they are properly initialize after opening from .npz
        self.n_threads     = int(self.meta['n_threads'])
        self.l_data        = int(self.meta['l_data'])
        self.l_kernel      = int(self.meta['l_kernel'])
        self.l_hc          = old_div(self.l_kernel,2) + 1
        self.l_kernel_sym  = self.l_hc                # Diviser par deux car on va symétrisé
        self.F             = int(self.meta['F'])
        self.period        = int(self.meta['period'])
        self.gz_phase_mes_params        = self.meta['gz_phase_mes_params']
        self.psg_A_phase_mes =  self.gz_phase_mes_params.pop('psg_A')
        self.phase_target_deg =  self.gz_phase_mes_params.pop('phase_target_deg')
        self.reps_phase_mes =  int(self.gz_phase_mes_params.pop('reps'))
        
        
class SIISyncExp(SIISyncInfo,Cross_Patern_Lagging_computation):
    """
    What it does :
        - Mesures the autocorrelation using class ACorrUpTo
            - in units of [V**2]
            - sweeping along Vdc and Vac [in a cross patern]
            - autocorrelation are computed only on len == l_kernel/2 + 1 (faster)
            - Retreive S2 from Acorr objects
        - Mesures the q and p quadrature
        - Accumulates results in histograms (one for each time point)
        - Does all this and saves results for each repetition of the experiment to giva maximal flexibility for the analysis
              
    - conditions == (n_measures,Vdc,Vac) 
    Options :
     
        
    Last Update
    -----------
    
    Todos :
    Bugs :
    """
    def _set_devices(self,devices):
        self.gz                        =   devices[0] 
        self.yoko                      =   devices[1]
        self.psg                       =   devices[2]
        self.colby                     =   devices[3]
    def _init_log(self):
        conditions  = self._conditions_core_loop_raw
        loop_sizes  = ( self._n_measures , len(conditions[0]),len(conditions[1]) )
        events      = ["Acquisition : {:04.2F} [s]","Computing : {:04.2F} [s] "]
        rate        = ( self.meta['l_data']*1.0e-9 ,"Rate : {:04.2F} [GSa/s] " )
        self._log    = logger(loop_sizes=loop_sizes,events=events,rate=rate,conditions=('{: .3f}','{: .0f}'))
    def _init_objects(self):
        self.pump_phase = _np.full( (self._n_measures+1,),_np.nan ) 
        self._init_acorr()
    def get_SII(self,data,data_type = 'int16'):
        acorr =  ACorrUpTo(self.l_kernel_sym,data_type)
        acorr(data)
        return (acorr.res).copy() # acorr.res is badbly implemented and unsafe. Copying the data removes some issues.
        
    def get_SII_phi (self,data,data_type = 'int16'):
        """
        Broken ??
        """
        acorr = ACorrUpTo(self.l_kernel_sym,data_type,phi=self.period)
        acorr(data)
        return (acorr.res).copy() # acorr.res is badbly implemented and unsafe. Copying the data removes some issues.
        
    #############
    # Utilities #
    ############# 
    def get_phase(self,F=12e9,samp_rate=32e9,reps=3,channel_idx=None):
        angles = []
        for i in range(reps):
            if channel_idx is None :
                data=self.gz.get()
            else :
                data=self.gz.get()[channel_idx]
            R=singleDFTterm(data,int(F),int(samp_rate))
            angles += [_np.angle(R,deg=True),]
        print("Angle : {}".format(_np.mean(angles)))
        return _np.mean(angles)
                
    def reset_phase(self,f=12e9,p_target = 0,reps=3,channel_idx=None):
        T = 1/f * 1e12 # ps
        if '312.5ps'==self.colby.get_mode() :
            dt = 0.25 # ps    
        elif '625ps'==self.colby.get_mode() :
            dt = 0.5 # ps
        else :
            raise Exception("Bad Mode!")
        epsilon = (dt/T)*360

        p_current = self.get_phase(F=f,reps=reps,channel_idx=channel_idx) #deg
        p_mov = p_target-p_current
        while abs(p_mov) > epsilon :
            n_step = round ( (p_mov/360)*(T/dt) )
            delay_add = n_step*dt
            delay_set = self.colby.get() + delay_add
            self.colby.set(delay_set)
            p_current = self.get_phase(F=f,reps=reps,channel_idx=channel_idx) #deg
            p_mov = p_target-p_current
        
    def _init_acorr(self):
        """
            Should I init for all or just for one ..?
        """
        n                       = self._n_measures
        l_vdc                   = len(self._conditions_core_loop_raw[0])
        l_vac                   = len(self._conditions_core_loop_raw[1])
        acorr_vdc_shape         = ( n,l_vdc, ) 
        acorr_vac_shape         = ( n,l_vac, )
        data_type = 'int16'
        self.SII_vdc            = _np.full((n,l_vdc            ,self.l_kernel_sym),_np.nan) 
        self.SII_vac            = _np.full((n,l_vac,self.period,self.l_kernel_sym),_np.nan)
   
    def _all_loop_open(self) :
        super(SIISyncExp,self)._all_loop_open()
        self.yoko.set_init_state(abs(self._conditions_core_loop_raw[0]).max())
        self.psg.set_ampl(-135)
        self.psg.set_output(True)
        
        self.psg.set_ampl(self.psg_A_phase_mes) 
        gz_config=self.gz.get_config_inputs()
        self.gz.config(**self.gz_phase_mes_params)
        self.reset_phase(f=self.F,p_target=self.phase_target_deg,reps=self.reps_phase_mes)
        self.pump_phase[0] = self.get_phase(F=self.F,reps=self.reps_phase_mes)
        self.psg.set_ampl(-135)
        self.gz.config(**gz_config)
        
        super(SIISyncExp,self)._loop_core(tuple(),tuple())

    def _repetition_loop_start(self,n,condition_it):
        Experiment._repetition_loop_start(self,n)
        
        self._first_conditions = next(condition_it)
        self._log.events_print(self._first_conditions)
        
        # Set first condition
        vdc_next,vac_next  = self._first_conditions
        self.psg.set_ampl(vac_next)                                    
        self.yoko.set_and_wait(vdc_next,Waittime=self.yo_wait)
        
    def _loop_core(self,index_tuple,condition_tuple,index_it,condition_it,n):
        """
            Works conditionnaly to the computing being slower than 0.4 sec
        """
        j,k                 = index_tuple     
        vdc_next,vac_next   = condition_tuple  
        # Gathering data from last point
        self.data_gz       = self.gz.get() # int16 
        # Setting next conditions
        if index_it.next_dim == 0 :
            self.yoko.set(vdc_next)
            self.psg.set_ampl(vac_next)
        else : # index_it.next_dim == 1            
            if index_it.current_dim == 0 : 
                # RESET PHASE
                self.psg.set_ampl(self.psg_A_phase_mes)
                gz_config=self.gz.get_config_inputs()
                self.gz.config(**self.gz_phase_mes_params)
                self.pump_phase[n+1] = self.get_phase(F=self.F,reps=self.reps_phase_mes)
                self.reset_phase(f=self.F,p_target=self.phase_target_deg,reps=self.reps_phase_mes)    
                self.gz.config(**gz_config)                
            self.psg.set_ampl(vac_next)                                  
            self.yoko.set(vdc_next)                                      
        
        self._log.event(0)
        if index_it.current_dim == 0 :
            self.SII_vdc[n,j]= self.get_SII(self.data_gz)
        else: # index_it.current_dim == 1 :
            self.SII_vac[n,k]= self.get_SII_phi(self.data_gz)
        self._log.event(1)
        super(SIISyncExp,self)._loop_core(index_tuple,condition_tuple)
        
    def _last_loop_core_iteration(self,n):
        self.data_gz   = self.gz.get() # int16 
        self._log.event(0)
        l_Vac             = len(self._conditions_core_loop_raw[1])
        self.SII_vac[n,-1]= self.get_SII_phi(self.data_gz) 
        
        self._log.event(1)
        super(SIISyncExp,self)._loop_core(tuple(),tuple())
            
    def _build_data(self):
        data = {\
        'data_gz'       : self.data_gz[:1<<20], # first millon points of the last measurement
        'S2_vdc'        : self.SII_vdc,
        'S2_vac'        : self.SII_vac,
        'Vdc'           : self._conditions_core_loop_raw[0],
        'Vac'           : self._conditions_core_loop_raw[1],
        'pump_phase'    : self.pump_phase
        }
        return data

   
from SBB.AutoCorr.util import symmetrize_SIIphi   
     
def ROUTINE_SII_SYNC_0(SII,F,R=int(32e9),fast=True,windowing=True,i=65) :
    """
    Symmetrize the photoexcited autocorrelation properly
    respecting the rule S_phi(-Tau) = S_{(phi-Omega tau)%2pi}(Tau)
    """
    if fast :
        SII  = _np.nanmean(SII ,axis = 0)[None,...]
    if windowing :
        SII    = window_after(SII , i=i, t_demi=1)
    return _np.fft.rfft(_np.fft.fftshift(symmetrize_SIIphi(SII,F,R)   ,axes=-1))*_dt
