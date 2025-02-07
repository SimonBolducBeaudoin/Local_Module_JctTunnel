#!/bin/env/python
#! -*- coding: utf-8 -*-
from __future__ import division
from past.utils import old_div
import numpy as _np

from SBB.Pyhegel_extra.Experiment                   import logger,Info, Cross_Patern_Lagging_computation, Experiment
# from SBB.Pyhegel_extra.Pyhegel_wrappers             import Yoko_wrapper, Guzik_wrapper , PSG_wrapper,DelayLine_wrapper
from SBB.AutoCorr.autocorr import autocorr_cyclo , autocorr_cyclo_m
from SBB.Numpy_extra.numpy_extra                    import find_nearest_A_to_a,build_array_of_objects
from SBB.FFT.DFT.utils                              import singleDFTterm

from fractions import Fraction as _Fraction

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
    def gen_meta_info(R_jct,R_tot,n_threads,l_data,l_kernel,F,sampling_rate,period,ref_idxs,yo_wait,gz_config_ac,sii_optimal,sii_m1_optimal):
        return {
            'R_jct':R_jct,'R_tot':R_tot,
            'n_threads':n_threads,
            'l_data':l_data,
            'l_kernel':l_kernel,
            'F':F,
            'sampling_rate':sampling_rate,
            'ref_idxs':ref_idxs,
            'yo_wait':yo_wait,
            'period':period,
            'gz_config_ac':gz_config_ac,
            'sii_optimal':sii_optimal,
            'sii_m1_optimal':sii_m1_optimal
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
        self.n_threads           = int(self.meta['n_threads'])
        self.l_data              = int(self.meta['l_data'])
        self.l_kernel            = int(self.meta['l_kernel'])
        
        self.F                   = int(self.meta['F'])
        self.sampling_rate       = int(self.meta['sampling_rate'])      # Sampling rate (int)(Hz)
        self.period              = int(self.meta['period'])
        self.gz_config_ac = self.meta['gz_config_ac']
        
        self.psg_A_phase_mes     =  self.gz_config_ac.pop('psg_A')
        self.phase_target_deg    =  self.gz_config_ac.pop('phase_target_deg')
        self.reps_phase_mes      =  int(self.gz_config_ac.pop('reps'))
        self.smallest_ac_possible =  self.gz_config_ac.pop('smallest_ac_possible')
        
        self.sii_optimal         = self.meta['sii_optimal']
        self.sii_optimal.update( **{'F':self.F,'R':self.sampling_rate})
        
        self.sii_m1_optimal      = self.meta['sii_m1_optimal']
        self.sii_m1_optimal.update( **{'F':self.F,'R':self.sampling_rate})
        
        
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
        # Only works for 1int16 fow now
        # This will need to be multiplied by dt^2 to correspond to the phyisical value
        # By default it is also normalized using numpy's "backward" convention for ffts
        return autocorr_cyclo_m(data,m=0,**self.sii_optimal)
        
    def get_SII_phi (self,data,data_type = 'int16'):
        # Only works for 1int16 fow now
        # This will need to be multiplied by dt^2 to correspond to the phyisical value
        # By default it is also normalized using numpy's "backward" convention for ffts
        return autocorr_cyclo_m(data,m=1,**self.sii_m1_optimal)
        
    #############
    # Utilities #
    ############# 
    @staticmethod
    def get_n_throw_points(theta,F,Clk=1e9,R=32e9,n_target=0):
        """
        Returns the number of point to throw away to be synchronized with F.
        theta = self.get_phase()
        n = get_n_throw_points(theta,F=...)
        data_sync = data[n:]
        
        It is the responsability of the user to ensure that the resolution on the 
        phase measurement is under 1/(2 pi den) (aka we can resolve all possible phases)
        """
        theta*=-1 
        den = _Fraction(F/Clk).denominator
        n_current = round(den*theta/(360)) # in [ -den//2 : den//2-1 ]
        return int(R/Clk)*((n_target-n_current+den)%den)
        
    def get_phase(self,F,samp_rate=32e9,reps=3,channel_idx=None):
        angles = []
        for i in range(reps):
            if channel_idx is None :
                data=self.gz.get()
            else :
                data=self.gz.get()[channel_idx]
            R=singleDFTterm(data,int(F),int(samp_rate))
            # The resolution on the phase must be good enough to find n_throw in a single measurement
            n_throw = self.get_n_throw_points( _np.angle(R,deg=True),int(F))
            R=singleDFTterm(data[n_throw:],int(F),int(samp_rate))
            print("{} : Angle : {} [deg]".format(i ,_np.angle(R,deg=True) ))
            angles   += [_np.angle(R,deg=True),]
        print("Angle : {} [deg]".format( _np.mean(angles) ))
        return _np.mean(angles)
        
    def gz_get(self,is_ref=False,idx_data=0,idx_phase=1,samp_rate=32e9):
        """
        Shifts the data in memory to make sure that it is aligned/in phase with self.F 
        FUTURE : add a parameter that truncates the works singleDFTterm does to make the aquisition faster.
        """
        if is_ref:
            return self.gz.get()[idx_data]
        else :
            data=self.gz.get()
            R=singleDFTterm(data[idx_phase],int(self.F),int(samp_rate))
            # The resolution on the phase must be good enough to find n_throw in a single measurement
            n_throw = self.get_n_throw_points( _np.angle(R,deg=True),int(self.F))
            return data[idx_data,n_throw:]
        
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
        self.SII_vdc            = _np.full((n,l_vdc            ,self.l_kernel-1),_np.nan,dtype=complex) 
        self.SII_vac            = _np.full((n,l_vac            ,self.l_kernel-1),_np.nan,dtype=complex)
   
    def _all_loop_open(self) :
        super(SIISyncExp,self)._all_loop_open()
        self.yoko.set_init_state(abs(self._conditions_core_loop_raw[0]).max())
        self.psg.set_ampl(-135)
        self.psg.set_output(True)
        
        # Pas nécessaire ###################################################################
        self.gz_config_dc     = self.gz.get_config_inputs() # Copies the current gz config
        self.psg.set_ampl(self.psg_A_phase_mes) 
        self.gz.config(**self.gz_config_ac)
        self.reset_phase(f=self.F,p_target=self.phase_target_deg,reps=self.reps_phase_mes,channel_idx=1)
        self.pump_phase[0] = self.get_phase(F=self.F,reps=self.reps_phase_mes,channel_idx=1)
        self.psg.set_ampl(-135)
        self.gz.config(**self.gz_config_dc)
        ####################################################################################
        
        super(SIISyncExp,self)._loop_core(tuple(),tuple())

    def _repetition_loop_start(self,n,condition_it):
        Experiment._repetition_loop_start(self,n)
        
        self._first_conditions = next(condition_it)
        self._log.events_print(self._first_conditions)
        
        # Set first condition
        self.gz.config(**self.gz_config_dc)
        vdc_next,vac_next  = self._first_conditions
        self.psg.set_ampl(vac_next)                                    
        self.yoko.set_and_wait(vdc_next,Waittime=self.yo_wait)
        
        # initialize 
        self.next_vac = -135
        self.is_ref = True
    def _loop_core(self,index_tuple,condition_tuple,index_it,condition_it,n):
        """
            Works conditionnaly to the computing being slower than 0.4 sec
        """
        j,k                 = index_tuple     
        vdc_next,vac_next   = condition_tuple  
        self.current_vac    = self.next_vac # Memorized in previous iteration
        self.skip = True if ( self.is_ref==False and self.current_vac<self.smallest_ac_possible) else False
        # Gathering data from last point
        if index_it.current_dim == 0 :
            self.data_gz       = self.gz.get() # int16 
        else :  # index_it.current_dim == 1            
            if self.skip :
                self.data_gz = None
            else :    
                self.data_gz = self.gz_get(is_ref=self.is_ref) # int16 
        # Setting next conditions
        if index_it.next_dim == 0 :
            self.yoko.set(vdc_next)
            self.psg.set_ampl(vac_next)
        else : # index_it.next_dim == 1            
            if index_it.current_dim == 0 : 
                # RESET PHASE
                self.psg.set_ampl(self.psg_A_phase_mes)
                self.gz.config(**self.gz_config_ac)
                self.pump_phase[n+1] = self.get_phase(F=self.F,reps=self.reps_phase_mes,channel_idx=1)
                self.reset_phase(f=self.F,p_target=self.phase_target_deg,reps=self.reps_phase_mes,channel_idx=1) 
            
            # Saving for next iteration #####################
            self.is_ref = True if vac_next == -135 else False
            self.next_vac = vac_next 
            #################################################
            # Forcing the reference to be Vdc = 0
            vdc_next = 0 if vac_next == -135 else vdc_next
            self.psg.set_ampl(vac_next)                                  
            self.yoko.set(vdc_next)                                      
        
        self._log.event(0)
        if index_it.current_dim == 0 :
            self.SII_vdc[n,j]= self.get_SII(self.data_gz)
        else: # index_it.current_dim == 1 :
            if self.skip :
                wait(self.yo_wait) 
            else :
                self.SII_vac[n,k]= self.get_SII_phi(self.data_gz)
        self._log.event(1)
        super(SIISyncExp,self)._loop_core(index_tuple,condition_tuple)
        
    def _last_loop_core_iteration(self,n):
        self.data_gz   = self.gz_get() # int16 
        self._log.event(0)
        l_Vac             = len(self._conditions_core_loop_raw[1])
        self.SII_vac[n,-1]= self.get_SII_phi(self.data_gz) 
        
        # Set the guzik for dc conditions
        self.gz.config(**self.gz_config_dc)
        
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