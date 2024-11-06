#!/bin/env/python
#! -*- coding: utf-8 -*-
from __future__ import division
from past.utils import old_div
import numpy as np

from SBB.Pyhegel_extra.Experiment                   import logger,Info, Cross_Patern_Lagging_computation, Experiment
# from SBB.Pyhegel_extra.Pyhegel_wrappers             import Yoko_wrapper, Guzik_wrapper , PSG_wrapper,DelayLine_wrapper
from SBB.Time_quadratures.time_quadratures          import TimeQuad_FFT_float_to_Hist2D_uint32_t_int16_t   as TimeQuad
from SBB.Time_quadratures.time_quadratures          import TimeQuadSync_FFT_float_to_Hist2D_uint32_t_int16_t as TimeQuadSync
from SBB.Time_quadratures.kernels                   import make_kernels

from SBB.Histograms.histograms_helper               import compute_moments2D
from SBB.AutoCorr.aCorrsOTF.acorrs_otf              import ACorrUpTo
from SBB.Numpy_extra.numpy_extra                    import find_nearest_A_to_a,build_array_of_objects
from SBB.Data_analysis.fit                          import polyfit_above_th
from SBB.Microwave.Microwave                        import dBm_to_V
from SBB.Phys.Tunnel_Junction                       import V_th
from SBB.FFT.DFT.utils                              import singleDFTterm

# Local
from .Routines_SII   import ROUTINE_AVG_GAIN
from .Quadratures    import gen_fmins_fmaxs,Cmpt_cumulants,Cmpt_std_cumulants,C_to_n,C4_correction,Add_Vac_to_Cdc
from .Methods        import build_imin_imax

class dn2SyncInfo(Info):
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
    def gen_meta_info(R_jct,R_tot,n_threads,l_data,dt,V_per_bin,l_kernel,F,t,alpha,betas,betas_info,Thetas,ks,Labels,nb_of_bin,period,max,l_fft,ref_idxs,gain_fit_params,yo_wait,moments_order,gz_phase_mes_params):
        return {
            'R_jct':R_jct,'R_tot':R_tot,'n_threads':n_threads,
            'l_data':l_data,'dt':dt,'V_per_bin':V_per_bin,'l_kernel':l_kernel,'F':F,
            't':t,'alpha':alpha,'betas':betas,'betas_info':betas_info,'Thetas':Thetas,'ks':ks,'Labels':Labels,
            'nb_of_bin':nb_of_bin,'max':max,'l_fft':l_fft, 'ref_idxs':ref_idxs,
            'gain_fit_params':gain_fit_params,'yo_wait':yo_wait,
            'moments_order':moments_order,'period':period,
            'gz_phase_mes_params':gz_phase_mes_params
            }
    def _set_options(self,options):
        super(dn2SyncInfo,self)._set_options(options)
        self._conditions_options    =   {'antisym':options.get('Vdc_antisym') }                                      # Sweeping on positive and negative DC current
        self._ref_options           =   {'interlacing': options.get('interlacing') , 'no_ref':options.get('no_ref')} # Referencing patern
    def _set_conditions(self,conditions):
        super(dn2SyncInfo,self)._set_conditions(conditions)
    @staticmethod
    def compute_interlacing(Vdc,ref=0):
        Vdc_interlaced = np.ones(2*len(Vdc))*ref
        Vdc_interlaced[1::2] = Vdc
        return Vdc_interlaced
    @staticmethod
    def compute_default_ref(Vdc,ref=0):
        return np.concatenate(([ref],Vdc))
    @staticmethod
    def add_antisym(Vdc,**sym_options):
        """
        Return Vdc_antisym conditional to sym_options
        """
        return np.concatenate(([(-1.0)*Vdc[::-1],Vdc])) if sym_options.get('antisym') else Vdc
    @staticmethod
    def add_ref_conditions(Vdc,ref=0,**ref_options):
        """
            Add the right referencing partern to Vdc conditionnal to ref_options
        """
        if    ref_options.get('no_ref'): 
            return Vdc
        elif  ref_options.get('interlacing'):
            return dn2SyncInfo.compute_interlacing(Vdc,ref)
        else :
            return dn2SyncInfo.compute_default_ref(Vdc,ref)
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
        super(dn2SyncInfo,self)._build_attributes()
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
        self.gain_fit_params              = self.meta['gain_fit_params']
        self.yo_wait                      = self.meta['yo_wait']
        self.moments_order                = self.meta['moments_order']
        
        ## Converting only int to make sure they are properly initialize after opening from .npz
        self.n_threads     = int(self.meta['n_threads'])
        self.l_data        = int(self.meta['l_data'])
        self.l_kernel      = int(self.meta['l_kernel'])
        self.l_hc          = old_div(self.l_kernel,2) + 1
        self.l_kernel_sym  = self.l_hc                # Diviser par deux car on va symétrisé
        self.l_fft         = int(self.meta['l_fft'])
        self.nb_of_bin     = int(self.meta['nb_of_bin'])
        self.F             = int(self.meta['F'])
        self.period        = int(self.meta['period'])
        self.gz_phase_mes_params        = self.meta['gz_phase_mes_params']
        self.psg_A_phase_mes =  self.gz_phase_mes_params['psg_A']
        self.phase_target_deg =  self.gz_phase_mes_params['phase_target_deg']
        self.reps_phase_mes =  int(self.gz_phase_mes_params['reps'])
        self.channel_idx_phase_mes =  int(self.gz_phase_mes_params['channel_idx'])
        
        ## important variables from filters
        self.t             = self.meta['t']
        self.alpha         = self.meta['alpha']
        self.betas         = self.meta['betas'] 
        self.Thetas        = self.meta['Thetas']
       
        self.make_kernels_d = dict(t=self.t,betas=self.betas[:,None],g=None,window=True,alpha=self.alpha,Z=self.meta['R_jct'],Theta=self.Thetas[None,:],half_norm=True)
        
        # ks[betas,quads,time]
        self.n_quads       = self.meta['ks'].shape[1]
        self.n_filters     = self.meta['ks'].shape[0]
        
class dn2SyncExp(dn2SyncInfo,Cross_Patern_Lagging_computation):
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
        self._log    = logger(loop_sizes=loop_sizes,events=events,rate=rate)
    def _init_objects(self):
        self.pump_phase = np.full( (self._n_measures+1,),np.nan ) 
        self._init_acorr()
        self._init_TimeQuad()
        self._init_Histograms()
    def get_SII(self,data,data_type = 'int16'):
        acorr =  ACorrUpTo(self.l_kernel_sym,data_type)
        acorr(data)
        return acorr.res
        
    def get_SII_phi (self,data,data_type = 'int16'):
        acorr = ACorrUpTo(self.l_kernel_sym,data_type,phi=self.period)
        acorr(data)
        return acorr.res
        
    def reset_objects(self):
        self.n_G_trck = 0
        self.X.reset()
        self.Y.reset()
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
            angles += [np.angle(R,deg=True),]
        print("Angle : {}".format(np.mean(angles)))
        return np.mean(angles)
                
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
        acorr_vdc_shape         = ( n+1,l_vdc, ) 
        acorr_vac_shape         = ( n+1,l_vac, )
        data_type = 'int16'
        self.SII_vdc            = np.full((n+1,l_vdc            ,self.l_kernel_sym),np.nan) 
        self.SII_vac            = np.full((n+1,l_vac,self.period,self.l_kernel_sym),np.nan)
    def _init_TimeQuad(self):
        g               = np.ones((self.l_hc,),dtype=complex) # dummy fillter for initialization 
        self.make_kernels_d['g'] = g
        self.make_kernels_d['half_norm'] = False
        self.ks_default , _ = make_kernels(**self.make_kernels_d)
        self.make_kernels_d['half_norm'] = True 
        self.data_gz   = self.gz.get()[0] # int16
        self.X         = TimeQuad(self.ks_default,self.data_gz, self.meta['dt'],self.l_fft,self.nb_of_bin,self.meta['max'],self.n_threads)
        self.Y         = TimeQuadSync(self.ks_default,self.data_gz, self.meta['dt'],self.l_fft,self.nb_of_bin,self.period,self.meta['max'],self.n_threads)
        self.X.execute( self.ks_default, self.data_gz ) # force the initialization of memory
        self.Y.execute( self.ks_default, self.data_gz ) # force the initialization of memory
    def _init_Histograms(self):
        n               = self._n_measures
        l_Vdc           = len(self._conditions_core_loop_raw[0])
        l_Vac           = len(self._conditions_core_loop_raw[1])
        max             = self.meta['max'] 
        nb_of_bin       = self.nb_of_bin
        period          = self.period
        n_filters       = self.n_filters
        # We accumulate over all reps now
        self.Hs_vacuum  = np.zeros((n_filters,nb_of_bin,nb_of_bin),dtype=np.uint64) 
        self.Hs_vac     = np.zeros((n_filters,l_Vac,period,nb_of_bin,nb_of_bin),dtype=np.uint32) 
        self.moments_vacuum = np.full( (n_filters,self.moments_order+1,self.moments_order+1), np.nan )  
        # self.moments_ac = np.full( (n+1,n_filters,l_Vac,period,self.moments_order+1,self.moments_order+1), np.nan )
        self.moments_ac = np.full( (n_filters,l_Vac,period,self.moments_order+1,self.moments_order+1), np.nan )  
        self.H_x        = TimeQuad.abscisse(max,nb_of_bin) 
   
    def compute_g_bin_v_per_v(self,G_of_f):
        """
        Converts G in A**2/A**2
        to       g in [bin_V/V]
        """
        return (1.0/(self.meta['V_per_bin']))*(50.0 / self.meta['R_jct'])*np.sqrt( G_of_f , dtype='complex')
    def _all_loop_open(self) :
        super(dn2SyncExp,self)._all_loop_open()
        self.yoko.set_init_state(abs(self._conditions_core_loop_raw[0]).max())
        self.psg.set_ampl(-135)
        self.psg.set_output(True)
        
        self.psg.set_ampl(self.psg_A_phase_mes) 
        self.reset_phase(f=self.F,p_target=self.phase_target_deg,reps=self.reps_phase_mes,channel_idx=self.channel_idx_phase_mes)
        self.pump_phase[0] = self.get_phase(F=self.F,reps=self.reps_phase_mes,channel_idx=self.channel_idx_phase_mes)
        self.psg.set_ampl(-135)
        
        ## Need to measure the fisrt G
        ### get an iterator only for Vdc
        idx_it, it = Experiment._super_enumerate(*self._conditions_core_loop_raw[:-1:])
        ### sets the first conditions and wait
        Experiment._repetition_loop_start(self,0)
        self._first_conditions = next(it)
        self._log.events_print(self._first_conditions)
        self.yoko.set_and_wait(self._first_conditions[0],Waittime=self.yo_wait)
        ### Iterate once on Vdc 
        core_it = self.core_iterator(idx_it,it)
        for (idx_tpl,cdn_tpl ) in core_it :
            j,                      = idx_tpl     
            vdc_next,               = cdn_tpl   
            self.data_gz            = self.gz.get()[0] # int16 
            self.yoko.set_and_wait(vdc_next,Waittime=self.yo_wait)
            self._log.event(0)
            self.SII_vdc[0,j]       = self.get_SII(self.data_gz)        # First accor is before all 
            self._log.event(1)
            super(dn2SyncExp,self)._loop_core(idx_tpl,cdn_tpl)
        
        ### Last iteration of that loop
        self.data_gz            = self.gz.get()[0] # int16 
        self._log.event(0)
        self.SII_vdc[0,-1]= self.get_SII(self.data_gz)
        
        self._log.event(1)
        super(dn2SyncExp,self)._loop_core(tuple(),tuple())
        
        ### Compute G avg################################################################################
        self.G_avg = ROUTINE_AVG_GAIN(self._conditions_core_loop_raw[0],self.SII_vdc,self.meta['R_tot'],self.meta['V_per_bin'],self.l_kernel,self.gain_fit_params,windowing=True,i=65)
        #################################################################################################

    def _repetition_loop_start(self,n,condition_it):
        Experiment._repetition_loop_start(self,n)
        
        self._first_conditions = next(condition_it)
        self._log.events_print(self._first_conditions)
        self._set_and_wait_all_devices(self._first_conditions)
        
        #Updating the gain
        g = self.compute_g_bin_v_per_v(self.G_avg)
        self.make_kernels_d['g'] = g
        self.ks , self.hn = make_kernels(**self.make_kernels_d)
        
        # Get vacuum level
        self.psg.set_ampl(-135)                                    
        self.yoko.set_and_wait(0,Waittime=self.yo_wait)
        self.data_gz  = self.gz.get()[0]
        # self.X.reset() # Clear histograms # We accumulate over a whole (all repetitions) experiment now
        self.X.execute( self.ks, self.data_gz ) 
        self.Hs_vacuum = self.X.Histograms()
        
        # Reseting the phase to 0
        self.psg.set_ampl(self.psg_A_phase_mes) 
        self.pump_phase[n+1] = self.get_phase(F=self.F,reps=self.reps_phase_mes,channel_idx=self.channel_idx_phase_mes)
        self.reset_phase(f=self.F,p_target=self.phase_target_deg,reps=self.reps_phase_mes,channel_idx=self.channel_idx_phase_mes)
        self.psg.set_ampl(-135)
        
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
        self.data_gz       = self.gz.get()[0] # int16 
        # Setting next conditions
        if index_it.next_dim == 0 :
            self.yoko.set(vdc_next)
            self.psg.set_ampl(vac_next)
        else : # index_it.next_dim == 1
            self.psg.set_ampl(vac_next)                                  
            self.yoko.set(vdc_next)                                      
        self._log.event(0)
        if index_it.current_dim == 0 :     
            self.SII_vdc[n+1,j]= self.get_SII(self.data_gz)
        else: # index_it.current_dim == 1 :
            #  self.Y.reset() # Clear histograms # We accumulate over a whole (all repetitions) experiment now
            self.Y.execute( self.ks, self.data_gz ) 
            self.Hs_vac[:,k] = self.Y.Histograms()
            # self.SII_vac[n+1,k]= self.get_SII_phi(self.data_gz)
        self._log.event(1)
        super(dn2SyncExp,self)._loop_core(index_tuple,condition_tuple)
    def _last_loop_core_iteration(self,n):
        self.data_gz   = self.gz.get()[0] # int16 
        self._log.event(0)
        # self.Y.reset()
        self.Y.execute( self.ks, self.data_gz ) 
        self.Hs_vac[:,-1] = self.Y.Histograms()
        # self.SII_vac[n+1,-1]= self.get_SII_phi(self.data_gz) 
        
        self.G_avg = ROUTINE_AVG_GAIN(self._conditions_core_loop_raw[0],self.SII_vdc,self.meta['R_tot'],self.meta['V_per_bin'],self.l_kernel,self.gain_fit_params,windowing=True,i=65)
        
        # self.moments_ac[...] = compute_moments2D(self.Hs_vac,self.H_x,order = self.moments_order,Cx=self.hn[:,0][:,None,None],Cy=self.hn[:,1][:,None,None],implementation='numba')
        
        self._log.event(1)
        super(dn2SyncExp,self)._loop_core(tuple(),tuple())
    
    def _all_loop_close(self):
        self.moments_vacuum[...] = compute_moments2D(self.Hs_vacuum,self.H_x,order = self.moments_order,Cx=self.hn[:,0],Cy=self.hn[:,1],implementation='numba')
        self.moments_ac[...] = compute_moments2D(self.Hs_vac,self.H_x,order = self.moments_order,Cx=self.hn[:,0][:,None,None],Cy=self.hn[:,1][:,None,None],implementation='numba')
        super(dn2SyncExp,self)._all_loop_close()
        
    def _build_data(self):
        data = {\
        'ks'            : self.ks_default , # with gain = 1.
        'betas'         : self.betas ,
        'data_gz'       : self.data_gz[:1<<20], # first millon points of the last measurement
        'hs_vacuum'     : self.Hs_vacuum, 
        'hs_vac'        : self.Hs_vac, 
        'S2_vdc'        : self.SII_vdc,
        'S2_vac'        : self.SII_vac,
        'moments_ac'    : self.moments_ac,
        'Vdc'           : self._conditions_core_loop_raw[0],
        'Vac'           : self._conditions_core_loop_raw[1],
        'G_avg'         : self.G_avg,
        'pump_phase'    : self.pump_phase
        }
        return data

     