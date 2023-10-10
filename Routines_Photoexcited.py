#!/bin/env/python
#! -*- coding: utf-8 -*-
import numpy as _np

from SBB.Pyhegel_extra.Experiment                   import logger,Info, Cross_Patern_Lagging_computation, Experiment
from SBB.Pyhegel_extra.Pyhegel_wrappers             import Yoko_wrapper, Guzik_wrapper , PSG_wrapper
from SBB.Time_quadratures.time_quadratures          import TimeQuad_FFT_uint64_t_int16_t as TimeQuad

from SBB.Histograms.histograms                      import Histogram_uint64_t_double
from SBB.Histograms.histograms_helper               import compute_moments
from SBB.AutoCorr.aCorrsOTF.acorrs_otf import ACorrUpTo
#from SBB.AutoCorr.AutoCorr_helper                   import binV2_to_A2, SII_dc_of_t_to_spectrum, compute_SII_sym_and_antisym
from SBB.Numpy_extra.numpy_extra                    import find_nearest_A_to_a,build_array_of_objects
from SBB.Data_analysis.fit                          import polyfit_above_th
from SBB.Microwave.Microwave                        import dBm_to_V
from SBB.Phys.Tunnel_Junction                       import V_th

# Local
from Routines_SII   import ROUTINE_AVG_GAIN
from Quadratures    import gen_fmins_fmaxs,Cmpt_cumulants,Cmpt_std_cumulants,C_to_n,C4_correction
from Methods        import build_imin_imax
from Quadratures    import gen_fmins_fmaxs,Add_Vac_to_Cdc

class dn2_photoexcited_info(Info):
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
    def gen_meta_info(R_jct,R_tot,n_threads,l_data,dt,V_per_bin,l_kernel,F,alpha,kernel_conf,filter_info,nb_of_bin,max,l_fft,ref_idxs,gain_fit_params,yo_wait,moments_order):
        return {
            'R_jct':R_jct,'R_tot':R_tot,'n_threads':n_threads,
            'l_data':l_data,'dt':dt,'V_per_bin':V_per_bin,'l_kernel':l_kernel,'F':F,
            'alpha':alpha,'kernel_conf':kernel_conf,'filter_info':filter_info,
            'nb_of_bin':nb_of_bin,'max':max,'l_fft':l_fft, 'ref_idxs':ref_idxs,
            'gain_fit_params':gain_fit_params,'yo_wait':yo_wait,
            'moments_order':moments_order
            }
    def _set_options(self,options):
        super(dn2_photoexcited_info,self)._set_options(options)
        self._conditions_options    =   {'antisym':options.get('Vdc_antisym') }                                      # Sweeping on positive and negative DC current
        self._ref_options           =   {'interlacing': options.get('interlacing') , 'no_ref':options.get('no_ref')} # Referencing patern
    def _set_conditions(self,conditions):
        super(dn2_photoexcited_info,self)._set_conditions(conditions)
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
            return dn2_photoexcited_info.compute_interlacing(Vdc,ref)
        else :
            return dn2_photoexcited_info.compute_default_ref(Vdc,ref)
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
        super(dn2_photoexcited_info,self)._build_attributes()
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
        self.meta                         = self._meta_info ## alias for self._meta_info
        
        self.Idc                          = self._conditions_core_loop_raw[0]/(self.meta['R_tot'])
        self.gain_fit_params              = self.meta['gain_fit_params']
        self.yo_wait                      = self.meta['yo_wait']
        self.moments_order                = self.meta['moments_order']
        
        ## Converting only int to make sure they are properly initialize after opening from .npz
        self.n_threads     = int(self.meta['n_threads'])
        self.l_data        = int(self.meta['l_data'])
        self.l_kernel      = int(self.meta['l_kernel'])
        self.l_hc          = self.l_kernel/2 + 1
        self.l_kernel_sym  = self.l_hc                # Diviser par deux car on va symétrisé
        self.l_fft         = int(self.meta['l_fft'])
        self.kernel_conf   = int(self.meta['kernel_conf'])
        self.nb_of_bin     = int(self.meta['nb_of_bin'])
        
        ## important variables from filters
        self.n_quads       = 2 if self.kernel_conf == 1 else 1         # Parcequ'on veut mesurer q et p 
        self.n_filters     = self.meta['filter_info']['length']   # alias
        self.labels        = self.meta['filter_info']['labels']   # alias
        self.Filters       = gen_Filters(self.l_kernel,self.meta['dt']*1.e9,self.meta['filter_info'])
    
        self.is_dc_max_set    = False
        self.is_ac_max_set    = False
    
class dn2_photoexcited_exp(dn2_photoexcited_info,Cross_Patern_Lagging_computation):
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
        self._gz                        =   devices[0] 
        self._yoko                      =   devices[1]
        self._psg                       =   devices[2]
    def _init_log(self):
        conditions  = self._conditions_core_loop_raw
        loop_sizes  = ( self._n_measures , len(conditions[0]),len(conditions[1]) )
        events      = ["Acquisition : {:04.2F} [s]","Computing : {:04.2F} [s] "]
        rate        = ( self.meta['l_data']*1.0e-9 ,"Rate : {:04.2F} [GSa/s] " )
        self._log   = logger(loop_sizes=loop_sizes,events=events,rate=rate)
    def _init_objects(self):
        self._init_acorr()
        self._init_TimeQuad()
        self._init_Histograms()
    def get_SII(self,data,data_type = 'int16'):
        acorr =  ACorrUpTo(self.l_kernel_sym,data_type)
        acorr(data)
        return acorr.res
    def reset_objects(self):
        self.n_G_trck = 0
        # Nothing to reset for Quads
        self._reset_Hs()
    #############
    # Utilities #
    #############   
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
        self.SII_vdc          = _np.full((n+1,l_vdc,self.l_kernel_sym),_np.nan) 
        self.SII_vac          = _np.full((n+1,l_vac,self.l_kernel_sym),_np.nan)
    def _init_TimeQuad(self):
        # Use R_jct if the gain has been premeasurement or R_tl if g = 1.0 i.e. no calibration
        g               = _np.ones((self.l_hc,),dtype=complex) # dummy fillter for initialization 
        self._X         = TimeQuad(self.meta['R_jct'],self.meta['dt']*1.e9,self.l_data,self.kernel_conf,self.Filters,g,self.meta['alpha'],self.l_fft,self.n_threads)
        self.betas      = self._X.betas()
        self.filters    = self._X.filters()
        self.ks         = self._X.ks()
        self.quads      = self._X.quads()[:,:,:]
        self._data_gz   = self._gz.get() # int16
        
        self._X.execute( self._data_gz ) # force the initialization of memory
    def _init_Histograms(self):
        n               = self._n_measures
        l_Vdc           = len(self._conditions_core_loop_raw[0])
        l_Vac           = len(self._conditions_core_loop_raw[1])
        max             = self.meta['max'] 
        n_threads       = self.n_threads
        nb_of_bin       = self.nb_of_bin
        n_quads         = self.n_quads
        n_filters       = self.n_filters
        self.Hs_vdc     = build_array_of_objects( (n_filters,l_Vdc)                             , Histogram_uint64_t_double , *(nb_of_bin,n_threads,max) )            
        self.Hs_vac     = build_array_of_objects( (n_quads,n_filters,l_Vac)   , Histogram_uint64_t_double , *(nb_of_bin,n_threads,max) )
        self.moments_dc = _np.full( (n+1,n_filters,l_Vdc,self.moments_order+1), _np.nan )
        self.moments_ac = _np.full( (n+1,n_quads,n_filters,l_Vac,self.moments_order+1), _np.nan )        
        self._H_x       = Histogram_uint64_t_double.abscisse(max,nb_of_bin) 
    def _reset_Hs(self):
        for h in self.Hs_vdc.flat :
            h.reset()
        for h in self.Hs_vac.flat :
            h.reset()
    def get_hs_vdc(self):
        """
            Converts histograms to _np.array and returns a single array
        """
        nb_of_bin  = self.nb_of_bin
        shape = self.Hs_vdc.shape + (nb_of_bin,)
        hs    = _np.full(shape,_np.nan,dtype=float) # histograms are doubles here
        hs.shape = (int(_np.prod(shape[:-1])),shape[-1])
        for Hs,h in zip(self.Hs_vdc.flat,hs):
                h[:] = Hs.get()
        hs.shape = shape
        return hs    
    def get_hs_vac(self):
        """
            Converts histograms to _np.array and returns a single array
        """
        nb_of_bin  = self.nb_of_bin
        shape = self.Hs_vac.shape + (nb_of_bin,)
        hs    = _np.full(shape,_np.nan,dtype=float) # histograms are doubles here
        hs.shape = (int(_np.prod(shape[:-1])),shape[-1])
        for Hs,h in zip(self.Hs_vac.flat,hs):
                h[:] = Hs.get()
        hs.shape = shape
        return hs
    def compute_g_bin_v_per_v(self,G_of_f):
        """
        Converts G in A**2/A**2
        to       g in [bin_V/V]
        """
        return (1.0/(self.meta['V_per_bin']))*(50.0 / self.meta['R_jct'])*_np.sqrt( G_of_f , dtype='complex')
    def _all_loop_open(self) :
        super(dn2_photoexcited_exp,self)._all_loop_open()
        self._yoko.set_init_state(abs(self._conditions_core_loop_raw[0]).max())
        self._psg.set_ampl(-135)
        self._psg.set_output(True)
        ## Need to measure the fisrt G
        ### get an iterator only for Vdc
        idx_it, it = Experiment._super_enumerate(*self._conditions_core_loop_raw[:-1:])
        ### sets the first conditions and wait
        Experiment._repetition_loop_start(self,0)
        self._first_conditions = it.next()
        self._log.events_print(self._first_conditions)
        self._yoko.set_and_wait(self._first_conditions[0],Waittime=self.yo_wait)
        ### Iterate once on Vdc 
        core_it = self.core_iterator(idx_it,it)
        for (idx_tpl,cdn_tpl ) in core_it :
            j,                      = idx_tpl     
            vdc_next,               = cdn_tpl   
            self._data_gz           = self._gz.get() # int16 
            self._yoko.set_and_wait(vdc_next,Waittime=self.yo_wait)
            self._log.event(0)
            self.SII_vdc[0,j]= self.get_SII(self._data_gz)        # First accor is before all 
            self._log.event(1)
            super(dn2_photoexcited_exp,self)._loop_core(idx_tpl,cdn_tpl)
        
        ### Last iteration of that loop
        self._data_gz            = self._gz.get() # int16 
        self._log.event(0)
        self.SII_vdc[0,-1]= self.get_SII(self._data_gz)
        
        self._log.event(1)
        super(dn2_photoexcited_exp,self)._loop_core(tuple(),tuple())
        
        ### Compute G avg################################################################################
        self.G_avg = ROUTINE_AVG_GAIN(self._conditions_core_loop_raw[0],self.SII_vdc,self.meta['R_tot'],self.meta['V_per_bin'],self.l_kernel,self.gain_fit_params,windowing=True,i=65)
        #################################################################################################

    def _repetition_loop_start(self,n,condition_it):
        Experiment._repetition_loop_start(self,n)
        
        self._first_conditions = condition_it.next()
        self._log.events_print(self._first_conditions)
        self._set_and_wait_all_devices(self._first_conditions)
        vdc_next,vac_next  = self._first_conditions
        self._psg.set_ampl(vac_next)                                    # First point setting psg first
        self._yoko.set_and_wait(vdc_next,Waittime=self.yo_wait)
        
        g = self.compute_g_bin_v_per_v(self.G_avg)
        self._X.set_g(g)
        self._reset_Hs()
    def _loop_core(self,index_tuple,condition_tuple,index_it,condition_it,n):
        """
            Works conditionnaly to the computing being slower than 0.4 sec
        """
        j,k                 = index_tuple     
        vdc_next,vac_next   = condition_tuple  
        # Gathering data from last point
        self._data_gz       = self._gz.get() # int16 
        # Setting next conditions
        if index_it.next_dim == 0 :
            self._yoko.set(vdc_next)
            self._psg.set_ampl(vac_next)
        else : # index_it.next_dim == 1
            self._psg.set_ampl(vac_next)                                  
            self._yoko.set(vdc_next)                                      
        self._log.event(0)
        
        self._X.execute(self._data_gz)
        if index_it.current_dim == 0 :     
            # if not(self.is_dc_max_set): # setting custom max for each kernels
                # for i in range(self.n_filters):
                    # self.Hs_vdc[i,j].max = self.quads[0,i,:].max()*1.05
                # self.is_dc_max_set = True
            for i in range(self.n_filters):
                self.Hs_vdc[i,j].accumulate( self.quads[0,i,:] )
            
            self.SII_vdc[n+1,j]= self.get_SII(self._data_gz)
        else: # index_it.current_dim == 1 :
            # if not(self.is_ac_max_set): # setting custom max for each kernels
                # for q in range(self.n_quads):
                    # for i in range(self.n_filters):
                        # self.Hs_vac[q,i,k].max = self.quads[q,i].max()*1.05
                # self.is_ac_max_set = True
            for q in range(self.n_quads):
                for i in range(self.n_filters):
                    self.Hs_vac[q,i,k].accumulate( self.quads[q,i] )
            self.SII_vac[n+1,k]= self.get_SII(self._data_gz)    
        self._log.event(1)
        super(dn2_photoexcited_exp,self)._loop_core(index_tuple,condition_tuple)
    def _last_loop_core_iteration(self,n):
        self._data_gz   = self._gz.get() # int16 
        self._log.event(0)
        self._X.execute(self._data_gz)
        for q in range(self.n_quads):
            for i in range(self.n_filters):
                self.Hs_vac[q,i,-1].accumulate( self.quads[q,i] )
        self.SII_vac[n+1,-1]= self.get_SII(self._data_gz) 
        
        self.G_avg = ROUTINE_AVG_GAIN(self._conditions_core_loop_raw[0],self.SII_vdc,self.meta['R_tot'],self.meta['V_per_bin'],self.l_kernel,self.gain_fit_params,windowing=True,i=65)
        ##############################################################################################
        # Compute moments and reset histograms #######################################################
        self.moments_dc[n+1,...] = compute_moments(self.Hs_vdc,self._H_x,order = self.moments_order,Cxs=self._X.half_norms())
        self.moments_ac[n+1,...] = compute_moments(self.Hs_vac,self._H_x,order = self.moments_order,Cxs=self._X.half_norms())
        ##############################################################################################
        self._log.event(1)
        super(dn2_photoexcited_exp,self)._loop_core(tuple(),tuple())
        
    def _build_data(self):
        data = {\
        'filters'       : self.filters,
        'ks'            : self.ks ,
        'quads'         : self.quads[...,:1<<20], # first millon points of the last quadrature calculation
        'betas'         : self.betas ,
        'data_gz'       : self._data_gz[:1<<20], # first millon points of the last measurement
        'hs_vdc'        : self.get_hs_vdc(), # last iteration's histogram
        'hs_vac'        : self.get_hs_vac(), # last iteration's histogram
        'S2_vdc'        : self.SII_vdc,
        'S2_vac'        : self.SII_vac,
        'moments_dc'    : self.moments_dc,
        'moments_ac'    : self.moments_ac,
        'Vdc'           : self._conditions_core_loop_raw[0],
        'Vac'           : self._conditions_core_loop_raw[1],
        'G_avg'         : self.G_avg
        }
        return data

def Std_moments_to_Cs_NO_CORR(std_m_dc,std_m_ac,fast=True,only_p=True):
    if fast :
        std_m_dc = _np.nanmean(std_m_dc,axis=0)[None,...]
        std_m_ac = _np.nanmean(std_m_ac,axis=0)[None,...]

    if only_p: # fix for when I choose the wronf kernel_conf
        std_m_ac = std_m_ac[:,0,...] # only p
        
    Cdc      = Cmpt_cumulants(std_m_dc)         # Cumulants
    Cdc      = _np.nanmean(Cdc,axis=2)          # Symmetrize
    Cac      = Cmpt_cumulants(std_m_ac)         # Cumulants
    
    # From here always work on averaged statistics
    Cdc = _np.nanmean(Cdc,axis=0)
    Cac = _np.nanmean(Cac,axis=0)
    
    return Cdc,Cac

def get_abscisses(Vac_dBm,alpha,R,F,Labels,separator='&'):
    Iac = alpha*dBm_to_V(Vac_dBm,R)/R/_np.sqrt(2)    
    f_mins,f_maxs = gen_fmins_fmaxs(Labels,separator=separator)
    return Iac,f_mins,f_maxs


def Std_moments_to_Cs(std_m_dc,std_m_ac,Vac_dBm,alpha,R,Te,F,Ipol,Labels,fast=True,only_p=True,separator='&'):
    if fast :
        std_m_dc = _np.nanmean(std_m_dc,axis=0)[None,...]
        std_m_ac = _np.nanmean(std_m_ac,axis=0)[None,...]

    if only_p: # fix for when I choose the wronf kernel_conf
        std_m_ac = std_m_ac[:,0,...] # only p
        
    Iac,f_mins,f_maxs = get_abscisses(Vac_dBm,alpha,R,F,Labels,separator=separator)
    
    If = V_th(F/2)/R # Courant correspondant à une certaine fréquence .. 
    _,ref_idx = find_nearest_A_to_a(If,Ipol)
    
    Cdc      = Cmpt_cumulants(std_m_dc)         # Cumulants
    Cdc      = _np.nanmean(Cdc,axis=2)          # Symmetrize
    Cac      = Cmpt_cumulants(std_m_ac)         # Cumulants
    
    C4dc_init = Cdc[...,4].copy()
    C4ac_init = Cac[...,4].copy()
    
    # Correction is done on the total noise (not sample noise)
    C4dc_corr,C4ac_corr,_ = C4_correction(Cdc,Cac,fuse_last_two_axis=True)

    Cdc[...,4]  = C4dc_corr
    Cac[...,4]  = C4ac_corr

    # From here always work on averaged statistics
    Cdc = _np.nanmean(Cdc,axis=0)
    Cac = _np.nanmean(Cac,axis=0)
    C4dc_init = _np.nanmean(C4dc_init,axis=0)
    C4ac_init = _np.nanmean(C4ac_init,axis=0)
   
    # Cumulants sample
    Cdc = (Cdc[...,1,:,:]-Cdc[...,0,:,:]) 
    Cac = (Cac[...,1,:,:]-Cac[...,0,:,:]) 
    C4dc_init = C4dc_init[...,1,:] - C4dc_init[...,0,:] 
    C4ac_init = C4ac_init[...,1,:] - C4ac_init[...,0,:] 
    
    # Adding measured vacuum to Cdc
    Cdc,Pdc = Add_Vac_to_Cdc(Ipol,Cdc,f_maxs,R=R,Te=Te,fmax=10000000000.0, imax=2.1e-06,epsilon=0.001)
    
    # Adding Cdc(Vdc=6GHz) to Cac
    Cac[...,2] = Cac[...,2] + Cdc[...,ref_idx:ref_idx+1,2]
    
    return Cdc,C4dc_init,Pdc,Iac,Cac,C4ac_init,f_mins,f_maxs,ref_idx
    
def Std_moments_to_ns(std_m_dc,std_m_ac,Vac_dBm,alpha,R,Te,F,Ipol,Labels,fast=True,only_p=True,separator='&'):
    Cdc,C4dc_init,Pdc,Iac,Cac,C4ac_init,f_mins,f_maxs,ref_idx = Std_moments_to_Cs(std_m_dc,std_m_ac,Vac_dBm,alpha,R,Te,F,Ipol,Labels,fast=fast,only_p=only_p,separator=separator)
    
    nsdc  = C_to_n(Cdc)
    ns_ac = C_to_n(Cac)
    return Cdc,C4dc_init,Pdc,nsdc,Iac,Cac,C4ac_init,ns_ac,f_mins,f_maxs,ref_idx