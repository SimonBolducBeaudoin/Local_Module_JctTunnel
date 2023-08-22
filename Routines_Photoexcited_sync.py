#!/bin/env/python
#! -*- coding: utf-8 -*-
import itertools
import numpy

from SBB.Pyhegel_extra.Experiment                   import logger,Info, Cross_Patern_Lagging_computation, Experiment
from SBB.Pyhegel_extra.Pyhegel_wrappers             import Yoko_wrapper, Guzik_wrapper , PSG_wrapper
from SBB.Time_quadratures.time_quadratures          import TimeQuad_uint64_t 
from SBB.Time_quadratures.TimeQuadrature_helper     import gen_Filters, gen_t_abscisse, gen_f_abscisse, moments_correction
from SBB.Histograms.histograms                      import Histogram_uint64_t_double
from SBB.Histograms.histograms_helper               import compute_moments
from SBB.AutoCorr.acorrs_otf                        import ACorrUpTo
from SBB.AutoCorr.AutoCorr_helper                   import binV2_to_A2, SII_dc_of_t_to_spectrum, compute_SII_sym_and_antisym
from SBB.Numpy_extra.numpy_extra                    import build_array_of_objects
from SBB.Data_analysis.fit                          import polyfit_above_th

from Routines import ROUTINE_AVG_GAIN

class dn2_photoexcited_sync_info(Info):
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
    def gen_meta_info(R_jct,R_tot,n_threads,l_data,dt,V_per_bin,l_kernel,F,alpha,kernel_conf,filter_info,nb_of_bin,max,l_fft,ref_idxs,n_time_histograms,gain_fit_params,yo_wait,moments_order):
        return {
            'R_jct':R_jct,'R_tot':R_tot,'n_threads':n_threads,
            'l_data':l_data,'dt':dt,'V_per_bin':V_per_bin,'l_kernel':l_kernel,'F':F,
            'alpha':alpha,'kernel_conf':kernel_conf,'filter_info':filter_info,
            'nb_of_bin':nb_of_bin,'max':max,'l_fft':l_fft, 'ref_idxs':ref_idxs,
            'n_time_histograms':n_time_histograms,'gain_fit_params':gain_fit_params,'yo_wait':yo_wait,
            'moments_order':moments_order
            }
    def _set_options(self,options):
        super(dn2_photoexcited_sync_info,self)._set_options(options)
        self._conditions_options    =   {'antisym':options.get('Vdc_antisym') }                                      # Sweeping on positive and negative DC current
        self._ref_options           =   {'interlacing': options.get('interlacing') , 'no_ref':options.get('no_ref')} # Referencing patern
        self.do_scope = options.get('do_scope')
    def _set_conditions(self,conditions):
        super(dn2_photoexcited_sync_info,self)._set_conditions(conditions)
    @staticmethod
    def compute_interlacing(Vdc,ref=0):
        Vdc_interlaced = numpy.ones(2*len(Vdc))*ref
        Vdc_interlaced[1::2] = Vdc
        return Vdc_interlaced
    @staticmethod
    def compute_default_ref(Vdc,ref=0):
        return numpy.concatenate(([ref],Vdc))
    @staticmethod
    def add_antisym(Vdc,**sym_options):
        """
        Return Vdc_antisym conditional to sym_options
        """
        return numpy.concatenate(([(-1.0)*Vdc[::-1],Vdc])) if sym_options.get('antisym') else Vdc
    @staticmethod
    def add_ref_conditions(Vdc,ref=0,**ref_options):
        """
            Add the right referencing partern to Vdc conditionnal to ref_options
        """
        if    ref_options.get('no_ref'): 
            return Vdc
        elif  ref_options.get('interlacing'):
            return dn2_photoexcited_sync_info.compute_interlacing(Vdc,ref)
        else :
            return dn2_photoexcited_sync_info.compute_default_ref(Vdc,ref)
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
        super(dn2_photoexcited_sync_info,self)._build_attributes()
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
        self.n_G_mod                      = self.meta['n_G_mod']
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
        self.n_time_histograms = int(self.meta['n_time_histograms'])
        
        ## important variables from filters
        self.n_quads       = 2 if self.kernel_conf == 1 else 1         # Parcequ'on veut mesurer q et p 
        self.n_filters     = self.meta['filter_info']['length']   # alias
        self.labels        = self.meta['filter_info']['labels']   # alias
        self.Filters       = gen_Filters(self.l_kernel,self.meta['dt']*1.e9,self.meta['filter_info'])
        
        self.is_max_set    = False
        
    
class dn2_photoexcited_sync_exp(dn2_photoexcited_sync_info,Cross_Patern_Lagging_computation):
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
        self.G_of_f = numpy.full((self._n_measures+1,self.l_hc),numpy.nan)
        self.B_of_f = numpy.full(self.G_of_f.shape,numpy.nan)
        self.freq   = gen_f_abscisse(self.l_kernel,self.meta['dt'])
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
        self.SII_vdc          = numpy.full((n+1,l_vdc,self.l_kernel_sym),numpy.nan) 
        self.SII_vac          = numpy.full((n+1,l_vac,self.l_kernel_sym),numpy.nan)
    def _init_TimeQuad(self):
        # Use R_jct if the gain has been premeasurement or R_tl if g = 1.0 i.e. no calibration
        g               = numpy.ones((self.l_hc,),dtype=complex) # dummy fillter for initialization 
        self._X         = TimeQuad_uint64_t(self.meta['R_jct'],self.meta['dt']*1.e9,self.l_data,self.kernel_conf,self.Filters,g,self.meta['alpha'],self.l_fft,self.n_threads)
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
        n_time_histograms = self.n_time_histograms
        n_quads         = self.n_quads
        n_filters       = self.n_filters
        self.Hs_vdc     = build_array_of_objects( (n_filters,l_Vdc)                             , Histogram_uint64_t_double , *(nb_of_bin,n_threads,max) )            
        self.Hs_vac     = build_array_of_objects( (n_quads,n_filters,l_Vac,n_time_histograms)   , Histogram_uint64_t_double , *(nb_of_bin,n_threads,max) )
        self.moments_dc = numpy.full( (n+1,n_filters,l_Vdc,self.moments_order+1), numpy.nan )
        self.moments_ac = numpy.full( (n+1,n_filters,l_Vac,self.moments_order+1), numpy.nan )        
        self._H_x       = Histogram_uint64_t_double.abscisse(max,nb_of_bin) 
    def _reset_Hs(self):
        for h in self.Hs_vdc.flat :
            h.reset()
        for h in self.Hs_vac.flat :
            h.reset()
    #################
    # Loop behavior #
    #################
    def _all_loop_open(self) :
        super(dn2_photoexcited_sync_exp,self)._all_loop_open()
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
            self._yoko.set(vdc_next)
            self._log.event(0)
            self.SII_vdc[0,j]= get_SII(self._data_gz)        # First accor is before all 
            self._log.event(1)
            super(dn2_photoexcited_sync_exp,self)._loop_core(idx_tpl,cdn_tpl)
        ### Last iteration of that loop
        self._data_gz            = self._gz.get() # int16 
        self._log.event(0)
        self.SII_vdc[0,-1]= get_SII(self._data_gz)
        self._log.event(1)
        super(dn2_photoexcited_sync_exp,self)._loop_core(tuple(),tuple())
        
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
        else # index_it.next_dim == 1 :
            self._psg.set_ampl(vac_next)                                  
            self._yoko.set(vdc_next)                                      
        self._log.event(0)
        self._X.execute(self._data_gz)
        if index_it.current_dim == 0 :            
            for i in range(self.n_filters):
                self.Hs_vdc[i,j].accumulate( self.quads[0,i,:] )
            self.SII_vdc[n+1,j]= get_SII(self._data_gz)
        else: # index_it.current_dim == 1 :
            n_time_hs = self.n_time_histograms
            for q in range(self.n_quads):
                for i in range(self.n_filters):
                    for t in range(n_time_hs):
                        self.Hs_vac[q,i,k,t].accumulate( self.quads[q,i,t::n_time_hs] )
            self.SII_vac[n+1,k]= get_SII(self._data_gz)    
        self._log.event(1)
        super(dn2_photoexcited_sync_exp,self)._loop_core(index_tuple,condition_tuple)
    def _last_loop_core_iteration(self,n):
        self._data_gz   = self._gz.get() # int16 
        self._log.event(0)
        self._X.execute(self._data_gz)
        n_time_hs = self.n_time_histograms
        for q in range(self.n_quads):
            for i in range(self.n_filters):
                for t in range(n_time_hs):
                    self.Hs_vac[q,i,-1,t].accumulate( self.quads[q,i,t::n_time_hs] )
        self.SII_vac[n+1,-1]= get_SII(self._data_gz) 
        
        self.G_avg = ROUTINE_AVG_GAIN(self._conditions_core_loop_raw[0],self.SII_vdc,self.meta['R_tot'],self.meta['V_per_bin'],self.l_kernel,self.gain_fit_params,windowing=True,i=65)
        ##############################################################################################
        # Compute moments and reset histograms #######################################################
        self.moments_dc[n+1,...] = compute_moments(self.Hs_vdc,self._H_x,order = self.moments_order,Cxs=self._X.half_norms())
        self.moments_ac[n+1,...] = compute_moments(self.Hs_vac,self._H_x,order = self.moments_order,Cxs=self._X.half_norms())
        self._reset_Hs()
        ##############################################################################################
        self._log.event(1)
        super(dn2_photoexcited_sync_exp,self)._loop_core(tuple(),tuple())
    ######################
    # Analysis Utilities #
    ######################
    def compute_g_bin_v_per_v(self,G_of_f):
        """
        Converts G in A**2/A**2
        to       g in [bin_V/V]
        """
        return (1.0/(self.meta['V_per_bin']))*(50.0 / self.meta['R_jct'])*numpy.sqrt( G_of_f , dtype='complex')
    def _build_data(self):
        data = {\
        'S2_vdc'        : self.SII_vdc,
        'S2_vac'        : self.SII_vac,
        'betas'         : self.betas ,
        'moments_dc'    : self.moments_dc,
        'moments_ac'    : self.moments_ac,
        }
        return data