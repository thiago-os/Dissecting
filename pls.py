##############################################################################
# Module for "Dissecting market expectations in 
# the cross-section of book-to-market ratios
# 
# Critical Finance Review, February 2021
# Thiago de Oliveira Souza
# https://sites.google.com/site/tdosthiago
##############################################################################


# =============================================================================
# Many of the docstrings correspond to previous versions of the code. 
# If the docstring seems at odds with the code, ignore the docstring.
# =============================================================================


import pandas as pd
import numpy as np
import pickle
import os
from pandas.tseries.offsets import MonthEnd
import statsmodels.api as sm
from functools import partial

# From the Jupyter part:
import itertools
import concurrent.futures
import time

from dateutil.relativedelta import *
from pandas.tseries.offsets import *


# Declare the Model class
#########################

class Model:
    def __init__(self, key):
        # Only attribute assigned when instance is created
        ##################################################
        self.key = key  # key is a tupple with all choices that define the
        # instance/model.
      

def standard_values(key, BM_dic, returns_dic, continuous=True, CRSPret=True, 
                    OOSdate='1979-12-31'):
    ''' This function fills the standard attributes of any model 
    (regressors, targets, meta information) 
    based on the key = tupple that identifies the model.
    Inputs:
        key = tupple that  follows Model.label_keys
        BM_dic = dictionary with BMs identified by the fields (target, cut30)
        returns_dic = dictionary with returns identified by fields (Nx, logBM,
                                                                    cut30)
    Output
        Instance of Model class with initial values in key +
            target
            BM panel
            standard (cont, CRSP, OOSdate)
    '''
    model = Model(key)
    # Copy all meta info from key into attributes
    for i in range(len(Model.label_keys)):
        setattr(model,
                Model.label_keys[i],
                model.key[i])
    # Fill return and BMs
    model.Y = returns_dic[model.target,
                          model.cut30]
    model.Xn = BM_dic[model.Nx,
                      model.logBM,
                      model.cut30]
    model.continuous = continuous
    model.CRSPret = CRSPret
    model.OOSdate = OOSdate
    return model    


# First pass regression
#######################
def get_phi(BMcol, target, minH, minR):
    ''' Regress BMcol on target (time-series). Returns slope phi_i for BMcol 
    series, given return target and sample size parameters.
    First pass: One per time-series of "BM". Apply on columns.
'''
    phi_i = np.nan
    # minH and minR are ad-hoc conditions KP use to get better results early 
    # in the sample.
    if (BMcol.notnull().sum() / len(BMcol) >= minH) & (target.count() > minR):
        # (too few observations for regression =>phi remains nan)
        results = sm.OLS(BMcol,
                         sm.add_constant(target),
                         missing='drop').fit()
        phi_i = results.params[1]
    return phi_i


# Second pass regression
########################
def get_ft(BMrow, phi):
    ''' Regress BMrow on phi. Returns slope, ft, for the *row*.
    (the group of BMs, in rows/cross-section), given phi 
    (with same dimension as Nrows).
    Second pass: One per period across BMs. Apply on rows.
'''    
    ft = np.nan
    # (too few observations for regression =>ft remains nan)
    if (np.count_nonzero(~np.isnan(phi)) > 2) and (not BMrow.isnull().all()):
        results = sm.OLS(BMrow,
                         sm.add_constant(phi),
                         missing='drop').fit()
        ft = results.params[1]
    return ft


# First + Second pass regressions
#################################

def estim_F(Xn, Ret, continuous, h, Std, Std2, minH, minR, IS, lagBM):
    """ function estimates latent factor (time series of) from panel of 
    regressors ("BMs") and target (future return, "MP", e.g., one year ahead,
                                   h = 12).
    First and second pass regressions together.
    
    Can be variance-standardized or not. 
    End of period convention:
        Ret(Jan) = realized during january
        F(Jan) = F(31/01)
    
Inputs:
    Xn, Ret must be indexed by date (must overlap)
    Xn = df with all "BMs"
    Ret = df vector with forecasting target: Realized returns in months 
    (decimals) or in year (if function is passed internally for OOS calculation
                           <=> IS=False)
    continuous = Continuous returns (true/false)
    h = forecasting horizon (e.g, 12 months)
    IS = estimation only in-sample? True/False (Yearly return needs adjustment)
    lagBM = number of periods BM must be (further) lagged before estimation
    
    None of these is mentioned by KP (but appear in their code):    
        Std = Standardize Xn?
        ...for better early OOS estimates
            Std2 = Standardize phi before second stage (for OOS) - No effect
            minH = Fraction of time-series needed to have valid estimate 
                    for the particular BM series (1st stage)
            minR = Min. number of returns for valid estimation (1st stage)
    
Output: 
    df with same index=dates as Xn
        df['F'] = series estimate of the latent factor
""" 
    # Copy to avoid modifying original df
    Ret = Ret.copy(deep=True)
    Xn = Xn.copy(deep=True)
    
    if lagBM != 0:
        Xn.index = Xn.index + MonthEnd(lagBM)
    
    # Main standardization choice (not mentioned in KP paper)
    if Std:
        Xn = Xn / Xn.std()

    # Transform to "annual" *realized* returns if needed, 
    # and only if function is run in-sample
    if h != 1 and IS:
        Ret = changeFreq(Ret, new_frequency=h, continuous=continuous,
                         percent=False)
    # else: Ret frequency already fine

    # Merge to enforce correct dates:
    data = pd.merge(Xn, Ret.shift(-h), how='left', 
                    left_index=True, right_index=True)
    idxR = Xn.shape[1]  # Index of 'RM' column
    #   =>  data.iloc[:,:idxR] = Xn
    #       data.iloc[:,idxR] = Ret.shift(-h)
    
    # First pass: Regress BMs on future returns (one time-series reg per BM)
    ########################################################################
    # Apply on columns of BMs:
    phi = np.array(data.iloc[:, :idxR].apply(get_phi,
                                             args=(data.iloc[:, idxR], minH,
                                                   minR),
                                             axis=0))
    
    # Here is the second normalization made by KP 
    # (only for OOS, also not mentioned in their paper)
    # But this one does not impact the results, in fact.
    if Std2:
        phi = phi / np.nanstd(phi)

    # Second pass: Regress BMs on phi (one cross-sec regression per t)
    ###################################################################
    data['F'] = data.iloc[:, :idxR].apply(get_ft,
                                          args=(phi,),
                                          axis=1)
    return data['F']


def changeFreq(Ret, new_frequency, continuous=False, percent=False):
    ''' transforms returns from one frequeny to another.
Inputs:
    Ret = pd series with the returns in the current frequency 
    (in decimals or percents)
    continuous = True/False for continuous returns
    new_frequency = new forecasting wanted (e.g, 12 months)
    percent = True/False if the original returns are in %
   
Output: 
    Transformed return series (in % if originally in %)
'''
    Ret = Ret.copy()
    if percent:
        Ret = Ret / 100
        
    if continuous:
        Ret = (Ret).rolling(window=new_frequency).apply(np.sum, raw=True)
    else:
        Ret = (1 + Ret).rolling(window=new_frequency
                                ).apply(np.prod, raw=True) - 1
        
    if percent:
        Ret = Ret * 100
        
    return Ret


# Recursive forecasts (and errors)
##################################
# Full time series. This is the main function that calls the ones above
def recursive(model_dic_item, Std2=True, skip=60, leave=0):
    """ function to estimate recursive (time series of) expectations of 
    the target (future return, "MP", e.g., one year ahead => freq h = 12)
    from panel of regressors ("BMs").
    Can be standardized or not.
    End of period convention:
        Ret(Jan) = P(31/12) -> P(31/01)
        F(Jan) = F(31/01)
        Prediction(Jan) = Prediction (31/1)


Inputs:
    Xn = matrix with all "BMs"
    Ret = Vector with forecasting target: Realized returns in months (decimals)
    continuous = Continuous returns (true/false)
    h = forecasting horizon (e.g, 12 months)
    
    
    None of these is mentioned by KP (but appear in their code):    
        Std = Standardize Xn?
        ...for better early OOS estimates
            Std2 = Standardize phi before second stage (for OOS)
            minH = Fraction of time-series needed to have valid estimate
                for the particular BM series (1st stage)
            minR = Min. number of return for valid estimation (1st stage)
            
    skip = N periods unused before first estimate
    leave = N periods unused after last estimate.
        Never used because it ignores data *completely*
    lagBM = number of periods BM must be (further) lagged before estimation

Output: 
    A object from Model class with:
        .forecast = df with same (end of period) index=dates as BMs
            df['Er'] = OOS Expected return each period from model ("larger")
            df['Ehist'] = OOS Expected return each period (historical/nested)
            df['eLarger'] = Forecasting error of the model ("larger" opposed 
                                                            to constant only)
            df['eNested'] = Forecasting error of historical mean (constant only
                                                                  => "nested")
            df['Future'] =  Returns that will be realized over h periods
                            starting *after* t. 
                            ex: t=Jan (end) => Ret(01/Feb - 31/Jan)
        .h = h (horizon)
        .ret = monthR
        .std = Std
        .std2 = Std2
        .continuous = continuous
        .shortBM = minH
        .shortR = minR
""" 

    key, model_obj = model_dic_item
    
    # Use internal variable names
    Xn = model_obj.Xn
    monthR = model_obj.target
    continuous = model_obj.continuous
    h = model_obj.horizon
    Std = model_obj.std
    minH = model_obj.minH
    minR = model_obj.minR
    lagBM = model_obj.lagBM

    if lagBM != 0:
        # Do not change original
        Xn = Xn.copy(deep=True)
        Xn.index = Xn.index + MonthEnd(lagBM)
        
    # Merge already copy the dfs => not change input df
    data = pd.merge(Xn, model_obj.Y, how='left', left_index=True, 
                    right_index=True)
    idxR = Xn.shape[1]  # Index of 'RM' column

    # Ret. in proper frequency, 'fR', (h=1 => no change)
    if h != 1:
        data['fR'] = changeFreq(data[monthR], new_frequency=h,
                                continuous=continuous, percent=False)        
    else:
        data['fR'] = data[monthR]
    
    # Returns that will be realized over h periods starting *after* t:
    # ex: t=Jan (end) => Ret(01/Feb - 31/Jan)
    data['Future'] = data['fR'].shift(-h)
        
    # historical mean prediction (observable at t):
    ###############################################
    data['Ehist'] = data['fR'].expanding().mean()
    # Errors historical:
    data['eNested'] = data['Ehist'] - data['Future']
    
    ############################################################
    # Forecasts from the model (larger) *at the end of month t*
    ############################################################
    data['Er'] = np.nan
    # make one prediction w/ info until (and including) t
    for t in range(skip, data.shape[0] - leave):
        
        # The last *position* is t!
        subsample = data.iloc[: t + 1, :].copy()
        lastDateIS = subsample.index[-1]  # last date in subsample
        
        subsample['F'] = estim_F(subsample.iloc[:, :idxR],  # the first t "BMs"
                                 subsample['fR'],
                                 continuous, h, Std, Std2, minH, minR,
                                 IS=False, lagBM=0)
        
        # because conditions minR, etc => F=Nan often
        if not subsample['F'].isnull().all():
            results = sm.OLS(subsample['fR'].shift(-h),  # .iloc[:-h]
                             sm.add_constant(subsample['F']),  # .iloc[:-h]
                             missing='drop').fit()
            # Save result in main df:
            data['Er'].loc[lastDateIS] = (
                results.params[0] 
                + results.params[1] * subsample['F'].loc[lastDateIS])

    # Prediction error (model)
    # NOTE: I register the prediction error *at the prediction date*,
    # not "1y" later, when the return is realized!
    # KP place the prediction in t+1 always, even for 12m. 
    # This can be confusing to compare the results.
    data['eLarger'] = data['Er'] - data['Future']
    
    # Return a tupple with the forecast DF and the key:
    return data.filter(['Er', 'Ehist', 'eLarger', 'eNested', 'Future']
                       ).copy(), model_obj.key

#######################################
# End of estimation functions.
#######################################


#########################################################
# Perfomance evaluation: ENC-NEW, critical values, OOS R2
#########################################################
# Inputs: Forecasting errors nested and (larger) model = Output of recursive().

# Variance of residuals with the Newey-West correction (for OOS errors)
def nw_var(errors, lags):
    ''' 
    Function returns the variance of the residuals after Newey-West correction.
    Needs the series of errors from the (larger) model.
    Implementation from Kelly and Pruitt (2013) code in Matlab.

Inputs:
    errors: the series of (heterosk. and autocorr) residuals 
    (not a full dataframe), ex. errors = X['eLarger']
    lags: Maximum number of lags for the Newey-West correction.
'''
    G = np.empty(lags + 1)
    u = errors.values  # .copy()
    T = len(u)

    G[0] = np.inner(u.T, u) / T

    for k in range(1, lags + 1):
        G[k] = np.inner(u[k:].T, u[:-k]) / T

    var = G[0]
    for k in range(1, lags):
        var = var + (1 - k / lags) * 2 * G[k]
    return var


def oos_stats(freq, ence, eNested='eNested', eLarger='eLarger',
              k=1, Er='Er', Ehist='Ehist'):
    '''
    function returns smaller df (without NaNs) with ENC-NEW statistic, 
    critical values, and OOS R2, for each date. Df must be indexed by date.
    Table with citical values are read locally: ENCNEWcriticalK1.csv, 
    ENCNEWcriticalK2.csv.
    (also here: https://sites.google.com/site/tdosthiago/code).
    It also returns:
        MAEd = difference in mean absolute error of the models
        AEd = difference in cumulative abs. error
        and versions sR2, sMAE, sAE = model that switch from hist 
        to model | t-1 error.
    
    Needs two series of OOS forecasting errors:
    a) for the nested model and b) the Larger model (with k extra param).
    
Inputs:
    freq = number of periods (e.g. 12 for year forecasts)
    df (ence) = indexed by date, contains fields with
        eNested = Name of series with error from "historical mean"
        eLarger = Name of series with errors from (larger) "model"
    k = number of extra parameters in larger model (1 or 2).
'''
    ence = ence.copy(deep=True)
    # first, drop missing errors (that have no forecast)
    ence.dropna(subset=[eNested, eLarger], inplace=True)

    # Square of the errors
    ence['eNested2'] = ence[eNested]**2
    ence['eLarger2'] = ence[eLarger]**2
        
    # "Cov" of the errors
    ence['CovLargerNested'] = ence[eLarger] * ence[eNested]
    
    # Now cumulative sum of these values from each point in time until the end
    ence['SumNested'] = np.cumsum(ence['eNested2'][::-1])[::-1]
    ence['SumLarger'] = np.cumsum(ence['eLarger2'][::-1])[::-1]
    ence['SumCov'] = np.cumsum(ence['CovLargerNested'][::-1])[::-1]
    # check on Excel:
    # ence.to_clipboard()
    
    # Here the OOS R-squared (in %):
    ence['R2'] = (1 - ence['SumLarger'] / ence['SumNested']) * 100
    # R2 calculated.
    
    # Goyal-Welch cumulative SSE
    ence['cumSSE'] = np.cumsum(ence['eNested2'] - ence['eLarger2']) 
    
    ######################################
    # Now the ENC-NEW test (and ~p-values)
    ######################################

    # First: .csv w/ critical values for ENC-NEW stat. for k2 = 1 or k2 = 2 
    # (recursive scheme)
    # comes from "Tests of Equal Forecast Accuracy and Encompassing for 
    # Nested Models" 
    # Online Appendix (Clark McCracken 2001). cv1 for one regressor, 
    # cv2 for two:

    t = os.path.join(os.getcwd(), 'local', "ENCNEWcriticalK1.csv")
    urlk = os.path.normpath(t)
    cv1 = pd.read_csv(urlk)
    
    t = os.path.join(os.getcwd(), 'local', "ENCNEWcriticalK2.csv")
    urlk = os.path.normpath(t)
    cv2 = pd.read_csv(urlk)
    
    # Keep date (because it's lost in merge asof)
    ence['date'] = ence.index

    # The test statistic needs number of OOS periods (P)
    # (and ratio of those, pi, for the critical values)
    TotalSample = ence['eNested2'].count()
    
    # If sample did not become empty after dropna, OK:
    if TotalSample != 0:
        # Numbers of IS obs. (R in CM,2001 paper)
        ence['R'] = (12 * (ence.index.year - ence.index[0].year) 
                     + (ence.index.month - ence.index[0].month))
    # otherwise, result doesnt matter
    else:
        ence['R'] = 1
    # Number of OOS obs.
    ence['P'] = TotalSample - ence['R']
    # Ratio OOS / IS
    ence['pi'] = ence['P'] / ence['R']
    
    # The ENC-NEW test statistic = Eq. (3) in Clark McCracken (2001):
    if freq == 1:
        ence['ENC-NEW'] = ence['P'] * (
            ence['SumNested'] - ence['SumCov']) / ence['SumLarger']
        
    else:  # use Newey-West std errors
        # Create Sum of adjusted var = (rolling) NW adjusted variance:
        ence['SumAdjvar'] = (ence['eLarger'][::-1]
                             .expanding()
                             .apply(partial(nw_var, lags=freq))[::-1])
        
        ence['ENC-NEW'] = (
            ence['SumNested'] - ence['SumCov']) / ence['SumAdjvar']

    # Get p-values: merge df with critical values *per date*.
    # Either cv1 or cv2:
    if k == 1:
        ence2 = pd.merge_asof(ence.sort_values(by=['pi']), cv1,
                              left_on='pi', right_on='pi', direction='nearest')
    if k == 2:
        ence2 = pd.merge_asof(ence.sort_values(by=['pi']), cv2,
                              left_on='pi', right_on='pi', direction='nearest')

    # bring back date as index and order df:
    ence2.sort_values(by=['date'], inplace=True)
    ence2.set_index('date', inplace=True)
    
    # closest approx. of critical values merged to df
    ence3 = ence2.filter(['ENC-NEW', 'R2', '99p', '95p', '90p', 'cumSSE'])
    
    return ence3


def non_recursive_estimates(model_obj, Std2=True):
    '''
    This functions changes the input.
    
    function to estimate all attributes of the Model class, given the data 
    inputs. See also description of recursive, estimF, oos_stats. As in
    (Xn, dfR, monthR='RM', continuous=True, h = 12, Std=True, Std2=False, 
    minH=0.8, minR=60, skip=60, leave=0, CRSPret=True, cut30=True)

Inputs 
    (Data indexed by date = last day of month):
    Xn = matrix with all "BMs"
    dfR = dataframe with realized returns
    monthR = label of the column in dfR to be used as the return target in
        months (decimals).
    continuous = Continuous returns? (true/false)
    h = forecasting horizon (e.g, 12 months)
    
    None of these is mentioned by KP (but appear in their code):
        Std = Standardize Xn?
        ...for better early OOS estimates
            Std2 = Standardize phi before second stage (for OOS), False 
                because it does not seem to have an effect.
            minH = Fraction of time-series required for valid estimate 
                in particular BM series (1st stage)
            minR = Min. number of return for valid estimation (1st stage)
            
    skip = N periods unused before first estimate
    leave = N periods unused after last estimate. Never used because it 
        ignores data *completely*
    
    - Just informative (no effect on estimation)
        CRSPret = True/False (simply say if returns are from CRSP (or FF)
        cut30 = True/False (Xn already excludes 1930-, like KP did?)
        
    OOSdate (= '1979-12-31' ex) = last day of month when prediction is made
    lagBM = number of periods BM must be (further) lagged before estimation

Output: 
    A object from Model class with:
    - Fields filled by recursive()
        .forecast = df with same (end of period) index=dates as Xn
            df['Er'] = OOS Expected return each period from model ("larger")
            df['Ehist'] = OOS Expected return each period (historical/nested)
            df['eLarger'] = Forecasting error of the model 
                ("larger" opposed to constant only)
            df['eNested'] = Forecasting error of historical mean 
                (constant only => "nested")
            df['Future'] =  Returns that will be realized over h periods 
                            starting *after* t. 
                            ex: t=Jan (end) => Ret(01/Feb - 31/Jan)
        .horizon = h (horizon)
        .ret = monthR
        .std = Std
        .std2 = Std2
        .continuous = continuous
        .shortBM = minH
        .shortR = minR
    
    - From oos_stats()
        .OOSstat = df with the time-series of the statistics ENC-NEW and 
            (OOS)R2,
        and critical values of ENC-NEW at each date: '99p','95p','90p'. 
            (ENC-NEW > CV => reject at that date)
    
    - From estim_F()
        .latestF =   df['F'] with same index=dates as
            Xn = last available estimate of the latent factor (time series)
    
    - simply filling the info:
        .CRSPret = CRSPret
        .cut30 = cut30
        
    
    .pNW = p-value of Ft in Newey-West (third pass) regression
    .ISR2 = In sample R2 of third pass regression
    .pENC30 = P-value of ENC-NEW test 
        (360 months before end of sample, as in KP)
    .OOSR230 = OOS R2 30y- end of sample (1980 in KP)

'''

    # Check for critical value files:
    t = os.path.join(os.getcwd(), 'local', "ENCNEWcriticalK1.csv")
    if not os.path.exists(t):
        print('Download the file and place it here: ' + t)
        return 'Missing: ' + t
    
    t = os.path.join(os.getcwd(), 'local', "ENCNEWcriticalK2.csv")
    if not os.path.exists(t):
        print('Download the file and place it here: ' + t)
        return 'Missing: ' + t 
    
    # Run the main estimation (already estimated before)
    # model_obj.forecast = recursive(
    # model_obj, Std2=Std2, skip=skip, leave=leave)
    
    model_obj.latest_F = estim_F(Xn=model_obj.Xn,
                                 Ret=model_obj.Y,
                                 continuous=model_obj.continuous,
                                 h=model_obj.horizon, 
                                 Std=model_obj.std,
                                 Std2=Std2,
                                 minH=model_obj.minH,
                                 minR=model_obj.minR,
                                 IS=True,
                                 lagBM=model_obj.lagBM)
    
    # Fill OOS quantities:
    model_obj.OOSstat = oos_stats(freq=model_obj.horizon,
                                  ence=model_obj.forecast,
                                  eNested='eNested',
                                  eLarger='eLarger',
                                  k=1)
    
    # IS quantities (Newey-West):
    ISresults = sm.OLS(model_obj.forecast['Future'],
                       sm.add_constant(model_obj.latest_F),
                       missing='drop'
                       ).fit(cov_type='HAC', 
                             cov_kwds={'maxlags': model_obj.horizon})
    model_obj.ISR2 = (ISresults.rsquared) * 100
    model_obj.pNW = ISresults.pvalues[1]
    
    # OOS sample statistics at date OOSdate:
    try:
        model_obj.OOSR230 = model_obj.OOSstat.loc[model_obj.OOSdate]['R2']
        if (model_obj.OOSstat.loc[model_obj.OOSdate]['ENC-NEW'] 
                > model_obj.OOSstat.loc[model_obj.OOSdate]['99p']):
            model_obj.pENC30 = '<0.01'
        elif (model_obj.OOSstat.loc[model_obj.OOSdate]['ENC-NEW'] 
              > model_obj.OOSstat.loc[model_obj.OOSdate]['95p']):
            model_obj.pENC30 = '<0.05'
        elif (model_obj.OOSstat.loc[model_obj.OOSdate]['ENC-NEW'] 
              > model_obj.OOSstat.loc[model_obj.OOSdate]['90p']):
            model_obj.pENC30 = '<0.10'
        else:
            model_obj.pENC30 = '-'
    except:
        model_obj.OOSR230 = 'OOS date error: Does not exist? (ex: 1979-12-31)'
        model_obj.pENC30 = 'OOS date error: Does not exist? (ex: 1979-12-31)'
        
# =============================================================================
# 
# 
#                                   MAIN
# 
# 
#
# =============================================================================
       
 
if __name__ == '__main__':      
    
    # =============================================================================
    #     Estimation parameters
    # =============================================================================

    use_pickles = False  # or CSV when available
    export_return_csv = False
    
    # All possible values
    Nx = [6, 25, 100]
    horizon = [1, 12] 
    target = ['RM', 'MP']
    std = [False, True]
    logBM = [False, True]
    lagBM = [0, 1] 
    minH = [0.2, 0.8]
    minR = [60]
    cut30 = [False, True]
    
    # default values in functions (not needed to be passed)
    OOSdate = '1979-12-31'
    continuous = True
    CRSPret = True
    
    BMfile = {6: '6_Portfolios_2x3_CSV_0720.zip',
              25: '25_Portfolios_5x5_CSV_0720.zip',
              100: '100_Portfolios_10x10_CSV_0720.zip'}
    
    # Set *class* attribute (to remember which variables are in the tuple/key 
    # for each model)
    Model.label_keys = ['Nx',
                        'horizon',
                        'target',
                        'std',
                        'logBM',
                        'lagBM',
                        'minH',
                        'minR',
                        'cut30']
    
# =============================================================================
#     Initiate dictionary container for all models (from Excel list)
# =============================================================================
    
    # Read tupples with all the keys that I estimate (using the codes above)
    all_keys = list(pd.read_excel(os.path.normpath(
        os.path.join(os.getcwd(), '../Disaggregate', 'Table model v2.xlsx')),
        sheet_name='All_keys')
        .itertuples(index=False, name=None))
    # all_keys

    model_dic = {key: None for key in all_keys}    
    
# =============================================================================
#     Load returns from file
# =============================================================================

    if use_pickles:
        CRSP = pd.read_pickle(os.path.join(os.getcwd(),
                                           "../CRSPindex/Output General",
                                           "mSR2019.pkl"))
        CRSP.rename(columns={'rm': 'RM',
                             'mp': 'MP'},
                    inplace=True)
        
        # *not* in logs
        Retfnlog = CRSP[['RM', 'MP']]
        
        # Detour maybe needed to distribute the code
        if export_return_csv:
            Retfnlog.to_csv(os.path.join(os.getcwd(), 'local', 'Retfnlog.csv'))
        # end of detour #
    else:
        Retfnlog = pd.read_csv(os.path.join(os.getcwd(),
                                            'local', 'Retfnlog.csv'),
                               index_col='date', parse_dates=True)

    # Returns in logs (that I use)
    Retf = Retfnlog.apply(lambda x: np.log1p(x) 
                          if np.issubdtype(x.dtype, np.number) else x)
    
    # ==================================
    # Dictionary to keep returns indexed
    # ==================================
    returns = {key: np.nan for key in itertools.product(target, cut30)}
    for itarget, icut30 in returns.keys():
        if icut30:
            returns[itarget,
                    icut30] = Retf[(Retf.index.year >= 1930) 
                                   & (Retf.index.year <= 2011)][itarget].copy()
        else:
            returns[itarget,
                    icut30] = Retf[itarget].copy()
            
    # print(returns[('RM', False)].head())
    
# =============================================================================
#     Load BMs + place them in dictionaries (identified by Nx, logBM, cut30)
# =============================================================================

    BMs = {key: np.nan for key in itertools.product(Nx,
                                                    logBM,
                                                    cut30)}
    
    for iNx, ilogBM, icut30 in BMs.keys():
        t = os.path.join(os.getcwd(), 'local', BMfile[iNx])
        urlbm = os.path.normpath(t)
        
    # =========================================================================
    #         Here is the procedure to calculate the BMs
    #          
    #         Change here if the FF file is updated (Position data =
    #         line number where start/end *values not header*
    #         number of firms, avg ME, VW Avg BE_FYt-1/ME_June t)
    # =========================================================================
    
        # number of firms (start/end):
        l1 = 2473
        l2 = 3599
    
        # Avg ME
        l3 = 3604
        l4 = 4730
    
        # Avg BM (Value Weight Average of BE_FYt-1/ME_June t Calculated for
        # June of t to June of t+1)
        l5 = 5874
        l6 = 7000
        
    # =========================================================================
    # Do not change below
    
        Nfirms = pd.read_csv(urlbm, skiprows=l1 - 2,
                             header=0,
                             nrows=l2 - l1 + 1)
        
        avgME = pd.read_csv(urlbm, skiprows=l3 - 2,
                            header=0,
                            nrows=l4 - l3 + 1)
        
        avgBM = pd.read_csv(urlbm, skiprows=l5 - 2,
                            header=0,
                            nrows=l6 - l5 + 1)
    
        Nfirms.replace(-99.99, np.nan, inplace=True)
        avgME.replace(-99.99, np.nan, inplace=True)
        avgBM.replace(-99.99, np.nan, inplace=True)
        
        # Save proper date in datetime + lag => end of period convention 
        # (July = end of June)! + drop FF dates
        Nfirms['date'] = pd.to_datetime(Nfirms['Unnamed: 0'] * 100 + 1,
                                        format="%Y%m%d") + MonthEnd(-1)
        Nfirms.set_index('date', inplace=True)
        Nfirms.drop(columns=['Unnamed: 0'], inplace=True)
        
        avgME['date'] = pd.to_datetime(avgME['Unnamed: 0'] * 100 + 1,
                                       format="%Y%m%d") + MonthEnd(-1)
        avgME.set_index('date', inplace=True)
        avgME.drop(columns=['Unnamed: 0'], inplace=True)
        
        avgBM['date'] = pd.to_datetime(avgBM['Unnamed: 0'] * 100 + 1,
                                       format="%Y%m%d") + MonthEnd(-1)
        avgBM.set_index('date', inplace=True)
        avgBM.drop(columns=['Unnamed: 0'], inplace=True)
        
        # All info loaded
        # ===============
        
    # =========================================================================
    #         Calculate BMs
    # =========================================================================
    
        # Total ME of ptf (per month)
        ME = Nfirms.mul(avgME)
        
        # Aux for BE: Only (End of) June (/begin july) is correct!
        # Others are weighted average BM... Mult by ME.
        auxBE = avgBM.mul(ME)
        
        # Filter only correct ones (in June) => df in yearly freq.
        BEy = auxBE[auxBE.index.month == 6]
        
        # Reshape y freq to m freq. Fill all months with june value,
        # until next june.
        # ...create empty df with same index as auxBE inside merge_asof()
        BE = pd.merge_asof(pd.DataFrame(index=auxBE.index),
                           BEy,
                           left_index=True, right_index=True)
        
        ################################################
        # BM in logs or not (notation in KP replication)
        ################################################
        
        # BM not in logs
        # ==============
        BMfnlog = BE.div(ME)
        BMfnlog
        
        # BM in logs
        # ==========
        BMf = BMfnlog.apply(lambda x: np.log(x) 
                            if np.issubdtype(x.dtype, np.number) else x)
        
        # Store BMs in dictionary:
        # =======================
        if ilogBM:
            if icut30:
                BMs[iNx, ilogBM, icut30] = BMf[(BMf.index.year >= 1930) 
                                               & (BMf.index.year <= 2011)
                                               ].copy()
            else:
                BMs[iNx, ilogBM, icut30] = BMf.copy()
        else:
            if icut30:
                BMs[iNx, ilogBM, icut30] = BMfnlog[(BMfnlog.index.year 
                                                    >= 1930)
                                                   & (BMfnlog.index.year 
                                                      <= 2011)
                                                   ].copy()
            else:
                BMs[iNx, ilogBM, icut30] = BMfnlog.copy()
                
# =============================================================================
# End of BM calculation + dictionary     
# =============================================================================

# =============================================================================
#     Construction of the models start properly by filling all standard values
# =============================================================================
    
    for key in model_dic:
        model_dic[key] = standard_values(key=key, BM_dic=BMs, 
                                         returns_dic=returns)
        
    print(Model.label_keys)
    
# =============================================================================
#     Recursive estimation that takes forever even multiprocessed
# =============================================================================    
    
    print('now the (loooong) recursive part starts...')
    start = time.time()  # how long this takes...
    
    c = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for forecast, key in executor.map(partial(recursive, Std2=True,
                                                  skip=60, leave=950),
                                          model_dic.items()):
            c += 1
            print(c)
            # print(f'Since start: {(time.time()-start)}')
            model_dic[key].forecast = forecast
            
    finish = time.time()
    print(f'Time needed: {(finish-start)}')
    
    # Save incomplete pickle
    s1 = os.path.join(os.getcwd(), '../Disaggregate/local', 'CFR72i.mod')
    pickle.dump(model_dic, open(s1, 'wb'))
    
# =============================================================================
#     Complete the simple estimations (faster, in simple loops)
# =============================================================================

    for key in model_dic:
        non_recursive_estimates(model_dic[key], Std2=True)
        
    # Save pickle
    s2 = os.path.join(os.getcwd(), '../Disaggregate/local', 'CFR72.mod')
    pickle.dump(model_dic, open(s2, 'wb'))
