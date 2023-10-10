# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 08:21:31 2023

@author: JW LEE
"""

import multiprocessing as mp
import datetime as dt
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
        str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return

def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asyn output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out

# tripple barrier & cusum  

#multiprocessing snippet  
def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):     
    import pandas as pd     
    #if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)     
    #else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)     
    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)     
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)     
         
    jobs=[]     
    for i in range(1,len(parts)):     
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}     
        job.update(kargs)     
        jobs.append(job)     
    if numThreads==1:out=processJobs_(jobs)     
    else: out=processJobs(jobs,numThreads=numThreads)     
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()     
    elif isinstance(out[0],pd.Series):df0=pd.Series()     
    else:return out     
    for i in out:df0=df0.append(i)     
    df0=df0.sort_index()     
    return df0     

#single-thread execution for debugging     
def processJobs_(jobs):     
    # Run jobs sequentially, for debugging     
    out=[]     
    for job in jobs:     
        out_=expandCall(job)     
        out.append(out_)     
    return out     

#Linear Partitions   
def linParts(numAtoms,numThreads):     
    # partition of atoms with a single loop     
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)     
    parts=np.ceil(parts).astype(int)     
    return parts     
def nestedParts(numAtoms,numThreads,upperTriang=False):     
    # partition of atoms with an inner loop     
    parts,numThreads_=[0],min(numThreads,numAtoms)     
    for num in range(numThreads_):     
        part=1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)     
        part=(-1+part**.5)/2.     
        parts.append(part)     
    parts=np.round(parts).astype(int)     
    if upperTriang: # the first rows are heaviest     
        parts=np.cumsum(np.diff(parts)[::-1])     
        parts=np.append(np.array([0]),parts)     
    return parts     

#Unwrapping the Callback
def expandCall(kargs):     
    # Expand the arguments of a callback function, kargs['func']     
    func=kargs['func']     
    del kargs['func']     
    out=func(**kargs)     
    return out     
   
#Daily Volatility Estimator  
def getDailyVol(close,days=1,span0=100):     
    # daily vol reindexed to close     
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=days))   
    df0=df0[df0>0]        
    df0=(pd.Series(close.index[df0-1],      
                   index=close.index[close.shape[0]-df0.shape[0]:]))        
    try:     
        df0=close.loc[df0.index]/close.loc[df0.values].values -1 # daily rets     
    except Exception as e:     
        print(f'error: {e}\nplease confirm no duplicate indices')     
    df0=df0.ewm(span=span0).std().rename('dailyVol')     
    return df0     

#Triple-Barrier Labeling Method  
def applyPtSlOnT1(close,events,ptSl,molecule):     
    # apply stop loss/profit taking, if it takes place before t1 (end of event)     
    events_=events.loc[molecule]     
    out=events_[['t1']].copy(deep=True)     
    if ptSl[0]>0: pt=ptSl[0]*events_['trgt']     
    else: pt=pd.Series(index=events.index) # NaNs     
    if ptSl[1]>0: sl=-ptSl[1]*events_['trgt']     
    else: sl=pd.Series(index=events.index) # NaNs     
    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():     
        df0=close[loc:t1] # path prices     
        df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns     
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss     
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking     
    return out     

#Gettting Time of First Touch (getEvents)   
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):     
    #1) get target     
    trgt=trgt.loc[tEvents]     
    trgt=trgt[trgt>minRet] # minRet     
    #2) get t1 (max holding period)     
    if t1 is False:t1=pd.Series(pd.NaT, index=tEvents)     
    #3) form events object, apply stop loss on t1     
    if side is None:side_,ptSl_=pd.Series(1.,index=trgt.index), [ptSl[0],ptSl[0]]     
    else: side_,ptSl_=side.loc[trgt.index],ptSl[:2]  
    
    events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)     
            .dropna(subset=['trgt']))     
    df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),     
                    numThreads=numThreads,close=close,events=events,     
                    ptSl=ptSl_)     
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan     
    if side is None:events=events.drop('side',axis=1)     
    return events     

#Adding Vertical Barrier 
def addVerticalBarrier(tEvents, close, numDays=1):     
    t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))     
    t1=t1[t1<close.shape[0]]     
    t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))     
    return t1     

#Expanding getBins to Incorporate Meta-Labeling  
#events = ma_events.copy()     
def getBins(events, close):     
    '''     
    Compute event's outcome (including side information, if provided).     
    events is a DataFrame where:     
    -events.index is event's starttime     
    -events['t1'] is event's endtime     
    -events['trgt'] is event's target     
    -events['side'] (optional) implies the algo's position side     
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action     
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)     
    '''     
    #1) prices aligned with events     
    events_=events.dropna(subset=['t1'])     
    px=events_.index.union(events_['t1'].values).drop_duplicates()     
    px=close.reindex(px,method='bfill')     
    #2) create out object     
    out=pd.DataFrame(index=events_.index)     
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1     
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling     
    out['bin']=np.sign(out['ret'])     
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling     
    return out     

#Dropping Unnecessary Labels    
def dropLabels(events, minPct=.05):     
    # apply weights, drop labels with insufficient examples     
    while True:     
        df0=events['bin'].value_counts(normalize=True)     
        if df0.min()>minPct or df0.shape[0]<3:break     
        print('dropped label: ', df0.argmin(),df0.min())     
        events=events[events['bin']!=df0.argmin()]     
    return events     

#Symmetric CUSUM Filter    
def getTEvents(gRaw, h):     
    tEvents, sPos, sNeg = [], 0, 0     
    diff = np.log(gRaw).diff().dropna()     
    for i in (diff.index[1:]):     
        try:     
            pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])     
        except Exception as e:     
            print(e)     
            print(sPos+diff.loc[i], type(sPos+diff.loc[i]))     
            print(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))     
            break     
        sPos, sNeg=max(0., pos), min(0., neg)     
        if sNeg<-h:     
            sNeg=0;tEvents.append(i)     
        elif sPos>h:     
            sPos=0;tEvents.append(i)     
    return pd.DatetimeIndex(tEvents)  

def cusum_filter(close, h) :
    '''
    cusum_filter 의한 Down-sampling을 실행한다.
    CUSUM(Cumulative Sum) filter란? 
    정상적인 기준에서 벗어난 정도(Anomaly)를 정적인 기준으로 판단하는 것이 아닌, 
    t기간 동안 누적된 양을 동적으로 고려하여 기준을 변화시키며 판단
    
    Parameters
    ------
    close : pd.Series. *[Date index]
    h : float. 필터 임계값. 일반적으로 vol을 사용함.
    
    Returns
    ------
    tEvents : pd.Index. DatetimeIndex 이며, 필터되고 남은 일자.
    '''
    tEvents = getTEvents(close,h=h)                 
    return tEvents

# 실사용 run module
def labeling_TrippleBarrier(close, fwd_t = 5, ptsl = [1,1], ptsl_target = 'dailyvol',
                            cpus = 1, minRet = 0, 
                            use_cusum_filter = True,
                            useplot = False) :
    '''
    groupby 예시 : df_triple = df.groupby(['Symbol'])['Close'].apply(labeling_TrippleBarrier, fwd_t = 10)
    
    Parameters
    ------
    close : pd.Series. *[Date index]
        (ex)
        Date
        2009-12-24    10800.0
        2009-12-28    10200.0
        2009-12-29    10300.0
        2009-12-30    10250.0
        2010-01-04    10700.0
    fwd_t : int. 트리플배리어에서 horizontal barrier의 길이를 결정. 
    
    ptsl : list. ptsl[0] 상단배리어너비, ptsl[1] 하단배리어너비.
        값이 0이면 해당 배리어는 사용하지 않는다. 예를 들어 [0,1]인 경우, 손절에 대해서만 반응하고 익절은 하지 않는셈.
        (공식 ; ptSl[i]*events_['trgt'] 이므로, DailyVol * ptSl[i]인 셈이다.)
    ptsl_target : str or float. ptsl * ptsl_target을 통해 HorizontalBarrier의 기준을 세운다.
        - float : 해당 값을 이용해 정적인 기준을 세운다. (종목, 일자 상관없이 항시 동일 기준)
        - 'dailyvol' : DailyVol * ptSl[i]
    
    minRet : float. 라벨링 적용위한 최소 임계치. 이 값을 넘어서야만 사용함. ( 0 = filter하지않음)        
    distinguish_by_name : bool.  만일 groupby.apply를 사용하고 있다면, 각각을 구분할 수 있는 Key가 필요하므로 넣은 기능.
        일반적으로 groupby의 Key value가 각각의 pd.Series의 name이 됨. 
        (ex)  df_triple = df.groupby(['Symbol'])['Close'].apply(labeling_TrippleBarrier, fwd_t = 10)
    useplot : bool.
    
    Returns 
    ------
    tb_result : pd.DataFrame. (shape) DateIndex * ['ret', 'bin']
       - ret : 최초로배리어에 도달했을때의 실현된 수익률
       - bin : 결과의 부호에 따른 함수로 된 레이블. (상단에 먼저 도달1, 하단에먼저도달 -1)
       - t1 : 최초로 배리어에 도달했을때의 Date (=predFor)
       - trgt : 도달시의 vol(?) 암튼 ret은아님.
        * DateIndex는 fwd(미래) 기준이 아닌, 현재 시점 기준.
        
    '''
    # close = df_pr['PRICE1']
    
    
    try :
        #-tripple barrier labeling- 
        dailyVol = getDailyVol(close)  
        
        if use_cusum_filter : tEvents = cusum_filter(close,h=dailyVol.mean())
        else : tEvents = close.index # all days(?)    
        
        t1 = addVerticalBarrier(tEvents, close, numDays = fwd_t)   # 최초로 배리어에 도달했을때의 타임스탬프
        
        if ptsl_target == 'dailyvol' :
            target = dailyVol
        if type(ptsl_target) == float :
            target = pd.Series(index = close.index)
            target[:] = ptsl_target
        
        events = getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1)
        # events : Date * ['t1', 'trgt']
        # t1 : 도달시의 일자
        # trge : 도달시의 vol(?) 암튼 ret은아님.
    
        labels = getBins(events, close)  
        # ret&bin은 verticalbarrier에 나타난 일자 간의 gap이다.
        # ret : 최초로배리어에 도달했을때의 실현된 수익률
        # bin : 결과의 부호에 따른 함수로 된 레이블. (상단에 먼저 도달1, 하단에먼저도달 -1)
        
        tb_result = pd.concat([events, labels], axis = 1)
        
        # test plot ; events
        if useplot :
            fig, axlst = plt.subplots(nrows = 2)
            
            ax, ax2 = axlst[0], axlst[1]
            ax.plot(close, color = 'black', alpha = 0.5)
            
            pt_idx = tb_result.index[ tb_result['bin'] == 1 ]
            ax.plot(close.loc[pt_idx], color = 'green', linestyle ='', marker ='^', alpha = 0.8, markersize = 0.8, label = 'Up')
            sl_idx = tb_result.index[ tb_result['bin'] == -1 ]
            ax.plot(close.loc[sl_idx], color = 'red', linestyle ='', marker ='v', alpha = 0.8, markersize = 0.8, label = 'Down')
            ax.legend()
            
            #ax2.plot(events['trgt'], color = 'blue', alpha = .5)
            ax2.plot(events['trgt'] * ptsl[0], color = 'green', alpha = .5, label = 'pt')
            ax2.plot(events['trgt'] * -ptsl[1], color = 'red', alpha = .5, label = 'sl')
            ax2.fill_between(labels.index, labels['ret'], [0], color = 'grey', alpha = .8, label = 'DailyReturn')
            
            #compare_ret = np.log(close.shift(-fwd_t) / close)
            #ax2.fill_between(compare_ret.index, compare_ret, [0], color = 'blue', alpha = .1, label = 'Real_ret')
            ax2.legend()
    
    except Exception as e : 
        print('(!Error raised!)', e)
        tb_result = pd.DataFrame([np.nan])
    
    return tb_result



