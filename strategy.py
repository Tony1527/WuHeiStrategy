# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels import regression
import time
import datetime
import re
import copy
from collections import defaultdict

#初始化账户       
#param:
#return:
def initialize(account):      
    #设置要交易的证券(600519.SH 贵州茅台)   
    log.info('开始执行init')
    get_iwencai('非ST 沪深300')
    ############################
    #设置手续费为交易额的0.025%，最少5元
    set_commission(PerShare(cost=0.00025,min_trade_cost=5.0))
    #设置滑点
    set_slippage(PriceSlippage(0.002))
    
    
    #多因子变量
    account.lags =4                         #多少期的财务报表
    account.T = 1                           #因子筛选周期
    #财务因子列表
    account.finance_factors=[       
        'valuation.pe',                     #市盈率           
        'profit.roa',                       #roa总资产报酬率
        'valuation.market_cap',             #总市值
        ]                                   #因子列表
    #财务因子与技术因子结合列表
    account.factors_extend=[
            'average_one_month_turnover_rate',  #月日均换手率率
        ]
    account.finance_factors=getLocalFactor(account.finance_factors)     #获得数据库中的因子名称
    account.factors_extend +=account.finance_factors                    #财务因子追加上额外技术因子的因子列表
    account.DB_factors = parseFactors(account.finance_factors)          #获得数据库中因子对象        
    #account.weight = pd.DataFrame([0.69,0.01,0.2,0.1],index = account.factors_extend)    
    
    #均线回归变量
    account.pre_days=30                                    #用来判定偏离度的前几日参数
    account.after_days=14                                  #均线回归的回归天数区间
    account.lose_param = 1                                 #输的参数，越小说明越保守
    account.win_param = 1                                  #赢的参数，越小说明当股票有较小波动时，动作越频繁
    account.loss_limit_param = 0.5                           #止损参数，越小说明越保守
    account.profit_limit_param = 0.8                         #止赢参数
    g.stats = {}                                     #股票输赢统计表
    g.history_securities={}       #历史上被选中过的股票
    account.avgreg_range=[i for i in range(-20,-1)]       #均值回归范围
    

    
    
        
    
    #全局重要变量
    g.stock_info = {}                                #股票信息记录表，包括记录（震荡期止盈线/止损线/已持有天数/胜率/方差/震荡方向），以及（全局的仓位线，初始分配金额）
    g.N = 0                                          #所有被多因子选出的股票数量
    g.securities=[]                                  #股票池，是指所有未停牌的沪深300的股票
    #account.max_drawback = 0.1                            #最大回撤参数    
    g.model = 'wave'                                 #模型初始状态为震荡态
    g.first_run = True                               #是否第一次运行
    account.trend_circle = 2                                            #震荡态与趋势态转换周期
    account.trend_times = 0                                             #变动次数，当达到转换周期时，则转换状态
    account.hold_N = 0.4                                                #默认长期持有股票数    
    account.loss_drawback = 0.07
    account.last_portfolio = 0    
    account.direction = True    
    
    #股票集合
    #其中g.tobuy = g.trend_tobuy.union(g.wave_tobuy) 
    g.tobuy=set()                                    #应买股票                   
    g.trend_tobuy=set()                              #趋势期股票
    g.wave_tobuy=set()                               #震荡期股票
    g.long_hold=set()                                #长期持仓股票    
    g.all_set=set()
    g.overbought=set()
    
    
    
    
    #常量
    account.MAX_PRICE = 100000                                          #最大金额，设一个极限
    account.MIN_PRICE = 0                                               #最小金额    
    account.const_positions={-3:0,                                      #金字塔加减仓配额
                             -2:1/4,
                             -1:1/3,
                             0:2/3,
                             1:7/8,
                             2:1
                             }

                             
    #每个月第二个交易日进行一次多因子的筛选
    schedule_function(func=strategy, date_rule=date_rules.month_start(20))
    log.info('执行完init')
    



# 设置买卖条件，每个交易频率（日/分钟/tick）调用一次   
def handle_data(account,data):
    log.info('开始执行handle_data')
    if not g.first_run:               #第一次应先运行多因子策略进行选股
        
        #每天增量添加数据
        for security in g.history_securities:      
            update_stat(account,data,security,g.stats[security])
        g.overbought=set()    
        
        remove=set();
        for security in account.positions:
            if sum(data.attribute_history(security, fields=['is_paused'], bar_count = 1, fre_step = '1d', skip_paused=False, fq=None)['is_paused'])==1:
                remove.add(security)
        for security in remove:
            order_target_value(security, 0)
            if security in g.long_hold:
                g.long_hold.remove(security)
            elif security in g.wave_tobuy:
                g.wave_tobuy.remove(security)
            elif security in g.trend_tobuy:
                g.trend_tobuy.remove(security)
            
        log.info(['handle_data开始',len(g.wave_tobuy),len(g.trend_tobuy),len(g.tobuy),len(g.long_hold),len(account.positions)])                
        
        
        #多只股票的布林线收开头统计                
        count=1                                                     
        for security in g.tobuy:         #注意，此处的迭代对象必须与下面跟count进行比较的一样！
            if(bolling(account,data,security)):
               count+=1
        
        #金字塔调仓
        adjustPyramidPositions(account,data,account.positions)
        
        for security in g.all_set:
            price = data.attribute_history(security, fre_step='1d', fields=['high','low'], skip_paused = False, fq = None, bar_count=9)

            Hn = max(price['high'])
            Ln = min(price['low'])
            last_price = data.current(security)[security].prev_close
            WR = (last_price - Hn)/(Hn-Ln)
            #     account.WR.append(WR)
            if WR>-0.15:
                g.stock_info[security]['WR'] = 1
                g.overbought.add(security)
            elif WR<=-0.15 and WR>=-0.85:
                g.stock_info[security]['WR'] = 0
            else:
                g.stock_info[security]['WR'] = -1

            
        #调整长期持有的股票
        switchLongHold(account,data)    
        
        
        log.info(["overbought",g.overbought])
        #卖出震荡期股票
        wave_tosell = wave_sell(account,data)
        g.wave_tobuy = g.wave_tobuy - wave_tosell
        
        
        #卖出趋势期股票
        g.trend_tobuy = g.trend_tobuy - trend_sell(account,data,g.trend_tobuy)
                   
                                       
        
        
        #day_market_info = data.history(g.securities, 'close', bar_count=1, fre_step='1d', skip_paused = True, fq = None, is_panel = 1)
        #month_market_info = data.history(g.securities, 'close', bar_count=20, fre_step='1d', skip_paused = True, fq = None, is_panel = 1)
        
        day_market_info = data.history(g.all_set, 'close', bar_count=1, fre_step='1d', skip_paused = True, fq = None, is_panel = 1)
        month_market_info = data.history(g.all_set, 'close', bar_count=15, fre_step='1d', skip_paused = True, fq = None, is_panel = 1)
        
        #log.info(['market_info',day_market_info,month_market_info])
        if not day_market_info.empty and not month_market_info.empty:
            day_market_price = day_market_info['close'].mean().mean()
            month_market_price = month_market_info['close'].mean().mean()
            if day_market_price>0.95*month_market_price:
                account.direction = True
            else:
                account.direction = False
        else:
            account.direction = True
        
        #log.info(['portfolio',(1-account.loss_drawback)*account.last_portfolio,account.portfolio_value])
        if (1-account.loss_drawback)*account.last_portfolio >= account.portfolio_value: #如果单日损失超过最大资金回测值，则将所有股票清空
            trend_tosell = []
            wave_tosell = []
            long_hold_tosell = []
            for security in g.trend_tobuy:
                price = data.attribute_history(security, fields=['close'], bar_count=2, fre_step='1d', skip_paused=True, fq=None)['close']
                if (price[1]-price[0])/price[0]<=-0.02:
                    trend_tosell.append(security)

            for security in g.wave_tobuy:
                price = data.attribute_history(security, fields=['close'], bar_count=2, fre_step='1d', skip_paused=True, fq=None)['close']
                if (price[1]-price[0])/price[0]<=-0.02:
                    wave_tosell.append(security)
                    
            for security in g.long_hold:
                price = data.attribute_history(security, fields=['close'], bar_count=2, fre_step='1d', skip_paused=True, fq=None)['close']
                if (price[1]-price[0])/price[0]<=-0.02:
                    long_hold_tosell.append(security)
                    
                    
            g.trend_tobuy = g.trend_tobuy - sell_stocks(account,data,trend_tosell)
            g.wave_tobuy = g.wave_tobuy - wave_sell_stock(account,data,wave_tosell)

            g.tobuy = g.tobuy.union(long_hold_tosell)
            g.long_hold = g.long_hold - sell_stocks(account,data,long_hold_tosell)
            

        elif account.direction:
            #如果当前市场是震荡型，则执行均线回归策略
            if g.model == 'wave':         
                if count >= len(g.tobuy)/2.0:                                 #如果半数以上的股票表现出趋势型，则认为将变动次数加1，否则将变动次数重置
                    account.trend_times+=1
                else:
                    account.trend_times=0                                   
            
                if  account.trend_times>=account.trend_circle:                      #在连续转换周期X天内都表现成趋势型，则认为现在是趋势市场
                    g.model = 'trend'
                    g.wave_tobuy = set([i for i in g.wave_tobuy if i in account.positions])#保留已经在仓位中的震荡期股票
                else:
                    g.wave_tobuy = g.tobuy - g.trend_tobuy        #初始化震荡期股票，待买股票集合与趋势期股票的差集
                    wave_buy = wave_buy_signal(account,data,g.wave_tobuy,g.stats)
                    wave_buy_stock(account,data,wave_tosell,wave_buy)    
                            
                        
            #如果当前市场是趋势型，则执行唐奇安策略    
            if g.model == 'trend':
                if count < len(g.tobuy)/2.0:                        #如果少于半数的股票表现出震荡型，则认为将变动次数减1，否则将变动次数重置为转换周期
                    account.trend_times-=1
                else:
                    account.trend_times=account.trend_circle
                
                if  account.trend_times<=0:                              #在连续转换周期X天内都表现成震荡型，则认为现在是震荡市场
                    g.model = 'wave'
                    g.trend_tobuy = set([i for i in g.trend_tobuy if i in account.positions])
                else:
                    g.trend_tobuy = g.tobuy-g.wave_tobuy        
                    trend_buy_stock(account,data)
                    
                
                #威廉指标出现超卖
                '''  upgrade_stocks = set()
                for security in g.tobuy:
                    if g.stock_info[security]['WR'] == -1:
                        upgrade_stocks.add(security)
                        if security in g.trend_tobuy:
                            g.trend_tobuy = g.trend_tobuy - set([security])
                        else:
                            init_wave_security(account,data,security)
                            g.wave_tobuy = g.wave_tobuy - set([security])
                for security in upgrade_stocks:
                    g.stock_info[security]['position'] = 2
                    log.info('超卖')
                    balance(account,data,security)
                g.tobuy = g.tobuy - upgrade_stocks
                g.long_hold = g.long_hold.union(upgrade_stocks)'''
            
            #威廉指标出现超卖
                for security in g.trend_tobuy:
                    if g.stock_info[security]['WR'] == -1:
                        g.stock_info[security]['position'] = 0
                        #log.info('超卖')
                        balance(account,data,security)
                    
                    
                
                   
                
        #打印当前的市场状态，服务器与浏览器的连接可能会导致出现丢包现象
        log.info(g.model)  
        log.info(['handle_data结束',len(g.wave_tobuy),len(g.trend_tobuy),len(g.tobuy),len(g.long_hold),len(account.positions)])    
        record(portfolio_value=account.capital_used)
        account.last_portfolio = account.portfolio_value

#多因子选股部分

def strategy(account,data):
    log.info('in strategy')
    if account.T == 1:
        #获得股票股票池
        g.securities =get_feasible_stocks(data,account.iwencai_securities,20)## get_index_stocks('000300.SH', get_datetime().strftime('%Y%m%d'))#account.iwencai_securities[80:160]#get_feasible_stocks(account.iwencai_securities,1)[:20]    #设置可行股票池
        
        
        #获得当期因子数据，
        curTime = get_datetime()
        year,lag = getLastQuarter(curTime)
        q = FinanceFactorsQuery(g.securities,account.DB_factors)
        df_now = getFundamentals(q,statDate="%dq%d"%(year,lag))     #获得当前股票信息
        
        count=1
        #如果当期数据取回来是空，则一直往上取，直到取到财务因子数据
        while df_now.empty:
            year,lag = getLastQuarter('%dq%d'%(year,lag))
            q = FinanceFactorsQuery(g.securities,account.DB_factors)
            df_now = getFundamentals(q,statDate="%dq%d"%(year,lag))     #获得当前股票信息
            count+=1
            if count==8:
                break
                        #扩充技术因子
        df_now = df_now.reindex(columns = account.factors_extend)
        for security in g.securities:
            extendTable(security,parseQuarter(year,lag),df_now)      
        
        
        
        #获取所有股票往期总共account.lags的财务与技术因子
        table = getTable(data,g.securities,account.finance_factors,curTime,account.lags)
        #log.info(table.head())
        
        
        if not df_now.empty:
            df_now.fillna(0)
            upplim = df_now.mean()+3*df_now.std()    
            downlim = df_now.mean()-3*df_now.std()    
            normalization = pd.DataFrame(np.zeros((len(df_now.index),len(df_now.columns))),columns = df_now.columns,index=df_now.index)
            for i in range(len(df_now.columns)):
                for j in range(len(df_now.index)):
                    if df_now.ix[j,i]>upplim[i]:
                        normalization.ix[j,i] = upplim[i]
                    elif df_now.ix[j,i]<downlim[i]:
                        normalization.ix[j,i] = downlim[i]
                    else:
                        normalization.ix[j,i] = df_now.ix[j,i]
            normalization = normalization.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
            
            weight = linreg(table[account.factors_extend],table['return'])              #权重矩阵,该平台还不是很稳定，有时候要重新一次运行才能出结果
            betas = weight[1:]                                                          #系数矩阵

            if df_now.shape[1] == 4:
                #points = normalization.dot(account.weight)                            #静态假设前一段时间的因子对下一刻仍然有效
                points = normalization.dot(betas)+weight[0]                                     #动态假设前一段时间的因子对下一刻仍然有效
                
                points = points.sort_values(0,ascending=False)                                #按照预测收益结果排序
                #将原先持仓的股票与多因子选出的股票结合，保留原先持仓的股票
                tobuy = set()
                len_tobuy = min(len(points),50)
                for i in range(len_tobuy):
                    security = points.index[i]
                    tobuy.add(security)
                    
                log.info(['stratagy开始',len(g.wave_tobuy),len(g.trend_tobuy),len(g.tobuy),len(g.long_hold),len(account.positions)])
                #调仓
                wave_tosell = [security for security in g.wave_tobuy if security not in tobuy]
                trend_tosell = [security for security in g.trend_tobuy if security not in tobuy]
                long_hold_tosell = [security for security in g.long_hold if security not in tobuy]
                
                #将不在待买股票池中的股票都卖出
                g.long_hold = g.long_hold - sell_stocks(account,data,long_hold_tosell)
                g.trend_tobuy = g.trend_tobuy  -  sell_stocks(account,data,trend_tosell)
                g.wave_tobuy = g.wave_tobuy - wave_sell_stock(account,data,wave_tosell)
                
                g.long_hold = g.long_hold.intersection(tobuy)
                g.trend_tobuy = g.trend_tobuy.intersection(tobuy)
                g.wave_tobuy = g.wave_tobuy.intersection(tobuy)


  #根据市场行情设定g.long_hold应该持有几只股票，行情好的话持有account.hold_N*len_tobuy，不好的话持有account.hold_N*len_tobuy/2                
                day_market_info = data.history(g.securities, 'close', bar_count=1, fre_step='1d', skip_paused = True, fq = None, is_panel = 1)
                month_market_info = data.history(g.securities, 'close', bar_count=15, fre_step='1d', skip_paused = True, fq = None, is_panel = 1)                

                if not day_market_info.empty and not month_market_info.empty:
                    day_market_price = day_market_info['close'].mean().mean()
                    month_market_price = month_market_info['close'].mean().mean()
                    if day_market_price>0.95*month_market_price:
                        padding_long_hold = account.hold_N*len_tobuy
                    else:
                        padding_long_hold = account.hold_N*len_tobuy/4
                else:
                    padding_long_hold = account.hold_N*len_tobuy/2
                
                for i in range(len_tobuy):
                    if padding_long_hold >len(g.long_hold) and points.index[i] not in g.long_hold:
                        g.long_hold.add(points.index[i])
                        if points.index[i] in g.trend_tobuy:
                            g.trend_tobuy.remove(points.index[i])
                        elif points.index[i] in g.wave_tobuy:
                            init_wave_security(account,data,points.index[i])
                            g.stock_info[points.index[i]]['position'] = 0
                            g.wave_tobuy.remove(points.index[i])
                    elif padding_long_hold <=len(g.long_hold):
                        break
                
                
                g.N = len(tobuy)
                
    
                g.tobuy = tobuy - g.long_hold
                g.all_set = tobuy         
                
                #对于stock_info表的初始化
                for security in tobuy:
                    g.history_securities[security]=1
                
                
                for security in g.history_securities:      
                    if security not in g.stock_info:  
                        init_stats(account,data,security)
                
                #设置平均分配
                for security in tobuy:
                    g.stock_info[security]['allocation'] = account.portfolio_value/g.N
                
                
                #长期持有的股票
                for security in g.long_hold:
                    g.stock_info[security]['position'] = 2
                
            log.info(['stratagy结束',len(g.wave_tobuy),len(g.trend_tobuy),len(g.tobuy),len(g.long_hold),len(account.positions)])                    
                
            
            account.T=0
            g.first_run = False
    account.T+=1


#长期持有股票的与待买股票的轮转
def switchLongHold(account,data):
    #将表现较差的股票从长期持有股票集合中删除
    bad_stocks =  trend_sell(account,data,g.long_hold)    
    g.long_hold = g.long_hold - bad_stocks
    g.tobuy = g.tobuy.union(bad_stocks)
    
    
    
    #升级应买股票为长期持有股票
    upgrade_stocks = set()
    for security in g.tobuy:
        if g.stock_info[security]['position'] == 2:
            upgrade_stocks.add(security)
            if security in g.trend_tobuy:
                g.trend_tobuy = g.trend_tobuy - set([security])
            else:
                init_wave_security(account,data,security)
                g.wave_tobuy = g.wave_tobuy - set([security])
    g.tobuy = g.tobuy - upgrade_stocks
    g.long_hold = g.long_hold.union(upgrade_stocks)

    
    
    #调整仓位
    for security in g.long_hold: 
        balance(account,data,security)




def trend_buy_stock(account,data):
    for security in g.trend_tobuy:
         current_price = data.current(security)[security].open
         high = getHigh(account,data,security)
         if current_price>=high and security not in g.overbought and security not in account.positions:
             log.info("security:{s},tobuy:high = {h} ,current = {o}".format(s = security,h=high,o=current_price))
             balance(account,data,security)



    
    
#获得过去几年的输赢统计表，并初始化股票信息记录表
#security:待初始化的股票
def init_stats(account,data,security):
    g.stats[security] = statistic_data(account,data,security)
    #股票信息记录表，包括记录（震荡期止盈线/止损线/已持有天数/胜率/方差/震荡方向），以及金字塔仓位线，初始分配金额
    g.stock_info[security] = {'avgreg_profit_limit':0,'avgreg_loss_limit':0,'days':account.after_days-1,'ratio':0,'sigma':0,'type':'plane','position':0,'allocation':0,'WR':0}




#布林线
######################
def bolling(account,data,security,trend=3,days=20):     #布林线三天的开收口状态最佳
    if data.current(security)[security].is_paused:      #如果已经停牌，则直接返回False,对统计的结果不产生影响
        return False
    price = data.attribute_history(security, fields=['close'], bar_count=trend+days, fre_step='1d', skip_paused=True, fq=None)['close']
    std = np.array([np.std(price[i:i+days]) for i in range(trend)])
    
    open=0
    for i in range(1,len(std)):
        if std[i]>=std[i-1]:
            open+=1
    
    if open == trend-1:                                 #如果连续三日开口，则是趋势信号
        return True
    else:                                               #如果中间有开有收，则认为是震荡信号
        return False

#获取前interval天的最高价（用于唐奇安通道）
def getHigh(account,data,security,interval=10):
    price = data.attribute_history(security,['close'],interval,'1d',skip_paused = True)['close'].dropna()
    if price.empty:                                     #如果历史数据为空，则将其置为最大值，则后面默认不对其买入
        high = account.MAX_PRICE
    else:
        high = max(price)
    return high
    
'''#获得布林线上轨
def getUpper(account,data,security,interval=20):
    price = data.attribute_history(security,['close'],interval,'1d',skip_paused = True)['close'].dropna()
    if price.empty:                                     #如果历史数据为空，则将其置为最大值，则后面不会对其进行买卖
        up = account.MIN_PRICE
    else:
        mid = np.mean(price)
        std = np.std(price)
        up = mid+2*std
    return up'''
    
    

    
#获得唐奇安通道的下线    
def getLow(account,data,security,interval=10):
    price = data.attribute_history(security,['low'],interval,'1d',skip_paused = True)['low'].dropna()
    if price.empty:                                     #如果历史数据为空，则将其置为最大值，则后面默认对其进行卖出
        low = account.MAX_PRICE
    else:
        low = min(price)
    return low


#这两个log可以方便的查看所有股票集合
# log.info(['开始',len(g.wave_tobuy),len(g.trend_tobuy),len(g.tobuy),len(g.long_hold),len(account.positions)])                
# log.info(['结束',len(g.wave_tobuy),len(g.trend_tobuy),len(g.tobuy),len(g.long_hold),len(account.positions)])                




#金字塔仓位调整
def adjustPyramidPositions(account,data,securities):
    trend_remove = set()
    wave_remove = set()
    long_hold_remove = set()
    for security in securities:
        price = data.attribute_history(security, fields=['close'], bar_count=6, fre_step='1d', skip_paused=True, fq=None)['close']
        MA5 = price[:-1].mean()
        if (price[-1]-MA5)/MA5>=0.02:                           #如果昨日价格大于MA5的2%,则认为是利好信号，进行逐步加仓
            if g.stock_info[security]['position']<0:      #如果原来仓位小于2/3,直接加到2/3
                g.stock_info[security]['position'] = 0 
            if g.stock_info[security]['position']!=2:   #不超过最大仓位1
                g.stock_info[security]['position']+=1
               
        elif (price[-1]-MA5)/MA5<=-0.02:                        #如果日涨跌幅小于3%，则不认为是利好信号，进行适当减仓
            #if g.stock_info[security]['position']>0:      #如果原来仓位大于于2/3,直接减到1/3
            #    g.stock_info[security]['position']=-1
            if g.stock_info[security]['position']!=-3:  #不超过最小仓位0
                g.stock_info[security]['position']-=1    
            if g.stock_info[security]['position'] == -3:
                if security in (g.tobuy-g.wave_tobuy):
                    trend_remove.add(security)                    
                elif security in g.long_hold:
                    long_hold_remove.add(security)
                elif security in g.wave_tobuy:
                    if g.stock_info[security]['type']=='up':
                        wave_remove.add(security)
                     
    for security in trend_remove:
        g.trend_tobuy = g.trend_tobuy - sell_stocks(account,data,[security])
        g.stock_info[security]['position']=0
        
        
    for security in long_hold_remove:
        g.long_hold = g.long_hold - sell_stocks(account,data,[security])
        g.stock_info[security]['position']=0
    
    for security in wave_remove:
        g.wave_tobuy = g.wave_tobuy - sell_stocks(account,data,[security])
        g.stock_info[security]['position']=0
        
    g.tobuy = g.tobuy.union(trend_remove)
    g.tobuy = g.tobuy.union(long_hold_remove)
    g.tobuy = g.tobuy.union(wave_remove)    
                 
#初始化震荡期的股票
def init_wave_security(account,data,security):
    g.stock_info[security]['avgreg_profit_limit']=0
    g.stock_info[security]['avgreg_loss_limit']=0
    g.stock_info[security]['days']=account.after_days-1
    g.stock_info[security]['ratio']=0
    g.stock_info[security]['sigma']=0
    g.stock_info[security]['type']='plane'


#根据金字塔仓位调整股票持仓数
#计算公式 目标持仓 = 单只股票最大金额 * 金字塔仓位
def balance(account,data,security):
    value = getMoney(account,data,security)*account.const_positions[g.stock_info[security]['position']]
    #log.info(['value',g.stock_info[security]['position']])
    if abs((value-account.positions[security].position_value))/account.positions[security].position_value>0.1:  #调整仓位时，最小分配规模为10%
        order_target_value(security, value)
        

#获得单只股票的最大金额
#计算公式：单只股票最大金额 = 初始分配金额-持仓成本+持仓市值
def getMoney(account,data,security):
    #log.info([g.stock_info[security]['allocation'],account.positions[security].cost_basis*account.positions[security].total_amount,account.positions[security].position_value])
    return g.stock_info[security]['allocation']-account.positions[security].cost_basis*account.positions[security].total_amount+account.positions[security].position_value

#利用凯利公式对均线回归的股票进行仓位管理
#计算公式:f = p/c-q/b
#其中p为前面得到的最佳范围对应的胜率，q为败率，b为赢钱率，c为赔钱率
def balance_for_avgreg(account,data,security):
    p = g.stock_info[security]['ratio']
    q = 1-p
    b = account.win_param*g.stock_info[security]['sigma']/data.current(security)[security].prev_close
    c = account.lose_param*g.stock_info[security]['sigma']/data.current(security)[security].prev_close
    if g.stock_info[security]['sigma']==0:
        order_targe_value=(security,0)
    else:
        f = p/c-q/b
        f = 1-1.2**(-f)
        log.info(['f:',f])
        if f<0:
            f=0
        elif f>1:
            f=1
        value = getMoney(account,data,security)
        
        if abs((value*f-account.positions[security].position_value))/account.positions[security].position_value>0.1:       #此处的设置是为了防止交易过于频繁
            order_target_value(security, value*f)
            if value*f == 0:
                init_wave_security(account,data,security)




#卖出震荡期股票
def wave_sell(account,data):
    wave_tosell = wave_sell_signal(account,data,g.wave_tobuy)
    g.wave_tobuy = g.wave_tobuy - set(wave_tosell)            #将待卖出股票从当前的震荡期股票删除（只要就是为了处理在状态转换时的余留的股票）
    return wave_sell_stock(account,data,wave_tosell)                      #此处返回待卖出的股票是为了防止止损后继续买入
                                                         

#卖出趋势期股票
def trend_sell(account,data,tosell):
    remove = set()                                                      #待卖股票集合
    for security in tosell:
        current_price = data.current(security)[security].open
        #drawback_price = (1-account.max_drawback)*np.mean(data.attribute_history(security, fields=['close'], bar_count=7, fre_step='1d', skip_paused=True, fq=None)['close'])       #20日最大回撤价格
        low = getLow(account,data,security)                             #唐奇安通道的下通道
        MA20 = data.attribute_history(security,fields=['close'],bar_count=20,skip_paused=True, fq=None,fre_step='1d')['close'].mean()
        
        if (current_price <=low or g.stock_info[security]['WR']==1):    #如果达到止损要求，则将要卖出股票加入remove集合，否则对其调整仓位current_price <= drawback_price or 
            if g.stock_info[security]['WR']==1:
                #log.info('超买')
                if current_price >= MA20 and account.direction ==True:
                    g.overbought.remove(security)
                    #log.info('超买取消')
                    continue;
    
            g.stock_info[security]['position']=0
            g.stock_info[security]['type']='plane'
            log.info("security:{s},tosell:current = {o},low={l}".format(s=security,o=current_price,l=low))
            remove.add(security)
        else:
            #if g.stock_info[security]['position'] == -3:
            #    remove.add(security)
            #else:
            balance(account,data,security)
    return sell_stocks(account,data,remove)
    
        
        
        
    


#判断当前是赢是输
#param:cur_price->当前价格，  dest_list->list,回归天数区间,   upper_c->回归上限常数,    lower_c->回归下限常数   sigma->标准差
#return:'win' or 'lose' or 'even'
def win_or_lose(cur_price,dest_list,upper_c,lower_c,sigma):
    if not iter(dest_list):
        raise TypeError('dest_list is not iterable!')
    upper_bound = cur_price+upper_c*sigma
    lower_bound = cur_price-lower_c*sigma
    for future_price in dest_list:
        if future_price<=lower_bound:
            return 'lose'
        elif future_price>=upper_bound:
            return 'win'
        
        
    return 'even'

#获得统计数据
#param:account->账户，security->待获得统计数据的股票
#return:输赢统计表
def statistic_data(account,data,security):
    price = get_price(security, start_date='20060601', end_date=getDate(get_datetime())[0], fre_step='1d', fields=['close'], skip_paused=True, fq=None)['close']#获得历史数据
    
    i = account.pre_days
    dic={}
    while i+account.after_days <len(price):                 #i用于迭代第pre_days+1天
        pre_price = price[i-account.pre_days:i]             #之前pre_days天的价格数据
        cur_price = price[i]                                #当前i天价格
        MA30 = pre_price.mean()                             #前30日MA
        sigma = pre_price.std()                             #标准差
        
        #此处乘5一方面是为了数据处理更加方便，另一方面此处倍数越大时get_best_range获得范围越小，则数据虽然取到了最优解，但很可能是噪声
        difference_times_sigma = int((float(cur_price-MA30)/float(sigma+1)*10))   
        if difference_times_sigma>=-20 and difference_times_sigma<=-2:               #偏离度窗口，太大时数据容易取极端值
            result = win_or_lose(cur_price,price[i+1:i+account.after_days+1].tolist(),account.win_param,account.lose_param,sigma)
            if difference_times_sigma not in dic:
                dic[difference_times_sigma]  = {'win':0,'lose':0,'even':0}
            dic[difference_times_sigma][result]+=1
        i+=1
    for i in range(-20,-1):
        if i not in dic:
            dic[i]  = {'win':0,'lose':0,'even':0}
    stats = pd.DataFrame(dic).T
    stats = stats.sort_index(ascending=True)                #升序排列
    # log.info(stats)
    return stats


#每日更新统计表
#param:account->账户,security->股票，stats->输赢统计表
#return:None
def update_stat(account,data,security,stats):
    paused = data.current(security)[security].is_paused
    if paused==0:
        price = data.attribute_history(security,fields=['close'],bar_count=account.pre_days+1+account.after_days,skip_paused=True, fq=None,fre_step='1d')['close']
            
        if len(price)==account.pre_days+1+account.after_days:       #排除取出数据为空的情况
            range_price = price[0:account.pre_days]
            cur_price = price[account.pre_days]
            MA30 = range_price.mean()
            sigma = range_price.std()
            difference_times_sigma = int((float(cur_price-MA30)/float(sigma+1)*10))
            if difference_times_sigma>=-20 and difference_times_sigma<=-2:
                result = win_or_lose(cur_price,price[account.pre_days+1:].tolist(),account.win_param,account.lose_param,sigma)
                stats.ix[difference_times_sigma,result]+=1



#均线回归买信号
#param:securities->待买股票,stats->输赢统计表
#return:
def wave_buy_signal(account,data,securities,stats):
    to_buy = []
    for security in securities:      
        paused = data.current(security)[security].is_paused
        if paused ==0:              #剔除停牌股票
            #返回均线回归最佳范围
            best_range = get_best_range(stats[security],0,int(len(stats[security])/2),len(stats[security])-1)           
            stock_best_range = {}
            stock_best_range['low'] = account.avgreg_range[best_range[0]]
            stock_best_range['high'] = account.avgreg_range[best_range[1]]
            range_win = 0
            range_lose = 0
            for i in range(stock_best_range['low'],stock_best_range['high']):
                range_win+=stats[security].ix[i,'win']
                range_lose+=stats[security].ix[i,'lose']
            
            if range_lose+range_win==0:
                stock_best_range['ratio'] = 0
            else:
                stock_best_range['ratio'] = range_win/(range_win+range_lose)
                
                
                     
            price = data.attribute_history(security,fields=['close'],bar_count=account.pre_days+1,skip_paused=True, fq=None,fre_step='1d')['close']
            price = price.dropna()
            pre_price_30 = price[:account.pre_days]
            cur_price = price[-1]


            '''#CCI指标计算            
            N=14
            last_price = data.attribute_history(security,fields=['close','high','low'],bar_count=1,skip_paused=True, fq=None,fre_step='1d')
            TP = (last_price['close'][0]+last_price['high'][0]+last_price['low'][0])/3
            pre_price_N = price[-N:]
            MA = pre_price_N.mean()
            MD = np.mean([abs(MA-i) for i in pre_price_N])
            CCI = (TP-MA)/(MD*0.015)'''
            
            MA30 = pre_price_30.mean()
            sigma = pre_price_30.std()
            if not np.isnan(MA30) and not np.isnan(sigma):  #数据不得为空
                
                #计算偏离度/最佳范围/止损线/止盈线，判断偏离度是否在最佳范围内
                difference_times_sigma = int((float(cur_price-MA30)/float(sigma+1)*10))
                #log.info(g.stock_info[security])    
                
                    
                    
                #当前偏移量在最佳范围内，则将其买入    
                if difference_times_sigma>=stock_best_range['low'] and difference_times_sigma<=stock_best_range['high'] and account.positions[security].total_amount==0:
                    if g.stock_info[security]['avgreg_loss_limit'] == 0 and g.stock_info[security]['avgreg_profit_limit'] == 0:
                        g.stock_info[security]['avgreg_loss_limit'] = cur_price-account.loss_limit_param*sigma
                        g.stock_info[security]['avgreg_profit_limit'] = cur_price+account.profit_limit_param*sigma
                        
                        
                        
    
                        
                        if range_lose+range_win!=0:
                            g.stock_info[security]['ratio'] = stock_best_range['ratio']
                            g.stock_info[security]['sigma'] = sigma
                        else:
                            g.stock_info[security]['ratio'] = 0
                            g.stock_info[security]['sigma'] = 0
                    g.stock_info[security]['type'] = 'down'
                    to_buy.append(security)
                    log.info(['WAVE Down tobuy',security])
                elif g.stock_info[security]['WR'] == -1 and security not in g.overbought and account.positions[security].total_amount==0: #CCI > 100
                    g.stock_info[security]['type'] = 'up'
                    to_buy.append(security)
                    log.info(['WAVE Up tobuy',security])
    return to_buy

#卖出信号
#param:account->账户
#return:to_sell->list,待卖出股票
def wave_sell_signal(account,data,securities):
    to_sell=[]
    for security in securities:
        if security in account.positions:
            #dic = data.attribute_history(security,bar_count=1, fre_step='1d', fields=['close','is_paused'], fq=None)
            paused = data.current(security)[security].is_paused
            #cur_price = dic['close'][0]
            cur_price = data.current(security)[security].open
            if paused ==0:
                #如果超出止赢点或者止损点或者到达股票最大持有时间，则卖出
                if g.stock_info[security]['type'] == 'up':
                    balance(account,data,security)
                elif g.stock_info[security]['avgreg_profit_limit']<=cur_price or g.stock_info[security]['avgreg_loss_limit']>=cur_price or g.stock_info[security]['days']<=0:
                    log.info(['wave tosell',security])
                    to_sell.append(security)
                elif g.stock_info[security]['type'] == 'down':
                    g.stock_info[security]['days']-=1
                    balance_for_avgreg(account,data,security)                                            
                    
    return to_sell

#卖出股票
#param:account->账户,data->数据,tosell->待卖出股票，tobuy->待买入股票
#return:None
def wave_sell_stock(account,data,tosell):
    remove = set()
    for security in tosell:
        if security in account.positions:
            order_id = order_target_value(security,0)
            order = get_order(order_id)
            
            if order != None:
                init_wave_security(account,data,security)
                g.stock_info[security]['position']=0
                remove.add(security)
    return remove
        
        
#震荡期买入股票
#param:account->账户,data->数据,tosell->待卖出股票，tobuy->待买入股票
def wave_buy_stock(account,data,tosell,tobuy):
    for security in tobuy:        
        if security not in account.positions and security not in tosell:
            if g.stock_info[security]['type'] == 'down':
                balance_for_avgreg(account,data,security)
                g.stock_info[security]['days']=account.after_days-1
            else:
                balance(account,data,security)
    
        

#获得最佳偏离度区间
#param:df->输赢统计表,p->开始区间位置，q->中间区间位置，r->结束区间位置
#return:最佳区间
def get_best_range(df,p,q,r):
    if p>=r:
        if df.iloc[p]['lose']+df.iloc[p]['win'] == 0:
            return (p,r,0)    
        else:
            return (p,r,float(df.iloc[p]['win'])/float(df.iloc[p]['lose']+1))
    (b1,e1,ratio1) = get_best_range(df,p,int((p+q)/2),q)
    (b2,e2,ratio2) = get_best_range(df,q+1,int((q+1+r)/2),r)
    total_win = 0
    total_lose = 0
    for i in range(b1,e2+1):
        total_win+=df.iloc[i]['win']
        total_lose+=df.iloc[i]['lose']
    ratio3 = float(total_win)/float(total_lose+1)
    if ratio1>=ratio2:
        if(ratio1>=ratio3):
            return (b1,e1,ratio1)
        else:
            return (b1,e2,ratio3)
    else:
        if(ratio2>=ratio3):
            return (b2,e2,ratio2)
        else:
            return (b1,e2,ratio3)
            

###############################################多因子部分################################################

        

#get unsuspended stocks
#param:securities->list,  days:unsigned int,days when securities hardly paused
#return:unsuspended stocks
def get_feasible_stocks(data,securities,days):
    current_dt = get_datetime()

    #当日所有股票的停牌信息
    all_stocks_df = data.history(securities, field='is_paused', bar_count=1, fre_step='1d', skip_paused = False, fq = None, is_panel = 1)['is_paused'].T
    # 当日未停牌股票的索引对象
    unsuspend_idx_live = all_stocks_df[all_stocks_df.ix[:,0]<1].index
    unsuspend_stocks = []
    for security in unsuspend_idx_live:
        if sum(data.attribute_history(security, fields=['is_paused'], bar_count = days, fre_step = '1d', skip_paused=False, fq=None)['is_paused'])==0:
             unsuspend_stocks.append(security)
    return unsuspend_stocks


#解析因子字符串，得到因子列表
#param:factors->list
#return:all_factor_list->form attribute list
def parseFactors(factors):
    d_factor={'valuation':[],'balance':[],'profit':[],'growth':[],'operating':[],'detrepay':[],'cashflow':[],'income':[]}
    
    #切割字符
    for i in factors:
        k,v = (i.split('_',1))
        if k in d_factor:
            # if k == 'valuation':        #every forms formats like formname_stat_attribute except valuation
            #     d_factor[k].append(v)
            # else:
            d_factor[k].append(v.split('_',1)[1])
        else:
            raise ValueError('the factors must be form.attr. (for example: valuation.symbol) ')
    
    #利用反射将输入的因子转化成对应财务信息的属性
    all_factor_list=[]
    if d_factor['valuation']:
        for factor in d_factor['valuation']:
            all_factor_list.append(getattr(valuation,factor))
    if d_factor['balance']:
        for factor in d_factor['balance']:
            all_factor_list.append(getattr(balance,factor))
    if d_factor['profit']:
        for factor in d_factor['profit']:
            all_factor_list.append(getattr(profit,factor))
    if d_factor['growth']:
        for factor in d_factor['growth']:
            all_factor_list.append(getattr(growth,factor))
    if d_factor['operating']:
        for factor in d_factor['operating']:
            all_factor_list.append(getattr(operating,factor))
    if d_factor['detrepay']:
        for factor in d_factor['detrepay']:
            all_factor_list.append(getattr(detrepay,factor))
    if d_factor['cashflow']:
        for factor in d_factor['cashflow']:
            all_factor_list.append(getattr(cashflow,factor))
    if d_factor['income']:
        for factor in d_factor['income']:
            all_factor_list.append(getattr(income,factor))
    return all_factor_list

#因子选择
#param:securities->list,  factor_list->list
#return:Query
def FinanceFactorsQuery(securities,factor_list):
    #解析sql语句
    if not isinstance(securities,list):
        raise TypeError('securities must be list')
    if factor_list:
        q = query(
                        *factor_list,
                        valuation.symbol
                 ).filter(
                        valuation.symbol.in_(securities)
                 )    
    else:
        raise ValueError('no factors are available')
    return q


#use query to fetch data
#param: q->query,  statDate->str,quarter information,  date->datetime,formatstr,date information
#return: pd.DataFrame->rows:securities,  ->columns:form_attribute
def getFundamentals(q,statDate=None,date=None):
    df = get_fundamentals(q,statDate=statDate,date=date)
    #   正则表示式的使用
    for column in df.columns:
        matchObj = re.search("symbol",column,re.I)
        if matchObj :
            df.index=np.array(df[column])
            df = df.drop(column,axis=1)
            break
    return df


def parseQuarter(year,quarter):
    if quarter == 1:
        year-=1
        return getDate(datetime.datetime.strptime('%d-12-2'%year,'%Y-%m-%d'),1)[0]
    elif quarter == 2:
        return getDate(datetime.datetime.strptime('%d-4-2'%year,'%Y-%m-%d'),1)[0]
    elif quarter == 3:
        return getDate(datetime.datetime.strptime('%d-7-2'%year,'%Y-%m-%d'),1)[0]
    elif quarter == 4:
        return getDate(datetime.datetime.strptime('%d-10-2'%year,'%Y-%m-%d'),1)[0]
        

#获得date之前的上一个quarter数据
#param:date-> datetime
#return:(year,quarter_number)
def getLastQuarter(date=datetime.datetime.now()):
    if isinstance(date,datetime.datetime):
        if date.month>=1 and date.month<4:
            return (date.year-1,4)
        elif date.month>=4 and date.month<7:
            return (date.year,1)
        elif date.month>=7 and date.month<10:
            return (date.year,2)
        elif date.month>=10 and date.month<13:
            return (date.year,3)
    elif isinstance(date,str):
        ObjQuarter = re.match('^\d{4}q[1,2,3,4]$',date,re.I)
        if ObjQuarter:
            ObjQuarter = ObjQuarter.group(0)
            year,quarter = ObjQuarter.split('q')
            year = int(year)
            quarter = int(quarter)
            if quarter==1:
                year-=1
                quarter=4
            else:
                quarter-=1
            return (year,quarter)
        else:
            date = datetime.datetime.strptime(date,'%Y%m%d')
            if date.month>=1 and date.month<4:
                return (date.year-1,4)
            elif date.month>=4 and date.month<7:
                return (date.year,1)
            elif date.month>=7 and date.month<10:
                return (date.year,2)
            elif date.month>=10 and date.month<13:
                return (date.year,3)
    else:
        raise TypeError('TypeError: date is not datetime.datetime nor quarter  like')

def getLastNMonth(date,step=1):
    ping_days = [30,30,28,30,30,30,30,30,30,30,30,30,30]
    if step>12 or step<1:
        raise ValueError('step Error')
    if not isinstance(date,datetime.datetime):
        dt = datetime.datetime.strptime(date,'%Y%m%d')
    dt = date
    day = 0
    day = ping_days[dt.month-1]
    #log.info(dt)
    try:
        if dt.month <= step:
            return datetime.datetime(year = dt.year-1,month = dt.month-step+12,day=day)
        else:
            return datetime.datetime(year = dt.year,month = dt.month-step,day=day)
    except:
        log.info(['error',dt,date])


def extendTable(security,date,df,rowName=None):
    dic = get_price(security, end_date = date, fre_step='1d', fields=['turnover_rate'], skip_paused = False, fq = None, bar_count=30,is_panel = 0)

    if rowName==None:
        df.ix[security,'average_one_month_turnover_rate'] = dic['turnover_rate'].mean()

    else:
        df.ix[rowName,'average_one_month_turnover_rate'] = dic['turnover_rate'].mean()



#get factor table
#param:securites->list,  factors->list,  curDate->datetime,  lags:unsigned int
#return: pd.DataFrame->rows:securities lags,  ->columns:form_attribute+'return'
def getTable(data,securities,factors,curDate,lags):
    # cols = factors+['factor_mtm',
    #         'average_one_month_turnover_rate',
    #         'one_month_volatility_rate','factor_cr','factor_vosc','return']
    cols = factors+['average_one_month_turnover_rate','return']
    col_t = factors.copy()
    if 'valuation_stat_stat_date' not in cols:        #add valuation_date to cols, it is used to calculate 'return'
        cols.append('valuation_stat_stat_date')
        col_t.append('valuation_stat_date')
    rows = [security+'_lag'+str(i) for security in securities for i in range(1,lags+1)]
    len_rows = len(securities*lags)
    len_cols = len(cols)
    values = np.zeros((len_rows,len_cols))
    values[:,:]=np.nan
    table = pd.DataFrame(values,index=rows,columns = cols)   #empty table
    
    
    cur_year,cur_quarter = getLastQuarter(curDate)
    year=[0 for i in range(lags)]
    quarter=[0 for i in range(lags)]
    year[0] = cur_year
    quarter[0] = cur_quarter
    for i in range(1,lags):
        if cur_quarter==1:
            cur_year-=1
            cur_quarter=4
        else:
            cur_quarter-=1
        year[i]=cur_year
        quarter[i]=cur_quarter
    #log.info([year,quarter])
        
    finance_factor_list = parseFactors(factors=col_t)
    #len_securities = len(securities)
    for idx,security in enumerate(securities):
        # log.info('['+str(idx)+','+str(len_securities)+']')
        q = FinanceFactorsQuery([security],finance_factor_list)
        D = getDate(curDate,1)[0]
        
        for i in range(1,lags+1):
            # log.info([year,quarter,i])
            df_temp = getFundamentals(q,statDate="%dq%d"%(year[i-1],quarter[i-1]))
            # log.info(df_temp)
            # log.info("**************************")
            # log.info("data from DB:",df_temp)
            
            if not df_temp.empty:     #if df is not empty
                rowName = security+'_lag'+str(i)
                
                table.ix[rowName] = df_temp.ix[security]   #move data
                

                LD = table.ix[rowName,'valuation_stat_stat_date']
                # log.info(table.head())
                
                LD = getDate(LD,1)[0]   #the day before publish date
                extendTable(security,D,table,rowName)
                # log.info(table.head())
               
                p1 = get_price(security, None,LD, '1d', ['close'],bar_count=1)
                p2 = get_price(security, None,D, '1d', ['close'],bar_count=1)
                
                if not p1.empty and not p2.empty:
                        p2=p2['close'][0]
                        p1=p1['close'][0]
                        differentDays = (D-LD).days
                        log_price = np.log(p2/p1)/differentDays              #calculate 'return'
        #                 print('log_price',log_price)
                        table.ix[rowName,'return'] = log_price
                        D = LD               
                       
                
    # log.info(table.head())
    table=table.fillna(0)
    return table

#linear regression
#param:factors->array like,  returns->array like
#return:betas
def linreg(factors,returns):
    X = sm.add_constant(factors)
    
    y = regression.linear_model.OLS(returns,X).fit()
    return y.params





#get date - delta
#param:date->datetime, date format like'20150701',   delta->int
def getDate(date,delta=1):
        if isinstance(date,datetime.datetime):
                return get_trade_days(end_date=date.strftime('%Y%m%d'),count=delta).to_pydatetime()
        else:
                return get_trade_days(end_date=date,count=delta).to_pydatetime()

def sell_stocks(account,data,tosell):
    remove = set()
    for security in tosell:
        if security in account.positions:
            order_id = order_target_value(security,0)
            order = get_order(order_id)

            if order != None:
                init_wave_security(account,data,security)
                g.stock_info[security]['position']=0
                remove.add(security)
    #log.info(['in sell_stocks remove:',remove])
    return remove
            
    

#switch finance factors into DB format
#param:factors->list
#return:DB format
def getLocalFactor(factors):
    for i,factor in enumerate(factors):
        f,v = tuple(factor.split('.',1))
        factors[i] = f+'_stat_'+v
    return factors    
