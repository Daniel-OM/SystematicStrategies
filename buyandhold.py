
import copy
from enum import Enum
import math
import datetime as dt

import numpy as np
import pandas as pd

import yfinance

from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Frequency(Enum):

    NATURAL: str = 'natural'
    BUSINESSDAY: str = 'businessday'
    DAY: str = 'day'
    WEEK: str = 'week'
    MONTH: str = 'month'
    YEAR: str = 'year'

class Transitions():

    BUSINESS_DAYS_IN_WEEK: int = 5
    BUSINESS_DAYS_IN_YEAR: int = 256
    DAYS_IN_WEEK: int = 7
    DAYS_IN_YEAR: float = 365.25
    WEEKS_IN_YEAR: float = 52.25
    MONTHS_IN_YEAR: int = 12
    HOURS_IN_YEAR: float = DAYS_IN_YEAR*24
    MINUTES_IN_YEAR: float = HOURS_IN_YEAR*60
    SECONDS_IN_YEAR: float = MINUTES_IN_YEAR*60

    def businessDaysToYear(self, days:float) -> float: return days/self.BUSINESS_DAYS_IN_YEAR
    def businessDaysToMonth(self, days:float) -> float: 
        return days/self.BUSINESS_DAYS_IN_YEAR*self.MONTHS_IN_YEAR
    def businessDaysToWeek(self, days:float) -> float: return days/self.BUSINESS_DAYS_IN_WEEK
    def daysToYear(self, days:float) -> float: return days/self.DAYS_IN_YEAR
    def daysToMonth(self, days:float) -> float: return days/self.DAYS_IN_YEAR*self.MONTHS_IN_YEAR
    def daysToWeek(self, days:float) -> float: return days/self.DAYS_IN_WEEK
    def weekToMonth(self, weeks:float) -> float: return weeks/self.WEEKS_IN_YEAR*self.MONTHS_IN_YEAR
    def weekToYear(self, weeks:float) -> float: return weeks/self.WEEKS_IN_YEAR
    def monthToYear(self, months:float) -> float: return months/self.MONTHS_IN_YEAR

    def simpleChange(self, data:float, first_frequency:Frequency, second_frequency:Frequency
               ) -> float:
        
        if first_frequency == Frequency.NATURAL:
            raise ValueError('The original Frequency can not be NATURAL. It must be explicit.')
        elif second_frequency == Frequency.NATURAL:
            return data
        
        if first_frequency == Frequency.DAY:
            if second_frequency == Frequency.DAY:
                return data
            elif second_frequency == Frequency.BUSINESSDAY:
                return data / self.DAYS_IN_WEEK * self.BUSINESS_DAYS_IN_WEEK
            elif second_frequency == Frequency.WEEK:
                return data / self.DAYS_IN_WEEK
            elif second_frequency == Frequency.MONTH:
                return data / self.DAYS_IN_YEAR * self.MONTHS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return data / self.DAYS_IN_YEAR
            
        elif first_frequency == Frequency.BUSINESSDAY:
            if second_frequency == Frequency.DAY:
                return data / self.BUSINESS_DAYS_IN_WEEK * self.DAYS_IN_WEEK 
            elif second_frequency == Frequency.BUSINESSDAY:
                return data
            elif second_frequency == Frequency.WEEK:
                return data / self.BUSINESS_DAYS_IN_WEEK
            elif second_frequency == Frequency.MONTH:
                return data / self.BUSINESS_DAYS_IN_YEAR * self.MONTHS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return data / self.BUSINESS_DAYS_IN_YEAR
            
        elif first_frequency == Frequency.WEEK:
            if second_frequency == Frequency.DAY:
                return data * self.DAYS_IN_WEEK 
            elif second_frequency == Frequency.BUSINESSDAY:
                return data * self.BUSINESS_DAYS_IN_WEEK
            elif second_frequency == Frequency.WEEK:
                return data
            elif second_frequency == Frequency.MONTH:
                return data / self.WEEKS_IN_YEAR * self.MONTHS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return data / self.WEEKS_IN_YEAR
            
        elif first_frequency == Frequency.MONTH:
            if second_frequency == Frequency.DAY:
                return data * self.DAYS_IN_YEAR / self.MONTHS_IN_YEAR 
            elif second_frequency == Frequency.BUSINESSDAY:
                return data * self.BUSINESS_DAYS_IN_YEAR / self.MONTHS_IN_YEAR 
            elif second_frequency == Frequency.WEEK:
                return data * self.WEEKS_IN_YEAR / self.MONTHS_IN_YEAR 
            elif second_frequency == Frequency.MONTH:
                return data
            elif second_frequency == Frequency.YEAR:
                return data / self.MONTHS_IN_YEAR
            
        elif first_frequency == Frequency.YEAR:
            if second_frequency == Frequency.DAY:
                return data * self.DAYS_IN_YEAR
            elif second_frequency == Frequency.BUSINESSDAY:
                return data * self.BUSINESS_DAYS_IN_YEAR
            elif second_frequency == Frequency.WEEK:
                return data * self.WEEKS_IN_YEAR
            elif second_frequency == Frequency.MONTH:
                return data * self.MONTHS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return data
            
        raise ValueError(f'Something went wrong for {data} with original frequency '+
                         f'beeing {first_frequency} and final frequency {second_frequency}')

    def compoundChange(self, data:float, first_frequency:Frequency, second_frequency:Frequency
               ) -> float:
        
        '''
        NOT APPLIED!!
        '''
        
        if first_frequency == Frequency.NATURAL:
            raise ValueError('The original Frequency can not be NATURAL. It must be explicit.')
        elif second_frequency == Frequency.NATURAL:
            return data
        
        if first_frequency == Frequency.DAY:
            if second_frequency == Frequency.DAY:
                return data
            elif second_frequency == Frequency.BUSINESSDAY:
                return data / self.DAYS_IN_WEEK * self.BUSINESS_DAYS_IN_WEEK
            elif second_frequency == Frequency.WEEK:
                return data / self.DAYS_IN_WEEK
            elif second_frequency == Frequency.MONTH:
                return data / self.DAYS_IN_YEAR * self.MONTHS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return data / self.DAYS_IN_YEAR
            
        elif first_frequency == Frequency.BUSINESSDAY:
            if second_frequency == Frequency.DAY:
                return data / self.BUSINESS_DAYS_IN_WEEK * self.DAYS_IN_WEEK 
            elif second_frequency == Frequency.BUSINESSDAY:
                return data
            elif second_frequency == Frequency.WEEK:
                return data / self.BUSINESS_DAYS_IN_WEEK
            elif second_frequency == Frequency.MONTH:
                return data / self.BUSINESS_DAYS_IN_YEAR * self.MONTHS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return data / self.BUSINESS_DAYS_IN_YEAR
            
        elif first_frequency == Frequency.WEEK:
            if second_frequency == Frequency.DAY:
                return data * self.DAYS_IN_WEEK 
            elif second_frequency == Frequency.BUSINESSDAY:
                return data * self.BUSINESS_DAYS_IN_WEEK
            elif second_frequency == Frequency.WEEK:
                return data
            elif second_frequency == Frequency.MONTH:
                return data / self.WEEKS_IN_YEAR * self.MONTHS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return data / self.WEEKS_IN_YEAR
            
        elif first_frequency == Frequency.MONTH:
            if second_frequency == Frequency.DAY:
                return data * self.DAYS_IN_YEAR / self.MONTHS_IN_YEAR 
            elif second_frequency == Frequency.BUSINESSDAY:
                return data * self.BUSINESS_DAYS_IN_YEAR / self.MONTHS_IN_YEAR 
            elif second_frequency == Frequency.WEEK:
                return data * self.WEEKS_IN_YEAR / self.MONTHS_IN_YEAR 
            elif second_frequency == Frequency.MONTH:
                return data
            elif second_frequency == Frequency.YEAR:
                return data / self.MONTHS_IN_YEAR
            
        elif first_frequency == Frequency.YEAR:
            if second_frequency == Frequency.DAY:
                return data * self.DAYS_IN_YEAR
            elif second_frequency == Frequency.BUSINESSDAY:
                return data * self.BUSINESS_DAYS_IN_YEAR
            elif second_frequency == Frequency.WEEK:
                return data * self.WEEKS_IN_YEAR
            elif second_frequency == Frequency.MONTH:
                return data * self.MONTHS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return data
            
        raise ValueError(f'Something went wrong for {data} with original frequency '+
                         f'beeing {first_frequency} and final frequency {second_frequency}')

    def getChange(self, first_frequency:Frequency, second_frequency:Frequency
               ) -> float:
        
        '''
        Obtain ratio to multiply.
        '''
        
        if first_frequency == Frequency.NATURAL:
            raise ValueError('The original Frequency can not be NATURAL. It must be explicit.')
        elif second_frequency == Frequency.NATURAL:
            return 1
        
        if first_frequency == Frequency.DAY:
            if second_frequency == Frequency.DAY:
                return 1
            elif second_frequency == Frequency.BUSINESSDAY:
                return self.BUSINESS_DAYS_IN_WEEK / self.DAYS_IN_WEEK
            elif second_frequency == Frequency.WEEK:
                return 1 / self.DAYS_IN_WEEK
            elif second_frequency == Frequency.MONTH:
                return self.MONTHS_IN_YEAR / self.DAYS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return 1 / self.DAYS_IN_YEAR
            
        elif first_frequency == Frequency.BUSINESSDAY:
            if second_frequency == Frequency.DAY:
                return self.DAYS_IN_WEEK / self.BUSINESS_DAYS_IN_WEEK
            elif second_frequency == Frequency.BUSINESSDAY:
                return 1
            elif second_frequency == Frequency.WEEK:
                return 1 / self.BUSINESS_DAYS_IN_WEEK
            elif second_frequency == Frequency.MONTH:
                return self.MONTHS_IN_YEAR / self.BUSINESS_DAYS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return 1 / self.BUSINESS_DAYS_IN_YEAR
            
        elif first_frequency == Frequency.WEEK:
            if second_frequency == Frequency.DAY:
                return self.DAYS_IN_WEEK 
            elif second_frequency == Frequency.BUSINESSDAY:
                return self.BUSINESS_DAYS_IN_WEEK
            elif second_frequency == Frequency.WEEK:
                return 1
            elif second_frequency == Frequency.MONTH:
                return self.MONTHS_IN_YEAR / self.WEEKS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return 1 / self.WEEKS_IN_YEAR
            
        elif first_frequency == Frequency.MONTH:
            if second_frequency == Frequency.DAY:
                return self.DAYS_IN_YEAR / self.MONTHS_IN_YEAR 
            elif second_frequency == Frequency.BUSINESSDAY:
                return self.BUSINESS_DAYS_IN_YEAR / self.MONTHS_IN_YEAR 
            elif second_frequency == Frequency.WEEK:
                return self.WEEKS_IN_YEAR / self.MONTHS_IN_YEAR 
            elif second_frequency == Frequency.MONTH:
                return 1
            elif second_frequency == Frequency.YEAR:
                return 1 / self.MONTHS_IN_YEAR
            
        elif first_frequency == Frequency.YEAR:
            if second_frequency == Frequency.DAY:
                return self.DAYS_IN_YEAR
            elif second_frequency == Frequency.BUSINESSDAY:
                return self.BUSINESS_DAYS_IN_YEAR
            elif second_frequency == Frequency.WEEK:
                return self.WEEKS_IN_YEAR
            elif second_frequency == Frequency.MONTH:
                return self.MONTHS_IN_YEAR
            elif second_frequency == Frequency.YEAR:
                return 1
            
        raise ValueError(f'Something went wrong with original frequency '+
                         f'beeing {first_frequency} and final frequency {second_frequency}')

class Leverage:
     
    class Type(Enum):
        SIZE = 'size'
        PRICE = 'price'

    capital_required = None

    def __init__(self, type:Type, value:float=1.0) -> None:

        self.type: self.Type = type
        self.value: float = value

    def calculateSize(self, capital:(float or pd.Series), risk:(float or pd.Series), 
                           price:(float or pd.Series), asset_pct_risk:(float or pd.Series)=1, 
                           currency:(float or pd.Series)=1, min_size:float=1
                           ) -> (float or pd.Series):
        
        if self.type == self.Type.SIZE:
            self.position = capital * self.value * risk / (currency * price * asset_pct_risk)
        elif self.type == self.Type.PRICE:
            self.position = capital * risk / (self.value * currency * price * asset_pct_risk)
        else:
            raise ValueError('Not a valid Type of leverage')
        
        if isinstance(self.position, pd.Series):
            # self.position = np.where(self.position * price * currency > capital, 
            #                          capital /(price * currency), self.position)
            self.position = np.where(self.position < min_size, min_size, np.floor(self.position))
        else:
            # if self.position * price * currency > capital:
            #     self.position = capital /(price * currency)
            self.position = max([min_size, math.floor(self.position)])
        
        return self.position
        
    def calculateRequiredCapital(self, risk:(float or pd.Series), 
                                 price:(float or pd.Series), 
                                 asset_pct_risk:(float or pd.Series)=1, 
                                 currency:(float or pd.Series)=1, 
                                 position:(float or pd.Series)=None, 
                                 unleveraged:bool=False
                                 ) -> (float or pd.Series):

        if not isinstance(position, pd.Series) and position == None:
            position = self.position

        if unleveraged:
            if self.type == self.Type.PRICE:
                return position * price * currency * asset_pct_risk * self.value / risk
        else:
            if self.type == self.Type.SIZE:
                return position * price * currency * asset_pct_risk / (self.value * risk)
            
        self.required_capital = position * price
            
        return self.required_capital
    
    def calculateReturn(self, entry:(float or pd.Series), 
                        exit:(float or pd.Series), 
                        rcapital: (float or pd.Series),
                        asset_pct_risk:(float or pd.Series),
                        position:(float or pd.Series)=None,
                        currency:(float or pd.Series)=1,
                        unleveraged:bool=False, risk:float=0.2,
                        pct:bool=True) -> (float or pd.Series):

        if not isinstance(position, pd.Series) and position == None:
            position = self.position

        if isinstance(currency, pd.Series) and isinstance(exit, pd.Series):
            currency = currency.reindex(exit.index, method="ffill")
            
        self.ret: (float or pd.Series) = (exit - entry) * position * currency # entry y exit son unadjusted, según RC tendrían que ser adjusted?
        if self.type == self.Type.PRICE:
            self.ret = self.ret * self.value

        if pct:
            capital_required: (float or pd.Series) = self.calculateRequiredCapital(
                                                         risk=risk,
                                                         price=rcapital, # No debería ser entry? Ahora es unadjusted 
                                                         position=position,
                                                         currency=currency,
                                                         asset_pct_risk=asset_pct_risk, 
                                                         unleveraged=unleveraged)
            self.pct_ret: bool = True
            self.ret: (float or pd.Series) = self.ret / capital_required.shift(1)

        return self.ret

class RiskCalculation:

    class Type(Enum):
        EQUAL: str = 'equal'
        KELLY: str = 'kelly'
        OPTIMALF: str = 'optimalf'

    def __init__(self, returns:pd.Series=None, risk_type:Type=Type.EQUAL, 
                 default_risk:float=0.01, min_risk:float=0.005, max_risk:float=0.1,
                 risk_mult:float=1.0, bounded:bool=False, scale:bool=True) -> None:

        '''
        Generate risk configuration.

        Parameters
        ----------
        risk_type: RiskType
            Type of calculation to use. It can be:
            - equal: to use the same for every trade, the default will be used
            - kelly: to calculate using the kelly criterion
            - optimalf: to calculate using the optimal f iterative method
        default_risk: float
            Risk in per unit to use by default.
        risk_mult: float
            Multiplier for the risk.
        min_risk: float
            Minimum risk to use.
        max_risk: float
            Maximum risk to use.
        bounded: bool
            True to bound returns to the max loss.
        scale: bool
            True to bound risk to the max loss.
        '''

        self.returns: pd.Series = returns
        self.risk_type: self.RiskType = risk_type
        self.default_risk: float = default_risk
        self.risk_mult: float = risk_mult
        self.min_risk: float = min_risk
        self.max_risk: float = max_risk
        self.bounded: bool = bounded
        self.scale: bool = scale
    
    def _wagerScale(self, risk:float, max_loss:float) -> float:
        return risk/max_loss
    
    def _boundReturns(self, returns:(np.ndarray or pd.Series)) -> (np.ndarray or pd.Series):

        '''
        returns: np.ndarray or pd.Series
            Series with per unit return for the trades to bound.
        '''
        return returns / np.abs(np.min(returns))

    def _kellyRisk(self, returns:pd.Series) -> float:
        
        wins: pd.Series = returns[returns > 0]
        losses: pd.Series = returns[returns <= 0]
        W: float = len(wins) / len(returns)
        R: float = np.mean(wins) / np.abs(np.mean(losses))

        risk: float = (W - ( (1 - W) / R )) * self.risk_mult

        return risk

    def _optimalFRisk(self, returns:pd.Series=None, n_curves:int=50, 
                      drawdown_limit:float=20.0, certainty_level:float=10.0
                      ) -> float:

        """
        Calculates ideal fraction to stake on an investment with given return distribution

        Args:
        returns: (array-like) distribution that's representative of future returns
        time_horizon: (integer) the number of returns to sample for each curve
        n_curves: (integer) the number of equity curves to generate on each iteration of f
        drawdown_limit: (real) user-specified value for drawdown which must not be exceeded
        certainty_level: (real) the level of confidence that drawdownlimit will not be exceeded

        Returns:
        'f_curve': calculated drawdown and ghpr value at each value of f
        'optimal_f': the ideal fraction of one's account to stake on an investment
        'max_loss': the maximum loss sustained in the provided returns distribution
        """

        bounded_f: pd.Series = np.cumsum(np.array([0.5]*200))
        f_curve: pd.DataFrame = pd.DataFrame(columns=['ghpr', 'drawdown'])
        for f in bounded_f:

            # Generate n_curves number of random equity curves
            reordered_returns = np.random.choice(f * returns, size= 
                (len(returns), n_curves))
            curves = (1 + reordered_returns).cumprod(axis=0)
            curves_df: pd.DataFrame = pd.DataFrame(curves)

            # Calculate Maximum Drawdown for each equity curve
            drawdown = curves_df / curves_df.cummax() - 1
            abs_drawdown = np.abs(drawdown)
            curves_drawdown = np.max(abs_drawdown) * 100
            
            # Calculate GHPR for each equity curve
            eq_arr: np.ndarray = np.array(curves_df)
            curves_ghpr = eq_arr[-1] / eq_arr[0] ** (1 / len(curves_df)) - 1

            # Calculate drawdown at our certainty level
            drawdown_percentile = np.percentile(curves_drawdown, 
                                                certainty_level)

            # Calculate median ghpr value
            curves_ghpr = np.nan_to_num(curves_ghpr)
            ghpr_median: float = np.median(curves_ghpr)
            f: float = round(f, 1)
            if drawdown_percentile <= drawdown_limit:
                _ghpr: float = ghpr_median
            else:
                _ghpr: float = 0
            f_curve.loc[f, 'ghpr'] = _ghpr
            f_curve.loc[f, 'drawdown'] = drawdown_percentile

        f_curve: pd.DataFrame = f_curve.fillna(0)
        risk: float = f_curve['ghpr'].idxmax() * self.risk_mult

        return risk

    def calculateRisk(self, returns:pd.Series=None, n:int=20, n_curves:int=50, 
                      drawdown_limit:float=20.0, certainty_level:float=10.0) -> float:

        if returns == None:
            returns = self.returns

        returns = returns.fillna(0)

        self.risk = self.default_risk
        max_loss: float = returns.abs().max()

        if self.risk_type != self.RiskType.EQUAL and len(returns) >= n:
            n_returns: pd.Series = returns.tail(n)
            if self.bounded:
                n_returns: pd.Series = self._boundReturns(returns=n_returns)

            if self.risk_type == 'kelly':
                self.risk = self._kellyRisk(returns=n_returns)
            elif self.risk_type == 'optimalf':
                self.risk = self._optimalFRisk(returns=n_returns, n_curves=n_curves, drawdown_limit=drawdown_limit, 
                                               certainty_level=certainty_level)

            max_loss: float = n_returns.abs().max()
        # print(self.risk, max_loss)
        if self.risk != self.risk:
            self.risk = self.min_risk
            
        if self.scale and max_loss == max_loss:
            self.risk = self._wagerScale(risk=self.risk, max_loss=max_loss)

        if self.risk > self.max_risk:
            self.risk = self.max_risk
        elif self.risk < self.min_risk:
            self.risk = self.min_risk

        return self.risk
    
    def to_dict(self) -> dict:

        return {k:v for k, v in self.__dict__.items() if k not in ['risk']}

class Metrics:
    
    avg_ret: float = None
    avg_loss: float = None
    avg_win: float = None
    wr: float = None
    std_dev: float = None
    drawdown: pd.Series = pd.Series()
    transition: Transitions = Transitions()
    
    def __init__(self, returns:pd.Series, from_frequency:Frequency=Frequency.DAY, 
                 to_frequency:Frequency=Frequency.NATURAL, compound:bool=True) -> None:
        
        self.orig_frequency: Frequency = from_frequency
        self.frequency: Frequency = to_frequency
        self.compound: bool = compound
        self.returns: pd.Series = returns.copy()
        self.cumret: pd.Series = self._cumReturns(returns=returns)
        self.transition: float = 1/Transitions().getChange(first_frequency=self.orig_frequency, 
                                    second_frequency=self.frequency)
        # self.annualized_ret: pd.Series = self._changeReturnsFrequency(returns=returns, frequency=Frequency.YEAR)
    
    def _changeReturnsFrequency(self, returns:pd.Series, frequency:Frequency=Frequency.NATURAL
                                ) -> pd.Series:
    
        if frequency == Frequency.NATURAL:
            return returns
        else:
            at_frequency_str_dict: dict = {Frequency.YEAR: "Y", Frequency.WEEK: "7D", 
                                    Frequency.MONTH: "1M", Frequency.DAY: "1D"}
            at_frequency_str: str = at_frequency_str_dict[frequency]
            if self.compound:
                cumret = lambda x: ((1+x).cumprod() - 1).iloc[-1]
                return returns.resample(at_frequency_str).apply(cumret)
            else:
                return returns.resample(at_frequency_str).sum()
        
    def _cumReturns(self, returns:pd.Series=pd.Series()) -> pd.Series:

        if returns.empty:
            returns = self.returns.copy()
            
        if self.compound:
            self.cumret = (1 + returns).cumprod() - 1
        else:
            self.cumret = returns.cumsum()

        return self.cumret
    
    def _drawdown(self) -> pd.Series:
        
        cumret: pd.Series = self.cumret + 1
        self.drawdown = 1 - cumret/cumret.cummax()
        
        return self.drawdown
        
    def _removeZeros(self, inplace:bool=False) -> pd.Series:
        
        temp = self.returns.copy()
        temp[temp == 0] = float('nan')
        if inplace:
            self.returns = temp
            
        return temp
    
    def _demean(self, returns:pd.Series=pd.Series()) -> pd.Series:
        
        if returns.empty:
            returns = self.returns.copy()
            
        return returns - returns.mean()
    
    def _tailRatio(self, quantiles:list=[], quantile_1:float=None, quantile_2:float=None,
              exact_norm:bool=True) -> float:
        
        if quantile_1 != None and quantile_2 != None:
            quantiles = [quantile_1, quantile_2]

        if len(quantiles) >= 2:
            q_extreme: float = [q for q in quantiles if abs(0.5-q) == \
                                max([abs(0.5-q) for q in quantiles])][0]
            q_std: float = [q for q in quantiles if abs(0.5-q) == \
                                min([abs(0.5-q) for q in quantiles])][0]
            
        demean: pd.Series = self._demean(returns=self._removeZeros(inplace=False))
        pr: float = demean.quantile(q_extreme) / demean.quantile(q_std)

        norm_dist_ratio: float = 4.43
        if exact_norm:
            from scipy.stats import norm
            norm_dist_ratio: float = norm.ppf(q_extreme) / norm.ppf(q_std)

        return pr / norm_dist_ratio

    def rateOfReturn(self) -> float:
        
        rate = self.transition * len(self.returns)

        if self.compound:
            self.rate_ret = (self.cumret.iloc[-1] + 1) ** (1/rate) - 1
        else:
            self.rate_ret = self.cumret.iloc[-1] / rate
        
        return self.rate_ret
    
    def averageReturn(self) -> float:
        
        self.avg_ret = self.returns.mean()

        if self.compound:
            self.avg_ret = (1 + self.avg_ret) ** self.transition - 1
        else:
            self.avg_ret = self.avg_ret * self.transition
        
        return self.avg_ret

    def averageWin(self) -> float:
        
        self.avg_win = self.returns[self.returns > 0].mean()
        
        if self.compound:
            self.avg_win = (1 + self.avg_win) ** self.transition - 1
        else:
            self.avg_win = self.avg_win * self.transition
        
        return self.avg_win
    
    def averageLoss(self) -> float:
        
        self.avg_loss = self.returns[self.returns < 0].mean()
        
        if self.compound:
            self.avg_loss = (1 + self.avg_loss) ** self.transition - 1
        else:
            self.avg_loss = self.avg_loss * self.transition
        
        return self.avg_loss
    
    def winrate(self) -> float:
        self.wr = len(self.returns[self.returns > 0])/len(self.returns)
        return self.wr

    def profitRatio(self) -> float:
        
        if self.avg_win == None:
            self.averageWin()
        if self.avg_loss == None:
            self.averageLoss()
            
        self.profit_ratio = self.avg_win/self.avg_loss
        
        return self.profit_ratio
    
    def expectancy(self) -> float:
        
        if self.avg_win == None:
            self.averageWin()
        if self.avg_loss == None:
            self.averageLoss()
        if self.wr == None:
            self.winrate()
            
        self.expec = self.avg_win * self.wr - (1-self.wr) * self.avg_loss
        
        return self.expec
    
    def kelly(self) -> float:
        
        if self.wr == None:
            self.winrate()
        if self.profit_ratio == None:
            self.profitRatio()
        
        self.kelly_size = self.wr - (1 - self.wr)/self.profit_ratio
        
        return self.kelly_size
    
    def skew(self) -> float:
        
        self.dist_skew = self.returns.skew() * 1/self.transition**(1/2)
        
        return self.dist_skew
    
    def standardDeviation(self) -> float:
        
        self.std_dev = self.returns.std() * (self.transition ** (1/2))
        
        return self.std_dev
    
    def sharpeRatio(self) -> float:
        
        if self.std_dev == None:
            self.standardDeviation()
        if self.avg_ret == None:
            self.averageReturn()
            
        self.sharpe_ratio = self.avg_ret/self.std_dev
        
        return self.sharpe_ratio
        
    def lowerTailRatio(self, exact_norm:bool=True) -> float:

        self.lower_tail = self._tailRatio(quantile_1=0.01, quantile_2=0.3, 
                                          exact_norm=exact_norm)
        
        return self.lower_tail
        
    def upperTailRatio(self, exact_norm:bool=True) -> float:

        self.upper_tail = self._tailRatio(quantile_1=0.7, quantile_2=0.99, 
                                          exact_norm=exact_norm)
        
        return self.upper_tail
    
    def averageDrawdown(self) -> float:
        
        if self.drawdown.empty:
            self._drawdown()
            
        self.avg_dd = self.drawdown.mean()
        
        return self.avg_dd
    
    def maxDrawdown(self) -> float:
        
        if self.drawdown.empty:
            self._drawdown()
            
        self.max_dd = self.drawdown.max()
        
        return self.max_dd

    def calculateMetrics(self, indicators:list=['expectancy', 'sharpeRatio', 
                        'maxDrawdown'], return_dict:bool=True) -> float:
        
        for ind in indicators:
            
            method = self.__getattribute__(ind)
            method()

        if return_dict:
            return self.to_dict()
            
    def to_dict(self) -> dict:
        
        return {k: (v._value_ if isinstance(v, Frequency) else v) \
                for k, v in self.__dict__.items() \
            if k not in ['returns', 'drawdown', 'compound', 'frequency', 
                         'cumret', 'transition', 'orig_frequency']}

class Returns:

    class Type(Enum):
        SIMPLE = 'simple'
        COMPOUND = 'compound'

    def __init__(self, adjusted:pd.Series, current:pd.Series, currency:pd.Series=pd.Series(), 
                 calc_type:Type=Type.COMPOUND) -> None:

        # Check if any error
        if adjusted.empty and current.empty:
            raise ValueError('You must pass at least one price type as argument')
        elif adjusted.empty and not current.empty:
            adjusted: pd.Series = current.copy()
        elif not adjusted.empty and current.empty:
            current: pd.Series = adjusted.copy()


        if isinstance(currency, float) or isinstance(currency, int):
            currency: pd.Series = pd.Series(currency, index=self.adjusted.index)
        elif isinstance(currency, pd.Series) and currency.empty:
            currency = pd.Series(1, adjusted.index)

        self.adjusted: pd.Series = adjusted.copy()
        self.current: pd.Series = current.copy()
        self.currency: pd.Series = currency.copy()
        self.calc_type: self.Type = calc_type
        self.volatility: pd.Series = self.calculateVolatilityEstimator()

    def calculateVolatilityEstimator(self, n:int=5, m:int=10, pct:bool=True, 
                                     annualized:bool=True) -> pd.Series:
        
        change: pd.Series = self.calculatePricePercChange(pct=pct)

        std: pd.Series = change.ewm(span=n).std()
        if annualized:
            std = std * (Transitions().getChange(Frequency.DAY, Frequency.YEAR) ** 0.5)
        self.volatility = 0.7*std + 0.3*std.rolling(m, min_periods=1).mean()

        return self.volatility
        
    def calculateSize(self, data:pd.DataFrame, capital:float, risk:float, min_size:int=1, 
                    leverage:Leverage=Leverage(type=Leverage.Type.SIZE, value=1), 
                    compound:Type=Type.COMPOUND) -> pd.DataFrame:

        leverage = copy.deepcopy(leverage)

        # Set default values
        price: pd.Series = self.adjusted.copy()

        # Check for errors
        if len(self.currency) != len(price): 
            raise ValueError(f'Length of currency and price are different: {len(self.currency)} vs. {len(price)}')
        if isinstance(self.volatility, pd.Series):
            if len(self.currency) != len(self.volatility): 
                raise ValueError(f'Length of currency and volat_target are different: {len(self.currency)} vs. {len(self.volatility)}')
            if len(self.volatility) != len(price): 
                raise ValueError(f'Length of volat_target and price are different: {len(self.volatility)} vs. {len(price)}')
        
        # Calculate data
        temp_df: pd.DataFrame = pd.DataFrame(index=data.index)
        temp_df['price'] = price
        temp_df['currency'] = self.currency
        temp_df['volat_target'] = self.volatility * price
        temp_df.dropna(inplace=True)

        if compound == Returns.Type.SIMPLE:
            size: pd.Series = leverage.calculateSize(capital=capital, risk=risk, price=temp_df['price'], 
                                                    currency=temp_df['currency'], 
                                                    asset_pct_risk=temp_df['volat_target'],
                                                    min_size=min_size)
            capital: pd.Series = ((temp_df['price'] - temp_df['price'].shift(periods=1)) * size).cumsum() + capital
        elif compound == Returns.Type.COMPOUND:
            size: list = [0]
            capital: list = [capital]
            prev_day = None
            for i in temp_df.index:
                day = temp_df.loc[i]
                if not isinstance(prev_day, pd.Series):
                    prev_day = day
                    continue
                print(f"{capital[-1]} * {leverage.value} * {risk} / ({day['price']} * {day['currency']} * {day['volat_target']})")
                size.append(leverage.calculateSize(capital=capital[-1], risk=risk, price=day['price'], 
                                                asset_pct_risk=day['volat_target'], currency=day['currency'],
                                                min_size=min_size))
                capital.append(capital[-1] + size[-1] * (day['price'] - prev_day['price']))
                prev_day = day

            capital: pd.Series = pd.Series(capital, index=temp_df.index)
            size: pd.Series = pd.Series(size, index=temp_df.index)
            
        last: int = min(len(size), len(capital))

        data: pd.DataFrame = data.tail(last)
        data['size'] = size[-last:]
        data['capital'] = capital[-last:]
        data['drawdown'] = 1 - data['capital']/data['capital'].cummax()

        return data

    def calculatePricePercChange(self, pct:bool=True) -> pd.Series:

        # Calculate Percentage Change
        percentage_changes: pd.Series = self.adjusted.diff(periods=1)
        if pct:
            percentage_changes = percentage_changes / self.current.shift(periods=1)

        return percentage_changes

    def calculatePercReturns(self, position_size:(float or pd.Series), risk:float=0.2,
                            leverage:Leverage=Leverage(type=Leverage.Type.SIZE, value=1.0), 
                            frequency:Frequency=Frequency.NATURAL) -> pd.Series:
        
        leverage = copy.deepcopy(leverage)

        if isinstance(position_size, float) or isinstance(position_size, int):
            position_size: pd.Series = pd.Series(position_size, index=self.adjusted.index)
        elif position_size.empty:
            position_size: pd.Series = pd.Series(1, index=self.adjusted.index)

        # Calculate returns
        perc_return: pd.Series = leverage.calculateReturn(entry=self.adjusted.shift(periods=1), 
                                            rcapital=self.current, exit=self.adjusted, 
                                            position=position_size.shift(periods=1),
                                            asset_pct_risk=self.volatility, risk=risk,
                                            currency=self.currency, unleveraged=True, pct=True)

        # Change returns frequency
        perc_return_at_freq: pd.Series = self.changeReturnsFrequency(returns=perc_return, frequency=frequency)

        return perc_return_at_freq
    
    def changeReturnsFrequency(self, returns:pd.Series, frequency:Frequency=Frequency.NATURAL) -> pd.Series:
        
        if frequency == Frequency.NATURAL:
            return returns
        else:
            at_frequency_str_dict: dict = {Frequency.YEAR: "Y", Frequency.WEEK: "7D", 
                                    Frequency.MONTH: "1M"}
            at_frequency_str: str = at_frequency_str_dict[frequency]
            if self.calc_type:
                return ((1+returns).cumprod()).resample(at_frequency_str).last()
            else:
                return returns.resample(at_frequency_str).sum()
            
    def cumReturns(self, returns:pd.Series) -> pd.Series:

        if self.calc_type:
            return (1 + returns).cumprod() - 1
        else:
            return returns.cumsum()


def dtIndexToDate(index_series) -> list:

    return [dt.date(*date_tuple).strftime('%Y-%m-%d') \
            for date_tuple in zip(index_series.year, index_series.month, index_series.day)]

def getData(ticker:str, tf:str='1d', interval:str='max', date_str:bool=False
            ) -> pd.DataFrame:

    data: pd.DataFrame = yfinance.Ticker(ticker).history(period=interval, interval=tf)
    if date_str:
        data.index = dtIndexToDate(data.index)

    return data

def getCurrencyForData(curr:str, data:pd.DataFrame, inverse:bool=False) -> pd.DataFrame:

    currency: pd.DataFrame = getData(curr)
    currency: pd.DataFrame = currency.reindex(data.index, method="ffill")
    currency.dropna(inplace=True)

    if len(currency) < len(data):
        data: pd.DataFrame = data.tail(len(currency))
    elif len(data) < len(currency):
        currency: pd.DataFrame = currency.tail(len(data))

    if inverse:
        for c in ['Open', 'High', 'Low', 'Close']:
            currency[c] = 1/currency[c]

    return data, currency

def calculateSize(data:pd.DataFrame, capital:float, risk:float, volat_target:(float or pd.Series), 
         currency:pd.Series=None, min_size:int=1, leverage:Leverage=Leverage(type=Leverage.Type.SIZE, value=1), 
         price_name:str='Close', compound:Returns.Type=Returns.Type.COMPOUND) -> pd.DataFrame:

    leverage = copy.deepcopy(leverage)

    # Set default values
    price: pd.Series = data[price_name]
    if not isinstance(currency, pd.Series):
        currency: pd.Series = pd.Series([1]*len(price))

    # Check for errors
    if len(currency) != len(price): 
        raise ValueError(f'Length of currency and price are different: {len(currency)} vs. {len(price)}')
    if isinstance(volat_target, pd.Series):
        if len(currency) != len(volat_target): 
            raise ValueError(f'Length of currency and volat_target are different: {len(currency)} vs. {len(volat_target)}')
        if len(volat_target) != len(price): 
            raise ValueError(f'Length of volat_target and price are different: {len(volat_target)} vs. {len(price)}')
    
    # Calculate data
    temp_df: pd.DataFrame = pd.DataFrame(index=data.index)
    temp_df['price'] = price
    temp_df['currency'] = currency
    temp_df['volat_target'] = volat_target * price
    temp_df.dropna(inplace=True)

    if compound != Returns.Type.COMPOUND:
        size: pd.Series = leverage.calculateSize(capital=capital, risk=risk, price=temp_df['price'], 
                                                 currency=temp_df['currency'], 
                                                 asset_pct_risk=temp_df['volat_target'])
        size: pd.Series = np.where(size < min_size, min_size, size)
        capital: pd.Series = ((temp_df['price'] - temp_df['price'].shift(periods=1)) * size).cumsum() + capital
    else:
        size: list = [0]
        capital: list = [capital]
        prev_day = None
        for i in temp_df.index:
            day = temp_df.loc[i]
            if not isinstance(prev_day, pd.Series):
                prev_day = day
                continue
            print(f"{capital[-1]} * {leverage.value} * {risk} / ({day['price']} * {day['currency']} * {day['volat_target']})")
            size.append(leverage.calculateSize(capital=capital[-1], risk=risk, price=day['price'], 
                                               asset_pct_risk=day['volat_target'], currency=day['currency']))
            capital.append(capital[-1] + size[-1] * (day['price'] - prev_day['price']))
            prev_day = day

        capital: pd.Series = pd.Series(capital, index=temp_df.index)
        size: pd.Series = pd.Series(size, index=temp_df.index)
        size: np.ndarray = np.where(size < min_size, min_size, size)
        
    last: int = min(len(size), len(capital))

    data: pd.DataFrame = data.tail(last)
    data['size'] = size[-last:]
    data['capital'] = capital[-last:]
    data['drawdown'] = 1 - data['capital']/data['capital'].cummax()

    return data

def calculateTurnover(position:pd.Series) -> float:

    pct_change: pd.Series = position.diff().abs() / position.shift(1)

    return pct_change.mean()

def volatAproximation(data:pd.DataFrame, exponential:bool=False, pct:bool=True) -> pd.DataFrame:

    data['std'] = data['Close'].rolling(22).std()
    data['std_mas'] = data['std'].rolling(20).mean()
    data['std_mal'] = data['std'].rolling(100).mean()
    if exponential:
        data['vol'] = 0.7*data['Close'].ewm(span=5).std() + 0.3*data['Close'].ewm(span=10).std()
    else:
        data['vol'] = 0.7*data['Close'].rolling(5).std() + 0.3*data['Close'].rolling(10).std()
    if pct:
        data['vol'] = data['vol'] / data['Close']
    
    data.dropna(inplace=True)

    return data

capital = 5000
risk = 0.20
min_size = 1
leverage = Leverage(type=Leverage.Type.SIZE, value=1)
compound = Returns.Type.SIMPLE

data: pd.DataFrame = getData('AAPL')
data, currency = getCurrencyForData('EURUSD=X', data=data, inverse=True)


returnsObj = Returns(adjusted=data['Close'], current=data['Close'], 
                     currency=currency['Close'], calc_type=compound)
returnsObj.calculateVolatilityEstimator(n=32, m=10*Transitions().BUSINESS_DAYS_IN_YEAR)
data = returnsObj.calculateSize(data=data, capital=capital, risk=risk, 
                                leverage=leverage, min_size=min_size, 
                                compound=compound)
returns = returnsObj.calculatePercReturns(position_size=data['size'],# pd.Series(5, index=data.index), 
                                          leverage=leverage, risk=risk, 
                                          frequency=Frequency.NATURAL)

import json

metrics = Metrics(returns=returns, from_frequency=Frequency.DAY, 
                  to_frequency=Frequency.YEAR, compound=False)
stats = metrics.calculateMetrics(indicators=['averageReturn','standardDeviation', 
                                             'sharpeRatio','skew', 'averageDrawdown', 
                                             'maxDrawdown','lowerTailRatio', 
                                             'upperTailRatio'], return_dict=True)
print(json.dumps({**stats, **{'turnouver': calculateTurnover(data['size'])}}, indent=4))

if True:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.0,
                        row_heights=[3,1],
                        specs=[[{'secondary_y': True}],[{'secondary_y': False}]])

    fig.add_trace(go.Scatter(x=data.index, y=data['capital'], name='Capital'), 
                    row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=data.index, y=data['drawdown'] * 10000//1/100, 
                                fill='tozeroy', name='DrawDown'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=data.index, y=data['capital'].cummax(), name='MaxBalance'), 
                    row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=data.index, y=data['size'], name='Size'), row=2, col=1)
    #fig.add_trace(go.Scatter(x=data.index, y=data['vol'], name='Vol'), row=2, col=1)

    fig.update_yaxes(title_text='Return ($)', row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text='DrawDown (%)', row=1, col=1, secondary_y=True)

    fig.update_yaxes(title_text='Size vs.Vol', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    fig.update_layout(title='Size vs. Vol', template='gridon')

    fig.show()