import pandas as pd
import numpy as np
import math
from bins import *
import copy
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

def KS(y, x):
    z = pd.concat([y, x], axis = 1)
    z.columns = ["label", "x"]
    z1 = z.groupby("x"). apply(lambda x:pd.Series({"cnt": x.shape[0], "bad": x["label"]. sum()}))
    z1["good"] = z1["cnt"] - z1["bad"]
    z_bad = z1["bad"]. cumsum() / z1["bad"]. sum()
    z_good = z1["good"].  cumsum() / z1["good"]. sum()
    return - (z_bad - z_good).min()


## TODO 改写为 引用 fit_func

def score_quant(x, quant = 10, single_tick = False):
    ef = bins_count(splt_eqfrq(x = x, quant = 10, single_tick = False).fi_list)
    efl = ef.show_all()["left"]. tolist()
    efl[0] = 0
    efl.append(1)
    return efl


class log_train:
    def __init__(self, x, y, C = 0.1,
                 cond = None,
                 quant = 10, 
                 penalty = "l2", sample_weight = None):
        if cond is not None:
            x = x.loc[cond]
            y = y.loc[cond]
            if sample_weight is not None:
                sample_weight = sample_weight.loc[cond]
            
        self.x = x
        self.y = y
        self.quant = quant
        self.sample_weight = sample_weight
        self.C = C
        self.woe = math.log(y.sum() / (y.shape[0] - y.sum()))
        self.exp = pd.Series(self.woe, index = y.index)
        self.log = pd.Series(y.mean(), index = y.index)
        self.st = score_quant(x = self.log, quant = self.quant, single_tick = False)
        self.cross = self.x.T@self.x
        self.sqr = pd.Series([self.cross.iat[i, i] for i in range(x.shape[1])], index = x.columns)
        self.corr = ((self.cross ** 2) / self.sqr).T / self.sqr
        self.intercept = self.woe
        self.cols = []
        self.coef_dict = dict()
        self.result = []
        self.update_hess()
        self.record_result()

    @property
    def coef(self):
        return pd.Series([self.coef_dict.get(i, 0) for i in self.cols], index = self.cols)
        
    def train(self):
        cols = self.cols
        coef = self.coef.tolist()
        
        lr = LogisticRegression(C = self.C,
                                penalty = "l2",
                                solver="lbfgs",
                                warm_start = True, 
                                max_iter=100,
                                #class_weight = {0: 0.1, 1: 0.9}
        )
        lr.coef_ = np.array([coef])
        lr.intercept_ = np.array([self.intercept])

        x = self.x[cols]
        y = self.y
        lr.fit(x, y, sample_weight = self.sample_weight)
        self.model = lr
        self.coef_dict = {cols[i]:lr.coef_[0][i] for i in range(len(cols))}
        self.intercept = lr.intercept_[0]
        self.log = pd.Series(lr.predict_proba(x)[:, 1], index = y.index)
        self.st = score_quant(x = self.log, quant = self.quant, single_tick = False)
        self.exp = self.log.apply(lambda x:math.log(x / (1 - x)))
        self.update_hess()
        self.record_result()

    @staticmethod
    def hess(x, y, log, cross = None, sqr = None, sample_weight = None):
        if cross is None:
            cross = x.T@x
        if sqr is None:
            sqr = pd.Series([cross.iat[i, i] for i in range(x.shape[1])], index = x.columns)
        log_sum = (1 - y - log).abs().apply(lambda x:math.log(x)).sum()
        w1 = y - log
        w2 = -log * (1 - log)

        if sample_weight is not None:
            w1 = w1 * sample_weight
            w2 = w2 * sample_weight

        ## g1 一阶导 g2 二阶导
        g1 = (w1 * x.T).sum(axis = 1)
        g2 = ((x.T * w2)@x)
        g2_inv = pd.DataFrame(np.linalg.pinv(g2.values), g2.columns, g2.index)
        g_newton = -g1@g2_inv


        ## 将 w1 和 g_newton 当成两个标准方向
        ## ang1: w1 与 xi 的夹角
        ## ang2: g_newton 与 xi 的夹角 (系数正交度量下)
        ang1 = g1 / (((w1 ** 2).sum() * (x.T ** 2).sum(axis = 1))**(1 / 2))
        sqr1 = (g_newton@cross)@g_newton
        ang2 = (g_newton@cross) / ((sqr * sqr1)**(1 / 2))

        ## ang3: g_newton 与 xi 的夹角 (g2 度量下)
        g2_idx_sqr = pd.Series(np.diag(-g2), index = g2.index)
        g2_idx_st = g_newton@ (- g2)
        g2_st_sqr = g2_idx_st@g_newton
        ang3 = g2_idx_st / ((g2_idx_sqr * g2_st_sqr)**(1 / 2))

        return {
            "log_sum": log_sum,
            "w1": w1,
            "w2": w2,
            "g1": g1,
            "g2": g2,
            "g2_inv": g2_inv,
            "g_newton": g_newton,
            "g2_st_sqr": g2_st_sqr, 
            "ang1": ang1,
            "ang2": ang2,
            "ang3": ang3, 
        } 
        
    def update_hess(self):
        res = self.hess(x = self.x, y = self.y, log = self.log, sqr = self.sqr,
                        cross = self.cross, sample_weight = self.sample_weight)
        for i, j in res.items():
            setattr(self, i, j)
        self.ang2_porp = self.ang2 * ((self.x.shape[1] - len(self.cols))**(1 / 2))
        self.ang3_porp = self.ang3 * ((self.x.shape[1] - len(self.cols))**(1 / 2))

    def record_result(self):
        ks = KS(self.y, self.log)
        auc = roc_auc_score(self.y, self.log)
        z = pd.concat([self.y, self.log], axis = 1)
        z.columns = ["label", "x"]
        binning = z.b1(x = "x", y = "label", ticks = self.st, single_tick = False)
        model = self.model if hasattr(self, "model") else None
        self.result.append({"log": self.log,
                            "st": self.st,
                            "coef": self.coef,
                            "intercept": self.intercept,
                            "cols": self.cols.copy(),
                            "log_sum": self.log_sum,
                            "g1": self.g1, 
                            "g2": self.g2, 
                            "ang2": self.ang2,
                            "ang3": self.ang3, 
                            "ang2_porp": self.ang2_porp,
                            "ang3_porp": self.ang3_porp, 
                            "g2_st_sqr": self.g2_st_sqr,
                            "KS": ks,
                            "model": model, 
                            "AUC": auc,
                            "binning": binning, 
        })
        
    def prepare_new(self, i):
        assert i not in self.cols
        self.cols += [i]
        self.coef_dict = {i: self.coef.get(i, 0) for i in self.cols}


    def recursive_train(self, **kwargs):
        while True:
            i, value = self.select_new(**kwargs)
            if i is None:
                break
            self.prepare_new(i)
            self.train()
            self.delete_neg()

    def select_new(self, **kwargs):
        raise

    def delete_neg(self, **kwargs):
        pass
    
    
class lt3(log_train):
    def __init__(self, x, y,
                 train_cond,
                 train_sample_weight = None,
                 quant = 10, 
                 valid_conds = [],
                 valid_sample_weight = None,
                 test_conds = [], 
                 test_sample_weight = None,
                 C = 0.1, penalty = "l2",
    ):

        assert isinstance(valid_conds, list)
        if len(valid_conds) > 0:
            if valid_sample_weight is None:
                valid_sample_weight = [None] * len(valid_conds)
            assert len(valid_sample_weight) == len(valid_conds)

        assert isinstance(test_conds, list)
        if len(test_conds) > 0:
            if test_sample_weight is None:
                test_sample_weight = [None] * len(test_conds)
            assert len(test_sample_weight) == len(test_conds)
        
        super().__init__(x = x.loc[train_cond], y = y.loc[train_cond], C = C, penalty = penalty,
                         sample_weight = train_sample_weight, quant = quant)

        self.valid = list()
        for i in range(len(valid_conds)):
            self.valid.append(log_train(
                x = x.loc[valid_conds[i]], y = y.loc[valid_conds[i]], C = C, penalty = penalty,
                sample_weight = valid_sample_weight[i], quant = quant))


        self.test = list()
        for i in range(len(test_conds)):
            self.test.append(log_train(
                x = x.loc[test_conds[i]], y = y.loc[test_conds[i]], C = C, penalty = penalty,
                sample_weight = test_sample_weight[i], quant = quant))
            
            
    def train(self):
        super().train()
        self.copy_param()

    def copy_param(self):
        for v in self.valid + self.test:
            v.coef_dict = self.coef_dict
            v.intercept = self.intercept
            v.cols = self.cols
            v.st = self.st
            v.model = self.model
            v.exp = v.x[v.cols]@v.coef + v.intercept
            v.log = pd.Series(v.model.predict_proba(v.x[v.cols])[:, 1], index = v.y.index)            
            v.update_hess()
            v.record_result()

    def select_new(self):
        raise

class lt3_ang3(lt3):
    def select_new(self, ang):
        i = self.ang3_porp.idxmax()
        v = self.ang3_porp.loc[i]
        print((i, v))
        if i in self.cols:
            print("idx in cols")
            return None, None

        if v < ang:
            print("ang less than angmin")
            return None, None
        
        return i, v

class lt3_ang2(lt3):
    def select_new(self, ang):
        i = self.ang2_porp.idxmax()
        v = self.ang2_porp.loc[i]
        print((i, v))
        if i in self.cols:
            print("idx in cols")
            return None, None

        if v < ang:
            print("ang less than angmin")
            return None, None
        return i, v
    
class lt3_double_ang3(lt3):
    def select_new(self, ang, valid_ang = 0.5):
        t_cols = set(self.x.columns)
        for j, k in enumerate(self.valid):
            ang3p = k.ang3_porp
            tmp_cols = ang3p[ang3p > valid_ang]. index.tolist()
            t_cols = t_cols & set(tmp_cols)
        t_cols = list(t_cols)

        if len(t_cols) <= 0:
            print("in valid check process, no cols left")

        ang3p = self.ang3_porp
        ang3p = ang3p.loc[t_cols]
        i = ang3p.idxmax()
        v = ang3p.loc[i]
        print(("add index", i, v))
        
        if i in self.cols:
            print("idx in cols")
            return None, None

        if v < ang:
            print("ang3 less than angmin")
            return None, None
            
        return i, v

    def drop_idx(self, i):
        self.cols = [j for j in self.cols if j != i]
        del self.coef_dict[i]
        
    def delete_neg(self):
        while True:
            ang3 = self.ang3_porp.loc[self.cols]
            ang3_neg = ang3[ang3 < 0.0]
            if len(ang3_neg) >= 1:
                i = ang3_neg.idxmin()
                value = ang3_neg.loc[i]
                print(("drop index", i, value))
                self.drop_idx(i)
                self.train()
            else:
                break
    


    ##############################################################################################################
    # def compare_coef(self, cond):                                                                              #
    #     cond1 = cond                                                                                           #
    #     cond2 = ~cond                                                                                          #
    #     log = pd.Series(self.y.mean(), index = self.y.index)                                                   #
    #     res1 = log_train.hess(x = self.x.loc[cond1], y = self.y.loc[cond1], log = log.loc[cond1])              #
    #     res2 = log_train.hess(x = self.x.loc[cond2], y = self.y.loc[cond2], log = log.loc[cond2])              #
    #     p1 = self.x@self.g_newton                                                                              #
    #     p1_1 = self.x@res1["g_newton"]                                                                         #
    #     p1_2 = self.x@res2["g_newton"]                                                                         #
    #     p12 = pd.concat([self.x.loc[cond1]@res1["g_newton"], self.x.loc[cond2]@res2["g_newton"]]).sort_index() #
    #     p2 = self.x[self.cols]@self.coef                                                                       #
    #     pc = pd.concat([p1, p1_1, p1_2, p12, p2], axis = 1)                                                    #
    #     pc.columns = ["n", "n1", "n2", "n12", "f"]                                                             #
    #     return pc.corr()                                                                                       #
    ##############################################################################################################





