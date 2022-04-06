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

def score_quant(x, quant = 10, single_tick = False):
    ef = bins_count(splt_eqfrq(x = x, quant = 10, single_tick = False).fi_list)
    efl = ef.show_all()["left"]. tolist()
    efl[0] = 0
    efl.append(1)
    return efl


class hess:
    def __init__(self, x, y, log, sample_weight=None) -> None:
        self.x=x
        self.y=y
        self.sample_weight=sample_weight
        self.log=log
        w1 = y - log
        w2 = -log * (1 - log)
        if sample_weight is not None:
            w1 = w1 * sample_weight
            w2 = w2 * sample_weight
        self.w1=w1
        self.w2=-w2
        self.ori=self.w1/self.w2
        self.limit_ori()

    def inner(self,x1,x2):
        return (x1*(self.w2)*x2).sum()

    def ang(self,x1,x2):
        return self.inner(x1,x2)/(self.inner(x1,x1)*self.inner(x2,x2))**(1/2)

    def inner_st(self,x1,x2):
        return x1@self.g2@x2

    def ang_st(self,x1,x2):
        return self.inner_st(x1,x2)/(self.inner_st(x1,x1)*self.inner_st(x2,x2))**(1/2)

    def limit_ori(self):
        g1 = (self.w1 * self.x.T).sum(axis = 1)
        g2 = ((self.x.T * self.w2)@self.x)
        g2_inv = pd.DataFrame(np.linalg.pinv(g2.values), g2.columns, g2.index)
        self.g1=g1
        self.g2=g2
        self.ori_st = g1 @ g2_inv
        self.ori_st_stack=self.x@self.ori_st
        self.ori_st_ang=self.ang(self.ori,self.ori_st_stack)

        self.sqr=pd.Series(np.diag(self.g2), index=self.g2.index)
        self.ori_sqr=self.inner_st(self.ori_st,self.ori_st)
        self.inner_idx=self.g2@self.ori_st
        self.ang_idx=self.inner_idx/((self.ori_sqr)*self.sqr)**(1/2)


    def split(self,cond):
        self.cond=cond
        sw1,sw2=(None,None) if self.sample_weight is None else (self.sample_weight.loc[cond],self.sample_weight.loc[~cond])
        h1=hess(x=self.x.loc[cond],y=self.y.loc[cond],log=self.log.loc[cond],
                sample_weight=sw1)
        h2=hess(x=self.x.loc[~cond],y=self.y.loc[~cond],log=self.log.loc[~cond],
                sample_weight=sw2)
        self.h1=h1
        self.h2=h2
        self.split_ang()

    def split_ang(self):
        v=pd.Series(0,index=self.y.index)
        v.loc[self.cond]=self.h1.ori_st_stack
        v.loc[~self.cond] = self.h2.ori_st_stack
        self.combine_v=v
        self.combine_ang=self.ang(self.combine_v,self.ori_st_stack)

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
        self.sample_weight = sample_weight
        if sample_weight is None:
            self.woe=self.y.mean()
        else:
            self.woe=(self.y*self.sample_weight).mean()

        self.quant = quant
        self.C = C
        self.penalty=penalty

        self.log = pd.Series(self.woe, index = y.index)
        self.st = score_quant(x = self.log, quant = self.quant, single_tick = False)

        self.intercept = self.woe
        self.cols = []
        self.coef_dict = dict()

        self.result=list()

        self.update_hess()
        #self.record_result()


    def update_hess(self):
        self.hess = hess(x=self.x, y=self.y, log=self.log, sample_weight=self.sample_weight)

    @property
    def coef(self):
        return pd.Series([self.coef_dict.get(i, 0) for i in self.cols], index = self.cols)
        
    def train(self):
        cols = self.cols
        coef = self.coef.tolist()
        lr = LogisticRegression(C = self.C,
                                penalty = self.penalty,
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
        self.update_hess()
        self.record_result()

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
                            "hess":self.hess,
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
        ## return i,value
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
            v.log = pd.Series(v.model.predict_proba(v.x[v.cols])[:, 1], index = v.y.index)            
            v.update_hess()
            v.record_result()

    def select_new(self):
        raise





class lt3_ang(lt3):
    def select_new(self, ang):
        i = self.hess.ang_idx.idxmax()
        v = self.hess.ang_idx.loc[i]
        print((i, v))
        if i in self.cols:
            print("idx in cols")
            return None, None
        if v < ang:
            print("ang less than angmin")
            return None, None
        return i, v


class lt3_ang_scale(lt3):
    def select_new(self, ang):
        i = self.hess.ang_idx.idxmax()
        v = self.hess.ang_idx.loc[i]*(self.x.shape[1]-len(self.cols))**(1/2)
        print((i, v))
        if i in self.cols:
            print("idx in cols")
            return None, None
        if v < ang:
            print("ang less than angmin")
            return None, None
        return i, v


class lt3_ang_scale_cv(lt3):
    def select_new(self, ang_min, ang_min_valid=0.5):
        t_cols = set(self.x.columns)
        for j, k in enumerate(self.valid):
            ang = k.hess.ang_idx
            ang_s = ang * (self.x.shape[1] - len(self.cols)) ** (1 / 2)
            tmp_cols = ang_s[ang_s > ang_min_valid].index.tolist()
            t_cols = t_cols & set(tmp_cols)
        t_cols = list(t_cols)

        if len(t_cols) <= 0:
            print("in valid check process, no cols left")
            return None,None

        ang = k.hess.ang_idx
        ang_s = ang * (self.x.shape[1] - len(self.cols)) ** (1 / 2)
        ang_s = ang_s.loc[t_cols]
        i = ang_s.idxmax()
        v = ang_s.loc[i]
        print(("add index", i, v))

        if i in self.cols:
            print("idx in cols")
            return None, None

        if v < ang_min:
            print("ang less than angmin")
            return None, None

        return i, v

    def drop_idx(self, i):
        self.cols = [j for j in self.cols if j != i]
        del self.coef_dict[i]

    def delete_neg(self):
        while True:
            ang = self.hess.ang_idx.loc[self.cols]
            ang_neg = ang[ang < 0.0]
            if len(ang_neg) >= 1:
                i = ang_neg.idxmin()
                value = ang_neg.loc[i]
                print(("drop index", i, value))
                self.drop_idx(i)
                self.train()
            else:
                break


conds=cond_part(pd.to_datetime(lg2.x["dt"]),[0.6,0.8])
gg=lt3_ang_scale_cv(x=lg2.woevalue,y=lg2.y,train_cond=conds[0],valid_conds=conds[1:2],test_conds=conds[2:])
gg.recursive_train(ang_min=1.5,ang_min_valid=2)
pd.DataFrame([i["coef"] for i in gg.result])




