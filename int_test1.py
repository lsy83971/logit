# pip install pandas,numpy,matplotlib,sklearn,scipy,xlsxwriter,xlrd.seaborn
from importlib import reload
import logit
import bins
from logit import lgt,cond_part
from logit_train import *
import pandas as pd


df1 = pd.read_csv("sample_train.csv", encoding='GBK',index_col=0)
df2 = pd.read_csv("sample_test.csv", encoding='GBK',index_col=0)
df3 = pd.read_csv("sample_oot.csv", encoding='GBK',index_col=0)
df=pd.concat([df1,df2,df3])



y=df["label"]
x=df.drop("label",axis=1)
lg2=lgt(x=x,y=y)


lg2.binning(cnt_min=300,pass_error=True)
lg2.binning_cnt()
# -*- coding: utf-8 -*-
lg2.woe_update()
lg2.corr_update()
conds = cond_part(pd.to_datetime(lg2.x["dt"]), 0.5)
lg2.sub_binning(conds = conds, labels = ["early", "late"])
lg2.sub_binning_plot()
lg2.draw_binning_excel()
lg2.bins.entL.sort_values(ascending=False)

lg2.cluster(min_value=0.6)
lg2.cluster_draw()

gp_cols=["feature_14","feature_15","feature_27","feature_11","feature_8"]
single_cols_drop=["feature_9"]
single_cols_left=[i for i in lg2.single_cols if i not in single_cols_drop]

cols2=single_cols_left+gp_cols
conds = cond_part(pd.to_datetime(lg2.x["dt"]), [0.6,0.8])
train_cond=conds[0]
valid_conds=conds[1:2]
test_conds=conds[2:3]
lg2.train(cols2,
          train_cond=train_cond,
          test_conds=test_conds,
          valid_conds=valid_conds,
          train_class=lt3_ang_scale_cv,
          train_param={"ang_min": 1.2,"ang_min_valid":0.5})

lg2.lt.hess
log=pd.Series(lg2.y.mean(),index=lg2.y.index)
hs=hess(x=lg2.woevalue[cols2],y=lg2.y,log=log)

cols=lg2.x.columns
combine_ang=pd.DataFrame(columns=["idx","less","split"])
for i in cols:
    for j in hs.x[i].value_counts().sort_index().index[1:]:
        cond=(hs.x[i]<j)
        hs.split(cond)
        v=hs.combine_ang
        combine_ang.loc[combine_ang.shape[0]]=[i,j,v]



