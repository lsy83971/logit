# -*- coding: utf-8 -*-  

# pip install pandas,numpy,matplotlib,sklearn,scipy,xlsxwriter,xlrd.seaborn
from importlib import reload
import logit
import bins
from logit import lgt,cond_part
from logit_train import *
from default_var import excel_sheets
import pandas as pd

import default_var
reload(default_var)

df = pd.read_csv("raw_data (1).csv")
cmt = pd.read_csv("C:/Users/48944/Downloads/raw_data_cmt.csv")
cmt = cmt.set_index("变量名")["描述"]

y=df["label"]
x=df.drop("label",axis=1)
lg2=lgt(x=x,y=y, cmt=cmt)

lg2.binning(cnt_min=300,pass_error=True)
lg2.binning_cnt()
# -*- coding: utf-8 -*-
lg2.woe_update()
lg2.corr_update()
lg2.x["dt"] = pd.to_datetime(lg2.x["init_repay_date"])
conds = cond_part(pd.to_datetime(lg2.x["dt"]), 0.5)

lg2.sub_binning(conds = conds, labels = ["early", "late"])
lg2.sub_binning_plot()
lg2.draw_binning_excel()

lg2.bins.entL.sort_values(ascending=False)



idx1 = lg2.bins.entL.index[lg2.bins.entL > 0.0112]
idx2 = lg2.psi.index[lg2.psi < 0.1]
idx3 = lg2.tsi.index[lg2.tsi < 0.3]


cols1 = idx1.intersection(idx2).intersection(idx3)
lg2.cluster(cols=cols1, min_value=0.6)
lg2.cluster_draw()

cols2 = ["sum_bal", 
"loan_cn", 
"online_setl_cn_180", 
"online_overdue_pay_term_cn_5", 
"online_overdue_pay_term_cn_90", 
#"fist_overdue_avg_d", 
#"two_overdue_avg_d", 
"max_loan_bal_180", 
"online_norm_pay_rt", 
"online_sum_capit_pay_90", 
"online_overdue_pay_term_cn_31", 
]


#single_cols_drop=["feature_9"]
# single_cols_left=[i for i in lg2.single_cols if i not in single_cols_drop]
#cols2=single_cols_left+gp_cols

conds = cond_part(pd.to_datetime(lg2.x["dt"]), [0.6,0.85])
train_cond=conds[0]
valid_conds=conds[1:2]
test_conds=conds[2:3]
lg2.train(cols2,
          train_cond=train_cond,
          test_conds=test_conds,
          valid_conds=valid_conds,
          train_class=lt3_ang_scale_cv,
          init_param={"penalty": "l2", "C": 0.1, "quant": 10},
          train_param={"ang_min": 0.4,"ang_min_valid":0.3})



#lg2.lt.hess
lg2.iter_info
lg2.iter_binning[ - 1]
lg2.lt.result[ - 1]["binning"]
lg2.lt.valid[0]. result[ - 1]["binning"]
lg2.lt.test[0]. result[ - 1]["binning"]
lg2.save_model_report()


log=pd.Series(lg2.y.mean(),index=lg2.y.index)
hs=hess(x=lg2.woevalue[cols2],y=lg2.y,log=log)
combine_ang=pd.DataFrame(columns=["idx","less","split"])

for i in cols2:
    for j in hs.x[i].value_counts().sort_index().index[1:]:
        cond=(hs.x[i]<j)
        hs.split(cond)
        v=hs.combine_ang
        combine_ang.loc[combine_ang.shape[0]]=[i,j,v]





