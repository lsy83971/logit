# pip install pandas,numpy,matplotlib,sklearn,scipy,xlsxwriter,xlrd
from importlib import reload
from types import MethodType
import logit
import bins
reload(logit)
reload(bins)
from logit import lgt
import pandas as pd

##for i in ["cluster","cluster_draw"]:
##    setattr(lg2,i,MethodType(getattr(lgt,i),lg2))

df1 = pd.read_csv("sample_train.csv", encoding='GBK',index_col=0)
df2 = pd.read_csv("sample_test.csv", encoding='GBK',index_col=0)
df3 = pd.read_csv("sample_oot.csv", encoding='GBK',index_col=0)

df=pd.concat([df1,df2,df3])
y=df["label"]
x=df.drop("label",axis=1)
lg2=lgt(x=x,y=y)
lg2.binning(cnt_min=300,pass_error=True)
len(lg2.bins)
lg2.binning_cnt()
lg2.woe_update()
lg2.corr_update()

x1=lg2.woevalue



conds = cond_part(pd.to_datetime(lg2.x["dt"]), 0.5)
lg2.sub_binning(conds = conds, labels = ["early", "late"])
lg2.sub_binning_plot()
lg2.draw_binning_excel()
lg2.bins["feature_24"].show()

lg2.psi.sort_values()
lg2.tsi.sort_values()
lg2.bins.entL.sort_values(ascending=False)


cols1 = [i for i in lg2.bins.entL.index if i != "feature_9"]
lg2.cluster(cols1,min_value=0.6)
lg2.cluster_draw()

gp_cols=["feature_14","feature_15","feature_27","feature_11","feature_8"]
single_cols_drop=[]
single_cols_left=[i for i in lg2.single_cols if i not in single_cols_drop]


cols2=single_cols_left+gp_cols
conds = cond_part(pd.to_datetime(lg2.x["dt"]), [0.6,0.8])
train_cond=conds[0]
valid_conds=conds[1:2]
test_conds=conds[2:3]
lg2.train(cols2,train_cond=train_cond,test_conds=test_conds,valid_conds=valid_conds,train_param={"ang": 1.2})
lg2.iter_info
lg2.coef


