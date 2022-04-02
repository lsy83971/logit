from common import *
from logit import *
from RAW.logit import lgt as lgt1

work_dir = "/home/bozb/notebook/chenruxiao/WORK/TMP/"
df = pd.concat([pd.read_csv(work_dir + i) for i in pd.Series(os.listdir(work_dir)).cc("csv").cc("g_offline")])
x = df["x1"]
y = df["y"]

from common import *



log2 = lgt(x = df.drop("y", axis = 1), y = y)
log2.binning(cols = log2.x.columns, pass_error = False, cnt_min = 200)
log2.woe_update()
log2.corr_update()
conds = cond_part(pd.Series(np.random.random(log2.x.shape[0])), 0.7)
log2.sub_binning(conds = conds, labels = ["early", "late"])
log2.sub_binning_plot()
log2.psi
log2.tsi
log2.bins.entL



log2.cluster()
log2.cluster_draw()







conds = cond_part(pd.Series(np.random.random(log2.x.shape[0])), 0.7)
train_cond = conds[0]
valid_conds = conds[1:]

self = log2
cols = log2.x.columns
valid_conds = conds[1:]
test_conds = []
train_class = lt3_ang3
init_param = {"penalty": "l2", "C": 0.5}
train_param = {"ang": 1.5}
quant = 10

import logit
reload(logit)
from logit import lgt
import logit_train
reload(logit_train)
import bins
reload(bins)
from logit_train import *



log2.train(
    cols = log2.x.columns,
    train_cond = train_cond,
    valid_conds = valid_conds,
    train_class = lt3_double_ang3,
    train_param = {"ang": 1, "valid_ang": 0.1}
)

log2.save_model_report()



sgg = pd.read_pickle("test_modelresult.pkl")


from RAW.logit import lgt as lgt1

gg = pd.read_pickle("/home/bozb/lsy/WORK/ONLINE_DOCKER/models/10007YQG_TY_V1.pkl")

('model', 'cols', 'trans', 'standard_woe')

del gg["ticks"]
del gg["trans"]["queryrecord_loanapproval_l12m_l24m_pct"]["result"]
del gg["trans"]["queryrecord_loanapproval_l12m_l24m_pct"]["info"]["bins"]
del gg["trans"]["queryrecord_loanapproval_l12m_l24m_pct"]["info"]["is_special"]
lt     rt    left   right woe

with open("test_modelresult.pkl", "wb") as f:
    pickle.dump(gg, f)


    
sb = lgt1.load_model_result("test_modelresult.pkl")
sb1 = pd.DataFrame(pd.Series(1, index = sb["cols"])).T
gg2 = sb["trans_func"](sb1)


len(log2.lt.cols)
log2.lt.model.coef_. shape



