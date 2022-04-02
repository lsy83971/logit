from default_var import *
from logit import *
from RAW.logit import lgt as lgt1

work_dir = "/home/bozb/notebook/chenruxiao/WORK/TMP/"
df = pd.concat([pd.read_csv(work_dir + i) for i in pd.Series(os.listdir(work_dir)).cc("csv").cc("g_offline")]).sample(10000)
y = df["y"]

log2 = lgt(x = df.drop("y", axis = 1), y = y)
log2.binning(cols = log2.x.columns, pass_error = False, cnt_min = 200)
log2.woe_update()
log2.corr_update()
conds = cond_part(pd.Series(np.random.random(log2.x.shape[0])), 0.7)
log2.sub_binning(conds = conds, labels = ["early", "late"])
log2.sub_binning_plot()
log2.draw_binning_excel()
log2.psi
log2.tsi
log2.bins.entL

conds = cond_part(pd.Series(np.random.random(log2.x.shape[0])), [0.6, 0.8])
train_cond = conds[0]
valid_conds = conds[1:2]
test_conds = conds[2:3]

log2.train(
    cols = list(log2.bins.keys()),
    train_cond = train_cond,
    valid_conds = valid_conds,
    test_conds = test_conds, 
    train_class = lt3_double_ang3,
    train_param = {"ang": 0.5, "valid_ang": 0.1}
)

log2.online(cn_name = "测试V1", en_name = "TEST_V1")









