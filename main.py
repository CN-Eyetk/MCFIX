import pandas as pd
import numpy as np
import sys, textstat, argparse, os.path
from  scipy.stats import pearsonr,spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearnex import patch_sklearn
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import numpy as np
from random import shuffle
from catboost import CatBoostRegressor
import lightgbm as lgb
from utils.utils import *
from sklearn.model_selection import KFold
import pandas as pd
import argparse
from transformers import AutoTokenizer
patch_sklearn()
import argparse
parser = argparse.ArgumentParser(description='Regression Analysis')
parser.add_argument('-i','--inputpath', type=str)
parser.add_argument('-l','--lang', type=str)
parser.add_argument('-mn','--modelname', type=str)
parser.add_argument('-m','--model', type=str)
parser.add_argument('-g','--usegpt', type=int)
parser.add_argument('-p','--usepoly', type=int)
parser.add_argument('-r','--resultpath', type=str)
parser.add_argument('-mp','--mappers', type=str)
args = parser.parse_args()

class Regressor:
    def __init__(self,lang,use_gpt,use_interation,data_path,mappers,ylabels,modelname='clue',gptmodel=None):
        self.lang = lang
        self.use_gpt = use_gpt
        self.use_interation = use_interation
        self.data_path = data_path
        self.files = os.listdir(self.data_path)
        if self.lang == "cantonese":
            self.files = [x for x in self.files if "trad" in x and "first" in x]
        else:
            self.files = [x for x in self.files if "simp" in x and "first" in x]
        print(f"Using files:{self.files}")
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.mappers = mappers.split()
        self.ylabels = ylabels.split(" ")
        self.modelname = modelname
        self.gptmodel = gptmodel
        self.feat_bundle_removed = []
        if self.use_gpt:
            self.load_df_with_gpt_vectors(self.gptmodel, self.modelname)
        else:
            self.load_df()
        self.model_performance = {}
    def load_df_with_gpt_vectors(self,model,modelname):
        X_vec_names = []
        scorer = WordSurprisal(model,"cuda")
        dfs = []
        for file in self.files:
            cur_df = load_dep(f"{self.data_path}/{file}")
            _,cur_df = parse_vec(cur_df,scorer,modelname)
            if "NR" in file:
                cur_df["TASK"] = 0
            else:
                cur_df["TASK"] = 1
            dfs.append(cur_df)
        df = pd.concat(dfs, axis=0)
        X_vec_names += [x for x in df.columns if f"Vecs_{modelname}" in x]
        print(f"with gpt df shape{df.shape}")
        self.X_vec_names = X_vec_names
        self.X_names = ["TASK"]
        df["TASK"] = df["TASK"].apply(lambda x:1 if x == "TSR" else 0)
        self.df = df
    def load_df(self):
        dfs = []
        for file in self.files:
            df = load_dep(f"{self.data_path}/{file}")
            if "NR" in file:
                df["TASK"] = 0
            else:
                df["TASK"] = 1
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        print(f"no gpt df shape{df.shape}")
        self.X_names = ["TASK"]
        df["TASK"] = df["TASK"].apply(lambda x:1 if x == "TSR" else 0)
        self.df = df
    def specify_features(self,pos=True,ldr=True, ldh=True, depth = True,adj=True,spr=True,freq=True,syl=True,prvpos=False,prvfreq=True,prvsyl=True,wp=True,punct=True):
        self.feat_bundle_removed = []
        df = self.df.copy()
        X_names = ["TASK"]
        if pos:
            df_pos_dummy = pd.get_dummies(df["upos*"], prefix = "CPOS")
            dummy_names = df_pos_dummy.columns
            df = pd.concat([df,df_pos_dummy], axis = 1)
            X_names += [x for x in df.columns if x in dummy_names]
        else:
            self.feat_bundle_removed.append("POS")
        if ldr:
            X_names += [x for x in df.columns if "LD2ROOT" in x]
        else:
            self.feat_bundle_removed.append("LDR")
        if ldh:
            X_names += [x for x in df.columns if "LD2HEAD" in x]
        else:
            self.feat_bundle_removed.append("LDH")
        if depth:
            X_names += [x for x in df.columns if "DEPTH2ROOT" in x]
        else:
            self.feat_bundle_removed.append("DEPTH")
        if adj:
            X_names += [x for x in df.columns if x == "Neighbors_Num"]
        else:
            self.feat_bundle_removed.append("Neighbor")
        if spr:
            X_names += [x for x in df.columns if "Sp" in x and self.modelname in x]
        else:
            self.feat_bundle_removed.append("Surprisal")
        if freq:
            df["Freq"] = df["Freq"].apply(lambda x:np.log(x+1))
            X_names += [x for x in df.columns if "Freq" in x and not "Prev" in x]
        else:
            self.feat_bundle_removed.append("Freq")
        if syl:
            X_names += [x for x in df.columns if "N_SYL" in x and not "Prev" in x]
        else:
            self.feat_bundle_removed.append("N_Syl")
        if prvfreq:
            df["Prev_Freq"] = df["Prev_Freq"].apply(lambda x:np.log(x+2) if x >= 0 else 0)
            X_names += [x for x in df.columns if "Freq" in x and "Prev" in x]
        else:
            self.feat_bundle_removed.append("PrevFreq")
        if prvsyl:
            df["Prev_N_SYL"] = df["Prev_N_SYL"].apply(lambda x:0 if x == -1 else x)
            X_names += [x for x in df.columns if "N_SYL" in x and "Prev" in x]
        else:
            self.feat_bundle_removed.append("PrevNSyl")
        if wp:
            X_names += ["Word_Position"]
        else:
            self.feat_bundle_removed.append("Word_Position")
        if punct:
            X_names += [x for x in df.columns if "_Punct" in x]
        else:
            self.feat_bundle_removed.append("Punct")
        self.df_to_use = df
        self.X_names = X_names
    def regress(self,result_path):
        mappers = self.mappers
        X_names = self.X_names
        X_Lingui = self.df_to_use[X_names]
        if self.use_interation:
            print("Using Interaction")
            pl = PolynomialFeatures(interaction_only=True)
            X = pl.fit_transform(X_Lingui)
            X_featnames = pl.get_feature_names_out()
            col_indexs = []
            n_feat_deletes = []
            for idx, feat in enumerate(X_featnames):
                if re.compile(r"^CPOS_\w+ CPOS_\w+$").match(feat) or re.compile(r"^PPOS_\w+ PPOS_\w+$").match(feat):
                    n_feat_deletes.append(feat)
                else:
                    col_indexs.append(idx)
            X = X[:,col_indexs]
        else:
            X = X_Lingui.to_numpy()
        if self.use_gpt:
            print("Using GPT Vectors")
            X_vec_names = self.X_vec_names
            X_vec = self.df_to_use[X_vec_names].to_numpy()
            X = np.concatenate((X,X_vec), axis = 1)
        for ylabel in self.ylabels:
            Y = self.df_to_use[[ylabel]]
            #Y = self.min_max_scaler.fit_transform(Y)
            Y = Y.to_numpy()
            Y = Y.squeeze()
            print(f"Shape of X:{X.shape}")
            print(f"Shape of Y:{Y.shape}")
            for mapper in mappers:
                kf = KFold(n_splits=5, shuffle = True, random_state = 42)
                kf.get_n_splits(X)
                r2s = 0
                maes = 0
                mses = 0
                prs = 0
                sprs = 0
                for i, (train_index, test_index) in enumerate(kf.split(X)):
                    print(f"mapper = {mapper}, Fold {i}:")
                    X_train = X[train_index]
                    X_test = X[test_index]
                    Y_train = Y[train_index]
                    Y_test = Y[test_index]
                    if mapper == "ls":
                        model = Lasso(alpha=1.0,max_iter=1000).fit(X_train, Y_train)
                    
                    if mapper == "gbdt":
                        #model = GradientBoostingRegressor(random_state=10).fit(X_train, Y_train)
                        model = CatBoostRegressor(task_type='GPU',random_state=10, verbose = False).fit(X_train, Y_train)
                    
                    if mapper == "plsr":
                        model = PLSRegression(n_components=5, max_iter=1000).fit(X_train, Y_train)

                    if mapper == "mlp":
                        model = MLPRegressor(hidden_layer_sizes=(5,5), activation='identity', early_stopping = True, solver='adam', max_iter=1000).fit(X_train, Y_train)

                    if mapper == "rf":
                        model = RandomForestRegressor(n_estimators=10).fit(X_train, Y_train)

                    if mapper == "lr":
                        model = LinearRegression().fit(X_train, Y_train)

                    if mapper == "rr":
                        model = Ridge(max_iter=1000).fit(X_train, Y_train)

                    if mapper == 'svr':
                        model = SVR(max_iter=1000).fit(X_train, Y_train)

                    if mapper == "brr":
                        model = BayesianRidge().fit(X_train, Y_train)

                    if mapper == "elast":
                        model = ElasticNet(alpha=1.0, l1_ratio=0.5, selection = "cyclic").fit(X_train, Y_train)

                    if mapper == 'lgb':
                        model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20,
                                            verbosity=0).fit(X_train, Y_train, verbose=False)
                    
                    if mapper == 'xgb':
                        model = xgb.XGBRegressor(objective="reg:squarederror", booster="gblinear", random_state=42).fit(X_train, Y_train)
                    Y_predict = model.predict(X_test)
                    r2 = r2_score(Y_test, Y_predict)
                    mae = mean_absolute_error(Y_test, Y_predict)
                    mse = mean_squared_error(Y_test, Y_predict)
                    pr = pearsonr(Y_test, Y_predict)[0]
                    spr = spearmanr(Y_test, Y_predict)[0]
                    print(f"fold={i}, r2={r2}, mae={mae}, mse={mse}, ps={pr}, spr={spr}")
                    r2s += r2
                    maes += mae 
                    mses += mse
                    prs += pr
                    sprs += spr
                with open(f"{result_path}","a+") as file:
                    if self.use_gpt:
                        model_id = self.modelname
                    else:
                        model_id = "-"
                    if self.use_interation:
                        interact = "+"
                    else:
                        interact = "-"
                    removed_feat = "-".join(x for x in self.feat_bundle_removed)
                    file.write(f"{self.lang}\t{mapper}\t{interact}\t{model_id}\t{r2s/5}\t{maes/5}\t{mses/5}\t{prs/5}\t{sprs/5}\t{ylabel}\t{X.shape[1]}\t{removed_feat}\t{self.modelname}\n")
        del model
if __name__ == "__main__":
    if args.usegpt == 0:
        use_gpt = False
    else:
        use_gpt = True
    if args.usepoly == 0:
        use_interation = False
    else:
        use_interation = True
    lang = args.lang
    modelname= args.modelname
    gptmodel=args.model
    data_path = args.inputpath
    regressor = Regressor(lang=lang, 
                            use_gpt = use_gpt,
                            use_interation = use_interation,
                            data_path = data_path,
                            mappers = args.mappers,
                            ylabels = "TOTAL_DURATION STD_TOTAL_DURATION IA_FIRST_FIXATION_DURATION IA_SECOND_FIXATION_DURATION STD_IA_FIRST_FIXATION_DURATION STD_IA_SECOND_FIXATION_DURATION",
                            modelname= modelname,
                            gptmodel=gptmodel)
    feat_setting = {"pos":True,
                    "ldh":True,
                    "ldr":True,
                    "depth":True,
                    "adj":True,
                    "spr":True,
                    "freq":True,
                    "syl":True,
                    "prvfreq":True,
                    "prvsyl":True,
                    "wp":True,
                    "punct":True}
    regressor.specify_features(**feat_setting)
    regressor.regress(args.resultpath)
    for key,_ in feat_setting.items():
        new_setting = {"pos":True,
                    "ldh":True,
                    "ldr":True,
                    "depth":True,
                    "adj":True,
                    "spr":True,
                    "freq":True,
                    "syl":True,
                    "prvfreq":True,
                    "prvsyl":True,
                    "punct":True,
                    "wp":True,
                    }
        new_setting[key] = False
        regressor.specify_features(**new_setting)
        regressor.regress(args.resultpath)
    