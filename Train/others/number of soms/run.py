import copy
import os, random, math
import numpy as np
from cypre import add_train_args,modify_train_args
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import sys
from tqdm import trange
from argparse import ArgumentParser
from cypre.models import MoleculeModel
from cypre.data import MoleculeDataset
from cypre.data.utils import get_data
from cypre.utils import get_loss_func, get_metric_func, save_checkpoint
from sklearn.metrics import recall_score, f1_score, precision_score, roc_auc_score, jaccard_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score,LeaveOneOut
import pickle
from scipy.misc import derivative
import xgboost

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps`
        steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(
            max(1.0, self.t_total - self.warmup_steps)))

class DiseaseModel(nn.Module):

    def __init__(self, args):
        super(DiseaseModel, self).__init__()
        self.encoder = MoleculeModel()
        self.encoder.create_encoder(args)

        self.encoder.ffn_bonds = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(args.hidden_size_ffn, 2),
        )

    def forward(self, mols, BD_v_batch, mode):
        r_bonds,t_bonds=self.encoder(mols,BD_v_batch)
        return r_bonds,t_bonds

def prepare_data(args):
    data,BD_v,BD_ty = get_data(path=args.path, args=args)
    args.train_data_size = len(data)
    return data,BD_v,BD_ty

def focal_loss_lgb_sk(label,y_pred):#,alpha=0.25, gamma=2

    def func(y_pred,label):
        preds = 1.0/(1.0+np.exp(-y_pred))
        pred = [int(i >= 0.5) for i in preds]
        tn=sum(label*pred)
        tp=sum((-(np.array(pred)-1))*(-(np.array(label)-1)))
        fn=sum(pred)-tp
        fp=sum(-(np.array(pred)-1))-tn

        score=(fp+3*fn)/len(pred)
        return score
    partial_func = lambda x: func(x, label)
    grad = [derivative(partial_func, y_pred, n=1, dx=1e-6)]*len(y_pred)
    hess = [derivative(partial_func, y_pred, n=2, dx=1e-6)]*len(y_pred)

    return grad, hess

def Feval_xg(y_pred, xgbtrain):
    label = xgbtrain.get_label()
    preds = 1.0/(1.0+np.exp(-y_pred))
    pred = [int(i >= 0.5) for i in preds]
    tn=sum(label*pred)
    tp=sum((-(np.array(pred)-1))*(-(np.array(label)-1)))
    fn=sum(pred)-tp
    fp=sum(-(np.array(pred)-1))-tn
    score=tp/(tp+fp+fn)
    return 'Feval_xg',score

def sort_xg(estimator,X_test,y_test):
    pred=estimator.predict(X_test)
    label = y_test
    tn=sum(label*pred)
    tp=sum((-(np.array(pred)-1))*(-(np.array(label)-1)))
    fn=sum(pred)-tp
    fp=sum(-(np.array(pred)-1))-tn
    score=tp/(tp+fp+fn)
    return score

def xg(v_bonds,t_bonds,v_test,t_test,xgmodel):
    v_bonds=v_bonds.cpu().detach().numpy()
    t_bonds_xg = torch.max(t_bonds.cpu(), 1)[1].numpy()
    if xgmodel==False:
        clf=XGBClassifier(objective='binary:logistic',use_label_encoder=False,learning_rate=1e-3,Feval=Feval_xg,tree_method='gpu_hist'
                          ,gamma=0.0,min_child_weight=2,n_estimators=1000,reg_lambda=50,max_depth=5,colsample_bytree=0.5)

        clf.fit(v_bonds,t_bonds_xg)

        pre_bond=clf.predict(v_bonds)
        pre_test=clf.predict(v_test.cpu().detach().numpy())
    else:
        clf=xgmodel
        pre_bond=xgmodel.predict(v_bonds)
        pre_test=xgmodel.predict(v_test.cpu().detach().numpy())
    return pre_bond,pre_test,clf

def xgtrain(model,args,xgmodel,train):
    if train:
        model.eval()

    v_bonds=torch.FloatTensor(np.zeros([1,args.hidden_size_ffn]))
    t_bonds=torch.FloatTensor([[0,0]])
    v_test=torch.FloatTensor(np.zeros([1,args.hidden_size_ffn]))
    t_test=torch.FloatTensor([[0,0]])
    data,BD_v,BD_ty = prepare_data(args)

    data = MoleculeDataset(data[:])
    BD_v = BD_v[:]

    for i in trange(0, len(data), args.batch_size):

        data_batch = MoleculeDataset(data[i:i + args.batch_size])
        BD_v_batch = BD_v[i:i + args.batch_size]

        # batch
        mols = data_batch.mols()
        r_bonds_o,t_bonds_o,v_bonds_o=model.encoder(mols,BD_v_batch)
        t_bonds_o=t_bonds_o[1:]
        v_bonds=torch.concat([v_bonds.cuda(),v_bonds_o.cuda()],dim=0)
        t_bonds=torch.concat([t_bonds.cuda(),t_bonds_o.cuda()],dim=0)

    v_bonds=v_bonds[1:]
    t_bonds=t_bonds[1:]

    path_now=sys.path[0]
    args.path=r''+path_now+'\data\EBoMD2.sdf'+''
    data,BD_v,BD_ty = prepare_data(args)
    data = MoleculeDataset(data[:])
    BD_v = BD_v[:]

    for i in trange(0, len(data), args.batch_size):

        data_batch = MoleculeDataset(data[i:i + args.batch_size])
        BD_v_batch = BD_v[i:i + args.batch_size]

        # batch
        mols = data_batch.mols()
        r_bonds_o,t_bonds_o,v_bonds_o=model.encoder(mols,BD_v_batch)
        t_bonds_o=t_bonds_o[1:]
        v_test=torch.concat([v_test.cuda(),v_bonds_o.cuda()],dim=0)
        t_test=torch.concat([t_test.cuda(),t_bonds_o.cuda()],dim=0)

    v_test=v_test[1:]
    t_test=t_test[1:]
    pre_bond,pre_test,clf=xg(v_bonds,t_bonds,v_test,t_test,xgmodel)
    pre_bond=list(pre_bond)
    pre_test=list(pre_test)

    t_bonds_label = list(torch.max(t_bonds.cpu(), 1)[1].numpy())
    t_test_label = list(torch.max(t_test.cpu(), 1)[1].numpy())

    random_bonds_label=[0,1]*len(t_bonds_label)
    random_bonds_label=random_bonds_label[:int(len(random_bonds_label)/2)]
    random.shuffle(random_bonds_label)
    random_score,list_name = calculate(t_bonds_label, random_bonds_label)

    score,list_name = calculate(t_bonds_label, pre_bond)
    score_test,list_name = calculate(t_test_label, pre_test)

    return score,score_test,list_name,clf,random_score

def calculate(t_bonds_label, r_bonds_label,ROC=True):
    score=[]
    list_name=['bonds_recall_score','bonds_recall_score_micro',
               'bonds_precision_score_macro','bonds_precision_score_micro','bonds_precision_score_every','bonds_f1_score_macro','bonds_f1_score_micro','roc_auc_bonds','jaccard_score_bonds',
               ]

    score.append(recall_score(t_bonds_label, r_bonds_label, average=None))
    score.append(recall_score(t_bonds_label, r_bonds_label, average='micro'))
    score.append(precision_score(t_bonds_label, r_bonds_label, average='macro'))
    score.append(precision_score(t_bonds_label, r_bonds_label, average='micro'))
    score.append(precision_score(t_bonds_label, r_bonds_label, average=None))
    score.append(f1_score(t_bonds_label, r_bonds_label, average='macro'))
    score.append(f1_score(t_bonds_label, r_bonds_label, average='micro'))
    if ROC==True:
        score.append(roc_auc_score(t_bonds_label, r_bonds_label))
    else:
        score.append('None')
    score.append(jaccard_score(t_bonds_label, r_bonds_label,average=None))

    return score,list_name

def train(model, optimizer, loss_func, args, train, xgmodel):

    model.eval()

    act_number=[]
    unact_number=[]
    name=[]

    path_now=sys.path[0]

    if train:
        args.path=r''+path_now+'\data\EBoMD.sdf'+''
    else:
        args.path=r''+path_now+'\data\EBoMD2.sdf'+''

    num=0

    ptype=['BOM_1A2','BOM_2A6','BOM_2B6','BOM_2C8','BOM_2C9','BOM_2C19','BOM_2D6','BOM_2E1','BOM_3A4']
    for j in ptype:
        args.Ptype=j
        data,BD_v,BD_ty = prepare_data(args)
        data = MoleculeDataset(data[:])
        BD_v = BD_v[:]

        r_bonds=torch.FloatTensor([[0,0]])
        t_bonds=torch.FloatTensor([[0,0]])
        for i in trange(0, len(data), args.batch_size):
            num+=1

            model.zero_grad()

            data_batch = MoleculeDataset(data[i:i + args.batch_size])
            BD_v_batch = BD_v[i:i + args.batch_size]

            # batch
            mols = data_batch.mols()
            r_bonds_o,t_bonds_o,v_bonds=model.encoder(mols,BD_v_batch)
            t_bonds_o=t_bonds_o[1:]
            r_bonds=torch.concat([r_bonds.cuda(),r_bonds_o.cuda()],dim=0)
            t_bonds=torch.concat([t_bonds.cuda(),t_bonds_o.cuda()],dim=0)

        r_bonds=r_bonds[1:]
        t_bonds=t_bonds[1:]

        # Take an equal number of positive and negative samples
        act_id_bonds=[]
        unact_id_bonds=[]

        t_bonds_e=list(torch.max(t_bonds.cpu(), 1)[1].numpy())
        for i in range(len(t_bonds_e)):
            if t_bonds_e[i]==0:
                act_id_bonds.append(i)
            else:
                unact_id_bonds.append(i)

        act_number.append(len(act_id_bonds))
        unact_number.append(len(unact_id_bonds))
        name.append(j)

    return act_number,unact_number,name

def run_training(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = DiseaseModel(args).cuda()

    params_name=[]
    for name,param in model.named_parameters():
        param.requires_grad = True
        params_name.append(name)
    print('params_name:',params_name)

    loss_func = get_loss_func(args).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    args.metric_func = get_metric_func(metric=args.metric)
    best_score = -float('inf')
    best_val = float('inf')
    best_epoch = 0

    act_number,unact_number,name = train(model, optimizer, loss_func, args, train=True, xgmodel=False)

    act_number_test,unact_number_test,name_test = train(model, optimizer, loss_func, args, train=False, xgmodel=False)


    result=pd.DataFrame(np.zeros([len(name),6]))
    path_now=sys.path[0]
    for i in range(len(name)):
        result.iloc[[i],[0]]=name[i]
        result.iloc[[i],[1]]=str(act_number[i])
        result.iloc[[i],[2]]=str(unact_number[i])
        result.iloc[[i],[3]]=('test_'+str(name_test[i]))
        result.iloc[[i],[4]]=str(act_number_test[i])
        result.iloc[[i],[5]]=str(unact_number_test[i])
    result.columns=['type','act','unact','test_type','act','unact']
    result.to_excel(r''+path_now+'\\result.xlsx'+'')

if __name__ == "__main__":
    parser=ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=2)
    add_train_args(parser)
    args = parser.parse_args()
    path_now=sys.path[0]
    args.path=r''+path_now+'\data\EBoMD.sdf'+''
    args.Ptype='BOM_2D6'
    args.dataset_type = 'multiclass'
    args.save_dir=path_now
    modify_train_args(args)
    args.batch_size=50
    args.epochs=500
    args.hidden_size=100
    args.hidden_size_ffn=args.hidden_size*1
    run_training(args)
