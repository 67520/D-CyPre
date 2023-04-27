import os, random
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
import pickle
from random import shuffle

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

class Feature_generator(nn.Module):

    def __init__(self, args):
        super(Feature_generator, self).__init__()
        self.encoder = MoleculeModel()
        self.encoder.create_encoder(args)

        self.encoder.ffn_bonds = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(args.hidden_size_ffn, 2),
        )

def prepare_data(args):
    data,BD_v,BD_ty = get_data(args=args)
    args.train_data_size = len(data)
    return data,BD_v,BD_ty

def Feval_xg(y_pred, xgbtrain):
    label = xgbtrain.get_label()
    preds = 1.0/(1.0+np.exp(-y_pred))
    pred = [int(i >= 0.5) for i in preds]
    tn = sum(label*pred)
    tp = sum((-(np.array(pred)-1))*(-(np.array(label)-1)))
    fn = sum(pred)-tp
    fp = sum(-(np.array(pred)-1))-tn
    score=tp/(tp+fp+fn)
    return 'Feval_xg', score

def sort_xg(estimator,X_test,y_test):
    pred=estimator.predict(X_test)
    label = y_test
    tn=sum(label*pred)
    tp=sum((-(np.array(pred)-1))*(-(np.array(label)-1)))
    fn=sum(pred)-tp
    fp=sum(-(np.array(pred)-1))-tn
    score=tp/(tp+fp+fn)
    return score

def xg(v,t,v_val,t_val,v_test,t_test,xgmodel,val):
    v=v.cpu().detach().numpy()
    v_val=v_val.cpu().detach().numpy()
    t_xg = torch.max(t.cpu(), 1)[1].numpy()
    pre_val=0

    # if train recall mode
    # S_1=sum(t_xg)
    # S_0=sum(1-t_xg)

    # get c from val
    # scale_weight=S_0/S_1*c

    if xgmodel==False:
        clf=XGBClassifier(objective='binary:logistic',use_label_encoder=False,learning_rate=1e-3,Feval=Feval_xg,tree_method='gpu_hist'
                          ,gamma=0.0,min_child_weight=2,n_estimators=1000,reg_lambda=0,max_depth=9,colsample_bytree=0.85)# if train recall mode: scale_pos_weight=scale_weight

        clf.fit(v,t_xg)
        if val:
            pre_val=clf.predict(v_val)
        pre=clf.predict(v)
        pre_test=clf.predict(v_test.cpu().detach().numpy())
    else:
        clf=xgmodel
        pre=xgmodel.predict(v)
        pre_test=xgmodel.predict(v_test.cpu().detach().numpy())
        if val:
            pre_val=clf.predict(v_val)
    return pre,pre_test,pre_val,clf

def xgtrain(data_xg,BD_v_xg,model,args,val,xgmodel,train,random):
    if train:
        model.eval()
    v_all=torch.FloatTensor(np.zeros([1,args.hidden_size_ffn]))
    t_all=torch.FloatTensor([[0,0]])
    v_all_val=torch.FloatTensor(np.zeros([1,args.hidden_size_ffn]))
    t_all_val=torch.FloatTensor([[0,0]])
    v_test=torch.FloatTensor(np.zeros([1,args.hidden_size_ffn]))
    t_test=torch.FloatTensor([[0,0]])

    data=data_xg
    BD_v=BD_v_xg
    socre_val=0

    if val:
        val_size=int(val*len(data))
        data_val=MoleculeDataset(data[:val_size])
        BD_v_val=BD_v[:val_size]
        data = MoleculeDataset(data[val_size:])
        BD_v = BD_v[val_size:]

    else:
        data = MoleculeDataset(data[:])
        BD_v = BD_v[:]

    for i in trange(0, len(data), args.batch_size):

        data_batch = MoleculeDataset(data[i:i + args.batch_size])
        BD_v_batch = BD_v[i:i + args.batch_size]

        mols = data_batch.mols()
        r_all_o,t_all_o,v_all_o=model.encoder(mols,BD_v_batch)
        t_all_o=t_all_o[1:]
        v_all=torch.concat([v_all.cuda(),v_all_o.cuda()],dim=0)
        t_all=torch.concat([t_all.cuda(),t_all_o.cuda()],dim=0)

    v_all=v_all[1:]
    t_all=t_all[1:]

    if val:
        for i in trange(0, len(data_val), args.batch_size):

            data_batch_val = MoleculeDataset(data_val[i:i + args.batch_size])
            BD_v_batch_val = BD_v_val[i:i + args.batch_size]

            mols = data_batch_val.mols()
            r_all_o,t_all_o,v_all_o=model.encoder(mols,BD_v_batch_val)
            t_all_o=t_all_o[1:]
            v_all_val=torch.concat([v_all_val.cuda(),v_all_o.cuda()],dim=0)
            t_all_val=torch.concat([t_all_val.cuda(),t_all_o.cuda()],dim=0)

        v_all_val=v_all_val[1:]
        t_all_val=t_all_val[1:]

    path_now=sys.path[0]
    args.path=r''+path_now+'\data\EBoMD2.sdf'+''
    data,BD_v,BD_ty = prepare_data(args)
    data = MoleculeDataset(data[:])
    BD_v = BD_v[:]

    for i in trange(0, len(data), args.batch_size):

        data_batch = MoleculeDataset(data[i:i + args.batch_size])
        BD_v_batch = BD_v[i:i + args.batch_size]

        mols = data_batch.mols()
        r_all_o,t_all_o,v_all_o=model.encoder(mols,BD_v_batch)
        t_all_o=t_all_o[1:]
        v_test=torch.concat([v_test.cuda(),v_all_o.cuda()],dim=0)
        t_test=torch.concat([t_test.cuda(),t_all_o.cuda()],dim=0)

    v_test=v_test[1:]
    t_test=t_test[1:]

    pre,pre_test,pre_val,clf=xg(v_all,t_all,v_all_val,t_all_val,v_test,t_test,xgmodel,val)
    pre=list(pre)
    pre_test=list(pre_test)
    if val:
        pre_val=list(pre_val)
        t_val_label = list(torch.max(t_all_val.cpu(), 1)[1].numpy())
        socre_val,list_name = calculate(t_val_label, pre_val)

    t_label = list(torch.max(t_all.cpu(), 1)[1].numpy())
    t_test_label = list(torch.max(t_test.cpu(), 1)[1].numpy())

    score,list_name = calculate(t_label, pre)
    score_test,list_name = calculate(t_test_label, pre_test)

    random_score=0
    if random:
        random_label=[0,1]*len(t_label)
        random_label=random_label[:int(len(random_label)/2)]
        shuffle(random_label)
        random_score,list_name = calculate(t_label, random_label)

    return score,score_test,socre_val,list_name,clf,random_score

def calculate(t_bonds_label, r_bonds_label,ROC=True):
    score=[]
    list_name=['bonds_recall_score','bonds_recall_score_micro',
               'bonds_precision_score_macro','bonds_precision_score_micro','bonds_precision_score_every',
               'bonds_f1_score_macro','bonds_f1_score_micro','roc_auc_bonds','jaccard_score_bonds',
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

def train(data_all, BD_v_all, data_xg, BD_v_xg, model, optimizer, scheduler, loss_func, args, train, xgmodel, val, random, xg_train):
    if train:
        model.train()
    else:
        model.eval()

    path_now=sys.path[0]
    args.path=r''+path_now+'\data\EBoMD.sdf'+''

    num=0
    r_all=torch.FloatTensor([[0,0]])
    t_all=torch.FloatTensor([[0,0]])
    r_target=torch.FloatTensor([[0,0]])
    t_target=torch.FloatTensor([[0,0]])
    type=['BOM_1A2','BOM_2A6','BOM_2B6','BOM_2C8','BOM_2C9','BOM_2C19','BOM_2D6','BOM_2E1','BOM_3A4']

    if train:
        for now_i in range(9):

            data=data_all[now_i]
            BD_v=BD_v_all[now_i]

            if type[now_i]==args.nowtype:
                if val:
                    val_size=int(val*len(data))
                    data = MoleculeDataset(data[val_size:])
                    BD_v = BD_v[val_size:]

            else:
                data = MoleculeDataset(data[:])
                BD_v = BD_v[:]

            for i in trange(0, len(data), args.batch_size):
                num+=1

                model.zero_grad()

                data_batch = MoleculeDataset(data[i:i + args.batch_size])
                BD_v_batch = BD_v[i:i + args.batch_size]

                mols = data_batch.mols()
                r_all_o,t_all_o,v_all=model.encoder(mols,BD_v_batch)
                t_all_o=t_all_o[1:]
                r_all=torch.concat([r_all.cuda(),r_all_o.cuda()],dim=0)
                t_all=torch.concat([t_all.cuda(),t_all_o.cuda()],dim=0)

                if type[now_i]==args.nowtype:
                    r_target=torch.concat([r_target.cuda(),r_all_o.cuda()],dim=0)
                    t_target=torch.concat([t_target.cuda(),t_all_o.cuda()],dim=0)

        r_target=r_target[1:]
        t_target=t_target[1:]

        r_all=r_all[1:]
        t_all=t_all[1:]

        # Take positive and negative samples
        act_id=[]
        unact_id=[]

        t_all_e=list(torch.max(t_all.cpu(), 1)[1].numpy())
        for i in range(len(t_all_e)):
            if t_all_e[i]==0:
                act_id.append(i)
            else:
                unact_id.append(i)

        loss_act=0
        loss_unact=0

        if len(act_id):
            r_act=r_all.cpu()[act_id]
            t_act=t_all.cpu()[act_id]
            r_unact=r_all.cpu()[unact_id]
            t_unact=t_all.cpu()[unact_id]

            loss_act = loss_func(r_act, t_act)
            loss_unact = loss_func(r_unact, t_unact)

    args.Ptype=args.nowtype
    if xg_train==True:
        score,score_test,socre_val,list_name,clf,random_score = xgtrain(data_xg,BD_v_xg,model,args,val,xgmodel,train,random)
    else:
        score,score_test,socre_val,clf,random_score=0,0,0,0,0

    t_target = list(torch.max(t_target.cpu(), 1)[1].numpy())
    r_target = list(torch.max(r_target.cpu(), 1)[1].numpy())
    score_NET,list_name = calculate(t_target, r_target, ROC=False)

    if train:
        loss = loss_act+loss_unact
        loss.backward()
        print('loss:',loss)
        optimizer.step()
        scheduler.step()

    return score,score_test,socre_val,list_name,clf,random_score,score_NET

def run_training(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = Feature_generator(args).cuda()

    params_name=[]
    for name,param in model.named_parameters():
        param.requires_grad = True
        params_name.append(name)
    print('params_name:',params_name)

    loss_func = get_loss_func(args).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    data_all=[]
    BD_v_all=[]
    for type in ['BOM_1A2','BOM_2A6','BOM_2B6','BOM_2C8','BOM_2C9','BOM_2C19','BOM_2D6','BOM_2E1','BOM_3A4']:
        args.Ptype=type
        data,BD_v,BD_ty = prepare_data(args) # get train_data_size
        data_all.append(data)
        BD_v_all.append(BD_v)

        if type==args.nowtype:
            scheduler = WarmupLinearSchedule(optimizer,
                                             warmup_steps=args.train_data_size / args.batch_size * 2,
                                             t_total=args.train_data_size / args.batch_size * (args.epochs)
                                             )
            data_xg=MoleculeDataset(data[:])
            BD_v_xg=BD_v[:]

    args.metric_func = get_metric_func(metric=args.metric)
    best_score = -float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        path_now=sys.path[0]
        args.path=r''+path_now+'\data\EBoMD.sdf'+''

        score,score_test,socre_val,list_name,clf,random_score,score_NET = train(data_all, BD_v_all, data_xg, BD_v_xg, model, optimizer, scheduler, loss_func, args, train=True, xgmodel=False, val=False, random=False, xg_train=True) # train
        score_now=float(score[8][0])

        if epoch >= 5 and (score_now>best_score):
            # save result
            model.eval()
            save_checkpoint(os.path.join(args.save_dir, 'model.pt'), model, args=args)
            best_score, best_epoch = score_now, epoch
            print('test:')
            for i in range(len(list_name)):
                print(list_name[i],':',score_test[i])

    print(f'best epoch {best_epoch}')
    ckpt_path = os.path.join(args.save_dir, 'model.pt')
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    # final model
    score,score_test,socre_val,list_name,clf,random_score,score_NET = train(data_all, BD_v_all, data_xg, BD_v_xg, model, optimizer, scheduler, loss_func, args, train=False, xgmodel=False, val=False, random=False, xg_train=True)

    pickle.dump(clf, open("pima.pickle.dat", "wb"))
    loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
    score,score_test,socre_val,list_name,clf,random_score,score_NET = train(data_all, BD_v_all, data_xg, BD_v_xg, model, optimizer, scheduler, loss_func, args, train=False, xgmodel=loaded_model, val=False, random=True, xg_train=True)

    print('all_result:')
    for i in range(len(list_name)):
        print(list_name[i],':',score[i])
    print('test:')
    for i in range(len(list_name)):
        print(list_name[i],':',score_test[i])

    result=pd.DataFrame(np.zeros([len(list_name),6]))
    for i in range(len(list_name)):
        result.iloc[[i],[0]]=list_name[i]
        result.iloc[[i],[1]]=str(score[i])
        result.iloc[[i],[2]]=('test_'+str(list_name[i]))
        result.iloc[[i],[3]]=str(score_test[i])
        result.iloc[[i],[4]]=('random_'+str(list_name[i]))
        result.iloc[[i],[5]]=str(random_score[i])
    result.to_excel(r''+path_now+'\\'+str(best_epoch)+'result.xlsx'+'')

if __name__ == "__main__":
    parser=ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=2)
    add_train_args(parser)
    args = parser.parse_args()
    path_now = sys.path[0]
    args.path = r''+path_now+'\data\EBoMD.sdf'+''
    args.nowtype = 'BOM_1A2'
    args.dataset_type = 'multiclass'
    args.save_dir = path_now
    modify_train_args(args)
    args.batch_size = 50
    args.epochs = 500
    args.hidden_size = 100
    args.hidden_size_ffn = args.hidden_size*1
    run_training(args)
