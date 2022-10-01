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

class DiseaseModel(nn.Module):

    def __init__(self, args):
        super(DiseaseModel, self).__init__()
        self.encoder = MoleculeModel()
        self.encoder.create_encoder(args)

        self.encoder.ffn_bonds = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(args.hidden_size_ffn, 2),
        )

def prepare_data(args):
    data,BD_v,BD_ty = get_data(path=args.path, args=args)
    args.train_data_size = len(data)
    return data,BD_v,BD_ty

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

def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = DiseaseModel(args).cuda()

    args.Ptype=args.nowtype
    data,BD_v,BD_ty = prepare_data(args) # get train_data_size
    val=0.2
    val_size=int(val*len(data))
    data=MoleculeDataset(data[:val_size])
    BD_v=BD_v[:val_size]

    loaded_model = pickle.load(open("atoms/pima.pickle.dat", "rb"))
    ckpt_path = os.path.join(args.save_dir, 'atoms/model_500.pt')
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    mols = data.mols()
    model.eval()
    r_all,t_all,v_all=model.encoder(mols,BD_v,type='atoms')
    t_all=t_all[1:]
    pre=loaded_model.predict(v_all.cpu().detach().numpy())
    pre_atoms=torch.FloatTensor(pre)
    t_all_atoms=torch.FloatTensor(t_all)

    pre_atoms_=list(pre_atoms)
    t_label_atoms = list(torch.max(t_all.cpu(), 1)[1].numpy())
    score,list_name = calculate(t_label_atoms, pre_atoms_)
    print('atoms_result:')
    for i in range(len(list_name)):
        print(list_name[i],':',score[i])

    loaded_model = pickle.load(open("bonds/pima.pickle.dat", "rb"))
    ckpt_path = os.path.join(args.save_dir, 'bonds/model_500.pt')
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    mols = data.mols()
    model.eval()
    r_all,t_all,v_all=model.encoder(mols,BD_v,type='bonds')
    t_all=t_all[1:]
    pre=loaded_model.predict(v_all.cpu().detach().numpy())
    pre_bonds=torch.FloatTensor(pre)
    t_all_bonds=torch.FloatTensor(t_all)

    pre_bonds_=list(pre_bonds)
    t_label_bonds = list(torch.max(t_all.cpu(), 1)[1].numpy())
    score,list_name = calculate(t_label_bonds, pre_bonds_)
    print('bonds_result:')
    for i in range(len(list_name)):
        print(list_name[i],':',score[i])

    pre=torch.concat([pre_atoms,pre_bonds],dim=0)
    t_all=torch.concat([t_all_atoms,t_all_bonds],dim=0)
    pre=list(pre)
    t_label = list(torch.max(t_all.cpu(), 1)[1].numpy())
    score,list_name = calculate(t_label, pre)

    print('all_result:')
    for i in range(len(list_name)):
        print(list_name[i],':',score[i])

    result=pd.DataFrame(np.zeros([len(list_name),2]))
    for i in range(len(list_name)):
        result.iloc[[i],[0]]=list_name[i]
        result.iloc[[i],[1]]=str(score[i])
    path_now=sys.path[0]
    result.to_excel(r''+path_now+'\\'+'result.xlsx'+'')

if __name__ == "__main__":
    parser=ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=2)
    add_train_args(parser)
    args = parser.parse_args()
    path_now=sys.path[0]
    args.path=r''+path_now+'\data\EBoMD.sdf'+''
    args.nowtype='BOM_2B6'
    args.dataset_type = 'multiclass'
    args.save_dir=path_now
    modify_train_args(args)
    args.batch_size=50
    args.epochs=1000
    args.hidden_size=100
    args.hidden_size_ffn=args.hidden_size*1

    run(args)
