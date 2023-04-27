from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
import pandas as pd
import numpy as np

def Read_mol(path,type):

    df = PandasTools.LoadSDF(path)
    Bom_1A2=df[type]

    BOM_list=[0]*Bom_1A2.shape[0]
    for i in range(Bom_1A2.shape[0]):
        BOM_list[i]=str(Bom_1A2[i]).replace('<','').split('>')

    df_Bom_1A2=pd.DataFrame()
    df_Bom_1A2['ID']=df.ID
    df_Bom_1A2['ROMol']=df.ROMol
    df_Bom_1A2['SMI']=df.ROMol

    for i in range(Bom_1A2.shape[0]):
        df_Bom_1A2['SMI'][i]=Chem.MolToSmiles(df.ROMol[i])

    BOM_list=pd.DataFrame(BOM_list)

    for i in range(BOM_list.shape[1]):
        BOM_list[BOM_list.columns[i]]=BOM_list[BOM_list.columns[i]].str.replace('\n', '').replace(' ', '').replace('\t', '').replace('\r', '').replace('nan', '')

    BD_v=[0]*Bom_1A2.shape[0]
    BD_ty=[0]*Bom_1A2.shape[0]
    list_zan=[]
    list_zan_ty=[]
    for i in range(BOM_list.shape[0]):
        if i>0:
            BD_v[(i-1)]=list_zan
            BD_ty[(i-1)]=list_zan_ty
            list_zan=[]
            list_zan_ty=[]
        for j in range(BOM_list.shape[1]):
            if not BOM_list.iloc[[i],[j]].values=='':
                if not BOM_list.iloc[[i],[j]].values==None:
                    d_zan=BOM_list.iloc[[i],[j]]
                    d_zan=str(d_zan.values)
                    d_zan=str(d_zan).split(';')
                    d_zan_bom=str(d_zan[0])
                    d_zan_bom=d_zan_bom.replace(']','')
                    d_zan_bom=d_zan_bom.replace('[','')
                    d_zan_bom=d_zan_bom.replace('\'','')
                    list_zan.append(d_zan_bom)
                    d_zan_ty=str(d_zan[1])
                    list_zan_ty.append(d_zan_ty)
        if i==(BOM_list.shape[0]-1):
            BD_v[i]=list_zan
            BD_ty[i]=list_zan_ty

    label=[]
    BD_v_new=[]
    BD_ty_new=[]
    for i in range(len(BD_v)):
        if len(BD_v[i])>0:
            label.append(i)
            BD_v_new.append(BD_v[i])
            BD_ty_new.append(BD_ty[i])
    df_Bom_1A2=df_Bom_1A2.iloc[label,:]
    df_Bom_1A2=df_Bom_1A2.reset_index()
    df_Bom_1A2= df_Bom_1A2.drop('index', 1)

    return df_Bom_1A2,BD_v_new,BD_ty_new
            

