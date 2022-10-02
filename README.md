# D-CyPre
D-CyPre: A Software Tool for Accurately Predicting the Site of Metabolism of Human Cytochrome P450 Metabolism\
D-CyPre is a convenient software developed based on D-CyPre model, and Train contains all the source code in the research process.\
The data is comes from CyProduct: A Software Tool for Accurately Predicting the Byproducts of Human Cytochrome P450 Metabolism. \
doi: https://doi.org/10.1021/acs.jcim.1c00144\
Requirements (source code):\
numpy==1.22.0\
scikit-learn==1.1.1\
pandas==1.4.2\
torch==1.11.0+cu113\
rdkit==2021.09.2\
tqdm==4.64.0\
pillow==9.0.0\
xgboost==1.5.2
# Description of folders/files in the repository
1. D-CyPre.zip is a convenient software. Please see the supplementary materials for specific usage methods.
2. Train contains the code for all the works.\
  2.1 original: Contains code for original D-CyPre.\
  2.2 model - DMPNN: Contains code for D-CyPre (including molecular features calculated by D-MPNN).\
  2.3 others: Contains other experiments in the study, such as training the models of atoms and bonds, separately.\
# Overview:
![overview](https://github.com/67520/D-CyPre/blob/master/overview.png)
