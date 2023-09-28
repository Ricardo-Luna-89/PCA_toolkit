import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns



#%%  ------   FUNCIONES   ------
  
def PCA_biplot(score, coeff, labels, explained, sample_code, sample_color) :          
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.figure()
    plt.scatter(xs * scalex,ys * scaley, c = sample_color, s=40)
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('PC1 ('+str(np.round(explained[0]*100,1))+'%)', fontsize = 20)
    plt.ylabel('PC2 ('+str(np.round(explained[1]*100,1))+'%)', fontsize = 20)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = '#BBB9B8', alpha = 0.5,  head_width = 0.01)
        if labels is None:
            plt.text(coeff[i,0]* 1.1, coeff[i,1] * 1.1, "Var"+str(i+1),
                     color = 'red', ha = 'center', va = 'center',
                     fontsize = 10)            
        else:
            plt.text(coeff[i,0]* 1.1, coeff[i,1] * 1.1, labels[i],
                 color = 'red', ha = 'center', va = 'center',
                 fontsize = 10, style = 'italic')
    for j in range(len(sample_code)):
        plt.text(xs[j]*0.89*scalex, ys[j]*0.89*scaley, sample_code[j], color = 'k',
                  ha = 'center', va = 'center', fontsize = 8)
        
        
        
def explained_plot(n_components, pca):
    
    explained = pca.explained_variance_ratio_ *100
    pc = np.arange(n_components) + 1
    cumulative_explained = np.cumsum(explained)
    plt.bar(pc, explained)
    plt.plot(pc, cumulative_explained, '-o', color = 'red')
    plt.ylabel('Explained Variance (%)')
    plt.xlabel('Number of principal components')    
    x_ticks = []
    
    for i in range(n_components):
        plt.text(pc[i]-0.1, cumulative_explained[i]+1, s = str(round(cumulative_explained[i]))+'%')
        x_ticks += ['PC'+str(i+1)]        
    plt.xticks(pc, x_ticks)
    
def scree_plot(n_components, pca):

    eigenvalues = pca.explained_variance_
    pc = np.arange(n_components) + 1
    plt.plot(pc,eigenvalues, marker='o', color = 'blue')
    plt.plot(pc,np.ones(n_components),color = 'red', linestyle = '--')
    plt.ylabel('Eigenvalue')
    plt.xlabel('Number of principal components')   
    
    x_ticks = []
    for i in range(n_components):
        x_ticks += ['PC'+str(i+1)]
    
    plt.xticks(pc, x_ticks)
    
def feature_contribution(n_components, pca, features):
    coeff = abs(pca.components_)
    y_values = np.linspace(0,1,6)
    for i in range(n_components):
        
        if n_components == 1:
            plt.bar(features, coeff[i,:])
            plt.ylabel('PC'+str(i+1))
            plt.xticks(rotation = 60)
            plt.yticks(y_values)
        
        elif n_components == 2:
            plt.subplot(1,2,i+1)        
            plt.bar(features, coeff[i,:])
            plt.ylabel('PC'+str(i+1))
            plt.xticks(rotation = 60)
            plt.yticks(y_values)
            
        elif n_components == 3:
            plt.subplot(1,3,i+1)        
            plt.bar(features, coeff[i,:])
            plt.ylabel('PC'+str(i+1))
            plt.xticks(rotation = 60)
            plt.yticks(y_values)
        
        elif n_components == 4:
            plt.subplot(2,2,i+1)        
            plt.bar(features, coeff[i,:])
            plt.ylabel('PC'+str(i+1))
            plt.xticks(rotation = 60)
            plt.yticks(y_values)
            
        elif n_components == 5 or n_components == 6:
            plt.subplot(3,2,i+1)
            plt.bar(features, coeff[i,:])
            plt.ylabel('PC'+str(i+1))
            plt.xticks(rotation = 60)
            plt.yticks(y_values)
    


#%%   ------   DATOS   ------
file = 'Analisis_diablo.xlsx'
sheet = 'diablo'
data = pd.read_excel(file, sheet_name = sheet)
code = data['Muestras']
data = data.drop(columns = ['Muestras', 'Código'])
features = list(data.columns)
n_samples, n_features = np.shape(data)
n_components = np.min([n_samples,n_features])
#%%   ------   CÁLCULOS Y VARIABLES   ------
scaler = StandardScaler().fit(data)
data_scale = scaler.transform(data)
data_scale = pd.DataFrame(data_scale)
data_scale.columns = features
pca = PCA(n_components=n_components,svd_solver='auto')
x_pca = pca.fit_transform(data_scale)
explained = pca.explained_variance_ratio_
# %% explained plot
plt.style.use('default')
explained_plot(n_components, pca)
scree_plot(n_components, pca)
feature_contribution(2, pca, features)

#%%   ------   PCA   ------
color = 'blue'
PCA_biplot(x_pca[:, 0:2], np.transpose(pca.components_[0:2, :]), features, explained, code, color)
plt.grid(True)
plt.title('Principal Component Analysis', fontsize = 20)
plt.xticks(size=12)
plt.yticks(size=12)











    
