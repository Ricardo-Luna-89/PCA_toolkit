import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%%  ------ clase PCA ---------------
class PCA_plot:     
  
    def __init__(self, data, n_components, sample_code, sample_color):
        """
        Parameters
        ----------
        data : DataFrame
            database with n samples and m features for PCA.
        n_components : int
            number of principal components.
        sample_code : array-like, Series object 
            list of the code names from the database samples.
        sample_color : array-like, Series object or color 
            list of colors from the database samples.
        """        
        
        self.features = list(data.columns)
        self.code = sample_code
        self.color = sample_color
        
        n_samples, n_features = np.shape(data)
                
        if n_components == None:
            self.n_components = np.min([n_samples, n_features])
        else:
            self.n_components = n_components  
        
        scaler = StandardScaler().fit(data)
        data_scale = scaler.transform(data)
        data_scale = pd.DataFrame(data_scale)
        data_scale.columns = self.features
        self.pca = PCA(n_components=self.n_components,svd_solver='auto')
        self.score = self.pca.fit_transform(data_scale)
        self.coeff = self.pca.components_
        self.explained = self.pca.explained_variance_ratio_*100    
  
    def biplot(self, color_arrow = 'blue', color_features = 'red', 
               color_code = 'black', fontsize_features = 10, fontsize_code = 10):   
        """
        Parameters
        ----------
        color_arrow : string, optional
            Color of the vectors (arrows) in the PCA biplot. The default is 'blue'.
        color_features : string, optional
            Color of the features names in the PCA biplot. The default is 'red'.
        color_code : string, optional
            Color of the samples names in the PCA biplot. The default is 'black'.
        fontsize_features : int, optional
            fontsize of the features names in the PCA biplot. The default is 10.
        fontsize_code : int, optional
            fontsize of the samples names in the PCA biplot. The default is 10.
        """
        
        
        xs = self.score[:,0]
        ys = self.score[:,1]
        n = self.coeff.shape[1]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        plt.figure()
        plt.scatter(xs * scalex,ys * scaley, c = self.color, s=40)        
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('PC1 ('+str(np.round(self.explained[0],1))+'%)', fontsize = 18)
        plt.ylabel('PC2 ('+str(np.round(self.explained[1],1))+'%)', fontsize = 18)
        plt.grid(True)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        
        for i in range(n):
            plt.arrow(0, 0, self.coeff[0,i], self.coeff[1,i], color = color_arrow, alpha = 0.5,  head_width = 0.01)
            if self.features is None:
                plt.text(self.coeff[i,0]* 1.1, self.coeff[i,1] * 1.1, "Var"+str(i+1),
                         color = color_features, ha = 'center', va = 'center',
                         fontsize = fontsize_features)            
            else:
                plt.text(self.coeff[0,i]* 1.1, self.coeff[1,i] * 1.1, self.features[i],
                     color = color_features, ha = 'center', va = 'center',
                     fontsize = fontsize_features, style = 'italic')
        for j in range(len(self.code)):
            plt.text(xs[j]*0.89*scalex, ys[j]*0.89*scaley, self.code[j], color = color_code,
                      ha = 'center', va = 'center', fontsize = fontsize_code)       
        
    def explained_plot(self):        
        
        pc = np.arange(self.n_components) + 1
        cumulative_explained = np.cumsum(self.explained)
        plt.figure()
        plt.bar(pc, self.explained)
        plt.plot(pc, cumulative_explained, '-o', color = 'red')
        plt.ylabel('Explained Variance (%)', fontsize = 18)
        plt.xlabel('Number of principal components', fontsize = 18)    
        x_ticks = []
        
        for i in range(self.n_components):
            plt.text(pc[i]-0.1, cumulative_explained[i]+1, 
                     s = str(round(cumulative_explained[i]))+'%', fontsize = 12)
            x_ticks += ['PC'+str(i+1)]        
        plt.xticks(pc, x_ticks, fontsize = 14)
        plt.yticks(fontsize = 14)
        
    def scree_plot(self):
    
        eigenvalues = self.pca.explained_variance_
        pc = np.arange(self.n_components) + 1
        plt.figure()
        plt.plot(pc,eigenvalues, marker='o', color = 'blue')
        plt.plot(pc,np.ones(self.n_components),color = 'red', linestyle = '--')
        plt.ylabel('Eigenvalue', fontsize = 18)
        plt.xlabel('Number of principal components', fontsize = 18)   
        
        x_ticks = []
        for i in range(self.n_components):
            x_ticks += ['PC'+str(i+1)]
        
        plt.xticks(pc, x_ticks, fontsize = 14)
        plt.yticks(fontsize = 14)
        
    def feature_contribution(self):
        
        coeff = abs(self.pca.components_)
        y_values = np.linspace(0,1,6)
        plt.figure()
        for i in range(self.n_components):
            
            if self.n_components == 1:                
                plt.bar(self.features, coeff[i,:])
                plt.ylabel('PC'+str(i+1))
                plt.xticks(rotation = 60)
                plt.yticks(y_values)
            
            elif self.n_components == 2:
                plt.subplot(1,2,i+1)        
                plt.bar(self.features, coeff[i,:])
                plt.ylabel('PC'+str(i+1))
                plt.xticks(rotation = 60)
                plt.yticks(y_values)
                
            elif self.n_components == 3:
                plt.subplot(1,3,i+1)        
                plt.bar(self.features, coeff[i,:])
                plt.ylabel('PC'+str(i+1))
                plt.xticks(rotation = 60)
                plt.yticks(y_values)
            
            elif self.n_components == 4:
                plt.subplot(2,2,i+1)        
                plt.bar(self.features, coeff[i,:])
                plt.ylabel('PC'+str(i+1))
                plt.xticks(rotation = 60)
                plt.yticks(y_values)
                
            elif self.n_components == 5 or self.n_components == 6:
                plt.subplot(3,2,i+1)
                plt.bar(self.features, coeff[i,:])
                plt.ylabel('PC'+str(i+1))
                plt.xticks(rotation = 60)
                plt.yticks(y_values)
        
