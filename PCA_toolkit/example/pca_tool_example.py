# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:06:48 2023

@author: rluna
"""
import pandas as pd
from pca_tool import PCA_plot
# %% Example
file = 'Analisis_diablo.xlsx'
sheet = 'diablo'
# import data
data = pd.read_excel(file, sheet_name = sheet)
# pre-procesing data
data_code = data['Muestras']
data_color = data['Color']
data = data.drop(columns = ['Muestras', 'Color'])
# create a pca object to generate differents plots
diablo = PCA_plot(data = data, n_components = None, sample_code = data_code, sample_color = data_color)
# Biplot
diablo.biplot(color_arrow='gray', color_features = 'blue', fontsize_code=8, fontsize_features=10)
# Explained plot
diablo.explained_plot()
# featrue contribution plot
diablo.feature_contribution()
# scree plot
diablo.scree_plot()
