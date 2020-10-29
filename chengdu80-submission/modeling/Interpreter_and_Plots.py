#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# In[ ]:


# pip install shap


# In[ ]:


import pandas as pd
# from feature_cleaning import missing_data as ms
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
# from google.colab import files


# In[ ]:


X = pd.read_csv('xdata.csv')
y = pd.read_csv('ydata.csv')


with open('rf_aapl.pickle', 'rb') as f:
    model = pickle.load(f)


# In[ ]:


X=X.set_index('date')

feature_list = X.columns
feature_list = list(feature_list)
# feature_list

y=y.set_index('date')
# y.head()

y1 =y.filter(items=['price4'])
# y1.head()

# X.describe()
date_list = X.index.tolist()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import shap

# ticker_list = df.ticker.unique()

j = 'aapl'


# # Feature Importance Plot
# 
# To get an overview of which features are most important for a model we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. The color represents the feature value (red high, blue low). 

# In[ ]:


# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


# In[ ]:


# summarize the effects of all the features
shap.summary_plot(shap_values[0], X)

# output = shap.summary_plot(shap_values, X)
# shap.save_html("Feature Importance Ranking - Aggregationg Bar Chart.html", output)


# In[ ]:


# explain the effects of all the features and distribution
import matplotlib.pyplot as plt

shap.summary_plot(shap_values, X)


# # Force Plot  - Feature Contribution Visualization Across Observatons

# In[ ]:


# load JS visualization code to notebook
shap.initjs()
# visualize the prediction's explanation (use matplotlib=True to avoid Javascript)
# Impact on Day 0 Price
output = shap.force_plot(explainer.expected_value[0], shap_values[0], X,plot_cmap=["#FF0000","#008000"])
shap.save_html("Price Influence by Features Across Observations.html", output)
shap.force_plot(explainer.expected_value[0], shap_values[0], X,plot_cmap=["#FF0000","#008000"])


# # Key Price Influencer Daily Slider

# In[ ]:


import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

X = pd.read_csv('xdata.csv')
date_list = X.date.tolist()
# date_list

# pip install shap

X = pd.read_csv('xdata.csv')
y = pd.read_csv('ydata.csv')

with open('rf_aapl.pickle', 'rb') as f:
    model = pickle.load(f) 

X=X.set_index('date')
feature_list = X.columns
y=y.set_index('date')

y1 =y.filter(items=['price4'])

date_list = X.index.tolist()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import shap

j = 'aapl'

# explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Prepare the List which Indices can be matched to the date for the Date Slider Output

import datetime

date_time_obj = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in date_list]
# date_time_obj = [datetime.datetime.strptime(date, '%d/%m/%Y').date() for date in date_list]

from datetime import datetime

start_date = datetime(2012, 1, 1)
end_date = datetime(2012, 10, 31)

dates = pd.date_range(start_date, end_date, freq='D')

options = [(date.strftime(' %d %b %Y '), date) for date in dates]
index = (0, len(options)-1)

selection_range_slider = widgets.SelectionRangeSlider(
    options=options,
    index=index,
    description='Date',
    orientation='horizontal',
    layout={'width': '500px'}
)

# selection_range_slider

import datetime 
def print_date_range(date_range):
    
    time_stamp = date_range[1]
    
    day_time = time_stamp.to_pydatetime()

    day = day_time.date()
    try:
        i = date_time_obj.index(day)
        print('          Bar Chart - Key Price Influencers on',day)
        shap.bar_plot(shap_values[0][i],feature_names=feature_list)
#         shap.force_plot(explainer.expected_value, shap_values[i,:], X.iloc[i,:])
    except:
        print("This day falls on weekend or public holiday, so no trading activity is found.")

widgets.interact(
    print_date_range,
    date_range=selection_range_slider
    
);

from IPython.display import Javascript
from nbconvert import HTMLExporter

from ipywidgets.embed import embed_minimal_html

embed_minimal_html('export.html', views=[selection_range_slider], title='Widgets export')


# In[ ]:


shap.bar_plot(shap_values[0][54],feature_names=feature_list)


# In[ ]:


expected_value = explainer.expected_value
shap.decision_plot(expected_value[0], shap_values[0], feature_names=feature_list,
                    highlight=0)


# In[ ]:


expected_value = explainer.expected_value
shap.decision_plot(expected_value[0], shap_values[0][54], feature_names=feature_list,
                    highlight=0)


# In[ ]:


# create a dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot("governance", shap_values[0], X)


# In[ ]:


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import plot_partial_dependence, partial_dependence
plot_fig = plot_partial_dependence(model, X, features=[0,1,2,5,6,7],n_jobs=2,feature_names = feature_list) 
fig = plt.gcf()
fig.subplots_adjust(hspace=0.6)


# In[ ]:


plot_fig

