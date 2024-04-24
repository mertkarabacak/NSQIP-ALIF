import os
HF_TOKEN = os.getenv("HF_TOKEN")

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from math import sqrt
from scipy import stats as st
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression

import shap
import gradio as gr
import random
import re
import textwrap
from datasets import load_dataset


#Read data training data.

x1 = load_dataset("mertkarabacak/NSQIP-ALIF", data_files="los_data_train.csv", use_auth_token = HF_TOKEN)
x1 = pd.DataFrame(x1['train'])
x1 = x1.iloc[:, 1:]

x2 = load_dataset("mertkarabacak/NSQIP-ALIF", data_files="discharge_data_train.csv", use_auth_token = HF_TOKEN)
x2 = pd.DataFrame(x2['train'])
x2 = x2.iloc[:, 1:]
x2 = x2.sample(n=1024, random_state=42)

x3 = load_dataset("mertkarabacak/NSQIP-ALIF", data_files="readmissions_data_train.csv", use_auth_token = HF_TOKEN)
x3 = pd.DataFrame(x3['train'])
x3 = x3.iloc[:, 1:]

x4 = load_dataset("mertkarabacak/NSQIP-ALIF", data_files="complications_data_train.csv", use_auth_token = HF_TOKEN)
x4 = pd.DataFrame(x4['train'])
x4 = x4.iloc[:, 1:]
x4 = x4.sample(n=1024, random_state=42)


#Read validation data.

x1_valid = load_dataset("mertkarabacak/NSQIP-ALIF", data_files="los_data_valid.csv", use_auth_token = HF_TOKEN)
x1_valid = pd.DataFrame(x1_valid['train'])
x1_valid = x1_valid.iloc[:, 1:]

x2_valid = load_dataset("mertkarabacak/NSQIP-ALIF", data_files="discharge_data_valid.csv", use_auth_token = HF_TOKEN)
x2_valid = pd.DataFrame(x2_valid['train'])
x2_valid = x2_valid.iloc[:, 1:]

x3_valid = load_dataset("mertkarabacak/NSQIP-ALIF", data_files="readmissions_data_valid.csv", use_auth_token = HF_TOKEN)
x3_valid = pd.DataFrame(x3_valid['train'])
x3_valid = x3_valid.iloc[:, 1:]

x4_valid = load_dataset("mertkarabacak/NSQIP-ALIF", data_files="complications_data_valid.csv", use_auth_token = HF_TOKEN)
x4_valid = pd.DataFrame(x4_valid['train'])
x4_valid = x4_valid.iloc[:, 1:]


#Define feature names.
f1_names = list(x1.columns)
f1_names = [f1.replace('__', ' - ') for f1 in f1_names]
f1_names = [f1.replace('_', ' ') for f1 in f1_names]

f2_names = list(x2.columns)
f2_names = [f2.replace('__', ' - ') for f2 in f2_names]
f2_names = [f2.replace('_', ' ') for f2 in f2_names]

f3_names = list(x3.columns)
f3_names = [f3.replace('__', ' - ') for f3 in f3_names]
f3_names = [f3.replace('_', ' ') for f3 in f3_names]

f4_names = list(x4.columns)
f4_names = [f4.replace('__', ' - ') for f4 in f4_names]
f4_names = [f4.replace('_', ' ') for f4 in f4_names]


#Prepare training data for the outcome 1 (prolonged LOS).
y1 = x1.pop('OUTCOME')

#Prepare validation data for the outcome 1 (prolonged LOS).
y1_valid = x1_valid.pop('OUTCOME')

#Prepare training data for the outcome 2 (non-home discharges).
y2 = x2.pop('OUTCOME')

#Prepare validation data for the outcome 2 (non-home discharges).
y2_valid = x2_valid.pop('OUTCOME')

#Prepare training data for the outcome 3 (30-day readmissions).
y3 = x3.pop('OUTCOME')

#Prepare validation data for the outcome 3 (30-day readmissions).
y3_valid = x3_valid.pop('OUTCOME')

#Prepare training data for the outcome 4 (unplanned reoperations).
y4 = x4.pop('OUTCOME')

#Prepare validation data for the outcome 4 (unplanned reoperations).
y4_valid = x4_valid.pop('OUTCOME')


#Assign hyperparameters.

y1_params =  {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 7.422806378061236e-06, 'lambda_l2': 1.8278682634415302e-08, 'num_leaves': 223, 'feature_fraction': 0.5747711884167779, 'bagging_fraction': 0.7629442040107172, 'bagging_freq': 5, 'min_child_samples': 26, 'metric': 'binary_logloss', 'verbosity': -1, 'random_state': 31}
y2_params =  {'objective': 'binary', 'boosting_type': 'gbdt', 'lambda_l1': 5.88769831138731e-06, 'lambda_l2': 2.9920914495963683e-07, 'num_leaves': 213, 'feature_fraction': 0.9553656216901187, 'bagging_fraction': 0.6727616196153284, 'bagging_freq': 5, 'min_child_samples': 25, 'metric': 'binary_logloss', 'verbosity': -1, 'random_state': 31}
y3_params =  {'criterion': 'entropy', 'max_features': None, 'max_depth': 28, 'n_estimators': 200, 'min_samples_leaf': 1, 'min_samples_split': 6, 'random_state': 31}
y4_params =  {'criterion': 'entropy', 'max_features': None, 'max_depth': 28, 'n_estimators': 200, 'min_samples_leaf': 1, 'min_samples_split': 6, 'random_state': 31}

#Training models.

from lightgbm import LGBMClassifier
lgb = LGBMClassifier(**y1_params)
y1_model = lgb

y1_model = y1_model.fit(x1, y1)
y1_explainer = shap.Explainer(y1_model.predict, x1)
y1_calib_probs = y1_model.predict_proba(x1_valid)
y1_calib_model = LogisticRegression()
y1_calib_model = y1_calib_model.fit(y1_calib_probs, y1_valid)


from lightgbm import LGBMClassifier
lgb = LGBMClassifier(**y2_params)
y2_model = lgb

y2_model = y2_model.fit(x2, y2)
y2_explainer = shap.Explainer(y2_model.predict, x2)
y2_calib_probs = y2_model.predict_proba(x2_valid)
y2_calib_model = LogisticRegression()
y2_calib_model = y2_calib_model.fit(y2_calib_probs, y2_valid)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(**y3_params)
y3_model = rf

y3_model = y3_model.fit(x3, y3)
y3_explainer = shap.Explainer(y3_model.predict, x3)
y3_calib_probs = y3_model.predict_proba(x3_valid)
y3_calib_model = LogisticRegression()
y3_calib_model = y3_calib_model.fit(y3_calib_probs, y3_valid)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(**y4_params)
y4_model = rf

y4_model = y4_model.fit(x4, y4)
y4_explainer = shap.Explainer(y4_model.predict, x4)
y4_calib_probs = y4_model.predict_proba(x4_valid)
y4_calib_model = LogisticRegression()
y4_calib_model = y4_calib_model.fit(y4_calib_probs, y4_valid)


output_y1 = (
    """          
        <br/>
        <center>The predicted risk of prolonged length of stay:</center>
        <br/>
        <center><h1>{:.2f}%</h1></center>
"""
)

output_y2 = (
    """          
        <br/>        
        <center>The predicted risk of non-home discharge:</center>
        <br/>        
        <center><h1>{:.2f}%</h1></center>
"""
)

output_y3 = (
    """          
        <br/>        
        <center>The predicted risk of 30-day readmission:</center>
        <br/>        
        <center><h1>{:.2f}%</h1></center>
"""
)

output_y4 = (
    """          
        <br/>        
        <center>The predicted risk of major complications:</center>
        <br/>        
        <center><h1>{:.2f}%</h1></center>
"""
)


#Define predict for y1.
def y1_predict(*args):
    df1 = pd.DataFrame([args], columns=x1.columns)
    pos_pred = y1_model.predict_proba(df1)
    pos_pred = y1_calib_model.predict_proba(pos_pred)
    prob = pos_pred[0][1]
    output = output_y1.format(prob * 100)
    return output

#Define predict for y2.
def y2_predict(*args):
    df2 = pd.DataFrame([args], columns=x2.columns)
    pos_pred = y2_model.predict_proba(df2)
    pos_pred = y2_calib_model.predict_proba(pos_pred)        
    prob = pos_pred[0][1]
    output = output_y2.format(prob * 100)
    return output

#Define predict for y3.
def y3_predict(*args):
    df3 = pd.DataFrame([args], columns=x3.columns)
    pos_pred = y3_model.predict_proba(df3)
    pos_pred = y3_calib_model.predict_proba(pos_pred)            
    prob = pos_pred[0][1]
    output = output_y3.format(prob * 100)
    return output

#Define predict for y4.
def y4_predict(*args):
    df4 = pd.DataFrame([args], columns=x4.columns)
    pos_pred = y4_model.predict_proba(df4)
    pos_pred = y4_calib_model.predict_proba(pos_pred)            
    prob = pos_pred[0][1]
    output = output_y4.format(prob * 100)
    return output


#Define function for wrapping feature labels.
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)
    

#Define interpret for y1.
def y1_interpret(*args):
    df1 = pd.DataFrame([args], columns=x1.columns)
    shap_values1 = y1_explainer(df1).values
    shap_values1 = np.abs(shap_values1)
    shap.bar_plot(shap_values1[0], max_display = 10, show = False, feature_names = f1_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

#Define interpret for y2.
def y2_interpret(*args):
    df2 = pd.DataFrame([args], columns=x2.columns)
    shap_values2 = y2_explainer(df2).values
    shap_values2 = np.abs(shap_values2)
    shap.bar_plot(shap_values2[0], max_display = 10, show = False, feature_names = f2_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

#Define interpret for y3.
def y3_interpret(*args):
    df3 = pd.DataFrame([args], columns=x3.columns)
    shap_values3 = y3_explainer(df3).values
    shap_values3 = np.abs(shap_values3)
    shap.bar_plot(shap_values3[0], max_display = 10, show = False, feature_names = f3_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig

#Define interpret for y4.
def y4_interpret(*args):
    df4 = pd.DataFrame([args], columns=x4.columns)
    shap_values4 = y4_explainer(df4).values
    shap_values4 = np.abs(shap_values4)
    shap.bar_plot(shap_values4[0], max_display = 10, show = False, feature_names = f4_names)
    fig = plt.gcf()
    ax = plt.gca()
    wrap_labels(ax, 20)
    ax.figure
    plt.tight_layout()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    plt.xlabel("SHAP value (impact on model output)", fontsize =12, fontweight = 'heavy', labelpad = 8)
    plt.tick_params(axis="y",direction="out", labelsize = 12)
    plt.tick_params(axis="x",direction="out", labelsize = 12)
    return fig


with gr.Blocks(title = "NSQIP-ALIF") as demo:
        
    gr.Markdown(
        """
    <br/>
    <center><h2>NOT FOR CLINICAL USE</h2><center>    
    <br/>    
    <center><h1>ALIF Outcomes</h1></center>
    <center><h2>Prediction Tool</h2></center>
    <br/>
    <center><h3>This web application should not be used to guide any clinical decisions.</h3><center>
    <br/>
    <center><i>The publication describing the details of this prediction tool will be posted here upon the acceptance of publication.</i><center>
        """
    )

    gr.Markdown(
        """
        <center><h3>Model Performances</h3></center>
          <div style="text-align:center;">
          <table style="width:100%;">
          <tr>
            <th>Outcome</th>
            <th>Algorithm</th>
            <th>Weighted Precision</th>
            <th>Weighted Recall</th>
            <th>Weighted AUPRC</th>
            <th>Balanced Accuracy</th>
            <th>AUROC</th>
            <th>Brier Score</th>
          </tr>
          <tr>
            <td>Prolonged LOS</td>
            <td>LightGBM</td>
            <td>0.783 (0.763 - 0.803)</td>
            <td>0.812 (0.793 - 0.831)</td>
            <td>0.423 (0.399 - 0.447)</td>
            <td>0.609 (0.586 - 0.632)</td>
            <td>0.735 (0.694 - 0.755)</td>
            <td>0.139 (0.122 - 0.156)</td>             
          </tr>
          <tr>
            <td>Non-home Discharges</td>
            <td>LightGBM</td>
            <td>0.863 (0.846 - 0.880)</td>
            <td>0.884 (0.869 - 0.889)</td>
            <td>0.377 (0.354 - 0.400)</td>
            <td>0.613 (0.590 - 0.636)</td>
            <td>0.814 (0.734 - 0.806)</td>
            <td>0.085 (0.072 - 0.098)</td>             
          </tr>
          <tr>
            <td>30-Day Readmissions</td>
            <td>Random Forest</td>
            <td>0.919 (0.906 - 0.932)</td>
            <td>0.907 (0.893 - 0.921)</td>
            <td>0.121 (0.105 - 0.137)</td>
            <td>0.555 (0.531 - 0.579)</td>
            <td>0.707 (0.589 - 0.710)</td>
            <td>0.044 (0.034 - 0.044)</td>             
          </tr>                        
          <tr>
            <td>Major Complications</td>
            <td>Random Forest</td>
            <td>0.909 (0.895 - 0.923)</td>
            <td>0.864 (0.848 - 0.880)</td>
            <td>0.123 (0.107 - 0.139)</td>
            <td>0.585 (0.561 - 0.609)</td>
            <td>0.701 (0.634 - 0.741)</td>
            <td>0.051 (0.040 - 0.062)</td>             
          </tr>        
        </table>
        </div>
        """
    )    

    with gr.Row():

        with gr.Column():

            Age = gr.Slider(label="Age", minimum = 18, maximum = 99, step = 1, value = 55)

            Sex = gr.Radio(label = "Sex", choices = ['Male', 'Female'], type = 'index', value = 'Male')
            
            Race = gr.Radio(label = "Race", choices = ['White', 'Black or African American', 'Asian', 'American Indian or Alaska Native', 'Native Hawaiian or Pacific Islander', 'Other/Unknown'], type = 'index', value = 'White')
            
            Hispanic_Ethnicity = gr.Radio(label = "Hispanic Ethnicity", choices = ['No', 'Yes', 'Unknown'], type = 'index', value = 'No')
            
            Height = gr.Slider(label = "Height (in meters)", minimum = 1, maximum = 2.5, step = 0.01, value = 1.75)
            
            Weight = gr.Slider(label = "Weight (in kilograms)", minimum = 20, maximum = 200, step = 1, value = 75)            
            
            Transfer_Status = gr.Radio(label = "Transfer Status", choices = ['Not transferred', 'Transferred', 'Unknown'], type = 'index', value = 'Not transferred')
            
            Current_Smoker_Status = gr.Radio(label = "Current Smoker Status", choices = ['No', 'Yes', 'Unknown'], type = 'index', value = 'No')   
            
            Diabetes_Mellitus_Requiring_Therapy = gr.Radio(label = "Diabetes Mellitus Requiring Therapy", choices = ['No', 'Yes',], type = 'index', value = 'No')   

            Dyspnea = gr.Radio(label = "Dyspnea", choices = ['No', 'Yes'], type = 'index', value = 'No')   

            Ventilator_Dependency = gr.Radio(label = "Ventilator Dependency", choices = ['No', 'Yes'], type = 'index', value = 'No')   
            
            History_of_Severe_COPD = gr.Radio(label = "History of Severe COPD", choices = ['No', 'Yes'], type = 'index', value = 'No')             

            Ascites_within_30_Days_Prior_to_Surgery = gr.Radio(label = "Ascites within 30 Days Prior to Surgery", choices = ['No', 'Yes'], type = 'index', value = 'No')   

            CHF_within_30_Days_Prior_to_Surgery = gr.Radio(label = "CHF within 30 Days Prior to Surgery", choices = ['No', 'Yes'], type = 'index', value = 'No')   

            Hypertension_Requiring_Medication = gr.Radio(label = "Hypertension Requiring Medication", choices = ['No', 'Yes'], type = 'index', value = 'No')

            Acute_Renal_Failure = gr.Radio(label = "Acute Renal Failure", choices = ['No', 'Yes'], type = 'index', value = 'No')   

            Currently_Requiring_or_on_Dialysis = gr.Radio(label = "Currently Requiring or on Dialysis", choices = ['No', 'Yes'], type = 'index', value = 'No')
            
            Disseminated_Cancer = gr.Radio(label = "Disseminated Cancer", choices = ['No', 'Yes'], type = 'index', value = 'No')   

            Open_Wound = gr.Radio(label = "Open Wound", choices = ['No', 'Yes'], type = 'index', value = 'No')   

            Steroid_or_Immunosuppressant_for_a_Chronic_Condition = gr.Radio(label = "Steroid/Immunosuppressant for a Chronic Condition", choices = ['No', 'Yes'], type = 'index', value = 'No')   
            
            Malnourishment = gr.Radio(label = "Malnourishment", choices = ['No', 'Yes'], type = 'index', value = 'No')   
            
            Bleeding_Disorder = gr.Radio(label = "Bleeding Disorder", choices = ['No', 'Yes'], type = 'index', value = 'No')              
            
            RBC_Transfusion_within_72_Hours_Prior_to_Surgery = gr.Radio(label = "RBC Transfusion within 72 Hours Prior to Surgery", choices = ['No', 'Yes'], type = 'index', value = 'No')   
                
            Functional_Status = gr.Radio(label = "Functional_Status", choices = ['Independent', 'Partially Dependent', 'Totally Dependent', 'Unknown'], type = 'index', value = 'Independent')   
            
            Preoperative_Serum_Sodium = gr.Slider(label="Preoperative Serum Sodium", minimum = 110, maximum = 150, step = 1, value = 135)
            
            Preoperative_Serum_BUN = gr.Slider(label="Preoperative Serum BUN", minimum = 0, maximum = 100, step = 1, value = 15)
            
            Preoperative_Serum_Creatinine = gr.Slider(label="Preoperative Serum Creatinine", minimum = 0, maximum = 20, step = 0.1, value = 0.9)
            
            Preoperative_WBC_Count = gr.Slider(label="Preoperative WBC Count (x1000)", minimum = 1, maximum = 50, step = 0.1, value = 5)
            
            Preoperative_Hematocrit = gr.Slider(label="Preoperative Hematocrit", minimum = 20, maximum = 70, step = 0.1, value = 45)
            
            Preoperative_Platelet_Count = gr.Slider(label="Preoperative Platelet Count (x1000)", minimum = 50, maximum = 1000, step = 1, value = 250)
            
            ASA_Classification = gr.Radio(label = "ASA Classification", choices = ['1-No Disturb', '2-Mild Disturb', '3-Severe Disturb',], type = 'index', value = '1-No Disturb')
            
            Surgical_Specialty = gr.Radio(label = "Surgical Specialty", choices = ['Neurosurgery', 'Orthopedic Surgery',], type = 'index', value = 'Neurosurgery')   
            
            Single_or_Multiple_Level_Surgery = gr.Radio(label = "Single or Multiple Level Surgery", choices = ['Single', 'Multiple',], type = 'index', value = 'Single')
                        
        with gr.Column():
            
            with gr.Box():
                
                gr.Markdown(
                    """
                    <center> <h2>Prolonged Length of Stay</h2> </center>
                    <br/>
                    <center> This model uses the LightGBM algorithm.</center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y1_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label1 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y1_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot1 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
            with gr.Box():
                gr.Markdown(
                    """
                    <center> <h2>Non-home Discharges</h2> </center>
                    <br/>
                    <center> This model uses the LightGBM algorithm.</center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y2_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label2 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y2_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot2 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
            with gr.Box():
                
                gr.Markdown(
                    """
                    <center> <h2>30-day Readmissions</h2> </center>
                    <br/>
                    <center> This model uses the Random Forest algorithm.</center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y3_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label3 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y3_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot3 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )  

            with gr.Box():
                
                gr.Markdown(
                    """
                    <center> <h2>Major Complications</h2> </center>
                    <br/>
                    <center> This model uses the Random Forest algorithm.</center>
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y4_predict_btn = gr.Button(value="Predict")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                label4 = gr.Markdown()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                with gr.Row():
                    y4_interpret_btn = gr.Button(value="Explain")
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )
                
                plot4 = gr.Plot()
                
                gr.Markdown(
                    """
                    <br/>
                    """
                    )  

           
                y1_predict_btn.click(
                    y1_predict,
                    inputs = [Sex, Race, Hispanic_Ethnicity, Transfer_Status, Age, Surgical_Specialty, Height, Weight, Diabetes_Mellitus_Requiring_Therapy, Current_Smoker_Status, Dyspnea, Functional_Status, Ventilator_Dependency, History_of_Severe_COPD, Ascites_within_30_Days_Prior_to_Surgery, CHF_within_30_Days_Prior_to_Surgery, Hypertension_Requiring_Medication, Acute_Renal_Failure, Currently_Requiring_or_on_Dialysis, Disseminated_Cancer, Open_Wound, Steroid_or_Immunosuppressant_for_a_Chronic_Condition, Malnourishment, Bleeding_Disorder, RBC_Transfusion_within_72_Hours_Prior_to_Surgery, Preoperative_Serum_Sodium, Preoperative_Serum_BUN, Preoperative_Serum_Creatinine, Preoperative_WBC_Count, Preoperative_Hematocrit, Preoperative_Platelet_Count, ASA_Classification, Single_or_Multiple_Level_Surgery],
                    outputs = [label1]
                )

                y2_predict_btn.click(
                    y2_predict,
                    inputs = [Sex, Race, Hispanic_Ethnicity, Transfer_Status, Age, Surgical_Specialty, Height, Weight, Diabetes_Mellitus_Requiring_Therapy, Current_Smoker_Status, Dyspnea, Functional_Status, Ventilator_Dependency, History_of_Severe_COPD, Ascites_within_30_Days_Prior_to_Surgery, CHF_within_30_Days_Prior_to_Surgery, Hypertension_Requiring_Medication, Acute_Renal_Failure, Currently_Requiring_or_on_Dialysis, Disseminated_Cancer, Open_Wound, Steroid_or_Immunosuppressant_for_a_Chronic_Condition, Malnourishment, Bleeding_Disorder, RBC_Transfusion_within_72_Hours_Prior_to_Surgery, Preoperative_Serum_Sodium, Preoperative_Serum_BUN, Preoperative_Serum_Creatinine, Preoperative_WBC_Count, Preoperative_Hematocrit, Preoperative_Platelet_Count, ASA_Classification, Single_or_Multiple_Level_Surgery],
                    outputs = [label2]
                )
                
                y3_predict_btn.click(
                    y3_predict,
                    inputs = [Sex, Race, Hispanic_Ethnicity, Transfer_Status, Age, Surgical_Specialty, Height, Weight, Diabetes_Mellitus_Requiring_Therapy, Current_Smoker_Status, Dyspnea, Functional_Status, Ventilator_Dependency, History_of_Severe_COPD, Ascites_within_30_Days_Prior_to_Surgery, CHF_within_30_Days_Prior_to_Surgery, Hypertension_Requiring_Medication, Acute_Renal_Failure, Currently_Requiring_or_on_Dialysis, Disseminated_Cancer, Open_Wound, Steroid_or_Immunosuppressant_for_a_Chronic_Condition, Malnourishment, Bleeding_Disorder, RBC_Transfusion_within_72_Hours_Prior_to_Surgery, Preoperative_Serum_Sodium, Preoperative_Serum_BUN, Preoperative_Serum_Creatinine, Preoperative_WBC_Count, Preoperative_Hematocrit, Preoperative_Platelet_Count, ASA_Classification, Single_or_Multiple_Level_Surgery],
                    outputs = [label3]
                )

                y4_predict_btn.click(
                    y4_predict,
                    inputs = [Sex, Race, Hispanic_Ethnicity, Transfer_Status, Age, Surgical_Specialty, Height, Weight, Diabetes_Mellitus_Requiring_Therapy, Current_Smoker_Status, Dyspnea, Functional_Status, Ventilator_Dependency, History_of_Severe_COPD, Ascites_within_30_Days_Prior_to_Surgery, CHF_within_30_Days_Prior_to_Surgery, Hypertension_Requiring_Medication, Acute_Renal_Failure, Currently_Requiring_or_on_Dialysis, Disseminated_Cancer, Open_Wound, Steroid_or_Immunosuppressant_for_a_Chronic_Condition, Malnourishment, Bleeding_Disorder, RBC_Transfusion_within_72_Hours_Prior_to_Surgery, Preoperative_Serum_Sodium, Preoperative_Serum_BUN, Preoperative_Serum_Creatinine, Preoperative_WBC_Count, Preoperative_Hematocrit, Preoperative_Platelet_Count, ASA_Classification, Single_or_Multiple_Level_Surgery],
                    outputs = [label4]
                )

                y1_interpret_btn.click(
                    y1_interpret,
                    inputs = [Sex, Race, Hispanic_Ethnicity, Transfer_Status, Age, Surgical_Specialty, Height, Weight, Diabetes_Mellitus_Requiring_Therapy, Current_Smoker_Status, Dyspnea, Functional_Status, Ventilator_Dependency, History_of_Severe_COPD, Ascites_within_30_Days_Prior_to_Surgery, CHF_within_30_Days_Prior_to_Surgery, Hypertension_Requiring_Medication, Acute_Renal_Failure, Currently_Requiring_or_on_Dialysis, Disseminated_Cancer, Open_Wound, Steroid_or_Immunosuppressant_for_a_Chronic_Condition, Malnourishment, Bleeding_Disorder, RBC_Transfusion_within_72_Hours_Prior_to_Surgery, Preoperative_Serum_Sodium, Preoperative_Serum_BUN, Preoperative_Serum_Creatinine, Preoperative_WBC_Count, Preoperative_Hematocrit, Preoperative_Platelet_Count, ASA_Classification, Single_or_Multiple_Level_Surgery],
                    outputs = [plot1],
                )
                
                y2_interpret_btn.click(
                    y2_interpret,
                    inputs = [Sex, Race, Hispanic_Ethnicity, Transfer_Status, Age, Surgical_Specialty, Height, Weight, Diabetes_Mellitus_Requiring_Therapy, Current_Smoker_Status, Dyspnea, Functional_Status, Ventilator_Dependency, History_of_Severe_COPD, Ascites_within_30_Days_Prior_to_Surgery, CHF_within_30_Days_Prior_to_Surgery, Hypertension_Requiring_Medication, Acute_Renal_Failure, Currently_Requiring_or_on_Dialysis, Disseminated_Cancer, Open_Wound, Steroid_or_Immunosuppressant_for_a_Chronic_Condition, Malnourishment, Bleeding_Disorder, RBC_Transfusion_within_72_Hours_Prior_to_Surgery, Preoperative_Serum_Sodium, Preoperative_Serum_BUN, Preoperative_Serum_Creatinine, Preoperative_WBC_Count, Preoperative_Hematocrit, Preoperative_Platelet_Count, ASA_Classification, Single_or_Multiple_Level_Surgery],
                    outputs = [plot2],
                )

                y3_interpret_btn.click(
                    y3_interpret,
                    inputs = [Sex, Race, Hispanic_Ethnicity, Transfer_Status, Age, Surgical_Specialty, Height, Weight, Diabetes_Mellitus_Requiring_Therapy, Current_Smoker_Status, Dyspnea, Functional_Status, Ventilator_Dependency, History_of_Severe_COPD, Ascites_within_30_Days_Prior_to_Surgery, CHF_within_30_Days_Prior_to_Surgery, Hypertension_Requiring_Medication, Acute_Renal_Failure, Currently_Requiring_or_on_Dialysis, Disseminated_Cancer, Open_Wound, Steroid_or_Immunosuppressant_for_a_Chronic_Condition, Malnourishment, Bleeding_Disorder, RBC_Transfusion_within_72_Hours_Prior_to_Surgery, Preoperative_Serum_Sodium, Preoperative_Serum_BUN, Preoperative_Serum_Creatinine, Preoperative_WBC_Count, Preoperative_Hematocrit, Preoperative_Platelet_Count, ASA_Classification, Single_or_Multiple_Level_Surgery],
                  outputs = [plot3],
                )

                y4_interpret_btn.click(
                    y4_interpret,
                    inputs = [Sex, Race, Hispanic_Ethnicity, Transfer_Status, Age, Surgical_Specialty, Height, Weight, Diabetes_Mellitus_Requiring_Therapy, Current_Smoker_Status, Dyspnea, Functional_Status, Ventilator_Dependency, History_of_Severe_COPD, Ascites_within_30_Days_Prior_to_Surgery, CHF_within_30_Days_Prior_to_Surgery, Hypertension_Requiring_Medication, Acute_Renal_Failure, Currently_Requiring_or_on_Dialysis, Disseminated_Cancer, Open_Wound, Steroid_or_Immunosuppressant_for_a_Chronic_Condition, Malnourishment, Bleeding_Disorder, RBC_Transfusion_within_72_Hours_Prior_to_Surgery, Preoperative_Serum_Sodium, Preoperative_Serum_BUN, Preoperative_Serum_Creatinine, Preoperative_WBC_Count, Preoperative_Hematocrit, Preoperative_Platelet_Count, ASA_Classification, Single_or_Multiple_Level_Surgery],
                  outputs = [plot4],
                )
                
    gr.Markdown(
                """    
                <center><h2>Disclaimer</h2>
                <center> 
                The American College of Surgeons National Surgical Quality Improvement Program and the hospitals participating in the ACS NSQIP are the source of the data used herein; they have not been verified and are not responsible for the statistical validity of the data analysis or the conclusions derived by the authors. The predictive tool located on this web page is for general health information only. This prediction tool should not be used in place of professional medical service for any disease or concern. Users of the prediction tool shouldn't base their decisions about their own health issues on the information presented here. You should ask any questions to your own doctor or another healthcare professional. The authors of the study mentioned above make no guarantees or representations, either express or implied, as to the completeness, timeliness, comparative or contentious nature, or utility of any information contained in or referred to in this prediction tool. The risk associated with using this prediction tool or the information in this predictive tool is not at all assumed by the authors. The information contained in the prediction tools may be outdated, not complete, or incorrect because health-related information is subject to frequent change and multiple confounders. No express or implied doctor-patient relationship is established by using the prediction tool. The prediction tools on this website are not validated by the authors. Users of the tool are not contacted by the authors, who also do not record any specific information about them. You are hereby advised to seek the advice of a doctor or other qualified healthcare provider before making any decisions, acting, or refraining from acting in response to any healthcare problem or issue you may be experiencing at any time, now or in the future. By using the prediction tool, you acknowledge and agree that neither the authors nor any other party are or will be liable or otherwise responsible for any decisions you make, actions you take, or actions you choose not to take as a result of using any information presented here.
                <br/>
                <h4>By using this tool, you accept all of the above terms.<h4/>
                </center>
                """
    )                
                
demo.launch()