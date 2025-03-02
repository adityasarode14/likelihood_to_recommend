import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def model_evaluation_report(y,y_pred,y_pred_proba):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn import metrics
    from sklearn.metrics import classification_report
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
    from IPython.display import display, HTML
    model_eval_df = pd.DataFrame()
    y = y.squeeze()
    if y.shape[0] == y_pred.shape[0] == y_pred_proba.shape[0]:
        model_eval_df['y'] = y
        model_eval_df['y_pred'] = y_pred
        model_eval_df['y_pred_proba'] = y_pred_proba
        
      
        print('Confusion Matrix:\n')
        cf_matrix = pd.crosstab(model_eval_df['y'],model_eval_df['y_pred'])
        fig, ax = plt.subplots(figsize=(3,3))
        sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
        plt.show()
        
        print('ROC Curve:\n')
        fpr, tpr, _ = metrics.roc_curve(y,  y_pred_proba)       
        fig, ax = plt.subplots(figsize=(3,3))
        auc = metrics.roc_auc_score(y, y_pred_proba)
        plt.plot(fpr,tpr,label="AUC="+str(round(auc,4)))
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.show()
        print('Classification Report:\n')
        print(classification_report(y,y_pred))
        print('Precision Score for Positive Class :', round(precision_score(y,y_pred),4))   
        print('Recall Score for Positive Class:',round(recall_score(y,y_pred),4))
        print('f1 Score for Positive Class:',round(f1_score(y,y_pred),4))
        print('Accuracy Score for Positive Class:',round(accuracy_score(y,y_pred),4))
        print("==================================================================")
        
#         decile_df = model_eval_df[['y_pred_proba','y']].dropna()
#         _,bins = pd.qcut(decile_df['y_pred_proba'],10,retbins=True)
#         bins[0] -= 0.001
#         bins[-1] += 0.001
#         bins_labels = ['%d.(%0.2f,%0.2f]'%(9-x[0],x[1][0],x[1][1]) for x in enumerate(zip(bins[:-1],bins[1:]))]
#         bins_labels[0] = bins_labels[0].replace('(','[')
#         decile_df['Decile']=pd.cut(decile_df['y_pred_proba'],bins=bins,labels=bins_labels)
#         decile_df['Population']=1
#         decile_df['Zeros']=1-decile_df['y']
#         decile_df['Ones']=decile_df['y']
#         decile_summary_df=decile_df.groupby(['Decile'])[['Ones','Zeros','Population']].sum()
#         decile_summary_df=decile_summary_df.sort_index(ascending=False)
#         decile_summary_df = decile_summary_df.reset_index()
#         decile_summary_df['yRate']=decile_summary_df['Ones']/decile_summary_df['Population']
#         decile_summary_df['CumulativeyRate']=decile_summary_df['Ones'].cumsum()/decile_summary_df['Population'].cumsum()
#         decile_summary_df['ysCaptured']=decile_summary_df['Ones'].cumsum()/decile_summary_df['Ones'].sum()
#         decile_summary_df['Lift']=(decile_summary_df['Ones'].cumsum()/decile_summary_df['Population'].cumsum())/(decile_summary_df['Ones'].sum()/decile_summary_df['Population'].sum())

#         tot_row = decile_df.groupby(['Population'])[['Ones','Zeros','Population']].sum()
#         tot_row['yRate']=tot_row['Ones']/tot_row['Population']
#         tot_row['CumulativeyRate']=tot_row['Ones'].cumsum()/tot_row['Population'].cumsum()
#         tot_row['ysCaptured']=tot_row['Ones'].cumsum()/tot_row['Ones'].sum()
#         tot_row['Decile'] = 'Total' 
#         tot_row['Lift']= 1.0
#         decile_summary_df=pd.concat([decile_summary_df,tot_row],axis = 0)
        
        
#         print('Decile Analysis:\n')s
        # display(HTML(decile_summary_df.to_html()))
        

        
        
    
def iv_woe(df, target,print_values = False):
    
    ivDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    cols = df.columns
    bins = 10
    for ivars in cols[~cols.isin([target])]:
        if (df[ivars].dtype.kind in 'bifc') and (len(np.unique(df[ivars]))>10):
            binned_x = pd.qcut(df[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': df[target]})
        else:
            d0 = pd.DataFrame({'x': df[ivars], 'y': df[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        ivDF=pd.concat([ivDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)
        if print_values == True:
            print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
            print("WOE:\n")
            print(d)

    return ivDF, woeDF

def oneway_anova_results(df,feature_list,tgt_list):
    import statsmodels.api as sm 
    from statsmodels.formula.api import ols 
    df_new = df.copy()
    result_df = pd.DataFrame()
    bins = 5
    for feat_attr in feature_list:
        print(feat_attr)
        for tgt_attr in tgt_list:
            if (df[feat_attr].dtype.kind in 'bifc') and (len(np.unique(df[feat_attr]))>10):
                binned_x = pd.qcut(df[feat_attr], bins,  duplicates='drop')
                anova_df = pd.DataFrame({'x': binned_x, 'y': df[tgt_attr]})
            else:
                anova_df = pd.DataFrame({'x': df[feat_attr], 'y': df[tgt_attr]})
            formula = "y ~ C(x)"
            model = ols(formula, data=anova_df).fit() 
            result = sm.stats.anova_lm(model, type=2) 
            # print(result.F[0])
            dtat_lst = []
            dtat_lst.append(feat_attr )
            dtat_lst.append(tgt_attr )
            dtat_lst.append(result.F[0] )
            result_df_tmp = pd.DataFrame([dtat_lst],columns=['Feature Attribute','Target Attribute','F-Value'])
            # print(result_df_tmp)
            result_df = pd.concat([result_df,result_df_tmp],axis = 0)
    return result_df


def handle_outlier(dframe,column,method,percentile = None,trim = False):
    if method == "IQR":
        Q1 = dframe[column].quantile(0.25)
        Q3 = dframe[column].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + (1.5 * IQR)
        lower_limit = Q1 - (1.5 * IQR)
        if trim == True:
            dframe1 = np.where(dframe[column] > upper_limit, True,
                                 np.where(dframe[column] < lower_limit,True,False))
            dframe2 = dframe.loc[~dframe1]
            return dframe2
        elif trim == False:
            dframe[column] = np.where(dframe[column] > upper_limit, upper_limit,
                                 np.where(dframe[column] < lower_limit,lower_limit,dframe[column]))
            
    elif method == "Percentile":
        lower_percentile = dframe[column].quantile(round(1 - (percentile/100),3))
        upper_percentile = dframe[column].quantile(round(percentile/100,3)) 
        if trim == True:
            dframe1 = np.where(dframe[column] > upper_percentile, True,
                                 np.where(dframe[column] < lower_percentile,True,False))
            dframe2 = dframe.loc[~dframe1]
            return dframe2
            
        elif trim == False:
            dframe[column] = np.where(dframe[column] > upper_percentile, upper_percentile,
                                 np.where(dframe[column] < lower_percentile,lower_percentile,dframe[column]))  
            

            
def add_deciles(df,score_attribute,score_thresholds,decile_attribute_name,decile_category_name):
    df_new = df.copy()
    # score_thresholds = [0, 0.0499900000 , 0.1199800000 , 0.1999600000 , 0.2896000000 , 0.3898200000 , 0.4799800000 , 0.5597700000 , 0.6399800000 , 0.7195900000,1]

    decile_conditions = [(df_new[score_attribute].isna()), 
                         ((df_new[score_attribute]<score_thresholds[1])),
                         ((df_new[score_attribute]>=score_thresholds[1]) & (df_new[score_attribute]<score_thresholds[2])),
                         ((df_new[score_attribute]>=score_thresholds[2]) & (df_new[score_attribute]<score_thresholds[3])),
                         ((df_new[score_attribute]>=score_thresholds[3]) & (df_new[score_attribute]<score_thresholds[4])),
                         ((df_new[score_attribute]>=score_thresholds[4]) & (df_new[score_attribute]<score_thresholds[5])),
                         ((df_new[score_attribute]>=score_thresholds[5]) & (df_new[score_attribute]<score_thresholds[6])),
                         ((df_new[score_attribute]>=score_thresholds[6]) & (df_new[score_attribute]<score_thresholds[7])),
                         ((df_new[score_attribute]>=score_thresholds[7]) & (df_new[score_attribute]<score_thresholds[8])),
                         ((df_new[score_attribute]>=score_thresholds[8]) & (df_new[score_attribute]<score_thresholds[9])),
                         ((df_new[score_attribute]>=score_thresholds[9]))] 
    decile_values = [99,10,9,8,7,6,5,4,3,2,1]
    df_new[decile_attribute_name] = np.select(decile_conditions,decile_values)
    
    decile_group_conditions = [(df_new[decile_attribute_name]==99), 
                         (df_new[decile_attribute_name] <= 3),
                         (df_new[decile_attribute_name] > 3) & (df_new[decile_attribute_name] <= 7),
                         (df_new[decile_attribute_name] > 7) & (df_new[decile_attribute_name] <= 10)] 
    decile_group_values = ['NA','Top 3 Deciles','Middle 4 Deciles','Bottom 3 Deciles']
    df_new[decile_category_name] = np.select(decile_group_conditions,decile_group_values)
    return df_new


# def get_model_from_pickle(model_file):
#     file = tarfile.open('model.tar.gz')
#     file.extractall(".")
#     with open('model.pkl', 'rb') as file:
#         model = pickle.load(file)
#     return model
# model = get_model_from_pickle('model.pkl')


def apply_model_scores(df,model):
    df_new = df.copy()
    
    for feature in model.feature_names_in_:
        df_new[feature] = df_new[feature].fillna(0)
        
    if df_new.shape[0] > 0 :
        df_new[model_score_attribute] = model.predict_proba(df_new[model.feature_names_in_])[:,1]
        print(df_new[model_score_attribute].quantile(np.arange(0,1.1,0.1)))
    else:
        df_new[model_score_attribute] = np.nan
    return df_new

def add_intercept(df):
    df_new = df.copy()
    df_new['f_intercept'] = 1
    return df_new