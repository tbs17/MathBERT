# =====For KC task prediction results conversion and evaluation================
def convert_test_result(orig_testPath,pred_testPath,org_dataPath,out_dir):
    from pathlib import Path
    import pandas as pd
    #read the original test data for the text and id
    df_test = pd.read_csv(orig_testPath, sep='\t',engine='python')
    df_test['guid']=df_test['guid'].astype(str)
    print(f'original test file has shape {df_test.shape}')
    #read the results data for the probabilities
    df_result = pd.read_csv(pred_testPath, sep='\t', header=None)
    print(f'predicted test file has shape {df_result.shape}')
    out_dir=Path(out_dir)
    Path.mkdir(out_dir,exist_ok=True)
    import numpy as np
    # df_map
    df_map_result = pd.DataFrame({'guid': df_test['guid'],
        'text': df_test['text'],
        'top1': df_result.idxmax(axis=1),
        'top1_probability':df_result.max(axis=1),
        'top2': df_result.columns[df_result.values.argsort(1)[:,-2]],
        'top2_probability':df_result.apply(lambda x: sorted(x)[-2],axis=1),
        'top3': df_result.columns[df_result.values.argsort(1)[:,-3]],
        'top3_probability':df_result.apply(lambda x: sorted(x)[-3],axis=1),
        'top4': df_result.columns[df_result.values.argsort(1)[:,-4]],
        'top4_probability':df_result.apply(lambda x: sorted(x)[-4],axis=1),
        'top5': df_result.columns[df_result.values.argsort(1)[:,-5]],
        'top5_probability':df_result.apply(lambda x: sorted(x)[-5],axis=1)
        })
    #view sample rows of the newly created dataframe
    df_map_result.head(10)
    df_map_result['top1']=df_map_result['top1'].astype(str)
    df_map_result['top2']=df_map_result['top2'].astype(str)
    df_map_result['top3']=df_map_result['top3'].astype(str)
    df_map_result.dtypes
    df_map_result['top4']=df_map_result['top4'].astype(str)
    df_map_result['top5']=df_map_result['top5'].astype(str)
    print(f'mapped test file has shape {df_map_result.shape}')
    data=pd.read_csv(org_dataPath,encoding='utf-8',names=['text','label'],header=0)
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    data['label_en']=le.fit_transform(data['label'])
    # data0=data.drop('Label',axis=1)
    data['Label']=le.inverse_transform(data['label_en'])
    key=pd.DataFrame({'code':data['label_en'].unique(),'Label':le.inverse_transform(data['label_en'].unique())})
    key.to_csv('further-pre-training/label-map.csv',index=False)

    key['code']=key.code.astype(str)
    label_map_dict=dict(key.to_dict(orient='split')['data'])
    marker=pred_testPath.split('/')[2].split('.')[0]
    df_map_result=df_map_result.replace({'top1':label_map_dict,'top2':label_map_dict,'top3':label_map_dict,'top4':label_map_dict,'top5':label_map_dict})
    df_map_result.to_csv('{}/{}_converted.csv'.format(out_dir,marker),index=False)
    print(df_map_result.shape)#(702, 12)
    return df_map_result


def match_top3_f1_acc(data_dir,label_Path,out_dir):
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import os
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    label_data=pd.read_csv(label_Path,names=['label'],header=0)
    print(f'test data shape{label_data.shape}')
    for i, file in enumerate(os.listdir(data_dir)):
    
        pred_dataPath=os.path.join(data_dir,file)
        print('======')
        print(i,pred_dataPath)
        
        if pred_dataPath.endswith('.csv'):

            pred_data=pd.read_csv(pred_dataPath,encoding='ISO-8859-1')
            print(f'predicted data shape: {pred_data.shape}')
            print('total {} classes are predicted as top class'.format(len(pred_data['top1'].unique())))
            match=pd.concat([pred_data,label_data],axis=1)
            print(f'merged data shape{match.shape}')
            correct_top1=match[match['top1']==match['label']].top1.unique()
            correct_top2=match[match['top2']==match['label']].top2.unique()
            correct_top3=match[match['top3']==match['label']].top3.unique()
            correct_all=list(set(list(correct_top1)+list(correct_top2)+list(correct_top3)))
            print('Correct top 1 label {}'.format(len(correct_top1)))
            print('Correct top 2 label {}'.format(len(correct_top2)))
            print('Correct top 3 label {}'.format(len(correct_top3)))
            print('Total top 3 Correct labels {}'.format(len(correct_all)))
            match['matched1']=np.where(match['top1']==match['label'],1,0)
            match['matched2']=np.where((match['top1']==match['label']) | (match['top2']==match['label']),1,0)
            match['matched3']=np.where((match['top1']==match['label']) | (match['top2']==match['label']) | (match['top3']==match['label']),1,0)
       
            marker=file.split('.')[0]
            from pathlib import Path
            out_dir=Path(out_dir)
            out_dir.mkdir(exist_ok=True)
            match.to_csv('{}/{}_matched.csv'.format(out_dir,marker),index=False)

            
            print('---Below is F1 Score(weighted)--')
            top1_f1=round(f1_score(match['label'], match['top1'], average='weighted')*100,3)
            print('Top1 label F1 score(weighted): {}%'.format(top1_f1))
            top2_f1=round(f1_score(match['label'], match['top2'], average='weighted')*100,3)
            print('Top1 label F1 score(weighted): {}%'.format(top2_f1+top1_f1))
            top3_f1=round(f1_score(match['label'], match['top3'], average='weighted')*100,3)
            print('Top1 label F1 score(weighted): {}%'.format(top3_f1+top2_f1+top1_f1))
            
            print ('---Below is Sklearn accuracy---')
            top1_accuracy=round(accuracy_score(match['label'], match['top1'])*100,3)
            print('Top1 label accuracy score: {}%'.format(top1_accuracy))
            top2_accuracy=round(accuracy_score(match['label'], match['top2'])*100,3)
            print('Top1 label accuracy score: {}%'.format(top2_accuracy+top1_accuracy))
            top3_accuracy=round(accuracy_score(match['label'], match['top3'])*100,3)
            print('Top1 label accuracy score: {}%'.format(top3_accuracy+top2_accuracy+top1_accuracy))


# example code to execute the functions

orig_testPath='TAPT/skill_code_prob_v2/CoLA/test.tsv'
pred_testPath='EVAL_RESULT/MathBERT/COLA_385_MathBERT_LR5E-5_BS64_MS512_EP25_customVocab_FIT_SEED4_problem-test_results.tsv'
org_dataPath='further-pre-training/CORPUS/ALL_GRADES/problem_text_all_grades_v2.csv'
out_dir='TEST_converted_PROB_V2_MATHBERT'
df_map_result=convert_test_result(orig_testPath,pred_testPath,org_dataPath,out_dir)
df_map_result.head()

data_dir='TEST_converted_PROB_V2_MATHBERT'
org_dataPath='TAPT/skill_code_prob_v2/CoLA/test_labels.csv'
out_dir='SKILL_CODE_PROB_V2_matched_MATHBERT-TAPT'
match_top3_f1_acc(data_dir,org_dataPath,out_dir)
# =====For auto-grading prediction results conversion and evaluation================

def convert_autoGrade(orig_testPath,pred_testPath,org_dataPath,out_dir):
    from pathlib import Path
    import pandas as pd
    #read the original test data for the text and id
    df_test = pd.read_csv(orig_testPath, sep='\t')#,engine='python'
    df_test['guid']=df_test.iloc[:,0].astype(str)
    print(f'original test file has shape {df_test.shape}')
    #read the results data for the probabilities
    df_result = pd.read_csv(pred_testPath, sep='\t', header=None)
    print(f'predicted test file has shape {df_result.shape}')
    out_dir=Path(out_dir)
    Path.mkdir(out_dir,exist_ok=True)
    import numpy as np
    # df_map
    df_map_result = pd.DataFrame({'guid': df_test['guid'],
        'question': df_test['question'],
        'answer': df_test['answer'],
        'top1': df_result.idxmax(axis=1),
        'top1_probability':df_result.max(axis=1),
        'top2': df_result.columns[df_result.values.argsort(1)[:,-2]],
        'top2_probability':df_result.apply(lambda x: sorted(x)[-2],axis=1),
        'top3': df_result.columns[df_result.values.argsort(1)[:,-3]],
        'top3_probability':df_result.apply(lambda x: sorted(x)[-3],axis=1),
        'top4': df_result.columns[df_result.values.argsort(1)[:,-4]],
        'top4_probability':df_result.apply(lambda x: sorted(x)[-4],axis=1),
        'top5': df_result.columns[df_result.values.argsort(1)[:,-5]],
        'top5_probability':df_result.apply(lambda x: sorted(x)[-5],axis=1)
        })
    #view sample rows of the newly created dataframe
#     display(df_map_result.head())
    df_map_result['top1']=df_map_result['top1'].astype(str)
    df_map_result['top2']=df_map_result['top2'].astype(str)
    df_map_result['top3']=df_map_result['top3'].astype(str)
    df_map_result.dtypes
    df_map_result['top4']=df_map_result['top4'].astype(str)
    df_map_result['top5']=df_map_result['top5'].astype(str)
    print(f'mapped test file has shape {df_map_result.shape}')
    label_map_dict={'0':1,'1':2,'2':3,'3':4,'4':5}
    marker=pred_testPath.split('/')[-1].split('.')[0]
    df_map_result=df_map_result.replace({'top1':label_map_dict,'top2':label_map_dict,'top3':label_map_dict,'top4':label_map_dict,'top5':label_map_dict})
    df_map_result.to_csv('{}/{}_converted.csv'.format(out_dir,marker),index=False)
    print(df_map_result.shape)#(702, 12)
    return df_map_result


def convert_auc_autoGrade(orig_testPath,pred_testDir,label_Path,org_dataPath,out_dir):
    from pathlib import Path
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import os
    #read the original test data for the text and id
    df_test = pd.read_csv(orig_testPath, sep='\t',engine='python')
    df_test['guid']=df_test.iloc[:,0].astype(str)
    print(f'original test file has shape {df_test.shape}')
    print('----')
    out_dir=Path(out_dir)
    Path.mkdir(out_dir,exist_ok=True)
    #read the results data for the probabilities
    for i, f in enumerate(os.listdir(pred_testDir)):
        if f.endswith('test_results.tsv'):
            print(i, f)
        
            df_result = pd.read_csv(os.path.join(pred_testDir,f), sep='\t', header=None)
            print(f'predicted test file has shape {df_result.shape}')
            label_data=pd.read_csv(label_Path,usecols=['label'])
            auc=round(roc_auc_score(label_data['label'], df_result,multi_class='ovo',average='weighted')*100,3)
            print(f'average auc for 5 classes is {auc} %!')
            print('----')

# example code to execute the functions
orig_testPath='mathBERT-downstreamTasks/auto_grade/test.tsv'
pred_testPath='EVAL_RESULT/MathBERT/AUTO-GRADE/auto_grade_MathBERT_LR2E-5_BS64_MS512_EP5_customVocab_FIT_seed5-v2_600k-test_results.tsv'
org_dataPath='mathBERT-downstreamTasks/auto_grade/valid_answers_clean_bert.csv'
out_dir='TEST_converted_autoGrade_MATHBERT'
df_map_result=convert_autoGrade(orig_testPath,pred_testPath,org_dataPath,out_dir)
df_map_result.head()

orig_testPath='mathBERT-downstreamTasks/auto_grade/test.tsv'
pred_testDir='EVAL_RESULT/MathBERT/AUTO-GRADE/'
org_dataPath='mathBERT-downstreamTasks/auto_grade/valid_answers_clean_bert.csv'
label_Path='mathBERT-downstreamTasks/auto_grade/test_labels.csv'
out_dir='TEST_converted_autoGrade_MATHBERT'
convert_auc_autoGrade(orig_testPath,pred_testDir,label_Path,org_dataPath,out_dir)

# =====For KT prediction results conversion and evaluation================
def convert_KT(orig_testPath,pred_testPath,org_dataPath,out_dir):
    from pathlib import Path
    import pandas as pd
    #read the original test data for the text and id
    df_test = pd.read_csv(orig_testPath, sep='\t')#,engine='python'
    df_test['guid']=df_test.iloc[:,0].astype(str)
    print(f'original test file has shape {df_test.shape}')
    #read the results data for the probabilities
    df_result = pd.read_csv(pred_testPath, sep='\t', header=None)
    print(f'predicted test file has shape {df_result.shape}')
    out_dir=Path(out_dir)
    Path.mkdir(out_dir,exist_ok=True)
    import numpy as np
    # df_map
    df_map_result = pd.DataFrame({'guid': df_test['guid'],
        'question': df_test['question'],
        'answer': df_test['answer'],
        'top1': df_result.idxmax(axis=1),
        'top1_probability':df_result.max(axis=1),
        'top2': df_result.columns[df_result.values.argsort(1)[:,-2]],
        'top2_probability':df_result.apply(lambda x: sorted(x)[-2],axis=1),
        
        })
    #view sample rows of the newly created dataframe
#     display(df_map_result.head())
    df_map_result['top1']=df_map_result['top1'].astype(str)
    df_map_result['top2']=df_map_result['top2'].astype(str)
    
    print(f'mapped test file has shape {df_map_result.shape}')

    label_map_dict={'0':0,'1':1}
    marker=pred_testPath.split('/')[-1].split('.')[0]
    df_map_result=df_map_result.replace({'top1':label_map_dict,'top2':label_map_dict})
    df_map_result.to_csv('{}/{}_converted.csv'.format(out_dir,marker),index=False)
    print(df_map_result.shape)#(702, 12)
    return df_map_result

def match_top3_f1_acc_KT(data_dir,label_Path,out_dir):
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import os
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from pathlib import Path
    label_data=pd.read_csv(label_Path,names=['guid','label'],header=0)
    print(f'test data shape{label_data.shape}')
    for i, file in enumerate(os.listdir(data_dir)):
    
        pred_dataPath=os.path.join(data_dir,file)
        print('======')
        print(i,pred_dataPath)
        

        if pred_dataPath.endswith('.csv'):

            pred_data=pd.read_csv(pred_dataPath,encoding='ISO-8859-1')
            print(f'predicted data shape: {pred_data.shape}')
            print('total {} classes are predicted as top class'.format(len(pred_data['top1'].unique())))
            match=pd.concat([pred_data,label_data],axis=1)
#             marker=file.split('.')[0]
#             out_dir=Path(out_dir)
#             out_dir.mkdir(exist_ok=True)
#             match.to_csv('{}/{}_matched.csv'.format(out_dir,marker),index=False)
            print(f'merged data shape{match.shape}')
#             display(match.head())
            correct_top1=match[match['top1']==match['label']].top1.unique()
            correct_top2=match[match['top2']==match['label']].top2.unique()
            
            correct_all=list(set(list(correct_top1)+list(correct_top2)))
            print('Correct top 1 label {}'.format(len(correct_top1)))
            print('Correct top 2 label {}'.format(len(correct_top2)))
           
            print('Total top 2 Correct labels {}'.format(len(correct_all)))
            match['matched1']=np.where(match['top1']==match['label'],1,0)
            match['matched2']=np.where((match['top1']==match['label']) | (match['top2']==match['label']),1,0)

       
            marker=file.split('.')[0]
            from pathlib import Path
            out_dir=Path(out_dir)
            out_dir.mkdir(exist_ok=True)
            match.to_csv('{}/{}_matched.csv'.format(out_dir,marker),index=False)

            
            print('---Below is F1 Score(binary)--')
            top1_f1=round(f1_score(match['label'], match['top1'])*100,3)
            print('Top1 label F1 score(binary): {}%'.format(top1_f1))

            
            print ('---Below is Sklearn accuracy---')
            top1_accuracy=round(accuracy_score(match['label'], match['top1'])*100,3)
            print('Top1 label accuracy score: {}%'.format(top1_accuracy))

def convert_auc_KT(orig_testPath,pred_testDir,label_Path,org_dataPath,out_dir):
    from pathlib import Path
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import os
    #read the original test data for the text and id
    df_test = pd.read_csv(orig_testPath, sep='\t',engine='python')
    df_test['guid']=df_test['Index'].astype(str)
    print(f'original test file has shape {df_test.shape}')
    print('----')
    out_dir=Path(out_dir)
    Path.mkdir(out_dir,exist_ok=True)
    #read the results data for the probabilities
    for i, f in enumerate(os.listdir(pred_testDir)):
        if f.endswith('test_results.tsv'):
            print(i, f)
        
            df_result = pd.read_csv(os.path.join(pred_testDir,f), sep='\t', header=None)
            print(f'predicted test file has shape {df_result.shape}')
            label_data=pd.read_csv(label_Path,names=['Index','label'],header=0)
            auc=round(roc_auc_score(label_data['label'], df_result.iloc[:,1])*100,3)
            print(f'average auc for 5 classes is {auc} %!')
            print('----')
# example code to execute the functions
orig_testPath='mathBERT-downstreamTasks/KT/test.tsv'
pred_testPath='EVAL_RESULT/MathBERT/KT/KT_MathBERT_LR5E-5_BS128_MS512_EP5_origVocab_FIT_seed2_600K-test_results.tsv'
org_dataPath='mathBERT-downstreamTasks/KT/valid_answers_clean_bert.csv'
out_dir='TEST_converted_KT_MATHBERT'
df_map_result=convert_KT(orig_testPath,pred_testPath,org_dataPath,out_dir)
df_map_result.head()

data_dir='TEST_converted_KT_MATHBERT'
label_Path='mathBERT-downstreamTasks/KT/test_labels.csv'
out_dir='KT_matched_MATHBERT'
match_top3_f1_acc_KT(data_dir,label_Path,out_dir)