import pandas as pd
from IPython import display
import os, time
# below is to calculate tokens
def cal_tokens(data_dir,out_dir=None):
    from nltk.tokenize import word_tokenize
    from pathlib import Path
    for i, file in enumerate(os.listdir(data_dir)):
        f=os.path.join(data_dir,file)
        print(f'{i}:{file} has size {round(os.path.getsize(f)/1024/1024,3)} MB')
        data=pd.read_csv(f,encoding='latin-1')
        data_tokens=data.iloc[:,0].apply(word_tokenize)
        data_corp=[]
        for i in data_tokens:
            for j in i:
                data_corp.append(j)
        data_corp_uni=np.unique(data_corp)
        print(f'{len(data_corp)} total tokens found\n'
             f'{len(data_corp_uni)} total unique tokens found')
    
    return data_corp,data_corp_uni
    
def cal_tokens2(data_dir,out_dir=None):
    from nltk.tokenize import word_tokenize
    from pathlib import Path
    for i, file in enumerate(os.listdir(data_dir)):
        f=os.path.join(data_dir,file)
        
        data=pd.read_csv(f,encoding='latin-1')
        print(f'{i}:{file} has shape {data.shape}')

        data['row_tokens']=data.iloc[:,0].apply(lambda x:len(word_tokenize(x)))
        print(data.sum(axis=0))

    
    return data


# below cleaning is for problem text which comes in with html tags and images. But the data is not shared in this repo due to provider's request.
def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
def tag_removal(data_path):

    data=pd.read_csv(data_path,encoding='latin-1')
    print('---df before removing html tags---')
    data.iloc[:,1]=data.iloc[:,1].str.strip()
    data.iloc[:,1]=list(map(remove_html_tags,data.iloc[:,1].astype(str)))
    data.iloc[:,1]=data.iloc[:,1].str.replace('&nbsp;',' ')
    data.iloc[:,1]=data.iloc[:,1].str.replace('&copy;',' ').str.replace('( ){2,10}',' ')
    data.iloc[:,1]=data.iloc[:,1].str.replace('[\r\n]{1,10}',' ')
    data.iloc[:,1]=data.iloc[:,1].replace('',np.nan)
    data=data.dropna()
    print('---df after removing html tags---')
    print(f'shape after removing empty rows{data.shape}')
    marker=data_path.split('.')[0]
    data.columns=['Index','question','answer','label']
    data.to_csv('{}_clean.csv'.format(marker),index=False)
    return data

def tag_removal_v2(data_path,text_col):
    import numpy as np

    data=pd.read_csv(data_path,encoding='latin-1')
#     print(f'{i}:{file} has size {round(os.path.getsize(f)/1024/1024,3)} MB and shape {data.shape}')
    print('---df before removing html tags---')
    display(data.tail())
    data[text_col+'_cleaned']=data[text_col].str.strip()
    data[text_col+'_cleaned']=list(map(remove_html_tags,data[text_col+'_cleaned']))
    data[text_col+'_cleaned']=data[text_col+'_cleaned'].str.replace('&nbsp;',' ')
    data[text_col+'_cleaned']=data[text_col+'_cleaned'].str.replace('&copy;',' ').str.replace('( ){2,10}',' ')
    data[text_col+'_cleaned']=data[text_col+'_cleaned'].str.replace('[\r\n]{1,10}',' ')
    data[text_col+'_cleaned']=data[text_col+'_cleaned'].replace('',np.nan)
    data=data.dropna()
    print('---df after removing html tags---')
    display(data.tail())
    print(f'shape after removing empty rows{data.shape}')
    marker=data_path.split('.')[0]
    data.to_csv('{}_clean.csv'.format(marker),index=False)
    return data

# =====Executing above code to clean auto-grade data===
valid_answer_clean=tag_removal_v2('mathBERT-downstream Tasks/auto_grade/valid_answer_texts.csv','problem_text')
# below is to split the data into train/dev/test into the ratio of 72:8:20 and output as .tsv files
# you will also need to convert the original .csv files (before splitting) to .txt files to create pre-training data.
def split_3data_label(org_path,out_dir):
    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split
    title_cc_code=pd.read_csv(org_path,encoding='utf-8',names=['text','label'],header=0)
    
    print(f'total sample is {title_cc_code.shape[0]}')
    df_train, df_test=train_test_split(title_cc_code,test_size=0.2,random_state=111)

    df_bert_train, df_bert_dev = train_test_split(df_train, test_size=0.1,random_state=111)
    #create new title_cc_codeframe for test title_cc_code

    #output tsv file, no header for train and dev
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_bert_train.to_csv('{}/train_with_label.csv'.format(out_dir), index=False)
    df_bert_dev.to_csv('{}/dev_with_label.csv'.format(out_dir),index=False)
    df_test.to_csv('{}/test_with_label.csv'.format(out_dir), index=False)
    print(f'training samples are {df_bert_train.shape[0]}\n'
        f'eval samples are {df_bert_dev.shape[0]}\n'
        f'testing samples are {df_test.shape[0]}'
        )
#     print('{} unique labels'.format(title_cc_code0['label_en'].nunique()))
    return df_bert_train,df_bert_dev,df_test
