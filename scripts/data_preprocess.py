

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
