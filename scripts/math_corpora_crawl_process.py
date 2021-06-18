import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io
import pandas as pd
import numpy as np
import os, re, json, csv
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import smart_open
import gensim
from node2vec import Node2Vec
import networkx as nx
from nltk.tokenize import word_tokenize

import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from string import ascii_lowercase
# -------crawl for utah math curriculum---------

def utah_math(url_list,folder_location):
    for url in url_list:
        if not os.path.exists(folder_location):os.mkdir(folder_location)

        response = requests.get(url)
        soup= BeautifulSoup(response.text, "html.parser")
        marker='middle-school-math-'+url.split('/')[-2]
        print(marker)
        for i, link in enumerate(soup.select("a[href]")):
        #     if i<15:
            if marker in link['href']:
                print(link['href'])

                soup2=BeautifulSoup(requests.get(link['href']).text,'html.parser')
        #             if soup2.a['class']=="dropdown-item":
                for l in soup2.select("a[class$='dropdown-item']"):
                    print(l['href'])

        #Name the pdf files using the last portion of each link which are unique in this case
                    filename = os.path.join(folder_location,l['href'].split('/')[-1])
                    print(filename)
                    with open(filename, 'wb') as f:
                        f.write(requests.get(l['href']).content)
    f.close()


# to execute
url_list=["http://utahmiddleschoolmath.org/6th-grade/student-materials.shtml","http://utahmiddleschoolmath.org/7th-grade/student-materials.shtml","http://utahmiddleschoolmath.org/8th-grade/student-materials.shtml"]
folder_location = r'../utah_math/'
utah_math(url_list,folder_location)

# ----crawl for engageNY curriculum---
def engage_NY(url, folder_location, keywords):

    if not os.path.exists(folder_location):os.mkdir(folder_location)

    for i, k in enumerate(keywords):
        if i>=0:
            link1=urljoin(url,k)
    #         print(link1)
            response = requests.get(link1)
            soup= BeautifulSoup(response.text, "html.parser")  
            for i, link in enumerate(soup.select("a[href]")):
                link_list=[]
                if 'module' in link['href']:
                    link_list.append(link['href'])
                for l in link_list:
                    join2=urljoin(url,l)
                    res2=requests.get(join2)
                    soup2= BeautifulSoup(res2.text, "html.parser")
                    for i, link in enumerate(soup2.select("a[href]")):
                        if '.pdf' in link['href']:
                            join3=urljoin(url,link['href'])
                            print(join3)
                            marker=link['href'].split('?')[0].split('/')[-1]
                            filename = os.path.join(folder_location,marker)
                            print(filename)
                            with open(filename, 'wb') as f:
                                f.write(requests.get(join3).content)
                                f.close()

# to execute                         
url = "https://www.engageny.org/"
folder_location = r'../engageNY/'
keywords=['resource/grade-1-mathematics','resource/grade-2-mathematics','resource/grade-3-mathematics',
          'resource/grade-4-mathematics','resource/grade-5-mathematics','resource/grade-6-mathematics',
          'resource/grade-7-mathematics','resource/kindergarten-mathematics','content/prekindergarten-mathematics',
          'resource/high-school-algebra-i','resource/high-school-algebra-ii','resource/high-school-geometry',
          'content/precalculus-and-advanced-topics','resource/grade-8-mathematics']
engage_NY(url, folder_location, keywords)


# ----crawl math books-----
def crawl_books(url,folder_location, booknames):
    def remove(value):
        deletechars='\/:*?"<>|'
        for c in deletechars:
            value = value.replace(c,'')
        return value

    if not os.path.exists(folder_location):os.mkdir(folder_location)

    response = requests.get(url)
    soup= BeautifulSoup(response.text, "html.parser")

    link_list=[]
    pdf_list=[]
    non_pdf=[]
    m=0
    for i, link in enumerate(soup.select("a[href]")):
    # link here is the book name level link
        try:
            if any(word in link.text for word in booknames):
                m+=1
                book_name=marker='_'.join(link.text.split(' '))
                print(f'{m}:Book name is {book_name}!')
                link_list.append(link['href'])

                if any(word in link['href'] for word in ['.PDF','pdf']):
                    print('pdf book:',link['href'])
                    marker='_'.join(link.text.split(' '))+'.pdf'
                    pdf_location = r'../FREE_MATH_BOOK/more/pdf'
                    if not os.path.exists(pdf_location):os.mkdir(pdf_location)
                    filename = os.path.join(pdf_location,marker)
                    pdf_list.append(marker)
                    print(filename)
    #                 with open(filename, 'wb') as f:
    #                     f.write(requests.get(link['href']).content)
    #                     f.close()
                else:
                    print('non-direct pdf',link['href'])

                    book_name=marker='_'.join(link.text.split(' '))
                    book_name=remove(book_name)
    #                 make a book name folder
                    folder_location = r'../FREE_MATH_BOOK/more/non_pdf'
                    if not os.path.exists(folder_location):os.mkdir(folder_location)

                    non_pdf.append(book_name)
                    non_pdf_direct=[]
                    res2=requests.get(link['href'])
                    soup2=BeautifulSoup(res2.text,'html.parser')
                    chapter_list=[]
    #                 link2 is the chapter level link
                    for i, link2 in enumerate(soup2.select('a[href]')):
                        non_pdf_direct.append(book_name)

                        if '.pdf' in link2['href']:

                            marker=link2['href'].split('/')[-1]
                            book_dir=os.path.join(folder_location,book_name)
                            if not os.path.exists(book_dir):os.mkdir(book_dir)
                            filename = os.path.join(book_dir,marker)
                            print(f'Save file at {filename}!')
    #                         join=urljoin(link,c)
    #                         print(f'download link{join}')

                            with open(filename, 'wb') as f:
                                f.write(requests.get(urljoin(link['href'],link2['href'])).content)
                                f.close()
                    print('\n')
        except:
            pass

    print(f'We extracted {len(pdf_list)} direct pdf books and {len(non_pdf)} total non-direct pdf books and {len(non_pdf_direct)} books have chapters!')                   
    print(f'We extracted {len(link_list)} books!')
# to execute
url="https://www.openculture.com/free-math-textbooks"

folder_location = r'../FREE_MATH_BOOK/more/'
fileObject=open('booknames.json','r')
jsonContent=fileObject.read()
booknames=json.loads(jsonContent)
crawl_books(url,folder_location, booknames)

# ----crawl mooc and arXiv paper abstract-----

# ----convert pdf to text-----
def pdfparser(in_path,out_path):

    fp = open(in_path, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec ='utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr,  laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data =  retstr.getvalue()
        with open(out_path,'w',encoding=codec) as f:
            f.write(data)
            
            f.close()
    return data


def pdfparser_multi(data_dir,out_path):
    tokens=0
    for i, f in enumerate(os.listdir(data_dir)):

        if os.path.getsize(os.path.join(data_dir,f))!=0 and f.endswith('.pdf'):
#             print(i,f)
            in_path=os.path.join(data_dir,f)
            fp = open(in_path, 'rb')
            rsrcmgr = PDFResourceManager()
            retstr = io.StringIO()
            codec ='utf-8'
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr,  laparams=laparams)
            # Create a PDF interpreter object.
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            # Process each page contained in the document.

            for page in PDFPage.get_pages(fp):
                interpreter.process_page(page)
                data =retstr.getvalue()
            token=word_tokenize(data)
            print(f'file {i}:{f}: {len(token)} tokens!')
            tokens+=len(token)

            with open(out_path,'a+',encoding=codec) as f:
                    f.write(data)
                    f.close()
        else:
            pass
    print(f'===Total token has {tokens}!===')
    return data, tokens

    # TO EXECUTE
    
data_dir='../FREE_MATH_BOOK/non_pdf/Basic_Concepts_of_Mathematics'
out_path='../FREE_MATH_BOOK_CVT/Basic_Concepts_of_Mathematics.txt'
data,token=pdfparser_multi(data_dir,out_path)


# -----cut on tokens for more efficient training---

# the output data will have 2 blank lines
def cut_on_tokens(data_path,out_dir,token_len):
    out_path=os.path.join(out_dir,data_path.split('/')[-1].split('.')[0]+'_2nl_{}_wD.txt'.format(token_len))
    
    b=open(data_path,'r',encoding='utf-8')
    pb=open(out_path,'w',encoding='utf-8')
    eff_line=''
    token_dict={}

    try:
        for i, line in enumerate(b):

            if line!='.' and len(line)>1:

                line=line.rstrip('\n')+' '
                eff_line+=line
                token=word_tokenize(eff_line)
                token_dict[i]=[len(token),eff_line]
                if token_dict.get(i)[0]>token_len:

                    print(i-1,token_dict[i-1][0])
                    pb.write(token_dict[i-1][1]+'\n\n')
                    eff_line=''
                    line=line.rstrip('\n')+' '
                    eff_line+=line
                    token2=word_tokenize(line)
                    token_dict[i]=[len(token2),eff_line]

        pb.close()
        b.close()
    except:
        pb.close()
        b.close()
            
# to execute the above function to get 256 tokens per sequence
data_dir='Corpus_data/mathCorpus/individual_txt/'
out_dir='Corpus_data/mathCorpus/individual_txt/processed1/'
token_len=256
for i, f in enumerate(os.listdir(data_dir)):
    if f.endswith('.txt'):
        print(i,f)
        data_path=os.path.join(data_dir,f)
        cut_on_tokens(data_path,out_dir,token_len)

# ----delete extra lines from original files---
# per tensorflow pre-training format request, there should be 1 blank line between each sequence, therefore, we delete one line from the above output
def delete_extra_line_v2(data_path,out_dir):
    out_path=data_path.split('.')[0]+'_1NL_short.txt'
    if not os.path.exists(out_dir):os.mkdir(out_dir)
    out_path=os.path.join(out_dir,out_path.split('/')[-1])
    b=open(data_path,'r',encoding='utf-8')
    pb=open(out_path,'w',encoding='utf-8')
#     no_break_lines=''
    try:
        for line in b.readlines()[1:]:
            line=line.rstrip('\n')
            lines=re.split('\.|\?', line.strip())
            GE2_lines=[l.strip() for l in lines if len(l.strip())!=1 and l.strip()!='']
            for l in GE2_lines:
                pb.write(l+'\n')
    except:
        pb.close()
        b.close()
    pb.close()
    b.close()

# to execute
    data_dir='K12BERT/Corpus_data/mathCorpus/individual_txt/dedup/dedup_files/'
for i, f in enumerate(os.listdir(data_dir)):
    print(i,f)
    data_path=os.path.join(data_dir,f)
    out_dir='K12BERT/Corpus_data/mathCorpus/individual_txt/dedup/processed1'
    if data_path.endswith('.txt'):
    
        delete_extra_line_v2(data_path,out_dir)

# ---to convert the pretrained data from csv files to txt format with 1 blank line between---
# we have data saved in csv format, we convert it to txt file for tensorflow training and insert 1 blank line in between
def convert_pretrain_csv0(data_path):
    out_path=data_path.split('.')[0]+'_short.txt'
    b=open(data_path,'r',encoding='utf-8')
    pb=open(out_path,'w',encoding='utf-8')
    for line in b.readlines()[1:]:
        lines=line.split('.')
        non_empty_lines=[l for l in lines if l.strip()!=""]
        if non_empty_lines!=['"']:
            valid_lines=''
            for line in non_empty_lines:
                valid_lines+=line+'\n\n'
                pb.write(valid_lines)