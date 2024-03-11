from bs4 import BeautifulSoup
import difflib
import pathlib
import pprint
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from unicodedata import normalize
import re
import string
import numpy as np
import numpy.linalg as LA
from collections import defaultdict, namedtuple

def read_doc(doc_path):
    if isinstance(doc_path, pathlib.Path):
        doc_path = doc_path.as_posix()
    with open(doc_path, 'r') as f:
        doc = f.read()
    return doc


def ixbrl_parser(ixbrl_doc):
    soup = BeautifulSoup(ixbrl_doc, 'html.parser')
    # get all tags
    tags = {tag.name for tag in soup.find_all()}
    ix_nonnumeric_key = 'ix:nonnumeric'
    tag = difflib.get_close_matches(ix_nonnumeric_key, tags, n=1)
    tag = tag and tag[0]
    ix_nonnumeric_tags = soup.find_all(tag)
    ix_nonnumeric_dictionary = dict()
    for ix_nonnumeric_tag_number, ix_nonnumeric_tag in enumerate(ix_nonnumeric_tags):
        ix_nonnumeric_dict = dict()
        attrs = ix_nonnumeric_tag.attrs
        ix_nonnumeric_dict['attrs'] = attrs
    
        contents = ix_nonnumeric_tag.contents
        contents_pattern = []
        for content in contents:
            content_type = type(content)
            content_type_name = content_type.__name__
            contents_pattern.append(content_type_name)
        ix_nonnumeric_dict['contents_pattern'] = contents_pattern
    
        text = ix_nonnumeric_tag.text
        ix_nonnumeric_dict['text'] = text
        ix_nonnumeric_dictionary[ix_nonnumeric_tag_number] = ix_nonnumeric_dict
    return ix_nonnumeric_dictionary


def html_parser(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    # get all tags
    tags = {tag.name for tag in soup.find_all()}
    div_key = 'div'
    tag = difflib.get_close_matches(div_key, tags, n=1)
    tag = tag and tag[0]
    div_tags = soup.find_all(tag)
    div_tags_dictionary = dict()
    for div_tag_number, div_tag in enumerate(div_tags):
        div_tag_dict = dict()
        attrs = div_tag.attrs
        div_tag_dict['attrs'] = attrs
    
        contents = div_tag.contents
        contents_pattern = []
        for content in contents:
            content_type = type(content)
            content_type_name = content_type.__name__
            contents_pattern.append(content_type_name)
        div_tag_dict['contents_pattern'] = contents_pattern
    
        text = div_tag.text
        div_tag_dict['text'] = text
        div_tags_dictionary[div_tag_number] = div_tag_dict
    return div_tags_dictionary



# clean a list of lines
def clean(line):
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # normalize unicode characters
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    # tokenize on white space
    tokens = line.split()
    # convert to lowercase
    tokens = [word.lower() for word in tokens]
    # remove punctuation from each token
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove non-printable chars form each token
    tokens = [re_print.sub('', w) for w in tokens]
    # remove tokens with numbers in them
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    with open("stopwords.txt", 'r') as f:
        text = f.read()
        stop_words = text.split('\n')
    tokens = [w for w in tokens if not w in stop_words]
    return tokens


def join_list(item):
    return ' '.join(item)





def create_ix_nonnumeric_model(ix_nonnumeric_texts):
    cleaned_ix_nonnumeric_texts = []
    for ix_nonnumeric_text in ix_nonnumeric_texts:
        ix_nonnumeric_text_tokens = clean(ix_nonnumeric_text)
        cleaned_ix_nonnumeric_text = join_list(ix_nonnumeric_text_tokens) 
        cleaned_ix_nonnumeric_texts.append(cleaned_ix_nonnumeric_text)
    
    
    vectorizer = TfidfVectorizer()
    X_vector = vectorizer.fit_transform(cleaned_ix_nonnumeric_texts).toarray()
    
    Model = namedtuple('Model', '''X_vector, vectorizer''')
    model = Model(
            X_vector = X_vector,
            vectorizer = vectorizer)
    return model



def predict(text, model):
    y_vector = model.vectorizer.transform([text]).toarray()
    cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)
    cosine_similarity_dict = defaultdict(list)
    max_cosine_score = 0
    for vector_row_number, vector in enumerate(model.X_vector):
        cosine = cx(vector, y_vector[0])
        if cosine > max_cosine_score:
            max_cosine_score = cosine
            cosine_similarity_dict[max_cosine_score] = vector_row_number
    return dict(cosine_similarity_dict)


def get_max_confidence(input_dict):
    max_conf = 0.0
    for conf, index in input_dict.items():
        if conf > max_conf:
            max_conf = conf
    return max_conf

def get_max_confidence_index(input_dict):
    max_conf = 0.0
    max_index = None
    for conf, index in input_dict.items():
        if conf > max_conf:
            max_conf = conf
            max_index = index
    return max_index 


def ix_nonnumeric_dictionary_contents_pattern(index, ix_nonnumeric_dictionary):
    if index is None or pd.isna(index):
        return
    else:
        ix_nonnumeric_dict = ix_nonnumeric_dictionary[int(index)]
        contents_pattern = ix_nonnumeric_dict.get('contents_pattern', None)
        return contents_pattern
    
def hashify(my_list):
    if my_list is None:
        return
    else:
        my_list = [l.strip() for l in my_list]
        my_text = ' '.join(my_list)
        hash_value_of_text = sum([ord(char) for char in my_text])
        return hash_value_of_text


def mapping(ixbrl_doc_path, html_doc_path):
    ixbrl_doc = read_doc(doc_path=ixbrl_doc_path)
    ix_nonnumeric_dictionary = ixbrl_parser(ixbrl_doc)
    # pprint.pprint(ix_nonnumeric_dictionary)
    
    # with open('ix_nonnumeric_dictionary.json', 'w') as f:
    #     json.dump(ix_nonnumeric_dictionary, f,  indent=4)
        
    html_doc = read_doc(doc_path=html_doc_path)
    div_tags_dictionary = html_parser(html_doc)
    # pprint.pprint(div_tags_dictionary)
    
    # with open('div_tags_dictionary.json', 'w') as f:
    #     json.dump(div_tags_dictionary, f,  indent=4)
    
    ix_nonnumeric_texts = []
    for num, dictionary in ix_nonnumeric_dictionary.items():
        text = dictionary.get('text', None)
        ix_nonnumeric_texts.append(text)
    
    # text1 = ix_nonnumeric_dictionary[30]['text']
    # text1_english_feature_words = clean(text1)
    # cleaned_text1 = join_list(text1_english_feature_words)

    # text2 = div_tags_dictionary[33]['text']
    # text2_english_feature_words = clean(text2)
    # cleaned_text2 = join_list(text2_english_feature_words)
    
    indexes, texts, contents_patterns, contextrefs, names, ids, escapes = [], [], [], [], [], [], []
    for index, dictionary in div_tags_dictionary.items():
        
        indexes.append(index)
        
        text = dictionary.get('text', None)
        texts.append(text)
        
        contents_pattern = dictionary.get('contents_pattern', None)
        contents_patterns.append(contents_pattern)
        
        attrs = dictionary.get('attrs', None)
        
        contextref = attrs.get("contextref", None)
        contextrefs.append(contents_pattern)
        
        name = attrs.get("name", None)
        names.append(name)

        id_ = attrs.get("id", None)
        ids.append(id_)
        
        escape = attrs.get("escape", None)
        escapes.append(escape)
        
    div_df = pd.DataFrame()
    div_df['indexes'] = indexes
    div_df['texts'] = texts
    div_df['contents_patterns'] = contents_patterns
    div_df['contextrefs'] = contextrefs
    div_df['names'] = names
    div_df['ids'] = ids
    div_df['escapes'] = escapes
   
    model = create_ix_nonnumeric_model(ix_nonnumeric_texts)
    div_df['text_token'] = div_df['texts'].apply(clean)
    div_df['cleaned_texts'] = div_df['text_token'].apply(join_list)
    div_df['cosine_similarity_dict'] = div_df['cleaned_texts'].apply(predict, model=model)
    div_df['ixbrl_max_confidence'] = div_df['cosine_similarity_dict'].apply(get_max_confidence)
    div_df['ixbrl_max_confidence_index'] = div_df['cosine_similarity_dict'].apply(get_max_confidence_index)
    
    div_df['ixbrl_contents_patterns'] = div_df['ixbrl_max_confidence_index'].apply(ix_nonnumeric_dictionary_contents_pattern, ix_nonnumeric_dictionary=ix_nonnumeric_dictionary)
    
    div_df['ixbrl_contents_patterns_hash'] = div_df['ixbrl_contents_patterns'].apply(hashify)
    div_df['contents_patterns_hash'] = div_df['contents_patterns'].apply(hashify)
    
    matched_index = []
    unique_ixbrl_index = []
    for item in sorted(set(div_df['ixbrl_max_confidence_index'])):
        if not pd.isna(item):
            unique_ixbrl_index.append(item)
    sorted_unique_ixbrl_index = sorted(unique_ixbrl_index)
    
    for ixbrl_index in sorted_unique_ixbrl_index:
        df = div_df[div_df['ixbrl_max_confidence_index'] == ixbrl_index]
        for index, contents_patterns_hash, ixbrl_contents_patterns_hash in zip(
                df['indexes'],
                df['contents_patterns_hash'],
                df['ixbrl_contents_patterns_hash']
            ):
            if contents_patterns_hash==ixbrl_contents_patterns_hash:
                matched_index.append(index)
    
    is_matched_index = []
    for index in div_df['indexes']:
        is_matched_index.append(index in matched_index)
        
    div_df['is_matched_index'] = is_matched_index
    
    matched_div_df = div_df[div_df['is_matched_index']==True]
    
    # Copy removes follwoing errors:
    # __main__:1: SettingWithCopyWarning: 
    # A value is trying to be set on a copy of a slice from a DataFrame.
    # Try using .loc[row_indexer,col_indexer] = value instead

    # See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    matched_div_df = matched_div_df.copy()
        
    matched_div_df['ixbrl_max_confidence_index'] = matched_div_df['ixbrl_max_confidence_index'].astype(int)
    matched_div_df['ixbrl_matched_text'] = matched_div_df['ixbrl_max_confidence_index'].apply(lambda x: ix_nonnumeric_dictionary[x].get('text', None))
    
    matched_div_df['ixbrl_mateched_text_length'] = matched_div_df['ixbrl_matched_text'].apply(lambda x: len(x))
    
    matched_div_df['texts_length'] = matched_div_df['texts'].apply(lambda x: len(x))
    
    matched_div_df['differences'] = abs(matched_div_df['ixbrl_mateched_text_length'] - matched_div_df['texts_length'])
    
    unique_ixbrl_index = []
    for item in sorted(set(matched_div_df['ixbrl_max_confidence_index'])):
        if not pd.isna(item):
            unique_ixbrl_index.append(item)
    sorted_unique_ixbrl_index = sorted(unique_ixbrl_index)
    
    
    mapped_index_list = []
    is_mapped_index = []
    for ixbrl_index in sorted_unique_ixbrl_index:
        df = matched_div_df[matched_div_df['ixbrl_max_confidence_index'] == ixbrl_index]
        min_difference = min(df['differences'])
        max_confidence = max(df['ixbrl_max_confidence'])
        for index, difference, ixbrl_max_confidence in zip(
            df['indexes'],
            df['differences'],
            df['ixbrl_max_confidence'] 
        ):
            if difference==min_difference and ixbrl_max_confidence==max_confidence:
                print("right")
                mapped_index_list.append(index)
                is_mapped_index.append(True)
            elif difference==min_difference:
                mapped_index_list.append(index)
                is_mapped_index.append(True)                
            elif ixbrl_max_confidence==max_confidence:
                mapped_index_list.append(index)
                is_mapped_index.append(True)
            else:
                mapped_index_list.append(None)
                is_mapped_index.append(False)
    matched_div_df['mapped_index_list'] = mapped_index_list
    matched_div_df['is_mapped_index'] = is_mapped_index

    mapped_div_df = matched_div_df[matched_div_df['is_mapped_index'] ==True]

    mapped_div_df = mapped_div_df.reset_index(drop=True)
    
    mapped_dictionary = dict()
    for num, div_index, ixbrl_index in zip(mapped_div_df.index, mapped_div_df['indexes'], mapped_div_df['ixbrl_max_confidence_index']):
        mapped_dict = dict()
        div = div_tags_dictionary[div_index]
        ixbrl = ix_nonnumeric_dictionary[ixbrl_index]
        mapped_dict['div'] = div
        mapped_dict['ixbrl'] = ixbrl
        mapped_dict['div_index'] = div_index
        mapped_dict['ixbrl_index'] = ixbrl_index
        mapped_dictionary[num] = mapped_dict
    
    return mapped_dictionary



if __name__ == '__main__':    
    ixbrl_doc_path = 'C:/Users/Celebal/OneDrive - Celebal Technologies Private Limited/Documents/Documents/IRIS Carbon/Cyberoptics/Q2 iXBRL/cybe-20210630.htm'
    html_doc_path = 'C:/Users/Celebal/OneDrive - Celebal Technologies Private Limited/Documents/Documents/IRIS Carbon/Cyberoptics/Q3 HTML/MainDocument.htm'
    mapped_dictionary = mapping(ixbrl_doc_path, html_doc_path)
    with open('mapped_dictionary.json', 'w') as f:
        json.dump(mapped_dictionary, f,  indent=4)
    
    
    
    
    
#     mapped_index = list(filter(None, mapped_index_list))
#     mapped_div_dictionary = dict()
#     for index in mapped_index_list:
#         mapped_div_dictionary[index] = div_tags_dictionary
    
#     df = pd.DataFrame()
#     df['indexes'] = matched_div_df['indexes']
#     df['texts'] = matched_div_df['texts']
#     df['ids'] =  matched_div_df['ids']
#     df['div'] = matched_div_df['contents_patterns_hash'] 
#     df["ix"] = matched_div_df['ixbrl_contents_patterns_hash']
#     df['is_matched_index'] = matched_div_df['is_matched_index']
#     df['ix_index']  = matched_div_df['ixbrl_max_confidence_index'] 
#     df['ix_score']  = matched_div_df['ixbrl_max_confidence'] 
#     df['difference'] = matched_div_df['difference']
#     df['mapped_index_list']  = matched_div_df['mapped_index_list']

#     div_tags_dictionary[388]
    
#     ix_nonnumeric_dictionary[72]
    
#     # cosine_similarity_dict = predict(text=cleaned_text2, model=model)
    
#     # for confidence, index in cosine_similarity_dict.items():
#     #     ix_nonnumeric_texts[index]
    
#     for a, b in zip(div_df['contents_patterns'][34:36+1] , div_df['ixbrl_contents_patterns'][34:36+1]):
#         print(len(a)==len(b))
#         print(a)
#         print(b)
#         print()







# ['NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString', 'Tag', 'NavigableString']



    
#     # print(attrs['id'])
#     # if attrs['id'] == 'Tag212':
#     #     print(attrs)
#     #     break
#     paras = ix_nonnumeric_tag.find_all('p')
#     for para in paras:
#         text = para.text
#         print(text)


# attributes = ["id", "name", ""]
# tag_attr = dict()
# # attr = "id"
# for attr in attributes:
#     if ix_nonnumeric_tag.has_attr(attr):
#         attr_value = ix_nonnumeric_tag[attr]
#         tag_attr[attr] = attr_value
    
    
    

# import re
# pattern = '<ix:nonNumeric.+<\/ix:nonNumeric>'
# ix_nonnumeric_tags = re.findall(pattern, ixbrl_doc)

# for ix_nonnumeric_tag in ix_nonnumeric_tags:
#     ix_nonnumeric_soup = BeautifulSoup(ix_nonnumeric_tag, 'html.parser')
#     print(ix_nonnumeric_soup)
#     print()
#     attrs = ix_nonnumeric_soup.attrs
