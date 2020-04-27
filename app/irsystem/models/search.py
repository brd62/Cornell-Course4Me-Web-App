import pandas as pd
import ast
from tabulate import tabulate
import re
import numpy as np
import math
import pickle

# Create a Pandas dataframe to act as our "table" for now
# Import csv file of scraped data from Class Roster
# Each row is a Cornell class
classes_dict = pickle.load(open( "./class_roster_api_dict.pickle", "rb"))

# In[3]:
# Import csv file of scraped data from RateMyProfessor.com
# ratemyprof_df = pd.read_csv("./ratemyprofessor_api_data.csv", na_filter= False)
# Get only the rows for professors that have reviews and tags
# reviewed_professors = ratemyprof_df[ratemyprof_df["review"] != ""]
#display(reviewed_professors)

# In[4]:
pd_data = pd.DataFrame(classes_dict).transpose()
pd_data = pd_data[["id"]+[col for col in pd_data.columns if col != "id"]]
np_dat = pd_data.to_numpy()
# COLUMNS id,subject,number,title,description,outcomes,professors
#headers = ["id", "subject", "number", "title", "description", "outcomes", "professors"]
# tabulate data
#table = tabulate(np_dat, headers, tablefmt="fancy_grid")
#print(table)


# In[5]:
#get individual columns as np.arrays
np_ids = np.array(np_dat[:, 0])
np_professors = np.array(np_dat[:, 1])
np_semesters = np.array(np_dat[:,3])
np_subject_number = np.array(np_dat[:, 2])
np_title = np.array(np_dat[:, 4])
np_descriptions = np.array(np_dat[:, 5])
np_outcomes = np.array(np_dat[:, 6])


# In[7]:
id_to_index = {}

for i in range(len(np_ids)):
    id_to_index[np_ids[i]] = i


# In[8]:
inverted_dict = {}
regex = r'\w+' # regular expression to find words (MAY NEED TO REVISE/EDIT THIS)

for i in range(len(np_descriptions)):
#for i in range(len(np_outcomes)):
    #if len(np_outcomes[i]) > 0:
     #   match = ''.join(np_outcomes[i])
    #else:
   #     match = np_descriptions[i]
    toks = re.findall(regex, np_descriptions[i].lower())
    seen = set()
    for t in toks:
        if t in inverted_dict.keys():
            if t not in seen:
                count = toks.count(t)
                inverted_dict[t].append((id_to_index[np_ids[i]], count))
                seen.add(t)
        else:
            count = toks.count(t)
            inverted_dict[t] = [(id_to_index[np_ids[i]], count)]
            seen.add(t)
    seen.clear()


# In[9]:
# inverted_dict['the'] #gives list of tuples (doc_index, frequency of 'the' in doc)


# In[10]:
min_df = 0
max_df_ratio = 0.17 #tuned this down until it printed out only very common words for a course description
num_docs = np.shape(np_dat)[0]

idf_dict = {}

for t in inverted_dict.keys():
    df = len(inverted_dict[t])
    if float(df/num_docs) < max_df_ratio:
        idf_dict[t] = math.log(num_docs/(1 + df), 2)
    # else:
    #     print(t)


# In[12]:
norms = np.zeros(num_docs)
for i in idf_dict:
    for tup in inverted_dict[i]:
        desc_idx = tup[0]
        desc_term_freq = tup[1]
        norms[desc_idx] += (desc_term_freq * idf_dict[i]) ** 2

norms = np.sqrt(norms)

# In[13]:
def cosine_sim(original_query):
    query = original_query.split()
    tuples = list()

    query_norm_sum = 0

    for q in query:
        if q in idf_dict.keys():
            q_count = query.count(q)
            q_idf = idf_dict[q]
            query_norm_sum += (q_count*q_idf) ** 2

    query_norm = math.sqrt(query_norm_sum)

    doc_scores = {}

    for q in query: #iterate over each query term
        if q in idf_dict.keys(): #if q has inverted doc frequency val
            for (doc_idx, value) in inverted_dict[q]: #iterate over each tuple in inverted_index[query_term]
                if doc_idx not in doc_scores.keys():
                    doc_scores[doc_idx] = query.count(q) * idf_dict[q] * value #begin accumulator
                else:
                    doc_scores[doc_idx] += query.count(q) * idf_dict[q] * value #add to accumulator
                #Additional score for query term in title
               # score_boost = 0.1
                #if q in np_title[doc_idx]:
                 #   print(q, np_title[doc_idx])
                  #  doc_scores[doc_idx] += score_boost

        #GET FROM DICT TO LIST OF TUPLES WHILE DIVIDING BY NORMS

    for doc_idx, value in doc_scores.items():
        tuples.append((value/(query_norm*norms[doc_idx]), doc_idx))

    tuples = sorted(tuples, key=lambda x: x[0], reverse=True)
    return tuples


# In[14]:
def cosine_sim_class(class_tag): #input is of the form 'INFO 4300' or 'INFO4300'

    subject = ("".join(re.split("[^a-zA-Z]*", class_tag))).upper()
    number = str("".join(re.split("[^0-9]*", class_tag)))

    result = [classes_dict[key] for key in classes_dict.keys()
                 if (subject + " " + number) in classes_dict[key]["subject-number"]][0]

    original_query = result["description"]

    print(original_query)

    query = original_query.split()
    tuples = list()

    query_norm_sum = 0

    for q in query:
        if q in idf_dict.keys():
            q_count = query.count(q)
            q_idf = idf_dict[q]
            query_norm_sum += (q_count*q_idf) ** 2

    query_norm = math.sqrt(query_norm_sum)

    doc_scores = {}

    for q in query: #iterate over each query term
        if q in idf_dict.keys(): #if q has inverted doc frequency val
            for (doc_idx, value) in inverted_dict[q]: #iterate over each tuple in inverted_index[query_term]
                if doc_idx not in doc_scores.keys():
                    doc_scores[doc_idx] = query.count(q) * idf_dict[q] * value #begin accumulator
                else:
                    doc_scores[doc_idx] += query.count(q) * idf_dict[q] * value #add to accumulator
                #Additional score for query term in title
               # score_boost = 0.1
                #if q in np_title[doc_idx]:
                 #   print(q, np_title[doc_idx])
                  #  doc_scores[doc_idx] += score_boost

        #GET FROM DICT TO LIST OF TUPLES WHILE DIVIDING BY NORMS

    for doc_idx, value in doc_scores.items():
        tuples.append((value/(query_norm*norms[doc_idx]), doc_idx))

    tuples = sorted(tuples, key=lambda x: x[0], reverse=True)
    return tuples

# In[15]:
def getKeywordResults(original_query):
    tuples = cosine_sim(original_query)

    data = []

    for score, doc_idx in tuples[:10]:
        data.append((" / ".join(np_subject_number[doc_idx])+
                    ": "+np_title[doc_idx],
                    np_descriptions[doc_idx],
                    ", ".join(np_professors[doc_idx]), score))

    return data

# In[16]:
def getClassResults(original_query):
    print("XXX",original_query,"XXX")
    tuples = cosine_sim_class(original_query)
    data = []

    for score, doc_idx in tuples[:10]:
        data.append((" / ".join(np_subject_number[doc_idx])+
                    ": "+np_title[doc_idx],
                    np_descriptions[doc_idx],
                    ", ".join(np_professors[doc_idx]), score))

    return data

#print professor tags
def professor_tags(class_tag):
    subject = ("".join(re.split("[^a-zA-Z]*", class_tag))).upper()
    number = str("".join(re.split("[^0-9]*", class_tag)))
    result = [classes_dict[key] for key in classes_dict.keys()
                 if (subject + " " + number) in classes_dict[key]["subject-number"]][0]
    professors = result['professors']

    tag_dict = {}

    for p in professors:
        tag_dict[p] = ratemyprof_dict[p]['tags']

    return tag_dict


# In[17]:
#SVD query expansion... modeled off Lecture Demo
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

vectorizer = TfidfVectorizer(stop_words = 'english', max_df = max_df_ratio,
                            min_df = 0)
my_matrix = vectorizer.fit_transform([x for x in np_descriptions]).transpose()

u, s, v_trans = svds(my_matrix, k=100)
words_compressed, _, docs_compressed = svds(my_matrix, k=40)
docs_compressed = docs_compressed.transpose()

word_to_index = vectorizer.vocabulary_
index_to_word = {i:t for t,i in word_to_index.items()}

words_compressed = normalize(words_compressed, axis = 1)

def getSuggestions(query, k=5):
    query_words = query.split()

    result = {}

    for w in query_words:
         if w in word_to_index:
            sims = words_compressed.dot(words_compressed[word_to_index[w],:])
            asort = np.argsort(-sims)[:k+1]
            for i in asort[1:]:
                word = index_to_word[i]
                if word not in result.keys():
                    result[word] = sims[i]/sims[asort[0]]
                else:
                    result[word] = result[word] + sims[i]/sims[asort[0]]

    if result == {}:
        return []

    x = sorted(result.items(),key=(lambda i: i[1]))

    suggestions = []
    for i in range(k):
        suggestions.append(x[-1-(1*i)][0])
        print(x[-1-(1*i)][0], " ", x[-1-(1*i)][1])

    return suggestions
