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
ratemyprof_dict = pickle.load(open( "./ratemyprofessor_api_dict.pickle", "rb"))
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
np_urls = np.array(np_dat[:, 7])


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
tokens = set()
idf_dict = {}

for t in inverted_dict.keys():
    df = len(inverted_dict[t])
    if float(df/num_docs) < max_df_ratio:
        tokens.add(t)
        idf_dict[t] = math.log(num_docs/(1 + df), 2)

tokens = sorted(tokens)

term_doc_matrix = np.zeros((len(np_descriptions), len(tokens)))

for t in tokens:
    term_idx = tokens.index(t)
    for doc_idx, freq in inverted_dict[t]:
        term_doc_matrix[doc_idx, term_idx] = freq

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
                    doc_scores[doc_idx] = query.count(q) * idf_dict[q] * value * idf_dict[q] #begin accumulator
                else:
                    doc_scores[doc_idx] += query.count(q) * idf_dict[q] * value * idf_dict[q]  #add to accumulator
                # Additional score for query term in title
                # Even more of a boost for queries that match a title verbatim
                score_boost = 0.1
                exact_title_boost = .5
                if q in np_title[doc_idx].lower():
                    doc_scores[doc_idx] += score_boost * (query_norm*norms[doc_idx])
                if(len(original_query.split()) > 1 and original_query in np_title[doc_idx].lower().replace("-", " ")):
                    print(query, np_title[doc_idx].lower())
                    doc_scores[doc_idx] += exact_title_boost * (query_norm*norms[doc_idx])

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
                    doc_scores[doc_idx] = query.count(q) * idf_dict[q] * value * idf_dict[q] #begin accumulator
                else:
                    doc_scores[doc_idx] += query.count(q) * idf_dict[q] * value * idf_dict[q] #add to accumulator
                # Additional score for query term in title
                # Even more of a boost for queries that match a title verbatim
                score_boost = 0.1
                exact_title_boost = .5
                if q in np_title[doc_idx].lower():
                    doc_scores[doc_idx] += score_boost * (query_norm*norms[doc_idx])
                if(len(original_query.split()) > 1 and original_query in np_title[doc_idx].lower().replace("-", " ")):
                    print(query, np_title[doc_idx].lower())
                    doc_scores[doc_idx] += exact_title_boost * (query_norm*norms[doc_idx])

        #GET FROM DICT TO LIST OF TUPLES WHILE DIVIDING BY NORMS

    for doc_idx, value in doc_scores.items():
        for tag in np_subject_number[doc_idx]:
            department = ("".join(re.split("[^a-zA-Z]*", tag))).upper()
            if department == subject:
                value += score_boost * (query_norm*norms[doc_idx])
        course = (subject + " " + number)
        if course not in np_subject_number[doc_idx]:
            tuples.append((value/(query_norm*norms[doc_idx]), doc_idx))

    tuples = sorted(tuples, key=lambda x: x[0], reverse=True)
    return tuples

# In[15]:
def getKeywordResults(original_query, classLevel_query, semester_query, major_query, k=10):
    tuples = cosine_sim(original_query)

    data = []
    i = 0

    while i < len(tuples) and len(data) < k:
        score, doc_idx = tuples[i] 
        semesters = set()
        semSatisfied = False
        classLevelSatisfied = False
        majorSatisfied = False

        for sem in np_semesters[doc_idx]:
            if(sem[:2] == 'SP'):
                semesters.add("Spring")
            else:
                semesters.add("Fall")

        if semester_query == ""  or semester_query == None:
            semSatisfied = True
        elif semester_query in semesters:
            semSatisfied = True

        majors = set()
        classLevels = set()
        for subject in np_subject_number[doc_idx]:
            subjectSplit = subject.split() 
            majors.add(subjectSplit[0])
            classLevels.add(subjectSplit[1])
        
        if classLevel_query == "" or classLevel_query == None:
            classLevelSatisfied = True
        else:
            for classLevel in classLevels:
                if int(classLevel) > int(classLevel_query[0:4]) and int(classLevel) <= int(classLevel_query[5:10]):
                    classLevelSatisfied = True

        
        if major_query == "" or major_query == None:
            majorSatisfied = True
        elif major_query in majors:
            majorSatisfied = True

        if majorSatisfied and classLevelSatisfied and semSatisfied:
            data.append((" / ".join(np_subject_number[doc_idx])+
                        ": "+np_title[doc_idx],
                        np_descriptions[doc_idx],
                        ", ".join(np_professors[doc_idx]),
                        professor_tags(np_professors[doc_idx]),
                        np_semesters[doc_idx],
                        np_urls[doc_idx],
                        doc_idx))

        i+=1

    return data

# In[16]:
def getClassResults(original_query):
    tuples = cosine_sim_class(original_query)
    data = []
    for score, doc_idx in tuples[:10]:
        data.append((" / ".join(np_subject_number[doc_idx])+
                    ": "+np_title[doc_idx],
                    np_descriptions[doc_idx],
                    ", ".join(np_professors[doc_idx]),
                    professor_tags(np_professors[doc_idx]),
                    np_semesters[doc_idx],
                    np_urls[doc_idx]))

    return data

#print professor tags
def professor_tags(professors):
    # subject = ("".join(re.split("[^a-zA-Z]*", class_tag))).upper()
    # number = str("".join(re.split("[^0-9]*", class_tag)))
    # result = [classes_dict[key] for key in classes_dict.keys()
    #              if (subject + " " + number) in classes_dict[key]["subject-number"]][0]
    # professors = result['professors']

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
    i=0
    while len(suggestions) < k:
        if(x[-1-(1*i)][0] not in query_words):
            suggestions.append(x[-1-(1*i)][0])
        i+=1

    return suggestions
    
# One-time Rocchio 

def rocchio(query, relevant_ids, irrelevant_ids, td_matrix=term_doc_matrix):
    alpha = 1
    beta = 1
    gamma = 1
    
    num_toks = np.shape(td_matrix)
    
    q_vec = np.zeros(num_toks[1])
    
    q_terms = query.split(' ')
    
    for t in q_terms:
        if (t in tokens):
            i = tokens.index(t)
            q_vec[i] = q_vec[i] + 1
        
    term1 = q_vec * alpha
    
    rel_sum = np.zeros(num_toks[1])
    irrel_sum = np.zeros(num_toks[1])
    
    for doc_id in relevant_ids:
        rel_sum = rel_sum + td_matrix[doc_id, ]

    for doc_id in irrelevant_ids:
        irrel_sum = irrel_sum + td_matrix[doc_id, ]
        
    if(len(relevant_ids) == 0):
        term2 = (rel_sum*beta)
    else:
        term2 = (rel_sum*beta)/len(relevant_ids)
    
    if(len(irrelevant_ids) == 0):
        term3 = (irrel_sum*gamma)
    else:
        term3 = (irrel_sum*gamma)/len(irrelevant_ids)
        
    new_vec = term1 + term2 - term3

    for i in range(len(new_vec)):
        if new_vec[i] < 0:
            new_vec[i] = 0
            
    new_query = ""        
    
    for i in range(len(new_vec)):
        word = tokens[i] + " "
        new_query = new_query + (word * int(round(new_vec[i])))
        
    return new_query
