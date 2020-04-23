import pandas as pd
import ast
from tabulate import tabulate
import re
import numpy as np
import math

# Create a Pandas dataframe to act as our "table" for now
# Import csv file of scraped data from Class Roster
# Each row is a Cornell class
classes_df = pd.read_csv("./roster_api_data.csv", na_filter= False)

# Find all classes with Subject ENGRD and number 2110
# subject = "ENGRD"
# number = 2110
# # Result is a table with one row
# result = classes_df[(classes_df["subject"] == subject) & (classes_df["number"] == number)]
#result = classes_df[(classes_df["id"] == 366268)]
# Display that table
#display(result)
# Since result is a table with one row
#   we need .iloc[0] to choose only that first row
#   then we get the "professors" column and "cast" it to a list
#   using ast.literal_eval
# professor_list = ast.literal_eval(result.iloc[0]["professors"])
#print(professor_list)


# In[3]:
# Import csv file of scraped data from RateMyProfessor.com
ratemyprof_df = pd.read_csv("./ratemyprofessor_api_data.csv", na_filter= False)
# Get only the rows for professors that have reviews and tags
reviewed_professors = ratemyprof_df[ratemyprof_df["review"] != ""]
#display(reviewed_professors)

# In[4]:
np_dat = pd.read_csv("./roster_api_data.csv", na_filter= False).to_numpy()
# COLUMNS id,subject,number,title,description,outcomes,professors
#headers = ["id", "subject", "number", "title", "description", "outcomes", "professors"]
# tabulate data
#table = tabulate(np_dat, headers, tablefmt="fancy_grid")
#print(table)


# In[5]:
#get individual columns as np.arrays
np_ids = np.array(np_dat[:, 0])
np_subject = np.array(np_dat[:, 1])
np_number = np.array(np_dat[:, 2])
np_title = np.array(np_dat[:, 3])
np_descriptions = np.array(np_dat[:, 4])
np_outcomes = np.array(np_dat[:, 5])
np_professors = np.array(np_dat[:, 6])
# print(np_outcomes[100])


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
def getResults(original_query):
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
                    doc_scores[doc_idx] = query.count(q) * idf_dict[q] * value * idf_dict[q]#begin accumulator
                else:
                    doc_scores[doc_idx] += query.count(q) * idf_dict[q] * value * idf_dict[q]#add to accumulator
                #Additional score for query term in title
               # score_boost = 0.1
                #if q in np_title[doc_idx]:
                 #   print(q, np_title[doc_idx])
                  #  doc_scores[doc_idx] += score_boost

        #GET FROM DICT TO LIST OF TUPLES WHILE DIVIDING BY NORMS

    for doc_idx, value in doc_scores.items():
        tuples.append((value/(query_norm*norms[doc_idx]), doc_idx))

    tuples = sorted(tuples, key=lambda x: x[0], reverse=True)

    data = []

    for score, doc_idx in tuples[:10]:
        data.append((np_subject[doc_idx]+""+str(np_number[doc_idx])+": "+np_title[doc_idx], np_descriptions[doc_idx],", ".join(eval(np_professors[doc_idx])), score))

    return data

    # print("#" * len(original_query))
    # print(original_query)
    # print("#" * len(original_query))
    #
    # for score, doc_idx in tuples[:10]:
    #     print("\n\n")
    #     print("Score: %s \n" % (score))
    #     print("Class: %s %s %s \n" % (np_subject[doc_idx], np_number[doc_idx], np_title[doc_idx]))
    #     print("Description: %s \n" % np_descriptions[doc_idx])
    #     print("\n\n")
