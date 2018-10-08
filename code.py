
# coding: utf-8

# In[125]:


#import libraries
import pandas as pd
import numpy as np
import scipy as sp
import nltk
nltk.download('averaged_perceptron_tagger')


# In[126]:


#import data into pandas data frame
reviews_df = pd.read_csv('yelp_review.csv')
bussiness_df = pd.read_csv('yelp_business.csv')


# In[127]:


# sample_review_df = reviews_df[0:5000]
# sample_business_df = bussiness_df[0:5000]


# In[128]:


# sample_review_df.to_csv('sample_review_df',encoding='utf-8')


# In[129]:


# sample_business_df.to_csv('sample_business_df',encoding='utf-8')


# In[130]:


reviews_df.head()


# In[131]:


#merge both data frames on business id
full_data_df = pd.merge(bussiness_df, reviews_df , on = 'business_id', how = 'left')

len(full_data_df)


# In[132]:


full_data_df[0:5]


# In[133]:


#filter data for las vegas city
Las_vegas_full_data_df = full_data_df[full_data_df['city']== 'Las Vegas']


# In[134]:


Las_vegas_full_data_df[0:5]


# In[135]:


# drop unwanted features
Las_vegas_full_data_df = Las_vegas_full_data_df.drop(['neighborhood','address','city','state','latitude','longitude','stars_x'],axis=1)


# In[136]:


Las_vegas_full_data_df = Las_vegas_full_data_df.drop(['review_count','is_open','review_id','user_id','date','useful','funny','cool'],axis=1)


# In[137]:


Las_vegas_full_data_df.head()


# In[138]:


# for index, data in Las_vegas_full_data_df.iterrows():
# #     print(index)
# #     if index == 182:
# #         break
#     my_string = Las_vegas_full_data_df.iloc[index]['categories']
#     my_string = str(my_string)    
#     my_list = my_string.split(";")

#     Las_vegas_full_data_df.set_value(index,'categories',my_list)


# In[139]:


# restaurants_Las_vegas_full_data_df = Las_vegas_full_data_df[Las_vegas_full_data_df['categories'].apply(lambda x: 'Restaurants' in x )
#                     ].reset_index(drop=True)


# In[140]:


# len(restaurants_Las_vegas_full_data_df)


# In[141]:


#filter only restaurant businesses
restaurants_Las_vegas_full_data_df= Las_vegas_full_data_df.loc[Las_vegas_full_data_df['categories']== 'Pizza;Restaurants']



# In[142]:


# restaurants_Las_vegas_full_data_df = restaurants_Las_vegas_full_data_df[0:60000]


# In[143]:


len(restaurants_Las_vegas_full_data_df)


# In[144]:


Las_vegas_full_data_df['categories'].value_counts()


# In[145]:


# restaurants_Las_vegas_full_data_df


# In[146]:


# get positive phrases that are in positivePOS_tag format
positivePOS_tag = "positive: {<JJ> <NN>|<JJ> <NNS>|<RB> <JJ>|<RBR> <JJ>}"
def get_positive_chunks(reviews):
    positive_score = 0.0
    ispositive = False
    p_parser = nltk.RegexpParser(positivePOS_tag)
    c_reviews = p_parser.parse(reviews)
    trees = c_reviews.subtrees()
    results = []
    for tree in trees:
        if tree.label() == "positive":
            noun = ""
            (terms, tags) = zip(*tree)
            for i in range(0, len(terms)):
                noun = noun + " " + terms[i]
            p_score = Pattern.sentiment(noun.strip())
            if p_score[0] >= (0.2) and p_score[1] >= 0.5:
                results.append(noun)
                positive_score += Pattern.sentiment(noun)[0]
                ispositive = True
    return ispositive, positive_score, results


# In[147]:


# get negative phrases that are in negativePOS_tag format

negativePOS_tag = "negative: {<JJ> <NN>|<JJ> <NNS>|<RB> <JJ>|<RBR> <JJ>}"

def get_negative_chunks(reviews):
    negative_score = 0.0
    isnegative = False
    results = []
    n_parser = nltk.RegexpParser(negativePOS_tag)
    c_reviews = n_parser.parse(reviews)
    trees = c_reviews.subtrees()
    for tree in trees:
        if tree.label() == 'negative':
            noun = ""
            (terms, tags) = zip(*tree)
            for i in range(0, len(terms)):
                noun = noun + " " + terms[i]
            p_score = Pattern.sentiment(noun.strip())
            if p_score[0] <= (-0.1) and p_score[1] >= 0.4:
                results.append(noun)
                negative_score += Pattern.sentiment(noun)[0]
                isnegative = True
    return isnegative, negative_score, results


# In[148]:


#generate 
import sys; sys.path.append('/Users/katari/Desktop/yelp_project/pattern-2.6')
import pattern.en as Pattern

def generate_results(restaurant_reviews):
    id = 0
    results_csv = []
    for each_review_list in restaurant_reviews:
        for each_review in each_review_list:
            id += 1
            results_csv_row = {}
            review_text = str(each_review['text'])
            results_csv_row["Reviews"] = review_text
            results_csv_row["Stars"] = each_review['stars_y']
            results_csv_row["Business Id"] = each_review['business_id']
            tokenize_reviews = nltk.word_tokenize(str(review_text.decode('utf-8')))
            POStagged_reviews = nltk.pos_tag(tokenize_reviews)
            detected_positive, positive_score, results_positive_phrases = get_positive_chunks(POStagged_reviews)
            detected_negative, negative_score, results_negative_phrases = get_negative_chunks(POStagged_reviews)
            if detected_positive or detected_negative:
                results_csv_row["Positive_Phrases"] = results_positive_phrases
                results_csv_row["Negative_Phrases"] = results_negative_phrases
                results_csv_row["Positive_Polarity"] = positive_score
                results_csv_row["Negative_Polarity"] = negative_score
                results_csv_row["Text sentiment"] = Pattern.sentiment(review_text.strip())
                results_csv.append(results_csv_row)
    return results_csv


# In[149]:


#convert data in daatframe to dictionary
results_csv = [(restaurants_Las_vegas_full_data_df).to_dict(orient='records')]


# In[150]:


results_csv_3 = generate_results(results_csv)


# In[151]:


print(results_csv[0][0]['text'])
reviews_list = results_csv


# In[152]:


#convert dictionary back to dataframe
Las_vegas_restaurants_review_results =pd.DataFrame(results_csv_3)


# In[153]:


Las_vegas_restaurants_review_results.columns.values


# In[154]:


#calculate average polarity score and add as a feature
average_pscore = []     
for row in Las_vegas_restaurants_review_results.iterrows(): 
    average_pscore = (Las_vegas_restaurants_review_results['Positive_Polarity'] + Las_vegas_restaurants_review_results['Negative_Polarity']) /2

Las_vegas_restaurants_review_results['average_pscore'] = average_pscore


# In[155]:


#create polarity labels
polarity_labels = []
for val in average_pscore:
    if val < 0:
        polarity_labels.append('Bad')
    else:
        polarity_labels.append('Good')


# In[156]:


Las_vegas_restaurants_review_results['polarity_labels'] = polarity_labels


# In[157]:


#plot corelation between polarity scores and star ratings
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
sns.lmplot('Stars','average_pscore',data=Las_vegas_restaurants_review_results, fit_reg=True)

plt.xlabel("Star Ratings")
plt.ylabel("Polarity Scores")
plt.title("Correlation Plot for Reviews in Las Vegas")


# In[158]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# In[159]:


df = Las_vegas_restaurants_review_results
X = df[['Stars','average_pscore']]
y = df['polarity_labels']


# In[160]:


# split data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=3057)

clf = SVC(C=0.5, kernel="linear")
#train the model using svm
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred)


# In[161]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=6)
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)
accuracy_score(y_test, y_pred)


# In[162]:


# train with naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(gnb, X, y, scoring="accuracy", cv=6)

import numpy as np
print("Average accuracy, 6-fold cross validation:")
print(np.mean(scores))


# In[163]:


df_business = restaurants_Las_vegas_full_data_df
df_business.head()


# In[164]:


# pd.set_option('display.max_colwidth', -1)
# df_business.head()


# In[165]:


# df_business['postal_code'].value_counts()


# In[166]:


# filter only businesses in 89109
df_postal = df_business[(df_business['postal_code']== '89109')]
# df_business['postal_code'][0]


# In[167]:


#filter restaurants serving pizza
df_pizza = df_postal[df_postal['categories'].apply(lambda x: 'Pizza' in x )
                    ].reset_index(drop=True)


# In[168]:


# df_pizza.groupby(["stars_y", "business_id"])["stars_y"].agg("count")


# In[169]:


low_star = df_pizza[df_pizza['stars_y'] < 3]
high_star = df_pizza[df_pizza['stars_y'] >= 3]
high_star_bid = list(high_star['business_id'])
low_star_bid = list(low_star['business_id'])
len(low_star_bid)


# In[170]:


# df_pizza_phrases = pd.read_csv("Phoenix_restaurant_reviews_Pizza_Results.csv", encoding="latin-1")


# In[171]:


df_pizza_phrases = Las_vegas_restaurants_review_results


# In[172]:


# df_pizza_phrases.head()
# # len(df_pizza_phrases)


# In[173]:


#extract negative phrases
df_pizza_Negatives_phrases = pd.DataFrame()
df_pizza_Negatives_phrases['Business Id'] = df_pizza_phrases['Business Id']
df_pizza_Negatives_phrases['Negative_Phrases'] = df_pizza_phrases['Negative_Phrases']
len(df_pizza_Negatives_phrases)


# In[174]:


df_pizza_Negatives_phrases = df_pizza_Negatives_phrases[df_pizza_Negatives_phrases.astype(str)['Negative_Phrases'] != '[]']
print


# In[175]:


# get business ids
df_pizza_negative_bid = set(list(df_pizza_Negatives_phrases['Business Id']))
len(df_pizza_negative_bid)


# In[176]:


# df_pizza_Negatives_phrases.loc[df_pizza_Negatives_phrases['Business Id'] == 'YQ--LJ7pvjiDSqNv0TuKTQ ', 'Negative_Phrases']



# In[177]:


for id in low_star_bid:
    if id in df_pizza_negative_bid:
        print"Business ID: ", id, "not performing well due to"
        print("-"*63)
        print df_pizza_Negatives_phrases.loc[df_pizza_Negatives_phrases['Business Id'] == id, 'Negative_Phrases']
        
    else:
        continue
    
    print("*"*63)
    break


# In[178]:


#extract positive phrases
df_pizza_Positive_phrases = pd.DataFrame()
df_pizza_Positive_phrases['Business Id'] = df_pizza_phrases['Business Id']
df_pizza_Positive_phrases['Positive_Phrases'] = df_pizza_phrases['Positive_Phrases']


# In[179]:


#get business ids
df_pizza_postive_bid = list(df_pizza_Positive_phrases['Business Id'])


# In[180]:


##results
for id in high_star_bid:
    if id in df_pizza_postive_bid:
        print("Business ID", id, "performing well due to")
        print("-"*63)
        print(df_pizza_phrases.loc[df_pizza_phrases['Business Id'] == id, 'Positive_Phrases'])
    else:
        continue
    print("*"*63)
    break

