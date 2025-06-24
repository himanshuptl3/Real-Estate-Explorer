import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# for Gurgaon
df = pd.read_csv('dataset/appartments.csv', encoding='utf-8')
df['NearbyLocations'] = df['NearbyLocations'].fillna('')
df['TopFacilities'] = df['TopFacilities'].fillna('')

loc_vectorizer = TfidfVectorizer()
loc_tfidf = loc_vectorizer.fit_transform(df['NearbyLocations'])

fac_vectorizer = TfidfVectorizer()
fac_tfidf = fac_vectorizer.fit_transform(df['TopFacilities'])

def recommend_by_location_input(user_input, top_n=5):
    user_vec = loc_vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, loc_tfidf).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['PropertyName', 'PriceDetails', 'NearbyLocations', 'TopFacilities']].assign(SimilarityScore=sim_scores[top_indices])

def recommend_by_facilities_input(user_input, top_n=5):
    user_vec = fac_vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, fac_tfidf).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['PropertyName', 'PriceDetails', 'NearbyLocations', 'TopFacilities']].assign(SimilarityScore=sim_scores[top_indices])

def get_unique_locations():
    return sorted(set(", ".join(df['NearbyLocations']).split(", ")))

def get_unique_facilities():
    return sorted(set(", ".join(df['TopFacilities']).split(", ")))


# for banglore
b=pd.read_csv('dataset/Bangalore  house data.csv')
loc_vectorizer1 = TfidfVectorizer()
loc_tfidf1 = loc_vectorizer1.fit_transform(b['location'].fillna('')) 

fac_vectorizer1 = TfidfVectorizer()
fac_tfidf1 = fac_vectorizer1.fit_transform(b['price'].astype(str))
def recommend_by_location_input1(user_input, top_n=5):
    user_vec = loc_vectorizer1.transform([user_input])
    sim_scores = cosine_similarity(user_vec, loc_tfidf1).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return b.iloc[top_indices][['location', 'size', 'society', 'total_sqft', 'bath', 'balcony', 'price']].assign(SimilarityScore=sim_scores[top_indices])

def recommend_by_price_input(user_input, top_n=5):
    user_vec = fac_vectorizer1.transform([user_input])
    sim_scores = cosine_similarity(user_vec, fac_tfidf1).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return b.iloc[top_indices][['location', 'size', 'society', 'total_sqft', 'bath', 'balcony', 'price']].assign(SimilarityScore=sim_scores[top_indices])

def get_unique_price1():
    return sorted(p['price'].astype(str).unique())

def get_unique_locations1():
    return sorted(set(", ".join(b['location'].fillna('').astype(str)).split(", ")))




# for pune
p=pd.read_csv('dataset/Pune house data.csv')
loc_vectorizer2 = TfidfVectorizer()
loc_tfidf2 = loc_vectorizer2.fit_transform(p['site_location'].fillna('')) 

fac_vectorizer2 = TfidfVectorizer()
fac_tfidf2 = fac_vectorizer2.fit_transform(b['price'].astype(str))
def recommend_by_location_input2(user_input, top_n=5):
    user_vec = loc_vectorizer2.transform([user_input])
    sim_scores = cosine_similarity(user_vec, loc_tfidf2).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return p.iloc[top_indices][['site_location', 'size', 'society', 'total_sqft', 'bath', 'balcony', 'price']].assign(SimilarityScore=sim_scores[top_indices])

def recommend_by_price_input2(user_input, top_n=5):
    user_vec = fac_vectorizer2.transform([user_input])
    sim_scores = cosine_similarity(user_vec, fac_tfidf2).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return p.iloc[top_indices][['site_location', 'size', 'society', 'total_sqft', 'bath', 'balcony', 'price']].assign(SimilarityScore=sim_scores[top_indices])

def get_unique_locations2():
    return sorted(set(", ".join(p['site_location'].fillna('').astype(str)).split(", ")))


def get_unique_price2():
    return sorted(p['price'].astype(str).unique())

# for delhi
d=pd.read_csv('dataset/Delhi house data.csv')
loc_vectorizer3 = TfidfVectorizer()
loc_tfidf3 = loc_vectorizer3.fit_transform(d['Locality'].fillna('')) 

fac_vectorizer3 = TfidfVectorizer()
fac_tfidf3 = fac_vectorizer3.fit_transform(d['Price'].astype(str))
def recommend_by_location_input3(user_input, top_n=5):
    user_vec = loc_vectorizer3.transform([user_input])
    sim_scores = cosine_similarity(user_vec, loc_tfidf3).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return d.iloc[top_indices][['Locality', 'Area', 'BHK', 'Bathroom', 'Price', 'Per_Sqft']].assign(SimilarityScore=sim_scores[top_indices])

def recommend_by_price_input3(user_input, top_n=5):
    user_vec = fac_vectorizer3.transform([user_input])
    sim_scores = cosine_similarity(user_vec, fac_tfidf3).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return d.iloc[top_indices][['Locality', 'Area', 'BHK', 'Bathroom', 'Price', 'Per_Sqft']].assign(SimilarityScore=sim_scores[top_indices])

def get_unique_locations3():
    return sorted(set(", ".join(d['Locality']).split(", ")))

def get_unique_price3():
    return sorted(d['Price'].astype(str).unique())




