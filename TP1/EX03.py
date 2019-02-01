from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd

documents = ["This little kitty came to play when I was eating at arestaurant.",
             "Merley has the best squooshy kitten belly.", "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.", "Best cat photo I've ever taken.",
             "Climbing ninja cat.", "Impressed with google map feedback.", "Key promoter extension for Google Chrome."]
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# k-mean
true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
print("====================", "Top terms per cluster:", sep="\n")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
print("====================", "terms:", terms, sep="\n")

for i in range(true_k):
    # print("Cluster %d:" % i)
    print("Cluster {}".format(i))
    for ind in order_centroids[i, :10]:
        print('\t %s' % terms[ind])

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print("====================", "Prediction:", prediction, sep="\n")

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)

print("====================", "Prediction:", prediction, sep="\n")
prediction = model.predict(Y)
print("====================", "Prediction:", prediction, sep="\n")

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print("====================", "Prediction:", prediction, sep="\n")


# print("====================", "taux de classification est:", accuracy_score(Y, ),sep="\n")

def fct_map(data):
    """
    map all columns in the data set
    :param data: DataFrame
    :return: DataFrame
    """
    cloned_data = data.copy()
    attributes = list(data.columns)
    for attr in attributes:
        unique_values = cloned_data[attr].unique()  # List Unique Values In A pandas Column
        map_to_int = {key: value for value, key in enumerate(unique_values)}
        cloned_data[attr] = cloned_data[attr].replace(map_to_int)
    return cloned_data


data = pd.DataFrame(terms)
