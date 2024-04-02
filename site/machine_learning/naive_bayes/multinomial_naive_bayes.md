# Multinomial Naive Bayes

![Classification](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Classification.svg)

Multinomial Naive Bayes is a variant of Naive Bayes algorithm suited for classification and espcially on tasks like counting or calculating frequencies.

## Assumption

Multinomial Naive Bayes should be used only on discrete or categorical independent variables. It also assumes that features follow a [multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution) (hence the name), which means that each feature represents the frequency with which it appears in an observation. 

## Sentiment classification on customer review

Multinomial naive bayes performs well on text classification. One example is the analysis of customer reviews on a product. The goal here is to build a model capable of recognizing whether the review is positive or negative to provide insights on each products.

I found a great dataset from [kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) containing e-commerce reviews on several products. We will try identify which product receives bad critics based on the reviews.

### Preprocessing

For this little analysis, we need a great preprocessing on our text. For this NLP task, we can simplify the text by 
- removing useless words such as **stop-words** (recurrent words such as: *I*, *will*, *can*, ...)
- remove punctuation
- reduce the meaning of similar words using techniques such as **Stemming** or **Lemmatization**

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/naive_bayes/NLP_Preprocessing_Pipeline.png
NLP Preprocessing techniques used
```

```python
# Imports
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download("stopwords")
nltk.download('punkt')

import re

df = pd.read_csv("review.csv")
df = df[["Review Text", "Recommended IND"]]
df.dropna(inplace=True)
y = df["Recommended IND"].to_numpy()

# Preprocessing
stemmer = PorterStemmer()
url_remove = re.compile(r'http?://\S+')
punkt_remove = re.compile(r'[^\w\s]')

def remove_url(text: str) -> str:
    return url_remove.sub('', text)

def lower_case(text :str) -> str:
    return text.lower()

def stemming(text: list) -> list:
    return [stemmer.stem(token) for token in text]

def remove_punct(text: list) -> list:
    return [word for word in text if not word in set()]

def remove_stop_words(text: list) -> list:
    return [word for word in text if not word in set(stopwords.words('english'))]

def remove_punctuation(text: list) -> list:
    text = [punkt_remove.sub('', word) for word in text]
    return [word for word in text if len(word) > 1]

def join_text(text: list) -> str:
    return text.join()

corpus = df["Review Text"].to_list()

actions = [lower_case, remove_url, word_tokenize, stemming, remove_stop_words, remove_punctuation, join_text]
for i in range(len(corpus)):
    for action in actions:
        corpus[i] = action(corpus[i])
```

Example of a preprocessed sentence:

*I love, love, love this jumpsuit. it's fun, flirty, and fabulous! every time i wear it, i get nothing but great compliments!*

becomes

*love love love thi jumpsuit fun flirti fabul everi time wear get noth great compliment*

Of course this method is not perfect but it is fast and appropriate for our dataset.

### Training

Now we have preprocessed our dataset, we can start training our dataset with a multinomial naive bayes model. Naive Bayes algorithm only works with number, we need to provide statistics over our reviews to the model rather than the preprocessed text.

There are crazy techniques such as using embedding vectors to keep the meaning of a sentence but since we consider that each words are unrelated.
We can provide more simple insights on the sentence. Here we calculate the number of words found in the review for each of the possible words of the entire dataset: this is called the **Bag-of-Word** (BoW) technique.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(corpus)

vectorizer.vocabulary_
```

We find 13048 different words in our text: that's a lot for only roughly 20000 reviews.

Now we can fit our Multinomial Naive Bayes with this BoW.

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X = vectorizer.transform(corpus)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)
```

The training and text transformation takes less than 1 second, the method is really efficient and easy to understand.

```{note}
BoW is appropriate as input to our model since it captures the frequency of each words in the corpus.
```

### Testing

```python
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
```

We get a great accuracy of almost **0.89**, this is great !
Now we can determine with a correct accuracy whether the customer liked or not the product based on its review.

## Problems with Multinomial Naive Bayes

This technique can be really useful for text analysis. However, it performs really bad when 
- reviews have words are really different from the original dataset. 
- the data are continuous

Naive Bayes model should be used in appropriate context, for example when we want a first insight on the data.
New techniques transforming a sentence into an embedding vector catching more meaningful realtionship between words can be used for more complex tasks.