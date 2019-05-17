
# Ranking McDonald's reviews using NLP based on rudeness

## Problem Statement

McDonald's receives **thousands of customer comments** on their website per day, and many of them are negative. Their corporate employees don't have time to read every single comment, but they do want to read a subset of comments that they are most interested in. In particular, the media has recently portrayed their employees as being rude, and so they want to review comments about **rude service**.

The goal is to develop a system that ranks each comment by the **likelihood that it is referring to rude service**. McDonald's corporate employees will use your system to build a "rudeness dashboard" for their corporate employees, so that employees can spend a few minutes each day examining the **most relevant recent comments**.

## Description of the data

Training data was collected by McDonald by using the [CrowdFlower platform](http://www.crowdflower.com/data-for-everyone) to pay humans to **hand-annotate** about 1500 comments with the **type of complaint**. The complaint types are listed below, with the encoding used in the data listed in parentheses:

- Bad Food (BadFood)
- Bad Neighborhood (ScaryMcDs)
- Cost (Cost)
- Dirty Location (Filthy)
- Missing Item (MissingFood)
- Problem with Order (OrderProblem)
- Rude Service (RudeService)
- Slow Service (SlowService)
- None of the above (na)

Our focus is on `RudeService`

## End result
The below output shows how the comments are ranked based on rudeness. 


```python
commentRankingByRudeness
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment</th>
      <th>rudeness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>My friend and I stopped in to get a late night snack and we were refused service. The store claimed to be 24 hours and the manager was standing right there doing paper work but would not help us. The cashier was only concerned with doing things for the drive thru and said that the manager said he wasn't allowed to help us. We thought it was a joke at first but when realized it wasn't we said goodbye and they just let us leave. I work in a restaurant and this is by far the worst service I have ever seen. I know it was late and maybe they didn't want to be there but it was completely ridiculous. I think the manager should be fired.Dallas</td>
      <td>0.999992</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Terrible service, I am never coming back here again. The lady was just dumb and idiotic, I hate this place. New Jersey</td>
      <td>0.962323</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ghetto lady helped me at the drive thru. Very rude and disrespectful to the co workers. Never coming back. Yuck!Los Angeles</td>
      <td>0.957038</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I've made at least 3 visits to this particular location just because it's right next to my office building.. and all my experience have been consistently bad.  There are a few helpers taking your orders throughout the drive-thru route and they are the worst. They rush you in placing an order and gets impatient once the order gets a tad bit complicated.  Don't even bother changing your mind oh NO! They will glare at you and snap at you if you want to change something.  I understand its FAST food, but I want my order placed right.  Not going back if I can help it.Portland</td>
      <td>0.399575</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Went through the drive through and ordered a #10 (cripsy sweet chili chicken wrap) without fries- the lady couldn't understand that I did not want fries and charged me for them anyways. I got the wrong order- a chicken sandwich and a large fries- my boyfriend took it back inside to get the correct order. The gentleman that ordered the chicken sandwich was standing there as well and she took the bag from my bf- glanced at the insides and handed it to the man without even offering to replace. I mean with all the scares about viruses going around... ugh DISGUSTING SERVICE. Then when she gave him the correct order my wrap not only had the sweet chili sauce on it, but the nasty (just not my first choice) ranch dressing on it!!!! I mean seriously... how lazy can you get!!!! I worked at McDonalds in Texas when I was 17 for about 8 months and I guess I was spoiled with good management. This was absolutely ridiculous. I was beyond disappointed.Las Vegas</td>
      <td>0.335338</td>
    </tr>
    <tr>
      <th>6</th>
      <td>This specific McDonald's is the bar I hold all other fast food joints to now. Been working in this area for 3 years now and gone to this location many times for drive-through pickup. Service is always fast, food comes out right, and the staff is extremely warm and polite.Atlanta</td>
      <td>0.271384</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Friendly people but completely unable to deliver what was ordered at the drive through.  Out of my last 6 orders they got it right 3 times.  Incidentally, the billing was always correct - they just could not read the order and deliver.  Very frustrating!Cleveland</td>
      <td>0.092585</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Why did I revisited this McDonald's  again.  I needed to use the restroom  facilities  and the women's bathroom didn't have soap, the floor was wet,  the bathroom stink, and the toilets were nasty. This McDonald's is very nasty.Houston</td>
      <td>0.059218</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Close to my workplace. It was well manged before. Now it's OK. The parking can be tight sometimes. Like all McDonald's, prices are getting expensive.New York</td>
      <td>0.039030</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Phenomenal experience. Efficient and friendly staff. Clean restrooms, good, fast service and bilingual staff. One of the best restaurants in the chain.Chicago</td>
      <td>0.008151</td>
    </tr>
  </tbody>
</table>
</div>



## Analysis
The analysis is divided into several tasks. The description of each task follows.

### Task 1

Read **`mcdonalds.csv`** into a pandas DataFrame and examine it. (It can be found in the **`data`** directory of the repository.)

- The **policies_violated** column lists the type of complaint. If there is more than one type, the types are separated by newline characters.
- The **policies_violated:confidence** column lists CrowdFlower's confidence in the judgments of its human annotators for that row (higher is better).
- The **city** column is the McDonald's location.
- The **review** column is the actual text comment.


```python
import pandas as pd
data = pd.read_csv('../data/mcdonalds.csv')
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1525 entries, 0 to 1524
    Data columns (total 11 columns):
    _unit_id                        1525 non-null int64
    _golden                         1525 non-null bool
    _unit_state                     1525 non-null object
    _trusted_judgments              1525 non-null int64
    _last_judgment_at               1525 non-null object
    policies_violated               1471 non-null object
    policies_violated:confidence    1471 non-null object
    city                            1438 non-null object
    policies_violated_gold          0 non-null float64
    review                          1525 non-null object
    Unnamed: 10                     0 non-null float64
    dtypes: bool(1), float64(2), int64(2), object(6)
    memory usage: 120.7+ KB


### Task 2

Remove any rows from the DataFrame in which the **policies_violated** column has a **null value**. Check the shape of the DataFrame before and after to confirm that you only removed about 50 rows.

- **Note:** Null values are also known as "missing values", and are encoded in pandas with the special value "NaN". This is distinct from the "na" encoding used by CrowdFlower to denote "None of the above". Rows that contain "na" should **not** be removed.
- **Hint:** [How do I handle missing values in pandas?](https://www.youtube.com/watch?v=fCMrO_VzeL8&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=16) explains how to do this.


```python
# remove rows where policies_violated is NaN
print(data.shape)
data = data[data.policies_violated.isna() == False]
print(data.shape)
```

    (1525, 11)
    (1471, 11)


### Task 3

Add a new column to the DataFrame called **"rude"** that is 1 if the **policies_violated** column contains the text "RudeService", and 0 if the **policies_violated** column does not contain "RudeService". The "rude" column is going to be your response variable, so check how many zeros and ones it contains.

- **Hint:** [How do I use string methods in pandas?](https://www.youtube.com/watch?v=bofaC0IckHo&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=12) shows how to search for the presence of a substring, and [How do I change the data type of a pandas Series?](https://www.youtube.com/watch?v=V0AWyzVMf54&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=13) shows how to convert the boolean results (True/False) to integers (1/0).


```python
# create a new column which contains 1 where policies_violated contains RudeService
data['rude'] = data.policies_violated.str.contains('RudeService').astype(int)
```


```python
# view the first few rows
data.loc[:5, ['policies_violated', 'rude']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>policies_violated</th>
      <th>rude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RudeService\nOrderProblem\nFilthy</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RudeService</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SlowService\nOrderProblem</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>na</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RudeService</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BadFood\nSlowService</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Task 4

1. Define X (the **review** column) and y (the **rude** column).
2. Split X and y into training and testing sets (using the parameter **`random_state=1`**).
3. Use CountVectorizer (with the **default parameters**) to create document-term matrices from X_train and X_test.


```python
# Define X (the review column) and y (the rude column).
X = data.review
y = data.rude
print(X.shape)
print(X.head())
print(y.shape)
print(y.head())
```

    (1471,)
    0    I'm not a huge mcds lover, but I've been to be...
    1    Terrible customer service. ŒæI came in at 9:30...
    2    First they "lost" my order, actually they gave...
    3    I see I'm not the only one giving 1 star. Only...
    4    Well, it's McDonald's, so you know what the fo...
    Name: review, dtype: object
    (1471,)
    0    1
    1    1
    2    0
    3    0
    4    1
    Name: rude, dtype: int64



```python
# Split X and y into training and testing sets (using the parameter random_state=1).
from sklearn.cross_validation import train_test_split
```


```python
# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(X_test.shape)
```

    (1103,)
    (368,)



```python
# use CountVectorizer to create document-term matrices from X_train and X_test
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
```


```python
# fit and transform X_train
X_train_dtm = vect.fit_transform(X_train)

# only transform X_test
X_test_dtm = vect.transform(X_test)

# print shape of X_train_dtm and X_test_dtm
print(X_train_dtm.shape)
print(X_test_dtm.shape)
```

    (1103, 7300)
    (368, 7300)



```python
# examine the last 50 features
print(vect.get_feature_names()[-50:])
```

    ['œæturns', 'œætwo', 'œæughhh', 'œæughhhhh', 'œæultimately', 'œæum', 'œæunfortunately', 'œæunreal', 'œæuntil', 'œæupon', 'œæuseless', 'œæusually', 'œævery', 'œæwait', 'œæwanna', 'œæwant', 'œæwas', 'œæwasn', 'œæway', 'œæwe', 'œæwell', 'œæwhat', 'œæwhatever', 'œæwhen', 'œæwhich', 'œæwhile', 'œæwho', 'œæwhy', 'œæwill', 'œæwish', 'œæwith', 'œæwon', 'œæword', 'œæwork', 'œæworkers', 'œæworst', 'œæwould', 'œæwow', 'œæwtf', 'œæya', 'œæyay', 'œæyeah', 'œæyears', 'œæyelp', 'œæyep', 'œæyes', 'œæyesterday', 'œæyet', 'œæyou', 'œæyour']



```python
vect
```




    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)



### Task 5

Fit a Multinomial Naive Bayes model to the training set, calculate the **predicted probabilites** (not the class predictions) for the testing set, and then calculate the **AUC**. Repeat this task using a logistic regression model to see which of the two models achieves a better AUC.

- **Note:** Because McDonald's only cares about ranking the comments by the likelihood that they refer to rude service, **classification accuracy** is not the relevant evaluation metric. **Area Under the Curve (AUC)** is a more useful evaluation metric for this scenario, since it measures the ability of the classifier to assign higher predicted probabilities to positive instances than to negative instances.
- **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains how to calculate predicted probabilities and AUC, and my [blog post and video](http://www.dataschool.io/roc-curves-and-auc-explained/) explain AUC in-depth.


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# define a function that accepts a vectorizer and calculates the accuracy
def tokenize_test(vect, model):
    
    # create document-term matrices using the vectorizer
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    
    # print the number of features that were generated
    print('Features: ', X_train_dtm.shape[1])
    
    # use passed model to predict the rudeness probabilities
    model.fit(X_train_dtm, y_train)
    y_pred_prob = model.predict_proba(X_test_dtm)[:,1]
    
    # print the AUC score of its predictions
    print('AUC score: ', metrics.roc_auc_score(y_test, y_pred_prob))
```


```python
# define MultinomialNB
nb = MultinomialNB()

# define CountVectorizer
vect = CountVectorizer()

# calculate the ROC-AUC-SCORE
print("Model: MultinomialNB")
tokenize_test(vect, nb)

# define LogisticRegression
lr = LogisticRegression()

# calculate the ROC-AUC-SCORE
print("Model: LogisticRegression")
tokenize_test(vect, lr)
```

    Model: MultinomialNB
    Features:  7300
    AUC score:  0.8426005404546177
    Model: LogisticRegression
    Features:  7300
    AUC score:  0.8233985058019394


### Task 6

Using either Naive Bayes or logistic regression (whichever one had a better AUC in the previous step), try **tuning CountVectorizer** using some of the techniques we learned in class. Check the testing set **AUC** after each change, and find the set of parameters that increases AUC the most.

- **Hint:** It is highly recommended that you adapt the **`tokenize_test()`** function from class for this purpose, since it will allow you to iterate quickly through different sets of parameters.


```python
# tune CountVectorizer to increase the AUC
vect = CountVectorizer(stop_words='english', max_df=0.3, min_df=4)
tokenize_test(vect, nb)
```

    Features:  1732
    AUC score:  0.8621522810364012


### Task 7

The **city** column might be predictive of the response, but we are not currently using it as a feature. Let's see whether we can increase the AUC by adding it to the model:

1. Create a new DataFrame column, **review_city**, that concatenates the **review** text with the **city** text. One easy way to combine string columns in pandas is by using the [`Series.str.cat()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.cat.html) method. Make sure to use the **space character** as a separator, as well as replacing **null city values** with a reasonable string value (such as 'na').
2. Redefine X as the **review_city** column, and re-split X and y into training and testing sets.
3. When you run **`tokenize_test()`**, CountVectorizer will simply treat the city as an extra word in the review, and thus it will automatically be included in the model! Check to see whether it increased or decreased the AUC of your **best model**.


```python
data['review_city'] = data.review.str.cat(data.city, sep=' ', na_rep='na')
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1471 entries, 0 to 1524
    Data columns (total 13 columns):
    _unit_id                        1471 non-null int64
    _golden                         1471 non-null bool
    _unit_state                     1471 non-null object
    _trusted_judgments              1471 non-null int64
    _last_judgment_at               1471 non-null object
    policies_violated               1471 non-null object
    policies_violated:confidence    1471 non-null object
    city                            1471 non-null object
    policies_violated_gold          0 non-null float64
    review                          1471 non-null object
    Unnamed: 10                     0 non-null float64
    rude                            1471 non-null int64
    review_city                     1471 non-null object
    dtypes: bool(1), float64(2), int64(3), object(7)
    memory usage: 190.8+ KB



```python
# Define X (the review_city column) and y (the rude column).
X = data.review_city
y = data.rude
print(X.shape)
print(X.head())
print(y.shape)
print(y.head())
```

    (1471,)
    0    I'm not a huge mcds lover, but I've been to be...
    1    Terrible customer service. ŒæI came in at 9:30...
    2    First they "lost" my order, actually they gave...
    3    I see I'm not the only one giving 1 star. Only...
    4    Well, it's McDonald's, so you know what the fo...
    Name: review_city, dtype: object
    (1471,)
    0    1
    1    1
    2    0
    3    0
    4    1
    Name: rude, dtype: int64



```python
# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(X_test.shape)
```

    (1103,)
    (368,)


Use the best model from the previous step and see if AUC improves using this `review_city` data.


```python
# Use the CountVectorizer that produced the best results
vect = CountVectorizer(stop_words='english', max_df=0.3, min_df=4)
tokenize_test(vect, nb)
```

    Features:  1739
    AUC score:  0.8648545541249404


This slightly improved the model's performance.

### Task 8

The **policies_violated:confidence** column may be useful, since it essentially represents a measurement of the training data quality. Let's see whether we can improve the AUC by only training the model using higher-quality rows!

To accomplish this, your first sub-task is to **calculate the mean confidence score for each row**, and then store those mean scores in a new column. For example, the confidence scores for the first row are `1.0\r\n0.6667\r\n0.6667`, so you should calculate a mean of `0.7778`. Here are the suggested steps:

1. Using the [`Series.str.split()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.split.html) method, convert the **policies_violated:confidence** column into lists of one or more "confidence scores". Save the results as a new DataFrame column called **confidence_list**.
2. Define a function that calculates the mean of a list of numbers, and pass that function to the [`Series.apply()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.apply.html) method of the **confidence_list** column. That will calculate the mean confidence score for each row. Save those scores in a new DataFrame column called **confidence_mean**.
    - **Hint:** [How do I apply a function to a pandas Series or DataFrame?](https://www.youtube.com/watch?v=P_q0tkYqvSk&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=30) explains how to use the `Series.apply()` method.


```python
import numpy as np

# calculate the mean of each entry in the Series
def calculateMean(value):
    valueList = value.split('\n')
    valueList = [float(j) for j in valueList]
    return np.mean(valueList)
```


```python
# apply a function to the Series
data['policies_violated:confidence'] = data['policies_violated:confidence'].apply(calculateMean)
```


```python
# check the column names
data.columns
```




    Index(['_unit_id', '_golden', '_unit_state', '_trusted_judgments',
           '_last_judgment_at', 'policies_violated',
           'policies_violated:confidence', 'city', 'policies_violated_gold',
           'review', 'Unnamed: 10', 'rude', 'review_city'],
          dtype='object')




```python
# rename a specific column
data.columns = data.columns.str.replace('policies_violated:confidence', 'confidence_mean')
```

Your second sub-task is to **remove lower-quality rows from the training set**, and then repeat the model building and evaluation process. Here are the suggested steps:

1. Remove all rows from X_train and y_train that have a **confidence_mean lower than 0.75**. Check their shapes before and after to confirm that you removed about 300 rows.
2. Use the **`tokenize_test()`** function to check whether filtering the training data increased or decreased the AUC of your **best model**.
    - **Hint:** Even though X_train and y_train are separate from the mcd DataFrame, they can still be filtered using a boolean Series generated from mcd because all three objects share the same index.
    - **Note:** It's important that we don't remove any rows from the testing set (X_test and y_test), because the testing set should be representative of the real-world data we will encounter in the future (which will contain both high-quality and low-quality rows).


```python
# create X_train and y_train
# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print("Before removing rows..")
print("X_train: %s" %(X_train.shape))
print("X_test: %s" %(X_test.shape))
print("y_train: %s" %(y_train.shape))
print("y_test: %s" %(y_test.shape))

# remove lower confidence rows from the training data
X_train = X_train[data.confidence_mean >= 0.75]
y_train = y_train[data.confidence_mean >= 0.75]

print("\nAfter removing rows..")
print("X_train: %s" %(X_train.shape))
print("X_test: %s" %(X_test.shape))
print("y_train: %s" %(y_train.shape))
print("y_test: %s" %(y_test.shape))
```

    Before removing rows..
    X_train: 1103
    X_test: 368
    y_train: 1103
    y_test: 368
    
    After removing rows..
    X_train: 799
    X_test: 368
    y_train: 799
    y_test: 368



```python
# Use the CountVectorizer that produced the best results
vect = CountVectorizer(stop_words='english', max_df=0.3, min_df=4)
tokenize_test(vect, nb)
```

    Features:  1353
    AUC score:  0.8496900333810206


Even after removing the noise, we were able to get similar result.

### Task 9

New comments have been submitted to the McDonald's website, and you need to **score them with the likelihood** that they are referring to rude service.

1. Before making predictions on out-of-sample data, it is important to re-train your model on all relevant data using the tuning parameters and preprocessing steps that produced the best AUC above.
    - In other words, X should be defined using either **all rows** or **only those rows with a confidence_mean of at least 0.75**, whichever produced a better AUC above.
    - X should refer to either the **review column** or the **review_city column**, whichever produced a better AUC above.
    - CountVectorizer should be instantiated with the **tuning parameters** that produced the best AUC above.
    - **`train_test_split()`** should not be used during this process.
2. Build a document-term matrix (from X) called **X_dtm**, and examine its shape.
3. Read the new comments stored in **`mcdonalds_new.csv`** into a DataFrame called **new_comments**, and examine it.
4. If your model uses a **review_city** column, create that column in the new_comments DataFrame. (Otherwise, skip this step.)
5. Build a document_term matrix (from the **new_comments** DataFrame) called **new_dtm**, and examine its shape.
6. Train your best model (Naive Bayes or logistic regression) using **X_dtm** and **y**.
7. Predict the "rude probability" for each comment in **new_dtm**, and store the probabilities in an object called **new_pred_prob**.
8. Print the **full text** for each new comment alongside its **"rude probability"**. (You may need to [increase the max_colwidth](https://www.youtube.com/watch?v=yiO43TQ4xvc&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=28) to see the full text.) Examine the results, and comment on how well you think the model performed!


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1471 entries, 0 to 1524
    Data columns (total 13 columns):
    _unit_id                  1471 non-null int64
    _golden                   1471 non-null bool
    _unit_state               1471 non-null object
    _trusted_judgments        1471 non-null int64
    _last_judgment_at         1471 non-null object
    policies_violated         1471 non-null object
    confidence_mean           1471 non-null float64
    city                      1471 non-null object
    policies_violated_gold    0 non-null float64
    review                    1471 non-null object
    Unnamed: 10               0 non-null float64
    rude                      1471 non-null int64
    review_city               1471 non-null object
    dtypes: bool(1), float64(3), int64(3), object(6)
    memory usage: 190.8+ KB



```python
# Define your X and y
X = data.review_city
y = data.rude
```


```python
# initialize CountVectorizer
# Use the CountVectorizer that produced the best results
vect = CountVectorizer(stop_words='english', max_df=0.3, min_df=4)
```


```python
# Create X-dtm
X_dtm = vect.fit_transform(X)
```


```python
print(X_dtm.shape)
```

    (1471, 2104)



```python
# read in new comments
new_comments = pd.read_csv('../data/mcdonalds_new.csv')
```


```python
# examine new comments
new_comments.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Las Vegas</td>
      <td>Went through the drive through and ordered a #...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chicago</td>
      <td>Phenomenal experience. Efficient and friendly ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Los Angeles</td>
      <td>Ghetto lady helped me at the drive thru. Very ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>New York</td>
      <td>Close to my workplace. It was well manged befo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Portland</td>
      <td>I've made at least 3 visits to this particular...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# combine review and city since we are using review_city for X
new_comments['review_city'] = new_comments.review.str.cat(new_comments.city)
```


```python
# examine new comments
new_comments.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>review</th>
      <th>review_city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Las Vegas</td>
      <td>Went through the drive through and ordered a #...</td>
      <td>Went through the drive through and ordered a #...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chicago</td>
      <td>Phenomenal experience. Efficient and friendly ...</td>
      <td>Phenomenal experience. Efficient and friendly ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Los Angeles</td>
      <td>Ghetto lady helped me at the drive thru. Very ...</td>
      <td>Ghetto lady helped me at the drive thru. Very ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>New York</td>
      <td>Close to my workplace. It was well manged befo...</td>
      <td>Close to my workplace. It was well manged befo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Portland</td>
      <td>I've made at least 3 visits to this particular...</td>
      <td>I've made at least 3 visits to this particular...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# build a dtm from new_comments for review_city
test_data = new_comments.review_city

# filter rows which contain na
test_data = test_data[test_data.notnull()]

# my custom comment
test_text = 'Terrible service, I am never coming back here again. \
The lady was just dumb and idiotic, I hate this place. New Jersey'

# append my comment to the original test data
test_data = test_data.append(pd.Series([test_text]), ignore_index=True)

# construct a document-term-matrix of test data
test_dtm = vect.transform(test_data)
print(test_dtm.shape)
```

    (10, 2104)



```python
# Train your best model

# define MultinomialNB
nb = MultinomialNB()

# Use the X_dtm and y to train the model
nb.fit(X_dtm, y)

# calculate the predicted probabilities on test_dtm
y_test_pred_prob = nb.predict_proba(test_dtm)[:, 1]

# print the predicted probabilities
print(y_test_pred_prob)
```

    [0.33533776 0.00815105 0.95703848 0.03903012 0.39957542 0.05921756
     0.27138442 0.99999236 0.092585   0.96232305]



```python
# widen the column display
pd.set_option('display.max_colwidth', 1000)
```


```python
# print the comment and it's probability of being rude
commentRankingByRudeness = pd.DataFrame({'comment': test_data, 
                                         'rudeness': y_test_pred_prob}).sort_values('rudeness', ascending=False)

commentRankingByRudeness
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment</th>
      <th>rudeness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>My friend and I stopped in to get a late night snack and we were refused service. The store claimed to be 24 hours and the manager was standing right there doing paper work but would not help us. The cashier was only concerned with doing things for the drive thru and said that the manager said he wasn't allowed to help us. We thought it was a joke at first but when realized it wasn't we said goodbye and they just let us leave. I work in a restaurant and this is by far the worst service I have ever seen. I know it was late and maybe they didn't want to be there but it was completely ridiculous. I think the manager should be fired.Dallas</td>
      <td>0.999992</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Terrible service, I am never coming back here again. The lady was just dumb and idiotic, I hate this place. New Jersey</td>
      <td>0.962323</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ghetto lady helped me at the drive thru. Very rude and disrespectful to the co workers. Never coming back. Yuck!Los Angeles</td>
      <td>0.957038</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I've made at least 3 visits to this particular location just because it's right next to my office building.. and all my experience have been consistently bad.  There are a few helpers taking your orders throughout the drive-thru route and they are the worst. They rush you in placing an order and gets impatient once the order gets a tad bit complicated.  Don't even bother changing your mind oh NO! They will glare at you and snap at you if you want to change something.  I understand its FAST food, but I want my order placed right.  Not going back if I can help it.Portland</td>
      <td>0.399575</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Went through the drive through and ordered a #10 (cripsy sweet chili chicken wrap) without fries- the lady couldn't understand that I did not want fries and charged me for them anyways. I got the wrong order- a chicken sandwich and a large fries- my boyfriend took it back inside to get the correct order. The gentleman that ordered the chicken sandwich was standing there as well and she took the bag from my bf- glanced at the insides and handed it to the man without even offering to replace. I mean with all the scares about viruses going around... ugh DISGUSTING SERVICE. Then when she gave him the correct order my wrap not only had the sweet chili sauce on it, but the nasty (just not my first choice) ranch dressing on it!!!! I mean seriously... how lazy can you get!!!! I worked at McDonalds in Texas when I was 17 for about 8 months and I guess I was spoiled with good management. This was absolutely ridiculous. I was beyond disappointed.Las Vegas</td>
      <td>0.335338</td>
    </tr>
    <tr>
      <th>6</th>
      <td>This specific McDonald's is the bar I hold all other fast food joints to now. Been working in this area for 3 years now and gone to this location many times for drive-through pickup. Service is always fast, food comes out right, and the staff is extremely warm and polite.Atlanta</td>
      <td>0.271384</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Friendly people but completely unable to deliver what was ordered at the drive through.  Out of my last 6 orders they got it right 3 times.  Incidentally, the billing was always correct - they just could not read the order and deliver.  Very frustrating!Cleveland</td>
      <td>0.092585</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Why did I revisited this McDonald's  again.  I needed to use the restroom  facilities  and the women's bathroom didn't have soap, the floor was wet,  the bathroom stink, and the toilets were nasty. This McDonald's is very nasty.Houston</td>
      <td>0.059218</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Close to my workplace. It was well manged before. Now it's OK. The parking can be tight sometimes. Like all McDonald's, prices are getting expensive.New York</td>
      <td>0.039030</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Phenomenal experience. Efficient and friendly staff. Clean restrooms, good, fast service and bilingual staff. One of the best restaurants in the chain.Chicago</td>
      <td>0.008151</td>
    </tr>
  </tbody>
</table>
</div>


