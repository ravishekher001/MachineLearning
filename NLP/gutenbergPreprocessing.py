import re
import nltk
import re
import string
from pprint import pprint
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

# text cleaning
def remove_guetenburg_headers(book_text):
    book_text = book_text.replace('\r', '')
    book_text = book_text.replace('\n', ' ')
    start_match = re.search(r'\*{3}\s?START.+?\*{3}', book_text)
    end_match = re.search(r'\*{3}\s?END.+?\*{3}', book_text)
    try:
        book_text = book_text[start_match.span()[1]:end_match.span()[0]]
    except AttributeError:
        print('No match found')    
    return book_text

def remove_gutenberg_headers_till_contents(book_text):
    if book_text.find('CONTENTS') != -1:
        book_text = book_text[book_text.find('CONTENTS'):]
    return book_text

def remove_gutenberg_headers_till_illustration(book_text):
    if book_text.find('[Illustration') != -1:
        book_text = book_text[book_text.find('[Illustration'):]
    return book_text

def remove_gutenberg_footer(book_text):
    if book_text.find('End of the Project Gutenberg') != -1:
        book_text = book_text[:book_text.find('End of the Project Gutenberg')]
    elif book_text.find('End of Project Gutenberg') != -1:
        book_text = book_text[:book_text.find('End of Project Gutenberg')]
    return book_text

# text pre-processing
def tokenize_text(book_text):
    TOKEN_PATTERN = r'\s+'
    regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN, gaps=True)
    word_tokens = regex_wt.tokenize(book_text)
    return word_tokens

def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation))) 
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens]) 
    return filtered_tokens

def convert_to_lowercase(tokens):
    return [token.lower() for token in tokens if token.isalpha()]

def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

def apply_stemming_and_lemmatize(tokens, ls=LancasterStemmer(), wnl=WordNetLemmatizer()):
    return [wnl.lemmatize(ls.stem(token)) for token in tokens]

