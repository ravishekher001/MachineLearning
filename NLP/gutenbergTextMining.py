import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import nltk
import string
from pprint import pprint
from gutenbergPreprocessing import *

def getTextFromURLByRemovingHeaders(book_urls):
    book_texts = []
    for url in book_urls:
        book_text = requests.get(url).text
        book_text = remove_guetenburg_headers(book_text)
        book_text = remove_gutenberg_headers_till_contents(book_text)
        book_text = remove_gutenberg_headers_till_illustration(book_text)
        book_texts.append(remove_gutenberg_footer(book_text))
    return book_texts

def cleanTextBooks(book_texts):
    clean_books = []
    for book in book_texts:
        book_i = tokenize_text(book)
        book_i = remove_characters_after_tokenization(book_i)
        book_i = convert_to_lowercase(book_i)
        book_i = remove_stopwords(book_i)
        book_i = apply_stemming_and_lemmatize(book_i)
        clean_books.append(book_i)
    return clean_books

def normalizedVocabularyScore(clean_books):
    v_size = [len(set(book)) for book in clean_books]
    max_v_size = np.max(v_size)
    v_raw_score = v_size/max_v_size
    v_sqrt_score = np.sqrt(v_raw_score)
    v_rank_score = pd.Series(v_size).rank()/len(v_size)
    v_final_score = (pd.Series(v_sqrt_score) + v_rank_score)/2
    
    return pd.DataFrame({'v_size': v_size,
                        'v_raw_score': v_raw_score,
                        'v_sqrt_score': v_sqrt_score,
                        'v_rank_score': v_rank_score,
                        'v_final_score': v_final_score})

def longWordVocabularySize(clean_book, minChar=10):
    V = set(clean_book)
    long_words = [w for w in V if len(w) > minChar]
    return len(long_words)

def normalizedLongWordVocabularyScore(clean_books):
    lw_v_size = [longWordVocabularySize(book) for book in clean_books]
    max_v_size = np.max(lw_v_size)
    v_raw_score = lw_v_size/max_v_size
    v_sqrt_score = np.sqrt(v_raw_score)
    v_rank_score = pd.Series(lw_v_size).rank()/len(lw_v_size)
    lw_v_final_score = (pd.Series(v_sqrt_score) + v_rank_score)/2
    
    return pd.DataFrame({'lw_v_size': lw_v_size,
                        'lw_v_final_score': lw_v_final_score})


def textDifficultyScore(clean_books):
    df_vocab_scores = normalizedVocabularyScore(clean_books)
    df_lw_vocab_scores = normalizedLongWordVocabularyScore(clean_books)
    lexical_diversity_scores = [len(set(book))/len(book) for book in clean_books]
    
    text_difficulty = (df_vocab_scores['v_final_score'] + \
                     df_lw_vocab_scores['lw_v_final_score'] + \
                     lexical_diversity_scores)/3
    
    return pd.DataFrame({'text_difficulty': text_difficulty})