import streamlit as st
import random
import numpy as np
import pandas as pd
import nltk
from transformers import set_seed

nltk.download('punkt')
nltk.download('punkt_tab')

set_seed(87)

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    data = data[data["model"] == "checkpoint-41448"]
    data = data.drop(["Unnamed: 0", "model"], axis=1)

    queries = []
    queries = [x for x in data["target_query"]]
    queries += [x for x in data["generated_output"]]

    return queries

@st.cache_data
def prepare_sentences(queries):
    sentences = [s.strip() for s in queries]
    sentences = [s for s in sentences if len(s) > 0]
    
    return sentences    

@st.cache_data
def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)

    return tokenized_sentences

@st.cache_data
def get_tokenized_data(file):
    
    queries = load_data(file)
    sentences = prepare_sentences(queries)
    tokenized_sentences = tokenize_sentences(sentences)
    
    return tokenized_sentences

@st.cache_data
def split_data(tokenized_sentences, split_percentage=0.8):
    random.shuffle(tokenized_sentences)

    train_size = int(len(tokenized_sentences) * split_percentage)
    train_data = tokenized_sentences[0:train_size]
    test_data = tokenized_sentences[train_size:]

    print("{} data are split into {} train and {} test set".format(
        len(tokenized_sentences), len(train_data), len(test_data)))

    print("First training sample:")
    print(train_data[0])
        
    print("First test sample")
    print(test_data[0])
    
    return train_data, test_data

def count_words(tokenized_sentences):

    word_counts = {}
    for sentence in tokenized_sentences: 
        for token in sentence: 

            if token not in word_counts:
                word_counts[token] = 1
            else:
                word_counts[token] += 1
    
    return word_counts

def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):

    closed_vocab = []
    word_counts = count_words(tokenized_sentences)

    for word, cnt in word_counts.items(): 
        if cnt >= count_threshold:
            closed_vocab.append(word)
    
    return closed_vocab

def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    vocabulary = set(vocabulary)
    
    replaced_tokenized_sentences = []
    
    for sentence in tokenized_sentences:

        replaced_sentence = []
        for token in sentence: 
            if token in vocabulary: 
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)
        
        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences

@st.cache_data
def preprocess_data(train_data, test_data, count_threshold, unknown_token="<unk>", get_words_with_nplus_frequency=get_words_with_nplus_frequency, replace_oov_words_by_unk=replace_oov_words_by_unk):
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token)
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token)
    
    return train_data_replaced, test_data_replaced, vocabulary

@st.cache_data
def count_n_grams(data, n, start_token='<s>', end_token = '<e>'):

    n_grams = {}
    for sentence in data: 
        sentence = [start_token] * (n) + sentence + [end_token]
        sentence = tuple(sentence)
        for i in range(len(sentence) -n +1): 
            n_gram = sentence[i:i+n]
            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    
    return n_grams

def estimate_probability(word, previous_n_gram, 
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + k * vocabulary_size
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    numerator = n_plus1_gram_count + k
    probability = numerator/ denominator
    
    return probability

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>",  k=1.0):

    previous_n_gram = tuple(previous_n_gram)    
    vocabulary = vocabulary + [end_token, unknown_token]    
    vocabulary_size = len(vocabulary)    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, 
                                           n_gram_counts, n_plus1_gram_counts, 
                                           vocabulary_size, k=k)
                
        probabilities[word] = probability

    return probabilities

def make_count_matrix(n_plus1_gram_counts, vocabulary):

    vocabulary = vocabulary + ["<e>", "<unk>"]
    
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]        
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    
    row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}    
    col_index = {word:j for j, word in enumerate(vocabulary)}    
    
    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix

def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>', end_token = '<e>', k=1.0):

    n = len(list(n_gram_counts.keys())[0]) 
    sentence = [start_token] * n + sentence + [end_token]
    sentence = tuple(sentence)
    N = len(sentence)

    product_pi = 1.0
    for t in range(n, N):
        n_gram = sentence[t-n:t]
        word = sentence[t]
        probability = estimate_probability(word, n_gram, 
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)
        product_pi *= (1/probability)

    perplexity = (product_pi)**(1/N)

    return perplexity

def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>", k=1.0, start_with=None):

    n = len(list(n_gram_counts.keys())[0])
    previous_tokens = ['<s>'] * n + previous_tokens
    previous_n_gram = previous_tokens[-n:]
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)
    
    suggestion = None
    max_prob = 0
    for word, prob in probabilities.items(): 
        if start_with is not None: 
            if not word.startswith(start_with): 
                continue
        if prob >= max_prob:
            suggestion = word
            max_prob = prob
    
    return suggestion, max_prob

def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions

def show_recommendations(recommendations):

    unique_recommendations = list(set(item[0] for item in recommendations))
    print(unique_recommendations)
    
    for item in unique_recommendations:
        if item not in ["<e>", "<unk>"]: 
            st.markdown(
                f'<div style="padding: 10px; background-color: #25252d; border-radius: 5px; margin-bottom: 5px;">'
                f'{item}</div>', 
                unsafe_allow_html=True
            )

def main():
    st.set_page_config(page_title="Autocomplete", page_icon="🌐", layout="centered")
    # tokenized_sentences = get_tokenized_data("data.csv")
    # train_data, test_data = split_data(tokenized_sentences)

    # minimum_freq = 1
    # train_data_processed, _, vocabulary = preprocess_data(train_data, 
    #                                                                     test_data, 
    #                                                                     minimum_freq)
    # n_gram_counts_list = []
    # for n in range(1, 6):
    #     print("Computing n-gram counts with n =", n, "...")
    #     n_model_counts = count_n_grams(train_data_processed, n)
    #     n_gram_counts_list.append(n_model_counts)

    if 'n_gram_counts_list' not in st.session_state:
        tokenized_sentences = get_tokenized_data("data.csv")
        train_data, test_data = split_data(tokenized_sentences)

        minimum_freq = 1
        train_data_processed, _, vocabulary = preprocess_data(train_data, test_data, minimum_freq)
        
        n_gram_counts_list = []
        for n in range(1, 6):
            n_model_counts = count_n_grams(train_data_processed, n)
            n_gram_counts_list.append(n_model_counts)

        # Store in session state to persist across reruns
        st.session_state.n_gram_counts_list = n_gram_counts_list
        st.session_state.vocabulary = vocabulary
        st.session_state.train_data_processed = train_data_processed

    # Retrieve from session state
    n_gram_counts_list = st.session_state.n_gram_counts_list
    vocabulary = st.session_state.vocabulary

    st.title(
        """Autocomplete tool testing"""
    )

    st.write("Enter text below and watch for the magic")

    if 'text_search' not in st.session_state:
        st.session_state.text_search = ""

    def update_text_search():
        st.session_state.text_search = st.session_state.input_text

    st.text_input("Type something...", 
                  key="input_text", 
                  value=st.session_state.text_search, 
                  on_change=update_text_search)

    if st.session_state.text_search:
        recommendations = get_suggestions(st.session_state.text_search.split(" "), n_gram_counts_list, vocabulary, k=1.0)
        show_recommendations(recommendations)

if __name__ == "__main__":
    main()