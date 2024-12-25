## Programming Assignment 2 for NLP

### Overview

This project involves various text preprocessing techniques applied to a dataset of text entries. The preprocessing steps include tokenization, finding emoticons, stop words, addresses, phone numbers, and account numbers, as well as text cleaning and stemming/lemmatization.

> **Important Note**: Ensure that the data files are placed under the `data` directory for the notebook to function correctly.

### Files

- `final.ipynb`: Jupyter notebook containing the code for text preprocessing and analysis.
- `data/train.csv`: The dataset used for training and analysis.

### Preprocessing Steps

1. **Tokenization**:
    - Sentence Tokenization
    - Word Tokenization

2. **Finding Specific Patterns**:
    - Emoticons
    - Stop Words
    - Addresses
    - Phone Numbers
    - Account Numbers

3. **Text Cleaning**:
    - Removing non-alphabetic characters
    - Converting text to lowercase
    - Removing stop words

4. **Stemming and Lemmatization**:
    - Applying Porter Stemmer
    - Applying WordNet Lemmatizer

### Usage

1. **Load the Dataset**:
    ```python
    train_df = pd.read_csv('./data/train.csv')
    ```

2. **Apply Preprocessing Functions**:
    ```python
    train_df['Sentences'] = train_df['Text'].apply(sentence_tokenizer)
    train_df['Words'] = train_df['Text'].apply(word_tokenizer)
    train_df['Emoticons'] = train_df['Text'].apply(find_emoticons)
    train_df['StopWords'] = train_df['Words'].apply(find_stop_words)
    train_df['Addresses'] = train_df['Text'].apply(find_address)
    train_df['PhoneNumbers'] = train_df['Text'].apply(find_phone_numbers)
    train_df['AccountNumbers'] = train_df['Text'].apply(find_account_numbers)
    ```

3. **Clean and Stem/Lemmatize Text**:
    ```python
    train_df_cleaned['Text'] = train_df_cleaned['Text'].apply(preprocess_text)
    train_df_cleaned['Words'] = train_df_cleaned['Words'].apply(stem_lem)
    ```

4. **Compute Statistics**:
    ```python
    stats_before = {
         'avg_sent_length': train_df['Sentences'].apply(len).mean(),
         'max_sent_length': train_df['Sentences'].apply(len).max(),
         'min_sent_length': train_df['Sentences'].apply(len).min(),
         'sent_count': train_df['Sentences'].apply(len).sum(),
         'word_count': train_df['Words'].apply(len).sum(),
         'vocab_size': len(set([word for words in train_df['Words'] for word in words])),
         'max_word_length': max([len(word) for words in train_df['Words'] for word in words]),
         'num_emoticons': train_df['Emoticons'].apply(len).sum(),
         'num_stop_words': train_df['StopWords'].apply(len).sum(),
         'num_lowercase': train_df['Words'].apply(lambda x: len([word for word in x if word.islower()])).sum(),
         'num_special_chars': train_df['Text'].apply(lambda x: len([char for char in x if not char.isalnum()])).sum(),
         'num_addresses': train_df['Addresses'].apply(len).sum(),
         'num_phone_numbers': train_df['PhoneNumbers'].apply(len).sum(),
         'num_account_numbers': train_df['AccountNumbers'].apply(len).sum(),
         'processing_time': normal_end - normal_start
    }
    ```

### Results

The notebook prints out statistics before and after text cleaning, including average sentence length, word count, vocabulary size, and the number of special characters, emoticons, addresses, phone numbers, and account numbers found.

### Dependencies

- pandas
- nltk
- pyap
- re
- time

Make sure to install the required libraries before running the notebook:
```bash
pip install pandas nltk pyap
```

### Running the Notebook

Open `final.ipynb` in Jupyter Notebook or JupyterLab and run all cells to see the preprocessing steps and results.