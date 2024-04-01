# %%
# !pip install pyiwn
# !pip install googletrans==4.0.0-rc1
# !pip install datasets

# %%
import pyiwn
pyiwn.download()
iwn = pyiwn.IndoWordNet()

# %%
synset=synsets[0]
synset.pos()
synset.lemmas()
synset.gloss()  
synset.examples()

# %%
synsets=iwn.synsets('समय', pos='noun')

# %%
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  

distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

input_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
attention_mask = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
output = distilbert_model([input_ids, attention_mask])[0]
output = tf.keras.layers.GlobalAveragePooling1D()(output)
output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs, batch_size=batch_size)

loss, accuracy = model.evaluate(test_dataset)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# %%
from nltk.corpus import wordnet

def augment_dataset(sentence, num_augmented_samples=5):
    augmented_samples = []
    synonyms = []
    for word in sentence.split():
        synonyms.extend([lemma.name() for synset in synsets for lemma in synset.lemmas()])
    synonyms = list(set(synonyms))  # Remove duplicates
    for synonym in synonyms[:num_augmented_samples]:
        augmented_samples.append(sentence.replace(word, synonym))
    return augmented_samples

original_sentence = "आज मौसम बहुत अच्छा है।"
augmented_sentences = augment_dataset(original_sentence)
print("Original Sentence:", original_sentence)
print("Augmented Sentences:", augmented_sentences) 

# %% [markdown]
# #this is for semantic similarity
# 
# first  Interval- Valued Fuzzy Hindi WordNet graph
# 
# intrigrating this as a layer to my distilled bert
# 
# training with the translated data
# 

# %% [markdown]
# next to find the redundant sentences using the ssimilarity score and find removing them.
# 
# then v summrize the paragraph

# %%
import torch
from transformers import AutoTokenizer, AutoModel

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%


# %%
device

# %% [markdown]
# PREPROCESSING

# %%







# %%
from datasets import load_dataset
dataset = load_dataset("stsb_multi_mt", name="en", split="train")


# %%
dataset

# %%
import pandas as pd
df = dataset.to_pandas()

# %%
df.head()
df.similarity_score.hist()


# %%
df.shape

# %%
# !pip install googletrans==4.0.0-rc1

# %%
from googletrans import Translator

def translate_to_hindi(text):
    translator = Translator()
    translated_text = translator.translate(text, src='en', dest='hi')
    return translated_text.text

def translate_to_english(text):
    translator = Translator()
    translated_text = translator.translate(text, src='hi', dest='en')
    return translated_text.text

# English text to be translated
english_text = "It is very cold tonight."
h2="आज शाम को ठंड बहुत ज्यादा है।"
# Translate English text to Hindi
hindi_text = translate_to_hindi(english_text)
e2=translate_to_english(h2)
print("English:", english_text)
print("Hindi:", hindi_text)


print("Hindi:", h2)
print("English:", e2)

# %%
from sklearn.preprocessing import MinMaxScaler

# Create a scaler object
scaler = MinMaxScaler()

# Fit the scaler to the 'similarity_score' column and transform it
df['similarity_score_1'] = scaler.fit_transform(df[['similarity_score']])

# %%
df.head()

# %%
from transformers import MarianTokenizer , MarianMTModel
def translate_to_hindi_batch(texts, model_name='Helsinki-NLP/opus-mt-en-hi'):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    tokenized_texts = tokenizer.prepare_seq2seq_batch(texts, return_tensors='pt')
    translated = model.generate(**tokenized_texts)
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_texts

# %%
english_text = "It is very cold tonight."
hindi_text = translate_to_hindi_batch(english_text)

print("English:", english_text)
print("Hindi:", hindi_text)


# %%
from multiprocessing import Pool
import numpy as np
def parallelize_dataframe(df, func, n_cores=16):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def apply_translate_to_hindi(df):
    df['s1'] = df['sentence1'].apply(translate_to_hindi)
    return df

# df = parallelize_dataframe(df, apply_translate_to_hindi)

# %%
from multiprocessing import Pool
import numpy as np

# %%
# Get the first 10 rows of the DataFrame
df_first_10 = df.iloc[:10]
df_first_10['s1']=df_first_10['sentence1'].apply(translate_to_hindi)
# Apply the parallelized translation to the first 10 rows


# %%
df.head()

# %%
df_first_10.head(10)
df_first_10['s2']=df_first_10['sentence2'].apply(translate_to_hindi)

# %%
import threading

# This will hold the results
results = [None] * 100  # Adjusted to match the number of threads

# Define a function to be executed in each thread
def thread_function(thread_id, rangge):
    df_first_10 = df.iloc[rangge:rangge+1]  # Adjusted to translate one line per thread
    df_first_10['s1'] = df_first_10['sentence1'].apply(translate_to_hindi)
    # Save the result to the shared list
    results[thread_id - 1] = df_first_10

# Create and start 100 threads
threads = []
for i in range(1, 101):
    thread = threading.Thread(target=thread_function, args=(i, i-1))  # Adjusted to correctly slice the DataFrame
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Now, 'results' contains the results from each thread
print("All threads have finished execution.")
result_df = pd.concat(results)

# %%
import numpy as np

# Ensure the DataFrame has 5000 rows
df_trimmed = df.iloc[:5000]

# Convert the DataFrame to a numpy array
arr = np.array(df_trimmed)

# Split the array into 100 sub-arrays
arrs = np.array_split(arr, 100)

print(arrs)

# %%
import pandas as pd

# Convert the numpy array back to a DataFrame
df0 = pd.DataFrame(arrs[0])

df0[3]=df0[0]
df0

# %%
import threading

results = [None] * 100

def thread_function(thread_id):
    print('thread id', thread_id)
    df_first_10 =  pd.DataFrame(arrs[thread_id])
    df_first_10[3] = df_first_10[0].apply(translate_to_hindi)
    results[thread_id] = df_first_10

threads = []
for i in range(0, 100):
    thread = threading.Thread(target=thread_function, args=(i,))  
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print("All threads have finished execution.")
# result_df = pd.concat(results)

# %%
# arr[]

# %%
# for(i in arrs):


# %%
results[0]

# %%
df['sentence1']

# %%
len(arrs)
arrs[99]['sentence1']   


