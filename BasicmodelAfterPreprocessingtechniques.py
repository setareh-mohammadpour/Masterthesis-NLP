# -*- coding: utf-8 -*-
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, GaussianNoise
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import layers
from keras.constraints import MaxNorm
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import initializers, regularizers, constraints
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.metrics import f1_score as sklearn_f1_score, precision_score, recall_score
from sklearn.utils import shuffle
import keras.backend as K
import keras.layers as layers
from keras.layers import Layer, Dense
import numpy as np
import pandas as pd
import csv
import os
import re
import random
import argparse
import tensorflow as tf
from matplotlib import pyplot as plt

class MaxoutDense(Layer):
    def __init__(self, output_dim, num_units=2, **kwargs):
        self.output_dim = output_dim
        self.num_units = num_units
        super(MaxoutDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_layers = [Dense(self.output_dim) for _ in range(self.num_units)]
        super(MaxoutDense, self).build(input_shape)

    def call(self, inputs):
        maxout_units = [layer(inputs) for layer in self.dense_layers]
        return tf.reduce_max(maxout_units, axis=0)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim



class MetricsAfterEpoch(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, result_path, last_epoch_metrics, epoch_train_accuracies, epoch_val_accuracies, current_fold):
        super().__init__()
        self.validation_data = validation_data 
        self.result_path = result_path
        self.last_epoch_metrics = last_epoch_metrics
        self.epoch_train_accuracies = epoch_train_accuracies  
        self.epoch_val_accuracies = epoch_val_accuracies 
        self.current_fold = current_fold  
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        self.result_file = open(result_path, 'w', newline='', encoding='utf8')
        self.result_writer = csv.writer(self.result_file)
        self.result_writer.writerow(['Epoch', 'MSE', 'MAE', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    def on_epoch_end(self, epoch, logs=None):
        X_test, y_test = self.validation_data
        y_pred = self.model.predict(X_test)
        y_pred_rounded = (y_pred >= 0.5).astype(int)  
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        prec = precision_score(y_test, y_pred_rounded, average='weighted')
        recall = recall_score(y_test, y_pred_rounded, average='weighted')
        f1 = f1_score(y_test, y_pred_rounded, average='weighted')

        # Get accuracy and validation accuracy from the logs (history)
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')

        # Log the metrics
        print(f"Epoch {epoch+1}: MSE: {mse}, MAE: {mae}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Write metrics to CSV
        self.result_writer.writerow([epoch+1, mse, mae, train_acc, prec, recall, f1])

        # Append accuracy and validation accuracy to the respective fold lists
        self.epoch_train_accuracies[self.current_fold].append(train_acc)
        self.epoch_val_accuracies[self.current_fold].append(val_acc)

        # Save last epoch metrics for averaging across folds
        if epoch == self.params['epochs'] - 1:
            self.last_epoch_metrics['acc'].append(train_acc)
            self.last_epoch_metrics['val_acc'].append(val_acc)
            self.last_epoch_metrics['prec'].append(prec)
            self.last_epoch_metrics['recall'].append(recall)
            self.last_epoch_metrics['f1'].append(f1)

    def on_train_end(self, logs=None):
        self.result_file.close()

def clean_text(text):
    """Function to clean text data"""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text
def getClass2(value):
    split = 2.5
    values = [0, 1]
    if value <= split:
        return values[0]
    return values[1]

def getClass5(value):
    splits = [1.0, 2.0, 3.0, 4.0]
    values = [0, 1, 2, 3, 4]
    if value <= splits[0]:
        return values[0]
    elif value <= splits[1]:
        return values[1]
    elif value <= splits[2]:
        return values[2]
    elif value <= splits[3]:
        return values[3]
    return values[4]
def getClass10(value):
    splits = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if value <= splits[0]:
        return values[0]
    elif value <= splits[1]:
        return values[1]
    elif value <= splits[2]:
        return values[2]
    elif value <= splits[3]:
        return values[3]
    elif value <= splits[4]:
        return values[4]
    if value <= splits[5]:
        return values[5]
    elif value <= splits[6]:
        return values[6]
    elif value <= splits[7]:
        return values[7]
    elif value <= splits[8]:
        return values[8]
    return values[9]

def dot_product(x, kernel):
    return tf.squeeze(tf.matmul(x, tf.expand_dims(kernel, axis=-1)), axis=-1)

class Attention(Layer):
    def __init__(self, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1],), initializer=self.init, regularizer=self.W_regularizer, constraint=self.W_constraint, name='{}_W'.format(self.name))
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],), initializer='zeros', name='{}_b'.format(self.name), regularizer=self.b_regularizer, constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = tf.tanh(eij)

        a = tf.exp(eij)

        if mask is not None:
            a *= tf.cast(mask, tf.float32)

        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.float32)
        a = tf.expand_dims(a, axis=-1)
        weighted_input = x * a
        return tf.reduce_sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def getModel(embedding_matrix, max_words, max_len, embedding_dim):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix], trainable=False))
    model.add(layers.Dropout(0.1))
    model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1, implementation=1)))
    model.add(Attention())
    model.add(Dropout(0.1))
    model.add(MaxoutDense(100))
    model.add(Dense(1, activation='sigmoid'))
    return model
def add_noise(text, noise_level=0.1):
    text_list = list(text)
    for i in range(len(text_list)):
        if random.random() < noise_level:
            text_list[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(text_list)

def main():
    parser = argparse.ArgumentParser(description='DL Sentiment Training')
    parser.add_argument('--traintest_file', dest='traintest_file', default='/content/drive/MyDrive/traintest_course_comments.csv', type=str, action='store', help='Train and test course comments CSV file')
    parser.add_argument('--comment_field', dest='comment_field', default='learner_comment', type=str, action='store', help='Field title for comments in CSV file')
    parser.add_argument('--score_field', dest='score_field', default='learner_rating', type=str, action='store', help='Field title for scores in CSV file')
    parser.add_argument('--max_len', dest='max_len', default=500, type=int, action='store', help='Max number of words in a comment')
    parser.add_argument('--n_classes', dest='n_classes', default=2, type=int, action='store', help='Number of prediction classes')
    parser.add_argument('--embs_dir', dest='embs_dir', default='/content/drive/MyDrive/embeddings_specific/specific/fasttext', type=str, action='store', help='Directory containing the embeddings files')
    parser.add_argument('--n_epochs', dest='n_epochs', default=20, type=int, action='store', help='Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=1024, type=int, action='store', help='Training batch size')
    parser.add_argument('--n_fold', dest='n_fold', default=10, type=int, action='store', help='Number of validation folds')
    args, unknown = parser.parse_known_args()
    print(args)
    file_path = r'/content/drive/MyDrive/traintest_course_comments.csv'
    reviews = pd.read_csv(file_path)
    cleaned_texts = [clean_text(text) for text in reviews[args.comment_field].tolist()]
    labels = reviews[args.score_field].tolist()
    # After loading the labels from the dataset
    combined_df = pd.DataFrame({'cleaned_texts': cleaned_texts, 'labels': labels})
    # Shuffle the DataFrame
    shuffled_df = shuffle(combined_df, random_state=42)
    # Extract the shuffled data back into lists
    cleaned_texts = shuffled_df['cleaned_texts'].tolist()
    labels = shuffled_df['labels'].tolist()
    max_len = args.max_len
    print(max_len)
    # Convert labels to binary based on threshold after shuffling**
    labels = [1 if rating > 2.5 else 0 for rating in labels]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(cleaned_texts)
    sequences = tokenizer.texts_to_sequences(cleaned_texts)

    word_index = tokenizer.word_index
    all_words = [word for word, i in word_index.items()]
    max_words = len(word_index) + 1
    print('Found %s unique words.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_len)
    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    print('* PRE-PROCESSING DATA')
    skf = StratifiedKFold(n_splits=args.n_fold)
    print('Shuffling data')
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    data = data[s]
    labels = labels[s]
    print('* LOADING EMBEDDINGS AND MODELS')
    print('Found', len(os.listdir(args.embs_dir)), 'embs_course_comments')
    embedding_files = [file for file in os.listdir(args.embs_dir) if os.path.isfile(os.path.join(args.embs_dir, file))]
    last_epoch_metrics = {'acc': [], 'val_acc': [],'prec': [], 'recall': [], 'f1': []}
    epoch_train_accuracies = [[] for _ in range(args.n_fold)]
    epoch_val_accuracies = [[] for _ in range(args.n_fold)]

    if len(embedding_files) == 0:
        print("No embedding files found in the specified directory:", args.embs_dir)
    else:
        print("Found", len(embedding_files), "embedding files")
        for embs_file in embedding_files:
            print('Loading embeddings from', embs_file)
            if '_' not in embs_file:
              print('Skipping file', embs_file, 'because it does not follow the expected naming convention.')
              continue

            embedding_dim_str = embs_file.split('_')[1]

            try:
              embedding_dim = int(embedding_dim_str)
            except ValueError:
              print('Skipping file', embs_file, 'because the embedding dimension is not a valid integer.')
              continue

        embedding_path = os.path.join(args.embs_dir,embs_file)

        embeddings_index = {}
        f = open(embedding_path)
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:-1], dtype='float32')
            if word in all_words:
                embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))
        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if i < max_words:
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
        early_stopping = EarlyStopping(
        monitor='val_loss',    
        patience=20,            
        restore_best_weights=True  
            )

        for id_fold, (train_index, test_index) in enumerate(skf.split(data, labels)):
            print('Training model for fold', id_fold)
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            initial_learning_rate = 0.0001
            lr_schedule = ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=100000,
                decay_rate=0.96,
                staircase=True
            )
            optimizer = Adam(learning_rate=lr_schedule)
            mcheck = ModelCheckpoint(os.path.join('models', 'class' + str(args.n_classes), embs_file.replace('.txt', '_fold' + str(id_fold) + '_model.keras')), monitor='val_loss', save_best_only=True)
            model = getModel(embedding_matrix, max_words, max_len, embedding_dim)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            fold_train_acc = []
            fold_val_acc = []
            result_path = os.path.join('results', f'class_{args.n_classes}', f'{embs_file}_fold_{id_fold}_results.csv')
            # Ensure directory exists before saving results
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            # Inside cross-validation loop
            metrics_callback = MetricsAfterEpoch(
                validation_data=(X_test, y_test),
                result_path=result_path,
                last_epoch_metrics=last_epoch_metrics,
                epoch_train_accuracies=epoch_train_accuracies,
                epoch_val_accuracies=epoch_val_accuracies,
                current_fold=id_fold    # Pass the current fold index here
                  )

            history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.n_epochs, batch_size=args.batch_size, shuffle=True, callbacks=[metrics_callback, mcheck, tensorboard_callback, early_stopping], verbose=1)
        # Calculate averages of last epoch metrics across all folds
        avg_train_acc = np.mean(last_epoch_metrics['acc'])
        avg_val_acc = np.mean(last_epoch_metrics['val_acc'])
        avg_prec = np.mean(last_epoch_metrics['prec'])
        avg_recall = np.mean(last_epoch_metrics['recall'])
        avg_f1 = np.mean(last_epoch_metrics['f1'])
        print(f"Average of Last Epochs Across All Folds: Training Accuracy: {avg_train_acc}, Validation Accuracy: {avg_val_acc}, Precision: {avg_prec}, Recall: {avg_recall}, F1 Score: {avg_f1}")
        # Plotting accuracies
        for i in range(len(epoch_train_accuracies)):
            plt.figure()
            plt.plot(epoch_train_accuracies[i], label='Training Accuracy')
            plt.plot(epoch_val_accuracies[i], label='Validation Accuracy')
            plt.title(f'Fold {i + 1} Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

!python /content/drive/MyDrive/Mainfile.py --traintest_file /content/drive/MyDrive/traintest_course_comments.csv --comment_field 'learner_comment' --score_field 'learner_rating' --max_len 500 --n_classes 2 --embs_dir /content/drive/MyDrive/embeddings_specific/specific/fasttext --n_epochs 20 --batch_size 1024 --n_fold 10


if __name__ == "__main__":
    main()
