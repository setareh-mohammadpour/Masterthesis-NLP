# -*- coding: utf-8 -*-

import os
import random
import tensorflow as tf
import numpy as np
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
import glob
import pandas as pd
import argparse
import time
import csv
from collections import Counter
import re
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, GlobalAveragePooling1D, BatchNormalization, Embedding, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.constraints import MaxNorm
from transformers import TFRobertaModel, RobertaTokenizer
from tensorflow.keras import mixed_precision
from transformers import TFDistilBertModel, DistilBertTokenizer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2



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
        
        X_test, test_mask = self.validation_data[0]  
        y_test = self.validation_data[1]
        y_pred = self.model.predict([X_test, test_mask])

        y_pred_rounded = (y_pred >= 0.5).astype(int)  

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        prec = precision_score(y_test, y_pred_rounded, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred_rounded, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred_rounded, average='binary', zero_division=0)

        # Get accuracy and validation accuracy from the logs (history)
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')

        # Log the metrics
        print(f"Epoch {epoch+1}: MSE: {mse}, MAE: {mae}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

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


# Define function to find the latest .h5 checkpoint manually
def find_latest_checkpoint(checkpoint_dir):
    # Search for all .h5 files in the checkpoint directory
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.h5"))

    if not checkpoint_files:
        return None  # Return None if no checkpoints are found

    # Sort the files by epoch number (extracted from the filename)
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Return the most recent checkpoint
    return checkpoint_files[-1]
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
    return tf.tensordot(x, kernel, axes=1)

class Attention(tf.keras.layers.Layer):
    def __init__(self, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.init = tf.keras.initializers.get('glorot_uniform')
        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)
        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)
        self.bias = bias

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer=self.init, regularizer=self.W_regularizer, constraint=self.W_constraint, name='{}_W'.format(self.name))
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1], 1), initializer='zeros', regularizer=self.b_regularizer, constraint=self.b_constraint, name='{}_b'.format(self.name))
        else:
            self.b = None
        self.built = True

    def call(self, x, mask=None):
        # Compute the attention scores
        eij = tf.keras.backend.squeeze(tf.keras.backend.dot(x, self.W), axis=-1)
        print(f"Shape of eij after squeeze: {eij.shape}")

        if self.bias:
            eij += tf.keras.backend.squeeze(self.b, axis=-1)  # Squeeze bias if needed
            print(f"Shape of eij after bias addition: {eij.shape}")

        eij = tf.keras.activations.tanh(eij)
        a = tf.keras.backend.exp(eij)
        print(f"Shape of attention scores a: {a.shape}")

        if mask is not None:
            a *= tf.cast(mask, tf.float32)
            print(f"Shape of attention scores a after masking: {a.shape}")

        a /= tf.keras.backend.sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon()
        a = tf.keras.backend.expand_dims(a)
        print(f"Shape of a after softmax normalization: {a.shape}")

        weighted_input = x * a
        output = tf.keras.backend.sum(weighted_input, axis=1)
        print(f"Shape of final output: {output.shape}")
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            'W_regularizer': tf.keras.regularizers.serialize(self.W_regularizer),
            'b_regularizer': tf.keras.regularizers.serialize(self.b_regularizer),
            'W_constraint': tf.keras.constraints.serialize(self.W_constraint),
            'b_constraint': tf.keras.constraints.serialize(self.b_constraint),
            'bias': self.bias
        })
        return config

class MyDistilBertLayer(tf.keras.layers.Layer):
    def __init__(self, model_name="distilbert-base-uncased", freeze_layers=1, **kwargs):
        super(MyDistilBertLayer, self).__init__(**kwargs)
        self.distilbert = TFDistilBertModel.from_pretrained(model_name)
        # Freeze the first `freeze_layers` layers
        for i, layer in enumerate(self.distilbert.layers):
            if i < freeze_layers:
                layer.trainable = False
            else:
                layer.trainable = True  # Unfreeze the remaining layers

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, training=training)
        return distilbert_output.last_hidden_state

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.distilbert.config.hidden_size)

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
def getModel(max_len):
    input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    distilbert_output = MyDistilBertLayer(freeze_layers=1)([input_ids, attention_mask])
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(distilbert_output)
    attention_output = Attention()(distilbert_output)

    # Combine Attention Output with Pooled Output
    x = tf.keras.layers.Concatenate()([pooled_output, attention_output])
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(pooled_output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    # Output layer for binary classification
    output = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32', kernel_regularizer=l2(0.001))(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
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
    parser.add_argument('--n_epochs', dest='n_epochs', default=20, type=int, action='store', help='Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=512, type=int, action='store', help='Training batch size')
    parser.add_argument('--n_fold', dest='n_fold', default=10, type=int, action='store', help='Number of validation folds')
    args, unknown = parser.parse_known_args()
    print(args)
    file_path = r'/content/drive/MyDrive/traintest_course_comments.csv'
    reviews = pd.read_csv(file_path)
    cleaned_texts = [clean_text(text) for text in reviews[args.comment_field].tolist()]
    labels = reviews[args.score_field].tolist()
    # Convert ratings to binary labels directly
    labels = reviews[args.score_field].apply(lambda x: 1 if x > 2.5 else 0).tolist()
    # Reshape labels to [batch_size, 1], then flatten to 1D
    labels = np.array(labels).reshape(-1, 1).flatten()

     # After loading the labels from the dataset
    combined_df = pd.DataFrame({'cleaned_texts': cleaned_texts, 'labels': labels})
    # Shuffle the DataFrame
    shuffled_df = shuffle(combined_df, random_state=42)
    # Extract the shuffled data back into lists
    cleaned_texts = shuffled_df['cleaned_texts'].tolist()
    labels = shuffled_df['labels'].tolist()

    # Verify label distribution after shuffling
    label_distribution = pd.Series(labels).value_counts()
    print("Label distribution after shuffling:")
    print(label_distribution)



    max_len = args.max_len
    print(max_len)
    discrete_labels = np.array(labels)
    # Check that the length of cleaned_texts matches the length of discrete_labels
    print(f"Length of cleaned_texts: {len(cleaned_texts)}")
    print(f"Length of discrete_labels: {len(discrete_labels)}")

    # Check that discrete_labels only contains valid labels (0 or 1)
    unique_labels = set(discrete_labels)
    print(f"Unique labels in discrete_labels: {unique_labels}")

    # Optionally, check the label distribution in discrete_labels
    from collections import Counter
    print(f"Label distribution in discrete_labels: {Counter(discrete_labels)}")

    print("Unique values in labels (y_test and y_train):", np.unique(labels))
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # Tokenize the entire dataset once
    encodings = tokenizer(cleaned_texts, truncation=True, padding=True, max_length=args.max_len)
    input_ids = np.array(encodings['input_ids'])
    attention_masks = np.array(encodings['attention_mask'])
    skf = StratifiedKFold(n_splits=args.n_fold, shuffle=True, random_state=42)
    last_epoch_metrics = {'acc': [], 'val_acc': [],'prec': [], 'recall': [], 'f1': []}
    epoch_train_accuracies = [[] for _ in range(args.n_fold)]
    epoch_val_accuracies = [[] for _ in range(args.n_fold)]
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20,        
        restore_best_weights=True  
          )




    for id_fold, (train_index, test_index) in enumerate(skf.split(input_ids, discrete_labels)):
        print('Training model for fold', id_fold)
        X_train, X_test = input_ids[train_index], input_ids[test_index]
        train_mask, test_mask = attention_masks[train_index], attention_masks[test_index]
        y_train, y_test = discrete_labels[train_index], discrete_labels[test_index]

        initial_learning_rate = 5e-5
        lr_schedule = ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True
            )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model = getModel(args.max_len)
        # Checkpoint directory and filename
        checkpoint_dir = f'/content/drive/MyDrive/Checkpoints/Checkpoints22/fold_{id_fold}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_filepath = os.path.join(checkpoint_dir, 'epoch_{epoch:02d}.keras')
        # Find the latest .h5 checkpoint manually
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
            model = tf.keras.models.load_model(latest_checkpoint, custom_objects={'MyDistilBertLayer': MyDistilBertLayer})
            initial_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])  # Extract the last epoch number from the checkpoint filename
        else:
            initial_epoch = 0  
        mcheck = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, save_best_only=False, monitor='val_loss', verbose=1)
        result_path = os.path.join('results', f'class_{args.n_classes}', f'roberta_fold_{id_fold}_results.csv')
        metrics_callback = MetricsAfterEpoch(
            validation_data=([X_test, test_mask], y_test),
            result_path=result_path,
            last_epoch_metrics=last_epoch_metrics,
            epoch_train_accuracies=epoch_train_accuracies,
            epoch_val_accuracies=epoch_val_accuracies,
            current_fold=id_fold
        )
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        fold_train_acc = []
        fold_val_acc = []
        history = model.fit(
                  [np.array(X_train), np.array(train_mask)], 
                  np.array(y_train),
                  validation_data=([np.array(X_test), np.array(test_mask)], np.array(y_test)),
                  epochs=args.n_epochs,  # Train one epoch at a time
                  initial_epoch=initial_epoch,  # Start from the current epoch
                  batch_size=args.batch_size,
                  shuffle=True,
                  callbacks=[mcheck, tensorboard_callback, early_stopping, metrics_callback],
                  verbose=1
                    )

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

!python /content/drive/MyDrive/Setare.py --traintest_file /content/drive/MyDrive/traintest_course_comments.csv --comment_field 'learner_comment' --score_field 'learner_rating' --max_len 500 --n_classes 2 --embs_dir /content/drive/MyDrive/embeddings_specific/specific/fasttext --n_epochs 20 --batch_size 1024 --n_fold 10


if __name__ == "__main__":
    main()
