## Helper functions for CNN and Transfer Learning

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import zipfile
import cv2
from sklearn.utils import shuffle
from sklearn import metrics
from tensorflow.keras.applications import resnet
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
K = tf.keras.backend


def load_json_as_df(filedir, filename):
    """
    This function loads data from json files in data folder. Also cross-checks
    whether each image is in images folder.
    inputs:
        filedir - project directory
        filename - file name, without .json
    output:
        tdf - dataframe
    """

    json_path = os.path.join(filedir, filename + '.json')
    tdf = pd.read_json(json_path, orient='index')

    img_filedir = os.path.join(filedir, 'images/')
    list_image_paths = glob.glob(img_filedir + '*.jpg')
    list_image_paths = [x.replace(img_filedir,'') for x in list_image_paths]
    check_image_bool = tdf['image-name'].apply(lambda x: x in list_image_paths)
    tdf = tdf[check_image_bool]

    print('loaded {}.json with shape {}\n'.format(filename, tdf.shape))

    return tdf
    

def plot_class_balances(df, col):
    """Plots the counts of classes for specified column"""

    counts = df[col].value_counts()
    counts.plot.bar()
    plt.title(col + ' Counts \n(classes={})'.format(counts.shape[0]))
    plt.show()
    
    
def sample_df(df, col, n_sample_per_class=120, replace = False):
    """
    Samples the dataframe based on a column, with or without replacement
    Replacement only applies when required sample size > available data
    """
    
    samples = df.groupby(col)
    list_cls = df[col].unique()
    df_lst = []
    for cls in list_cls:
        cls_df = samples.get_group(cls)
        if (cls_df.shape[0] < n_sample_per_class) and (replace==False):
            cls_sample = cls_df
        else:
            cls_sample = cls_df.sample(n=n_sample_per_class,replace=replace,random_state=42)
        df_lst.append(cls_sample)
      
    df_sampled = pd.concat(df_lst, sort=False)
    df_sampled = shuffle(df_sampled)
    
    return df_sampled


def plot_training_history(history, metric):
    """
    Function for plotting training history.
    This plots the classification accuracy and loss values recorded during training with the Keras API.
    """
    
    val_metric = 'val_'+metric
    acc = history.history[metric]
    val_acc = history.history[val_metric]
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = history.epoch
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Acc.')
    plt.plot(epochs_range, val_acc, label='Validation Acc.')
    plt.legend(loc='best',)
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Training and Validation Loss')
    plt.show()


def create_res_labels_df(test_generator, test_history):
    """
    Converts array returned by the predict generator into a dataframe with true & predicted labels and the image names
    Inputs: 
        test_history - n-dimensional array returned by the mode.predict_generator() method
        test_generator - ImageDataGenerator Object that was used used by .predict_generator()    
    """
    
    df_test_results = pd.DataFrame()
    test_len = test_history.shape[0]
    df_test_results['y_true'] = test_generator.labels[:test_len]
    df_test_results['y_pred'] = tf.math.argmax(test_history, axis=1).numpy().ravel()
    df_test_results['image_path'] = test_generator.filepaths[:test_len]
    
    return df_test_results


def plot_confusion_matrix(df_res_labels):
    """function to plot confusion matrix heatmap"""

    y_tr = df_res_labels['y_true']
    y_pre = df_res_labels['y_pred']  

    labels = np.sort(y_tr.unique())
    conf_mat = metrics.confusion_matrix(y_tr, y_pre,labels=labels )
    df_conf = pd.DataFrame(conf_mat, columns=labels, index=labels)

    mask = np.ones(conf_mat.shape) 
    mask = (mask - np.diag(np.ones(conf_mat.shape[0]))).astype(np.bool)
    max_val = np.amax(conf_mat[mask])
    
    fig, ax = plt.subplots(figsize=(14,10))
    ano = True if df_conf.shape[0] < 100 else False
    sns.heatmap(df_conf,vmax=max_val, annot=ano, ax=ax)
    plt.show()
    print('- '*50)
        
    return


def create_test_report(test_generator, test_history):
    """Function to create whole test report uses create_res_labels function and plot_confusion_matrix function"""
    
    df_res_labels = create_res_labels_df(test_generator, test_history)
    
    lvls=['']
    
    metrics_dict = {}
    
    n_samples = df_res_labels.shape[0]
    print('.'*50)
    print('showing test metrics for {} samples'.format(n_samples))
    print('`'*50)
    
    lvl_metrics_dict = {}
    for lvl in lvls:
        y_tr = df_res_labels['y_true' + lvl]
        y_pre = df_res_labels['y_pred' + lvl]  
    
        lvl_metrics_dict = {}
        
        # Macro / Micro Driven Metrics
        for avg in ['macro', 'micro']:
            
            met_name = 'precision' + ('_'+ avg)    
            res = metrics.precision_score(y_tr, y_pre, average=avg)
            lvl_metrics_dict[met_name] = res
            
            met_name = 'f1' + ('_'+ avg)    
            res = metrics.f1_score(y_tr, y_pre, average=avg)
            lvl_metrics_dict[met_name] = res
            
            met_name = 'recall' + ('_'+ avg)    
            res = metrics.recall_score(y_tr, y_pre, average=avg)
            lvl_metrics_dict[met_name] = res
            
        met_name = 'accuracy'    
        res = metrics.accuracy_score(y_tr, y_pre)
        lvl_metrics_dict[met_name] = res
        
        metrics_dict[lvl] = lvl_metrics_dict
    
    df_test_results = pd.DataFrame(metrics_dict).sort_values(by=lvls, ascending=False)
    df_test_results=df_test_results.reindex(columns=lvls)
    
    print(df_test_results)
    print('- '*70)
    
    plot_confusion_matrix(df_res_labels)
    
    return df_res_labels


def plot_images(images_arr):
    """This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column."""
    
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return


def preprocess_images(im_path):
  # load the original image from gdrive (in OpenCV format)
  # resize the image to its target dimensions

  orig = cv2.imread(im_path)
  resized = cv2.resize(orig, (224, 224))

  # load the input image from gdrive (in Keras/TensorFlow format)
  # basic image pre-processing

  image = load_img(im_path, target_size=(224, 224))
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = resnet.preprocess_input(image)

  return resized, image, orig


def get_class_predictions(class_rank, predictions):
  classes_ranked = np.argsort(predictions)[::-1]
  i = classes_ranked[class_rank]
  decoded = resnet.decode_predictions(predictions, 3)
  (_, label, prob) = decoded[0][class_rank]
  
  label = "{}: {:.2f}%".format(label, prob * 100)
  print('Class with highest probability:')
  print("{}".format(label))

  return i, decoded


def plot_gradcam(original, output, heatmap):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    # images are in cv2 format 'BGR' but pyplot uses 'RGB'
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    ax3.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

    # add titles
    ax1.set_title('Original')
    ax2.set_title('GradCAM Output')
    ax3.set_title('Heatmap')

    plt.show()

    return
