#!/usr/bin/env python
# coding: utf-8

# # **Imports**

# In[1]:


import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# # **Dataset**

# In[2]:


raw_dataset_path = r"C:\Users\hp\Downloads\archive\Air Pollution Image Dataset\Air Pollution Image Dataset\Country_wise_Dataset\India\Mumbai"
dataset_path = "air_pollution_dataset_Mumbai"

os.makedirs(dataset_path, exist_ok=True)


# In[3]:


CATEGORIES = ['a_Good','b_Moderate','d_Unhealthy']


# In[4]:


train_percent, test_percent, val_percent = 0.7, 0.2, 0.1

os.makedirs(dataset_path, exist_ok=True)
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")
test_dir = os.path.join(dataset_path, "test")

# class_freq = []

random.seed(123)

for afolder in CATEGORIES:
    folder_path = os.path.join(raw_dataset_path, afolder)
    files = os.listdir(folder_path)
    file_count = len(files)
    random.shuffle(files)
    print(f"{afolder}: {file_count}")

    train_len = int(file_count * train_percent)
    val_len = int(file_count * val_percent)

    train_files = files[:train_len]
    val_files = files[train_len:train_len+val_len]
    test_files = files[train_len+val_len:]

    train_folder = os.path.join(train_dir, afolder)
    val_folder = os.path.join(val_dir, afolder)
    test_folder = os.path.join(test_dir, afolder)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for afile in train_files:
        shutil.copy(os.path.join(folder_path, afile), os.path.join(train_folder, afile))

    for afile in val_files:
        shutil.copy(os.path.join(folder_path, afile), os.path.join(val_folder, afile))

    for afile in test_files:
        shutil.copy(os.path.join(folder_path, afile), os.path.join(test_folder, afile))


# In[5]:


tot_train = 0
tot_val = 0
tot_test = 0
for folder in os.listdir(train_dir):
    if os.path.isdir(os.path.join(train_dir,folder)):
        train_len = len(os.listdir(os.path.join(train_dir, folder)))
        val_len = len(os.listdir(os.path.join(val_dir, folder)))
        test_len = len(os.listdir(os.path.join(test_dir, folder)))

        tot_len = train_len+val_len+test_len
        train_percent = float(train_len*100)/tot_len
        val_percent = float(val_len*100)/tot_len
        test_percent = float(test_len*100)/tot_len

        tot_train += train_len
        tot_val += val_len
        tot_test += test_len

        print(f"{folder:10}:\t train: {train_len:4d} ({train_percent:5.2f}%)  val: {val_len:3d} ({val_percent:5.2f}%)  test: {test_len:3d} ({test_percent:5.2f}%)")

tot_len = tot_train + tot_val + tot_test
train_percent = float(tot_train*100)/tot_len
val_percent = float(tot_val*100)/tot_len
test_percent = float(tot_test*100)/tot_len
print("-"*55)
print(f"Total     \t train: {tot_train:4d} ({train_percent:5.2f}%)  val: {tot_val:3d} ({val_percent:5.2f}%)  test: {tot_test:3d} ({test_percent:5.2f}%)")


# ## **Create tensorflow datasets**

# In[6]:


# image_size = (224, 224)
image_size = (60, 60)
batch_size = 32
seed = 123


# In[7]:


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)


# In[8]:


val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)


# In[9]:


test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False
)


# In[10]:


class_names = train_ds.class_names
print(class_names)


# In[11]:


normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# In[12]:


# visualize

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(class_names[labels[i]])
        plt.axis("off")


# In[13]:


# cache and prefetch the datasets for better performance
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# # **Model**

# In[14]:


def get_model(input_shape, num_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))

    # Keep the output layer as 3 for three air quality classes
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model


model = get_model(
    input_shape=image_size+(3,),
    num_classes=len(class_names)
)


# In[15]:


model.summary()


# In[16]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)


# In[17]:


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
)


# # **Evaluate**

# In[18]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'])
plt.show()


# In[19]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()


# In[20]:


from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


# In[21]:


def plot_confusion_matrix(cmat, title=None):
    sns.heatmap(
        cmat,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
        fmt='g',
        cmap='Blues'
    )
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    if title is not None:
        plt.title(title)
    plt.show()


# In[22]:


pred_y = model.predict(test_ds)
pred_y = tf.argmax(pred_y, axis=1)
true_y = tf.concat(list(test_ds.map(lambda s,lab: lab)), axis=0)

acc = accuracy_score(true_y, pred_y)
print(f"Accuracy: {acc}")


# In[23]:


cmat = confusion_matrix(true_y, pred_y)
plot_confusion_matrix(cmat)


# In[ ]:




