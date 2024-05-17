#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# # Load Dataset

# In[17]:


x_train=np.loadtxt('C:/Users/1030G2/OneDrive/Desktop/classification/input.csv', delimiter=',')
y_train=np.loadtxt('C:/Users/1030G2/OneDrive/Desktop/classification/labels.csv', delimiter=',')
x_test=np.loadtxt('C:/Users/1030G2/OneDrive/Desktop/classification/input_test.csv', delimiter=',')
y_test=np.loadtxt('C:/Users/1030G2/OneDrive/Desktop/classification/labels_test.csv',delimiter=',' )


# In[18]:


x_train=x_train.reshape(len(x_train), 100, 100,3)
y_train=y_train.reshape(len(y_train), 1)
x_test=x_test.reshape(len(x_test), 100, 100,3)
y_test=y_test.reshape(len(y_test), 1)
#rescaling it we have
x_train=x_train/255
x_test=x_test/255


# In[19]:


print('shape of x_train', x_train.shape)
print('shape of y_train', y_train.shape)
print('shape of x_test', x_test.shape)
print('shape of y_test', y_test.shape)


# In[20]:


x_train[1,:]


# # Displaying the image

# In[50]:


idx=random.randint(0,len(x_train))
plt.imshow(x_train[idx,:])
plt.show()


# # Model

# In[49]:


model=Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
           MaxPooling2D((2,2)),
           
           Conv2D(32, (3,3), activation='relu'),
           MaxPooling2D((2,2)),        
           
           Flatten(),
           Dense(64, activation='relu'),
           Dense(1,activation='sigmoid')
    
])


# In[66]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[69]:


model.fit(x_train, y_train, epochs=5, batch_size=64)


# In[70]:


model.evaluate(x_test,y_test)


# # Making prediction

# In[65]:


idx2=random.randint(0,len(y_test))
plt.imshow(x_test[idx2,:])
plt.show()
y_pred=model.predict(x_test[idx2,:].reshape(1,100,100,3))
y_pred=y_pred > 0.5
if (y_pred==0):
    pred='dog'
else:
    pred='cat'
print('our model say it is a:', pred)


# In[ ]:





# In[ ]:





# In[ ]:




