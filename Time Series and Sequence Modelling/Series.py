# %%
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
for val in dataset:
    print(val.numpy(), end = " ")

# %%
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(size = 5, shift=1)
#for window_dataset in dataset:
#    print(window_dataset)
for window_dataset in dataset:
    print([item.numpy() for item in window_dataset])

# %%
# Switching drop_remainder to True.
dataset  = tf.data.Dataset.range(10)
dataset = dataset.window(size = 5, shift=1, drop_remainder=True)
for window_dataset in dataset:
    print([item.numpy() for item in window_dataset])

# %% [markdown]
# ## Flatten the Windows
# 
# In training the model later, you will want to prepare the windows to be [tensors](https://www.tensorflow.org/guide/tensor) instead of the `Dataset` structure. You can do that by feeding a mapping function to the [flat_map()](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#flat_map) method. This function will be applied to each window and the results will be [flattened into a single dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#flatten_a_dataset_of_windows_2). To illustrate, the code below will put all elements of a window into a single batch then flatten the result.

# %%
# Flatten the windows.
dataset  = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset  = dataset.flat_map(lambda window: window.batch(5)) # Flatten the windows by putting its elements in a single batch
 
for window_dataset in dataset:
    print(window_dataset.numpy())

# %%
# Group into features and Labels :
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
# Create tuples with features (first four elements of the window) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

for x,y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())
    print()

# %%
# Shuffle the data.
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(size = 5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
dataset = dataset.shuffle(buffer_size=10)

for x, y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())
    print()

# %% [markdown]
# ## Create batches for training
# 
# Lastly, you will want to group your windows into batches. You can do that with the [batch()](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch) method as shown below. Simply specify the batch size and it will return a batched dataset with that number of windows. As a rule of thumb, it is also good to specify a [prefetch()](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch) step. This optimizes the execution time when the model is already training. By specifying a prefetch `buffer_size` of `1` as shown below, Tensorflow will prepare the next one batch in advance (i.e. putting it in a buffer) while the current batch is being consumed by the model. You can read more about it [here](https://towardsdatascience.com/optimising-your-input-pipeline-performance-with-tf-data-part-1-32e52a30cac4#Prefetching).

# %%
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(size = 5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)

for x, y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())
    print()



