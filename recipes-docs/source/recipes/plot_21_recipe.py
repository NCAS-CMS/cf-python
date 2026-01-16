"""
Applying functions and mathematical operations to data
======================================================

In this recipe we will explore various methods to apply a mathematical operation or a function to a set of data in a field. For the purposes of the example, we will look at various ways of calculating the sine of each element in a data array.

There are various options to do this, the recommended option is to use `cf native functions <https://ncas-cms.github.io/cf-python/class/cf.Field.html#mathematical-operations>`_, as they preserve units and metadata associated with fields. Sometimes, however, the function you need is not implemented in cf, so there are alternative methods.
"""

# %% [markdown]
#
# .. figure:: ../images/data-operations-flowchart.png
#    :scale: 50 %
#    :alt: flowchart showing process of location a function in cf, then in Dask, then in NumPy, and finally vectorising it with NumPy.
#
#    It is recommended to use the highest possible implementation of a given function as shown by the chart.
#

# %%
# 1. Import cf-python:

import cf

# %%
# 2. Read the template field constructs from the example:

f = cf.example_field(1)
print(f)

# %% [markdown]
#
# 1: Native cf
# ------------
#
# As mentioned, cf supports a handful of `field operations <https://ncas-cms.github.io/cf-python/class/cf.Field.html#mathematical-operations>`_ that automatically update the domain and metadata alongside the data array.
#
# Additionally, where a function or operation has a specific domain, cf will mask any erroneous elements that were not processed properly.
#

# %%
# 1. Create an instance of the template field to work with:

field1 = f.copy()

# %%
# 2. Calculate the sine of the elements in the data array:

new_field = field1.sin()

print(new_field.data)

# %%
# Alternatively, we can update the original field in place using the ``inplace`` parameter:

field1.sin(inplace=True)

print(field1.data)

# cf will automatically update the units of our field depending on the operation.
# Here, since the sine is a dimensionless value, we get the units "1".

print(f.units)  # Original
print(field1.units)  # After operation

# %% [markdown]
#
# 2: Dask
# -------
#
# When it comes to computing mathematical operations on a data array,
# cf utilises two different libraries under the hood: Dask and NumPy.
#
# In the event that cf does not natively support an operation, the next
# port of call is Dask (or specifically, the``dask.array``module).
#
# Dask implements `a number of functions <https://docs.dask.org/en/stable/array-numpy-compatibility.html>`_, either as pass-throughs for
# NumPy functions (see below) or as its own implementations.
#
# To preserve the metadata associated with the origin field, we will
# have to create a duplicate of it and rewrite the data array using the
# ``f.Field.set_data()`` method. However, care must be taken to also
# update metadata such as units or coordinates when applying a function
# from outside of cf.
#

# %%
# 1. Import the necessary Dask module:

import dask as da

# %%
# 2. Create an instance of the template field to work with:

field2 = f.copy()

# %%
# 3. Load the data from the field as a Dask array:

data = field2.data

dask_array = data.to_dask_array()

# %%
# 4. Create a new field, calculate the sine of the elements,
# and write the array to the new field:

new_field = field2.copy()

calculated_array = da.array.sin(dask_array)

new_field.set_data(calculated_array)

print(new_field.data)

# %%
# 5. Manually update the units:

new_field.override_units("1", inplace=True)

print(new_field.units)

# %%
# To instead update the original field in place, as before:

calculated_array = da.array.sin(dask_array)

field2.set_data(calculated_array)

field2.override_units("1", inplace=True)

print(field2.data)
print(field2.units)

# %% [markdown]
#
# 3: NumPy Universal Functions
# ----------------------------
#
# Applying an operation with Dask and NumPy is a similar process,
# and some Dask functions are effectively aliases for equivalent NumPy
# functions. NumPy has so-called `universal functions <https://numpy.org/doc/stable/reference/ufuncs.html>`_ that improve
# performance when working on large arrays compared to just iterating
# through each element and running a function on it.
#
# As above, take care to manually update any metadata for the new field.
#

# %%
# 1. Import NumPy:

import numpy as np

# %%
# 2. Create an instance of the template field to work with:

field3 = f.copy()

# %%
# 3. Create a new field, compute the sine of the elements,
# and write the array to the new field:

new_field = field3.copy()

calculated_array = np.sin(field3)

new_field.set_data(calculated_array)

print(new_field.data)

# %%
# 4. Manually update the units:

new_field.override_units("1", inplace=True)

print(new_field.units)

# %% [markdown]
#
# 4: NumPy Vectorization
# ----------------------
#
# In the event that the operation you need is not supported in cf, Dask,
# or NumPy, then any standard Python function can be vectorized using
# NumPy. In essence, this simply allows the function to take an array as
# input, and return the updated array as output. There is no improvement
# in performance to simply iterating through each element in the data
# array and applying the function.
#

# %%
# 1. Import our third-party function; here, from the ``math`` module:

import math

# %%
# 2. Create an instance of the template field to work with:

field4 = f.copy()

# %%
# 3. Vectorize the function with NumPy:

vectorized_function = np.vectorize(math.sin)

# %%
# 4. Create a new field, calculate the sine of the elements,
# and write the array to the new field:

new_field = field4.copy()

calculated_array = vectorized_function(field4)

new_field.set_data(calculated_array)

print(new_field.data)

# %%
# 5. Manually update the units:

new_field.override_units("1", inplace=True)

print(new_field.units)

# %% [markdown]
#
# Performance
# -----------
#
# NumPy and Dask tend to work the quickest thanks to their universal
# functions. NumPy vectorization works much slower as functions cannot
# be optimised in this fashion.
#
# Operations in cf, whilst running NumPy and Dask under the hood, still
# come with all the performance overheads necessary to accurately adapt
# metadata between fields to ensure that resultant fields are still
# compliant with conventions.
#
