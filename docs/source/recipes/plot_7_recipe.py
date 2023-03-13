"""
Plotting members of a model ensemble
====================

In this recipe, we will plot the members of a model ensemble.
"""

# %%
# 1. Import cf-python and cf-plot:

import cf
import cfplot as cfp

# %%
# 2. Read the field constructs using read function and store it in the variable ``f``. The * in the filename is a wildcard character which means the function reads all files in the directory that match the specified pattern. [0:5] selects the first five elements of the resulting list:

f = cf.read('~/recipes/realization/PRMSL.1941_mem*.nc')[0:5]
print(f)

# %%
# 3. The desciption of one of the fields from the list shows ``'realization'`` as a property by which the members of the model ensemble are labelled:

f[1].dump()

# %%
# 4. An ensemble of the members is then created by aggregating the data in ``f`` along the ``'realization'`` using the cf.aggregate function and storing the result in the variable ``ensemble``. ``'relaxed_identities=True'`` allows for missing coordinate identities to be inferred. [0] selects the first element of the resulting list. ``id%realization`` now shows as an auxiliary coordinate for the ensemble:

ensemble = cf.aggregate(f, dimension=('realization',), relaxed_identities=True)[0]
print(ensemble)


# %%
# 5. To see the constructs for the ensemble, the *constructs* attribute is passed in print:

print(ensemble.constructs)

# %%
# 6. Loop over the realizations in the ensemble using the *range* function and the *domain_axis* to determine the size of the realization dimension. For each realization, extract a subspace of the ensemble using the *subspace* method and the ``'id%realization'`` keyword argument along specific latitude and longitude and plot the realizations from the 4D field using `cfplot.lineplot <http://ajheaps.github.io/cf-plot/lineplot.html>`_. The *squeeze* method removes any dimensions of size 1 from the field to produce a 2D field.
# The ensemble mean is plotted similarly by using the *collapse* method to take the mean along the ``'realization'`` dimension of the ensemble.


cfp.gopen()

for realization in range(1, ensemble.domain_axis('id%realization').size + 1):
    cfp.lineplot(
        ensemble.subspace(**{'id%realization': realization},
            latitude=[0], longitude=[0]).squeeze(),
        label=f'Member {realization}',
        linewidth=1.0)

cfp.lineplot(ensemble.collapse('mean', 'id%realization')[:, :, 0, 0].squeeze(),
    label='Ensemble mean', linewidth=2.0, color='black',
    title='Model Ensemble Pressure')
    
cfp.gclose()