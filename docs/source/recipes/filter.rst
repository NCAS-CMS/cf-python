**Recipes using cf**
====================

----

Version |release| for version |version| of the CF conventions.

Click on the keywords below to filter the recipes according to their function:

.. raw:: html

   <script>
   document.addEventListener('DOMContentLoaded', function() {
       showFilteredRecipes('all');

       function showFilteredRecipes(filter) {
           const recipes = document.querySelectorAll('.sphx-glr-thumbcontainer');
           recipes.forEach(recipe => {
               if (filter === 'all') {
                   recipe.style.display = 'inline-block';
               } else if (recipe.classList.contains(filter)) {
                   recipe.style.display = 'inline-block';
               } else {
                   recipe.style.display = 'none';
               }
           });
       }

       const filterButtons = document.querySelectorAll('.filter-menu button');
       filterButtons.forEach(button => {
           button.addEventListener('click', () => {
               const filter = button.getAttribute('data-filter');
               showFilteredRecipes(filter);
           });
       });
   });
   </script>

.. raw:: html

   <style>
      .filter-menu {
          margin-bottom: 20px;
      }
   </style>

   <div class="filter-menu">
       <button data-filter="all">All</button>
       <button data-filter="aggregate">Aggregate</button>
       <button data-filter="collapse">Collapse</button>
       <button data-filter="contourmap">Contourmap</button>
       <button data-filter="lineplot">Lineplot</button>
       <button data-filter="regrid">Regrid</button>
       <button data-filter="subspace">Subspace</button>
   </div>

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer collapse lineplot" tooltip="In this recipe we will calculate and plot monthly and annual global mean temperature timeseries...">

.. only:: html

  .. image:: /recipes/images/thumb/sphx_glr_plot_1_recipe_thumb.png
    :alt: Calculating global mean temperature timeseries

  :ref:`sphx_glr_recipes_plot_1_recipe.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calculating global mean temperature timeseries</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer collapse lineplot subspace" tooltip="In this recipe we will calculate and plot the global average temperature anomalies.">

.. only:: html

  .. image:: /recipes/images/thumb/sphx_glr_plot_2_recipe_thumb.png
    :alt: Calculating and plotting the global average temperature anomalies

  :ref:`sphx_glr_recipes_plot_2_recipe.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calculating and plotting the global average temperature anomalies</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer collapse contourmap" tooltip="In this recipe, we will plot the global mean temperature spatially.">

.. only:: html

  .. image:: /recipes/images/thumb/sphx_glr_plot_3_recipe_thumb.png
    :alt: Plotting global mean temperatures spatially

  :ref:`sphx_glr_recipes_plot_3_recipe.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting global mean temperatures spatially</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer regrid" tooltip="In this recipe, we will regrid two different datasets with different resolutions. An example us...">

.. only:: html

  .. image:: /recipes/images/thumb/sphx_glr_plot_4_recipe_thumb.png
    :alt: Comparing two datasets with different resolutions using regridding

  :ref:`sphx_glr_recipes_plot_4_recipe.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparing two datasets with different resolutions using regridding</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer subspace contourmap" tooltip="In this recipe we will plot wind vectors, derived from northward and eastward wind components, ...">

.. only:: html

  .. image:: /recipes/images/thumb/sphx_glr_plot_5_recipe_thumb.png
    :alt: Plotting wind vectors overlaid on precipitation data

  :ref:`sphx_glr_recipes_plot_5_recipe.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting wind vectors overlaid on precipitation data</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer regrid contourmap" tooltip="In this recipe, we will be regridding from a rotated latitude-longitude source domain to a regu...">

.. only:: html

  .. image:: /recipes/images/thumb/sphx_glr_plot_6_recipe_thumb.png
    :alt: Converting from rotated latitude-longitude to regular latitude-longitude

  :ref:`sphx_glr_recipes_plot_6_recipe.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Converting from rotated latitude-longitude to regular latitude-longitude</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer aggregate lineplot subspace" tooltip="In this recipe, we will plot the members of a model ensemble.">

.. only:: html

  .. image:: /recipes/images/thumb/sphx_glr_plot_7_recipe_thumb.png
    :alt: Plotting members of a model ensemble

  :ref:`sphx_glr_recipes_plot_7_recipe.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting members of a model ensemble</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer collapse contourmap" tooltip="In this recipe, we will analyse and plot temperature trends from the HadCRUT.5.0.1.0 dataset fo...">

.. only:: html

  .. image:: /recipes/images/thumb/sphx_glr_plot_8_recipe_thumb.png
    :alt: Plotting statistically significant temperature trends with stippling

  :ref:`sphx_glr_recipes_plot_8_recipe.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plotting statistically significant temperature trends with stippling</div>
    </div>

.. raw:: html

    </div>

.. toctree::
   :hidden:

   /recipes/plot_1_recipe
   /recipes/plot_2_recipe
   /recipes/plot_3_recipe
   /recipes/plot_4_recipe
   /recipes/plot_5_recipe
   /recipes/plot_6_recipe
   /recipes/plot_7_recipe
   /recipes/plot_8_recipe

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: recipes_python.zip </recipes/recipes_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: recipes_jupyter.zip </recipes/recipes_jupyter.zip>`

.. only:: html

  .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_