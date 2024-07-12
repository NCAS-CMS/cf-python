* The generation of recipes using Sphinx-Gallery requires:

  ```txt
  pip install sphinx-gallery==0.11.0
  pip install sphinx-copybutton==0.5.1
  pip install sphinx-toggleprompt==0.2.0
  pip install sphinxcontrib-spelling==4.3.0
  pip install sphinxcontrib.serializinghtml==1.1.5
  pip install sphinxcontrib.htmlhelp==1.0.3
  pip install sphinxcontrib-devhelp==1.0.2
  pip install sphinxcontrib-serializinghtml==1.1.3
  pip install sphinxcontrib-qthelp==1.0.3
  pip install alabaster==0.7.13
  pip install sphinx==2.4.5
  ```

* The `.py` files to generate recipes are stored in `docs/source/recipes/`.
  
* The netCDF and PP files for generation of the same are available to download at: 
  https://drive.google.com/drive/folders/1gaMHbV0A37hzQHH_C5oMBWOoxgoQPdCT?usp=sharing

* For every new added recipe, `docs/source/recipes/recipe_list.txt` has to be
  edited to include the newly added recipe with its corresponding filters in the
  standardised manner as given for the previous recipes:

  ```txt
  plot_1_recipe.html#sphx-glr-recipes-plot-1-recipe-py
  <div class="sphx-glr-thumbcontainer collapse lineplot" tooltip="Collapse, Lineplot">
  plot_2_recipe.html#sphx-glr-recipes-plot-2-recipe-py
  <div class="sphx-glr-thumbcontainer collapse lineplot subspace" tooltip="Collapse, Lineplot, Subspace">
  plot_3_recipe.html#sphx-glr-recipes-plot-3-recipe-py
  <div class="sphx-glr-thumbcontainer collapse contourmap" tooltip="Collapse, Contourmap">
  ...
  plot_n_recipe.html#sphx-glr-recipes-plot-n-recipe-py
  <div class="sphx-glr-thumbcontainer collapse test" tooltip="Collapse, Test">
  ```

* If the filter doesn't exist, it has to be added in `<div class="filter-menu">` 
  alphabetically for `/docs/source/recipes/README.rst`, e.g.,

  ```html
  <div class="filter-menu">
      <button data-filter="all">All</button>
      <button data-filter="aggregate">Aggregate</button>
      <button data-filter="collapse">Collapse</button>
      <button data-filter="contourmap">Contourmap</button>
      <button data-filter="lineplot">Lineplot</button>
      <button data-filter="regrid">Regrid</button>
      <button data-filter="subspace">Subspace</button>
      <button data-filter="testfilter">Test</button>
  </div>
  ```
