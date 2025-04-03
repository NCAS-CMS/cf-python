# Instructions for the cf-python gallery of code recipes

## Environment

* The generation of recipes using Sphinx-Gallery requires `Python<3.12.0` and:

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
  pip install sphinxcontrib.applehelp==1.0.4
  pip install alabaster==0.7.13
  pip install sphinx==2.4.5
  ```
  
* **Note** this is a very old very of Sphinx and some of its plugins etc., which will be updated soon so we
use the latest, but is necessary for now to maintain certain features. When installing libraries, take
care not to allow the newest version of Sphinx to be installed by default during the operation.


## Notes

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
  alphabetically for `docs/source/recipes/README.rst`, e.g.,

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


## Building for a release

* Since the datasets to use are often very large, it is best to build only new recipes and re-build any
  which have been updated (usually to replace deprecated keywords or methods, etc.).
  * To force the prevention of re-builds, move the relevant `docs/source/recipes/plot_*_recipe.py` script(s)
  out of that directory (or `rm` them, they can be recovered by version control) and they won't be built.
  Use that to build only recipes that are new or updated.
  * The HTML files, code and generated notebooks under `_downloads` and generated plots
     and thumbnail images under  `_images`, for any previously-built recipes, can then be manually copied
     across from stored archive builds, to create the full recipe listing.
  * The `index.html` will need to be updated to list the previously-generated recipes, also.
  * Note that recipes built for the `dev` or `archive` documentation builds will have paths in the
     recipes HTML files beginning with the relative path `../../` whereas for the `latest` documentation
     build this needs to be `../` only i.e. is one level up in the tree structure, so if copying anything over
     note this might need to be updated.
