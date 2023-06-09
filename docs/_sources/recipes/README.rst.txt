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
       <button data-filter="histogram">Histogram</button>
       <button data-filter="lineplot">Lineplot</button>
       <button data-filter="maths">Mathematical Operations</button>
       <button data-filter="regrid">Regrid</button>
       <button data-filter="subspace">Subspace</button>
   </div>

.. raw:: html

