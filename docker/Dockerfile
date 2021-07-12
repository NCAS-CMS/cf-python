FROM continuumio/miniconda3

RUN /opt/conda/bin/conda update -n base -c defaults conda
RUN /opt/conda/bin/conda install -c ncas -c conda-forge cf-python cf-plot udunits2==2.2.25
RUN /opt/conda/bin/conda install -c conda-forge mpich esmpy
