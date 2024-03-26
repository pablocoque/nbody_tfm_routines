# Main routines for an easy galaxy catalog generation and cosmological reconstruction

The main routines used for the research used on the TFM project are stored under the
`./mymocks/` directory. These routines make use of the nbodykit python package, both
for the catalog generation, reconstruciton and analysis. The only thing missing from
said directory is the generation of paired initial conditions that is stored in the
main directory, the routines for this are in the `generate_ICs.py` script. Apart from
these main routines, the repository has many other routines from different attempts of
generating mock catalogues, preparing plots, or some other things. It also has many
repeated code here and there, this hopefully gets ordered.

## Simple mock catalogue generation

Starts with the generation of paired ICs (in `generate_ICs.py`), seeding non-linearities,
forward Zeld'ovich evolution, to finally populate with tracers, e.g. galaxies. Main
example of this is inside the `./mymocks/generate_catalog.ipynb` notebook.

## Cosmological reconstruction

Routines for the iterative reconstruction scheme. Subject to improvements from reconstruction
validation and ability to recover expected bias. Main routines for it are stored in
`./mymocks/reconstruction_iter.ipynb` notebook.