source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate igwn-py311-20240910

python make_events_hdf.py \
	--gwtc1-path /home/cbc/CatalogDraftReleases/gwtc1/P1900392_3/ \
	--gwtc2-path /home/cbc/CatalogDraftReleases/gwtc2p1/6513631_3/ \
	--gwtc3-path /home/cbc/CatalogDraftReleases/gwtc3/8177023_3/ \
	--gwtc4-path /home/cbc/CatalogDraftReleases/gwtc4/GWTC4-Stable_Release-3/ \
	--remove-er15-events \

conda deactivate
