source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate igwn-py311-20240910

python sample_purity_calc.py \
	--input-hdf-filename 'events.hdf' \
	--remove-er15 \
	--output-filename 'samplePurityEstimate.json' \

conda deactivate
