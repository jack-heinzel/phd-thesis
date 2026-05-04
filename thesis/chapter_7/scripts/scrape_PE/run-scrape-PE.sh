DATAPATH=/home/cbc/CatalogDraftReleases/gwtc4/GWTC4-Stable_Release-9/
TAG=38214bd95_724
OUTDIR=/home/rp.o4/catalogs/GWTC-4/GWTC4-Stable_Release-9/$TAG
#OUTDIR=/home/aditya.vijaykumar/catalogs/GWTC-4/GWTC4-Stable_Release-1/$TAG
mkdir -p $OUTDIR

ANALYSIS_TYPE="bbh_only"
FAR_THRESHOLD="3.169e-8" # https://git.ligo.org/publications/o4/cbc/gwtc-4-catalog/-/blob/main/catalog_recipes/CorePipelineConfig.yml
./scrape-catalog-PE --analysis-type $ANALYSIS_TYPE --far-threshold $FAR_THRESHOLD --data-path $DATAPATH --tag $TAG --output-dir $OUTDIR # --dryrun
cp {run-scrape-PE.sh,scrape-catalog-PE} $OUTDIR/$ANALYSIS_TYPE

ANALYSIS_TYPE="ns_containing_only"
FAR_THRESHOLD="7.922e-9"
./scrape-catalog-PE --analysis-type $ANALYSIS_TYPE --far-threshold $FAR_THRESHOLD --data-path $DATAPATH --tag $TAG --output-dir $OUTDIR # --dryrun
cp {run-scrape-PE.sh,scrape-catalog-PE} $OUTDIR/$ANALYSIS_TYPE

ANALYSIS_TYPE="full_spectrum"
FAR_THRESHOLD="7.922e-9"
./scrape-catalog-PE --analysis-type $ANALYSIS_TYPE --far-threshold $FAR_THRESHOLD --data-path $DATAPATH --tag $TAG --output-dir $OUTDIR # --dryrun
cp {run-scrape-PE.sh,scrape-catalog-PE} $OUTDIR/$ANALYSIS_TYPE
