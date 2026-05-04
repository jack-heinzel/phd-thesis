## GWTC-4.0: Population Properties of Merging Compact Binaries

Welcome to the data release associated with **GWTC-4.0: Population Properties of Merging Compact Binaries**. This is a large data release; if you run into problems please [alert us](mailto:aditya.vijaykumar@ligo.org).

### Description of files in this data release

1. `analyses_AllCBC.tar`: Contains `popsummary` format files for all analyses that are part of the section "Binary Merger Population Across All Masses"
2. `analyses_NS.tar`: Contains `popsummary` format files for all analyses that are part of the section "Population Properties of Mergers Containing Neutron Stars"
3. `analyses_BBH.tar`: Contains `popsummary` format files for all analyses that are part of the section "Binary Black Hole Population", except those that use the Binned Gaussian Processes (BGP) model.
4. `analyses_BGP.tar`: Contains `popsummary` format files for all analyses that are part of the section "Binary Black Hole Population" and use the Binned Gaussian Processes (BGP) model. 
5. `figure_scripts.tar`: Contains all `.py` scripts needed to produce figures in the paper.
6. `figures.tar`: Contains `.pdf` files of all figures in the paper
7. `o4a_event_list.tar`: Contains list of events used in the paper, along with the label used in the PE samples file.
8. `download_gwtc3_data.py`: Download necessary GWTC-3 data to reproduce figures.
9. `popsummary_tutorial.ipynb`: A short tutorial that shows how to deal with `popsummary` files.

### Setup instructions

#### Creating a conda environment

Use the following commands to set up a conda environment called `o4a-astro` and install required packages:

```
conda create --name o4a-astro python=3.11 -y
conda activate o4a-astro
pip install numpy==2.3.2 scipy==1.16.1 matplotlib==3.10.5 seaborn==0.13.2 bilby==2.6.0 popsummary==0.0.1 zenodo-get==2.0.0
```

#### Setup directory structure

Run the following on the bash terminal

```
mkdir o4a-astro
cd o4a-astro
mkdir data_release figures figure_scripts

# If you want to get the figure .py scripts
zenodo_get 16911563 -g figure_scripts.tar -o figure_scripts
tar -xvf figure_scripts/figure_scripts.tar -C figure_scripts

# If you want to get the figure pdf files
zenodo_get 16911563 -g figures.tar -o figures
tar -xvf figures/figures.tar -C figures
```

Optionally, if you want to download GWTC-3 data, run the following:

```
zenodo_get 16911563 -g download_gwtc3_data.py
python download_gwtc3_data.py
```

Running download_gwtc3_data.py will create a directory called `gwtc3_data` in the `o4a-astro` directory.

The final ingredient to run the figure scripts is downloading the `popsummary` themselves. Below is a table that says which TAR file each script uses, and which input h5 files from inside the TAR archive it uses. Download the requisite TAR files, and extract them inside the `o4a-astro/data_release`. Thereafter, you can enter the `o4a-astro/figure_scripts` directory and run scripts using `python figure_<figureID>.py`. The figure file will be saved in the `o4a-astro/figures` directory (any existing files will be overwritten).

| Figure Script              | TAR File                                             | Input Files                                                                                                                                                                                                                                                                                                                                                                                                                         |
| -------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **figure\_1.py**           | `analyses_AllCBC.tar`                                | `AllCBC_FullPop.h5`<br>`AllCBC_FullPopBGP.h5`                                                                                                                                                                                                                                                                                                                                                                                       |
| **figure\_2.py**           | `analyses_AllCBC.tar`                                | `AllCBC_FullPop.h5`<br>`AllCBC_FullPopBGP.h5`                                                                                                                                                                                                                                                                                                                                                                                       |
| **figure\_3.py**           | `analyses_BBH.tar`, `gwtc3_data`                     | `BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5`<br>`BBHMassSpinRedshift_BSplineIID.h5`<br>`gwtc3_data/PowerLawPeak`                                                                                                                                                                                                                                                                         |
| **figure\_4.py**           | `analyses_BBH.tar`                                   | `BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5`<br>`BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift_GWTC3.h5`                                                                                                                                                                                                                                            |
| **figure\_5.py**           | `analyses_BBH.tar`, `gwtc3_data`                     | `BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5`<br>`BBHMassSpinRedshift_BSplineIID.h5`<br>`gwtc3_data/PowerLawPeak`                                                                                                                                                                                                                                                                         |
| **figure\_6.py**           | `analyses_BBH.tar`                                   | `BBHMassSpinRedshift_BSplineIsopeakIID.h5`<br>`BBHMass_VaryingBetaQs_DominantMode.h5`                                                                                                                                                                                                                                                                                                                                               |
| **figure\_7.py**           | `analyses_BBH.tar`, `gwtc3_data`                     | `BBHMassSpinRedshift_BSplineIID.h5`<br>`BBHSpin_MagTruncnormIidTiltIsotropicTruncnormNid.h5`<br>`gwtc3_data/PowerLawPeak`                                                                                                                                                                                                                                                                                                           |
| **figure\_8.py**           | `analyses_BBH.tar`                                   | `BBHSpin_MagTruncnormIidTiltIsotropicTruncnormSpinSorting.h5`                                                                                                                                                                                                                                                                                                                                                                       |
| **figure\_9.py**           | `analyses_BBH.tar`, `gwtc3_data`                     | `BBHSpin_EpsSkewNormalChiEff.h5`<br>`gwtc3_data/GaussianSpin`                                                                                                                                                                                                                                                                                                                                                                       |
| **figure\_10.py**          | `analyses_BBH.tar`, `gwtc3_data`                     | `BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5`<br>`BBHMassSpinRedshift_BSplineIID.h5`<br>`gwtc3_data/PowerLawPeak`                                                                                                                                                                                                                                                                         |
| **figure\_11\_and\_14.py** | `analyses_BBH.tar`                                   | `BBHCorr_qchieffLinearCorrelationModel.h5`<br>`BBHCorr_zchieffLinearCorrelationModel.h5`<br>`BBHCorr_qchieffSplineCorrelationModel.h5`<br>`BBHCorr_zchieffSplineCorrelationModel.h5`                                                                                                                                                                                                                                                |
| **figure\_12.py**          | `analyses_BBH.tar`                                   | `BBHCorr_qchieffCopulaCorrelationModel.h5`<br>`BBHCorr_qchieffLinearCorrelationModel.h5`                                                                                                                                                                                                                                                                                                                                            |
| **figure\_13.py**          | `analyses_BGP.tar`                                   | `BBHCorr_MassSpinCorrelatedBGPModel.h5`                                                                                                                                                                                                                                                                                                                                                                                             |
| **figure\_15.py**          | `analyses_BBH.tar`, `gwtc3_data`                     | `BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5`<br>`BBHMassSpinRedshift_BrokenPowerLawOnePeak_GaussianComponentSpins_PowerLawRedshift.h5`<br>`BBHMassSpinRedshift_BrokenPowerLawThreePeaks_GaussianComponentSpins_PowerLawRedshift.h5`<br>`BBHMassSpinRedshift_PowerLawPeak_GaussianComponentSpins_PowerLawRedshift.h5`<br>`BBHMassSpinRedshift_BSplineIID.h5`<br>`gwtc3_data/PowerLawPeak` |
| **figure\_16.py**          | `analyses_BBH.tar`                                   | `BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift_DominantMode.h5`<br>`BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift_SubdominantMode.h5`<br>`BBHMassSpinRedshift_BrokenPowerLawOnePeak_GaussianComponentSpins_PowerLawRedshift.h5`                                                                                                                           |
| **figure\_17.py**          | `analyses_BBH.tar`                                   | `BBHSpin_MagTruncnormIndTiltIsotropicTruncnormNnd.h5`<br>`BBHSpin_MagTruncnormIidTiltIsotropicTruncnormNid.h5`                                                                                                                                                                                                                                                                                                                      |
| **figure\_18\_and\_19.py** | `analyses_BBH.tar`, `gwtc3_data`                     | `BBHSpin_GaussianChiEffChiP.h5`<br>`BBHSpin_GaussianChiEffChiP_NeffCut.h5`<br>`gwtc3_data/GaussianSpin`                                                                                                                                                                                                                                                                                                                             |
| **figure\_20.py**          | `analyses_BBH.tar`, `analyses_BGP.tar`               | `BBHMass_Autoregressive.h5`<br>`BBHCorr_MassRedshiftCorrelatedBGPModel.h5`<br>`BBHMassSpinRedshift_BSplineIID.h5`<br>`BBHMass_FlexibleMixtures.h5`<br>`BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5`<br>`BBHMassSpinRedshift_BSplineIID_NoMinM.h5`                                                                                                                                         |
| **figure\_21.py**          | `analyses_BBH.tar`, `analyses_BGP.tar`, `gwtc3_data` | `BBHCorr_MassSpinCorrelatedBGPModel.h5`<br>`BBHSpin_EpsSkewNormalChiEff.h5`<br>`BBHSpin_BSplineChiEff.h5`<br>`BBHSpin_GaussianChiEffChiP.h5`<br>`BBHCorr_qchieffSplineCorrelationModel.h5`<br>`gwtc3_data/GaussianSpin`                                                                                                                                                                                                             |
| **figure\_22.py**          | `analyses_BBH.tar`, `analyses_BGP.tar`               | `BBHCorr_zchieffCopulaCorrelationModel.h5`<br>`BBHCorr_MassRedshiftCorrelatedBGPModel.h5`<br>`BBHMass_FlexibleMixtures.h5`                                                                                                                                                                                                                                                                                                          |
