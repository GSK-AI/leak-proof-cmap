# leak proof cmap
Standardized L1000 metric benchmarking using leak proof splits of the the CMAP database

## What is it
The Connectivity Map (CMap) is a large publicly available database of cellular transcriptomic responses to chemical and genetic perturbations built using a standardized acquisition protocol known as the L1000 technique. Databases such as CMap provide an exciting opportunity to enrich drug discovery efforts, providing a 'known' phenotypic landscape to explore and enabling the development of state of the art techniques for enhanced information extraction and better informed decisions. Whilst multiple methods for measuring phenotypic similarity and interrogating profiles have been developed, the ecosystem is severely lacking standardized benchmarks using appropriate data splitting for training and unbiased evaluation of machine learning methods. To address this, we have developed ‘Leak Proof CMap’ and exemplified its application to a set of common transcriptomic and generic phenotypic similarity methods along with an exemplar triplet loss-based method. Benchmarking in three critical performance areas (compactness, distinctness, and uniqueness) is conducted using carefully crafted data splits ensuring no similar cell lines or treatments with shared or closely matching responses or mechanisms of action are present in training, validation, or test sets. This enables testing of models with unseen samples akin to exploring treatments with novel modes of action in novel patient derived cell lines. With a carefully crafted benchmark and data splitting regime in place, the tooling now exists to create performant phenotypic similarity methods for use in personalized medicine (novel cell lines) and to better augment higher throughput phenotypic primary screening technologies with the often complementary L1000 transcriptomic technology.

This 'leakproofcmap' Python package contains all code and functionality used in preparation of the 'Leak Proof CMap; a framework for training and evaluation of cell line agnostic L1000 similarity methods' manuscript, covering the retrieval of leakproof splits, training of new models for phenotypic similarity evaluation and profiling of similarity methods.

## Installation
leakproofcmap requires pytorch built with cuda 11.8. A good approach is to use a new conda environment for python 3.10 and then install pytorch, torchaudio, and torchvision into the a new pyhon environment with the index-url specified. This ensures use of whls built with cuda 11.8 which was required in the build HPC environment. The following commands can be used to set up a new conda environment and install leakproofcmap:
```bash
conda create --name leakproofcmap python=3.10
source activate leakproofcmap
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install leakproofcmap
```

Remember to add the new environment to jupyter ipykernels (replacing leakproofcmap with the name of your python environment):
```bash
python -m ipykernel install --user --name=leakproofcmap
```

The leakproofcmap code should be run in a directory containing a "working_dir" subdirectory, but this will be created if absent. Running any function from this module which requires CMAP data will cause it to check for the presence of a phe_ob_CMAP_CLUE_MOAs_downsampled.pkl located in working_dir/pickles.  If this is not present, then it will use the Phenonaut (python package) dataloader to download and extract CMAP Level 4 data. Additionally, the clue.io Drug Repurposing Hub Drug Information dataset (release version 2/24/2020) will be downloaded, which contains information on compounds including identifiers and mechanisms of action (MOA) and protein target.  It is upon these mode of action groups and also cell line splits that splitting of the CMAP database will be performed. With CMAP Level4 data and compound MOA data, each compound within CMAP has its Clue.io Drug Repurposing Hub MOA assigned to it. All compounds within CMAP which are not assigned a MOA through matching of their 'pert_iname' fields are removed from the CMAP pool. DMSO is a special case and is allowed to remain in the compound pool with no assigned MOA. To ensure the dataset is not skewed by many thousands of repeats of special interest and control compounds, the CMAP pool is downsampled, iterating through each unique treatment and downsampling if required to 200 profiles. 

## Quick usage/functionality
To access prescribed Leak Proof CMap splits, e.g. cell line split 1, MOA split 2 the following function call may be used:
```python
leakproofcmap.get_split_train_validation_test_dfs_from_cmap(1,2)
```

## Notebooks
### leak_proof_cmap_00_define_splits.ipynb
This notebook performs actions to define splits used in Leak Proof CMap. Splits contain training, validation, and test data for the training and validation of machine learning models. Running this notebook will perform the following actions.
- Generate statistics on CMAP and associated compound MOAs, including counts and counts per MOA group. This data is written to TSV files (Tab Separated Values), not CSVs, due to the inclusion of commas in compound names found within pert_iname fields. This is also inkeeping with the clue.io Drug Repurposing Hub Drug Information dataset. TSVs are written to working_dir/split_data and named ‘moa_counts.tsv’ and  ‘pert_iname_counts.tsv’.
- Produce 5 cell line splits and 5 MOA splits based on diversity between splits, and similarity within splits.
- Enumerate these 5 cell line splits and 5 MOA splits to produce 25 splits, covering the holding out of all cell lines and MOAs for model training.
- Write split information to CMapSplit object files as well as ascii files.

### leak_proof_cmap_01_model_selection
Used to create the triplet-loss-bassed phenotypic similarity method as a demonstration of how simple models can return good performance in the chosen benchmarks. Contains code to perform and analyse:
- Hyperparameter scanning
- Cross fold validation
- Training of models for every split

### leak_proof_cmap_02_investigate_model_training.ipynb
With a triplet-loss based established, this notebook contains code to evaluate reproducibility, stability and the types of triplets encountered during training.

### leak_proof_cmap_03_compactness_eval.ipynb
Notebook containing code for analysis of the compactness benchmark task (percent replicating).

### leak_proof_cmap_04_distinctness_eval.ipynb
Notebook containing code for analysis of the distinctness benchmark task (permutation testing of treatments vs the negative DMSO control)

### leak_proof_cmap_05_uniqueness_eval.ipynb
Notebook containing code for analysis of the uniquness benchmark task (AUROC)

### leak_proof_cmap_06_additional_analysis_count_unique_in_splits.ipynb
Notebook containing code for the analysis of unique MOA counts within each split.

### leak_proof_cmap_07_uniqueness_vs_compactness_treatments.ipynb
Notebook containing code used to plot performance of all three benchmark tasks (all CMAP filtered compounds)

### leak_proof_cmap_07_uniqueness_vs_compactness_moas.ipynb
Notebook containing code used to plot performance of all three benchmark tasks (JUMP MOA compounds)


Development has been supported by GSK.