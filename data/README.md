# Data acquisition

This directory documents how to obtain the datasets used by GenoPerfVAE. Raw and processed genotype and phenotype files are not distributed in this repository. Download the data from the original providers and keep local data files under `data/`.

## Rice — updated IRRI dataset

**Source:** the International Rice Research Institute (IRRI) irrigated rice breeding population used by Spindel et al. (2015).

- [Original publication](https://doi.org/10.1371/journal.pgen.1004982)
- [Dryad data repository](https://doi.org/10.5061/dryad.7369p)

Open the Dryad record and download the genotype and phenotype files from the **Data files** section:

- `MET_crfilt_.90_outliers_removed_for_RRBlup_line_corrected.csv`: the post-imputation genotype matrix prepared for genomic-selection cross-validation in the original study.
- `RYT_plotdata_by_GHID_corrected_PUBLIC_ACCESS.zip`: the raw phenotypic records for the 2009–2012 dry and wet seasons.

The archive also provides a HapMap genotype release and marker subsets. These releases have different filtering histories; use the release corresponding to the study's preprocessing protocol.

The updated manuscript describes a genotype–phenotype-matched subset of **349 accessions**, with the phenotype labels `YLD`, `PH`, `TILLER`, `FL`, `LDG`, `MAT`, and `Kg_ha`. The 349-accession subset is specific to this study and is not the sample count of the original public population. Reproducing it requires the same accession matching and phenotype preparation used in the study.

## Wheat2000

**Source:** the Iranian bread wheat landrace collection from CIMMYT, described by Crossa et al. (2016).

- [Original publication](https://doi.org/10.1534/g3.116.029637)
- [CIMMYT replication data](https://hdl.handle.net/11529/10548918)

Open the CIMMYT record and obtain the available genotype and phenotype archives from the **Files** tab. Select the Iranian landrace data corresponding to the 2,000-accession benchmark. The six prediction traits used in this project are thousand-kernel weight, test weight, grain length, grain width, grain hardness, and grain protein content.

If a required archive is marked as restricted, follow the repository's access-request procedure. The original publication also identifies the Iranian data files, including `standarizedData_univariate.RData`, in its **Data availability** section.

## Wheat599

**Source:** the historical CIMMYT wheat dataset distributed with the R package BGLR.

- [BGLR package on CRAN](https://CRAN.R-project.org/package=BGLR)
- [Official BGLR reference manual](https://cran.r-project.org/web/packages/BGLR/refman/BGLR.html)

Install BGLR and load its bundled wheat data in R:

```r
install.packages("BGLR")
library(BGLR)
data("wheat", package = "BGLR")
```

The loaded objects include `wheat.X` (marker genotypes) and `wheat.Y` (grain yield in four environments). The packaged marker matrix contains 599 lines and 1,279 markers. Use the four columns of `wheat.Y` for the four environment-specific tasks.

## Maize1404

**Source:** the CUBIC maize population, comprising 1,404 offspring derived from 24 founders.

- [MaizeCUBIC database](http://cubicmaize.hzau.edu.cn/)
- [Database publication and data-access description](https://doi.org/10.1093/database/baaa044)

Access MaizeCUBIC and follow its download links for the CUBIC population genotype and phenotype data. Select the offspring population used in the Maize1404 benchmark; the founders are additional records.

If the legacy database cannot be reached, consult the publication's data-access information or contact the data providers for the corresponding release.

## Local use

Download links provide source data, which may require conversion, sample alignment, and study-specific filtering before they can be read by the training script. The source archives do not necessarily use the local filenames expected by the code. Follow the manuscript's preprocessing protocol and the dataset reader in the version of the code being used.

Keep this `README.md` in version control and exclude downloaded and processed data files. Cite the original publications and dataset records when using these resources.

