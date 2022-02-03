# Implementation for FastVAE
"Fast Variational AutoEncoder with Inverted Multi-Index for Collaborative Filtering" (WWW2022)

## Data
Example datasets in dictory ` /datasets/`.
Format : Rating matrix in sparse matrix (`.mat`)

## Sampler 
See more details in `sampler_gpu_mm.py`
+ SamplerBase: Uniform Sampling
+ PopularSampler: Popularity-based Sampling
+ MidxUniform：Midx Sampler with Uniform
+ MidxUniPop：Midx Sampler with Popularity

## Run example
`python run_mm.py`
or
`sh test_running.sh`
