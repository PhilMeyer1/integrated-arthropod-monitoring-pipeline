# Reproduction environment for analysis/scripts/Data_Analysis.Rmd (v1.1.0).
# Built and used by run_all.sh; see analysis/README.md, "Reproduction in Docker".
# R 4.5.0 as recorded in analysis/renv.lock; packages are restored into a
# Docker volume by restore.R, not baked into the image.
FROM rocker/r-ver:4.5.0

# System libraries for the binary/source packages in renv.lock and pandoc
# for rmarkdown::render().
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      pandoc \
      libxml2-dev libcurl4-openssl-dev libssl-dev \
      libfontconfig1-dev libfreetype6-dev libharfbuzz-dev libfribidi-dev \
      libpng-dev libtiff5-dev libjpeg-dev libwebp-dev \
      libgmp-dev libmpfr-dev libglpk-dev libudunits2-dev \
      libuv1-dev libicu-dev zlib1g-dev libzstd-dev cmake \
    && rm -rf /var/lib/apt/lists/*

ENV R_LIBS_USER=/opt/rlib
WORKDIR /run

# run.sh applies reproduction.patch to a scratch copy of the Rmd.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends patch \
    && rm -rf /var/lib/apt/lists/*
