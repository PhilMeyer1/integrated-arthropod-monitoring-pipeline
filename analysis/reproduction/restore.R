# Restore the package library for the reproduction run (inside the container).
# Versions: analysis/renv.lock (R 4.5.0, newest package date 2025-11-09).
# The lockfile is "explicit" (DESCRIPTION only); packages the Rmd loads but
# the lockfile does not pin are installed from the same CRAN snapshot.
snapshot <- Sys.getenv("P3M_SNAPSHOT", "2025-11-19")
repo <- sprintf("https://p3m.dev/cran/__linux__/noble/%s", snapshot)
lib <- Sys.getenv("R_LIBS_USER", "/opt/rlib")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(lib)
options(repos = c(CRAN = repo), Ncpus = max(1L, parallel::detectCores() - 1L))

if (!requireNamespace("renv", quietly = TRUE) ||
    packageVersion("renv") != "1.1.5") {
  install.packages("renv", lib = lib)
}
Sys.setenv(RENV_CONFIG_REPOS_OVERRIDE = repo)
proj <- "/tmp/rproj"
dir.create(proj, showWarnings = FALSE)
renv::restore(project = proj, lockfile = "/lock/renv.lock", library = lib,
              repos = c(CRAN = repo), prompt = FALSE)

# Loaded by Data_Analysis.Rmd but not recorded in renv.lock.
extras <- c("MuMIn", "dotwhisker", "broom.mixed", "ggeffects", "emmeans",
            "sjPlot", "flextable", "officer", "GUniFrac", "betapart",
            "indicspecies", "metR", "png", "rmarkdown", "knitr")
missing <- setdiff(extras, rownames(installed.packages(lib.loc = lib)))
if (length(missing)) install.packages(missing, lib = lib)

ip <- installed.packages(lib.loc = lib)[, c("Package", "Version")]
write.csv(ip, "/out/installed_packages.csv", row.names = FALSE)
