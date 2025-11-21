# R Package Setup Script
# Convenience script for installing all required R packages
# for the arthropod monitoring analysis

cat("==================================================\n")
cat("R Package Setup for Arthropod Monitoring Analysis\n")
cat("==================================================\n\n")

# Check R version
if (getRversion() < "4.0.0") {
  stop("This analysis requires R version 4.0.0 or higher. ",
       "You are running R version ", getRversion())
}

cat("R version:", as.character(getRversion()), "✓\n\n")

# Define required packages
required_packages <- c(
  "DT", "tidyverse", "rio", "data.table", "lubridate",
  "ggpubr", "ggsci", "mgcv", "ggstatsplot", "patchwork",
  "funspace", "car", "broom", "glmnet", "MASS",
  "DHARMa", "codyn", "viridis", "iNEXT", "vegan",
  "cluster", "ggordiplots", "openxlsx", "svglite", "glmmTMB"
)

cat("This script will install", length(required_packages), "required packages.\n\n")

# Present installation options
cat("Choose an installation method:\n")
cat("  [1] Using renv (recommended for exact reproducibility)\n")
cat("  [2] Using DESCRIPTION file (installs latest compatible versions)\n")
cat("  [3] Manual installation (installs only missing packages)\n\n")

choice <- readline(prompt = "Enter your choice (1, 2, or 3): ")

if (choice == "1") {
  # Method 1: renv
  cat("\n--- Installing using renv ---\n")

  if (!requireNamespace("renv", quietly = TRUE)) {
    cat("Installing renv package...\n")
    install.packages("renv")
  }

  if (file.exists("renv.lock")) {
    cat("Found renv.lock file. Restoring exact package versions...\n")
    renv::restore()
    cat("\n✓ Installation complete using renv!\n")
    cat("All packages installed with exact versions from renv.lock\n")
  } else {
    cat("Warning: renv.lock not found.\n")
    cat("Initializing new renv environment...\n")
    renv::init()
    cat("\n✓ renv initialized. Please install packages and run renv::snapshot()\n")
  }

} else if (choice == "2") {
  # Method 2: DESCRIPTION file
  cat("\n--- Installing using DESCRIPTION file ---\n")

  if (!requireNamespace("remotes", quietly = TRUE)) {
    cat("Installing remotes package...\n")
    install.packages("remotes")
  }

  if (file.exists("DESCRIPTION")) {
    cat("Installing dependencies from DESCRIPTION...\n")
    remotes::install_deps(dependencies = TRUE)
    cat("\n✓ Installation complete using DESCRIPTION!\n")
    cat("All packages installed with latest compatible versions\n")
  } else {
    stop("DESCRIPTION file not found!")
  }

} else if (choice == "3") {
  # Method 3: Manual installation
  cat("\n--- Manual installation ---\n")
  cat("Checking which packages are already installed...\n\n")

  installed <- installed.packages()[, "Package"]
  missing <- required_packages[!required_packages %in% installed]

  if (length(missing) == 0) {
    cat("✓ All required packages are already installed!\n")
  } else {
    cat("Missing packages:", length(missing), "\n")
    cat(paste("-", missing, collapse = "\n"), "\n\n")

    cat("Installing missing packages...\n")
    install.packages(missing, dependencies = TRUE)

    cat("\n✓ Installation complete!\n")
  }

} else {
  stop("Invalid choice. Please run the script again and choose 1, 2, or 3.")
}

# Verify installation
cat("\n--- Verifying installation ---\n")
all_installed <- all(required_packages %in% installed.packages()[, "Package"])

if (all_installed) {
  cat("✓ Success! All required packages are installed.\n\n")
  cat("You can now run the analysis with:\n")
  cat("  rmarkdown::render('scripts/Data_Analysis.Rmd')\n")
} else {
  still_missing <- required_packages[!required_packages %in% installed.packages()[, "Package"]]
  cat("⚠ Warning: Some packages are still missing:\n")
  cat(paste("-", still_missing, collapse = "\n"), "\n\n")
  cat("Please try installing them manually:\n")
  cat("  install.packages(c('", paste(still_missing, collapse = "', '"), "'))\n", sep = "")
}
