# Initialize renv for reproducible R environment
# This script sets up renv and creates the renv.lock file

cat("Initializing renv for reproducible R environment...\n\n")

# Check if renv is installed
if (!requireNamespace("renv", quietly = TRUE)) {
  cat("Installing renv package...\n")
  install.packages("renv", repos = "https://cloud.r-project.org/")
}

# Initialize renv
cat("Setting up renv environment...\n")
renv::init(bare = TRUE, restart = FALSE)

# Install all packages from DESCRIPTION
cat("\nInstalling packages from DESCRIPTION file...\n")
if (file.exists("DESCRIPTION")) {
  renv::install()
} else {
  cat("Warning: DESCRIPTION file not found!\n")
}

# Create snapshot
cat("\nCreating renv.lock snapshot...\n")
renv::snapshot()

cat("\n✓ renv initialization complete!\n")
cat("The renv.lock file has been created with all package versions.\n")
