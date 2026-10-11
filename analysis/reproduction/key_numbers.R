# Read the key numbers of the analysis (sample counts, model coefficients and
# P-values, PERMANOVA/PERMDISP, beta diversity, indicator taxa) from the
# workspace of a finished run (session.RData written by run.sh) and write them
# as key,value into key_numbers.csv. Nothing is refitted except the post-hoc
# contrast, which is evaluated on the run's own abu_model exactly as in the Rmd
# (pairs of emmeans(~ treatment)). make_table_s15.py builds Table S15 from the
# key_numbers.csv of the main and the sensitivity run.
suppressPackageStartupMessages({
  library(tidyverse); library(glmmTMB); library(emmeans); library(vegan)
})
load(Sys.getenv("SESSION", "/in/session.RData"))
out_file <- Sys.getenv("KEY_NUMBERS_OUT", "/out/key_numbers.csv")

rows <- list()
put <- function(key, value) rows[[length(rows) + 1]] <<- tibble(key = key, value = as.numeric(value))

# Samples, sites, specimens
put("samples_n", nrow(env_prep))
put("sites_n", n_distinct(env_prep$site))
put("eco_sites_n", n_distinct(env_prep$site[env_prep$treatment == "eco"]))
put("specimens_with_order_n", nrow(results_prep))
put("units_in_results_prep", n_distinct(results_prep$unit))
put("taxa_n", nlevels(droplevels(results_prep$species)))

# GLMMs of the "Big Picture" chunk (same formulas, data and families as the
# summary tables of the diversity, abundance and biomass sections)
coefs <- function(model, prefix) {
  cf <- summary(model)$coefficients$cond
  for (term in rownames(cf)) {
    k <- gsub("[^A-Za-z0-9]+", "_", term)
    put(paste0(prefix, "_", k, "_b"), cf[term, "Estimate"])
    put(paste0(prefix, "_", k, "_p"), cf[term, "Pr(>|z|)"])
  }
  put(paste0(prefix, "_N"), nobs(model))
}
coefs(div_model, "div")
coefs(abu_model, "abu")
coefs(bio_model, "bio")
ph <- summary(pairs(emmeans(abu_model, ~ treatment), adjust = "tukey"))
put("abu_posthoc_p", ph$p.value[1])
put("abu_posthoc_size", ref_grid(abu_model)@grid$size[1])

# Community composition (results[[1]] = chosen metric, as printed in Table S8/S9)
pm <- results[[1]]$permanova
for (term in c("treatment", "temp_mean", "yday")) {
  put(paste0("permanova_", term, "_R2"), pm[term, "R2"])
  put(paste0("permanova_", term, "_p"), pm[term, "Pr(>F)"])
}
put("permdisp_p", as.numeric(permdisp_result[["p-value"]][1]))
# Axis labels of Fig. 4a (CAP1, MDS1 of the CAP ordination): share of each real
# eigenvalue in the sum of all real eigenvalues (eigenvals() of the capscale)
ev <- eigenvals(cc_select$ord_cap)
put("cap1_pct", 100 * ev[1] / sum(ev))
put("mds1_pct", 100 * ev[2] / sum(ev))
for (i in seq_along(permanova_list)) {
  put(paste0("beta_", metric_names[i], "_p"), permanova_list[[i]][1, "Pr(>F)"])
  put(paste0("beta_", metric_names[i], "_R2"), permanova_list[[i]][1, "R2"])
}
bd <- beta_permdisp_result
for (i in seq_len(nrow(bd))) put(paste0("betadisp_", bd[["Metric"]][i], "_p"), as.numeric(bd[["p-value"]][i]))

# Indicator species
put("ind_n", nrow(indicator_all))
put("ind_eco_n", sum(indicator_all$indicator == "eco", na.rm = TRUE))
put("ind_control_n", sum(indicator_all$indicator == "control", na.rm = TRUE))
put("ind_p005_n", sum(indicator_all$p.value < 0.05))
put("ind_tested_n", length(keep_spp))
put("ind_mean_size_mm", mean(as.numeric(sub("k1", "0.1", sub(".*#", "", indicator_all$species_name)))))

# Model accuracy (test set; independent of the sample exclusion, listed as a control)
st <- cor.test(~ train_taxon_count + accuracy_without_threshold,
               data = taxa_summary_prep, method = "spearman", exact = FALSE)
put("spearman_training_rho", st$estimate); put("spearman_training_p", st$p.value)
put("spearman_size_rho", spearman_res$estimate); put("spearman_size_p", spearman_res$p.value)
put("vline_90", vline_90$xint); put("vline_95", vline_95$xint)

write_csv(bind_rows(rows), out_file)
cat("written:", out_file, "rows:", length(rows), "\n")
