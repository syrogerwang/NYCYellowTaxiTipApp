library(config)
library(readr)

source("model.R")

cfg <- config::get()

files <- paste0(cfg$base_url, cfg$files)

zone_lookup_url <- paste0(cfg$base_url, cfg$zone_lookup_file)

borough_zone <- read_csv(zone_lookup_url)

taxi <- prepare_taxi_data(files, borough_zone)
tip_model <- train_model(taxi)

saveRDS(tip_model, "tip_model.rds")
