load_taxi_data <- function(
    path_data = "TaxiApp/data",
    lookup_file = "TaxiApp/data/taxi_zone_lookup.csv",
    sample_n = 500,
    seed = 123
) {
  library(arrow)
  library(dplyr)
  library(lubridate)
  
  # List all parquet files
  files <- list.files(path_data, pattern = "\\.parquet$", full.names = TRUE)
  
  if (length(files) == 0) {
    stop("No parquet files found in: ", path_data)
  }
  
  if (!file.exists(lookup_file)) {
    stop("Lookup file not found: ", lookup_file)
  }
  
  # Read and row-bind
  taxi <- files %>%
    lapply(arrow::read_parquet) %>%
    dplyr::bind_rows()
  
  # Join taxi pickup/dropoff location IDs to taxi zone lookup information
  locations <- read.csv(lookup_file, header = TRUE)
  
  taxi <- taxi %>%
    dplyr::left_join(locations, by = c("PULocationID" = "LocationID")) %>%
    dplyr::left_join(
      locations,
      by = c("DOLocationID" = "LocationID"),
      suffix = c("_pickup", "_dropoff")
    ) %>%
    dplyr::mutate(
      # Creating date/time variables
      month = lubridate::month(tpep_pickup_datetime, label = TRUE, abbr = FALSE),
      weekday = lubridate::wday(tpep_pickup_datetime, label = TRUE, abbr = FALSE),
      duration = as.numeric(difftime(
        tpep_dropoff_datetime,
        tpep_pickup_datetime,
        units = "mins"
      )),
      pickup_hour = lubridate::hour(tpep_pickup_datetime),
      
      pickup_time_of_day = factor(
        dplyr::case_when(
          pickup_hour >= 5  & pickup_hour < 12 ~ "Morning",
          pickup_hour >= 12 & pickup_hour < 17 ~ "Afternoon",
          pickup_hour >= 17 & pickup_hour < 21 ~ "Evening",
          TRUE ~ "Night"
        ),
        levels = c("Morning", "Afternoon", "Evening", "Night")
      ),
      
      # Vendor labels
      Vendor_name = dplyr::case_when(
        VendorID == 1 ~ "Creative Mobile Technologies",
        VendorID == 2 ~ "Curb Mobility",
        VendorID == 6 ~ "Myle Technologies",
        VendorID == 7 ~ "Helix",
        TRUE ~ "Unknown"
      ),
      
      # Rate code labels
      Ratecode_name = dplyr::case_when(
        RatecodeID == 1 ~ "Standard",
        RatecodeID == 2 ~ "JFK",
        RatecodeID == 3 ~ "Newark",
        RatecodeID == 4 ~ "Nassau/Westchester",
        RatecodeID == 5 ~ "Negotiated",
        RatecodeID == 6 ~ "Group ride",
        RatecodeID == 99 ~ "Unknown",
        TRUE ~ "Unknown"
      ),
      
      # Payment type labels
      payment_type_name = dplyr::case_when(
        payment_type == 0 ~ "Flex fare",
        payment_type == 1 ~ "Credit card",
        payment_type == 2 ~ "Cash",
        payment_type == 3 ~ "No charge",
        payment_type == 4 ~ "Dispute",
        payment_type == 5 ~ "Unknown",
        payment_type == 6 ~ "Voided trip",
        TRUE ~ "Unknown"
      ),
      
      tip_pct = tip_amount / fare_amount
    ) %>%
    dplyr::filter(
      tip_amount > 0,            # only where tips occurred
      tip_pct < 1,               # only reasonable tips < 100% of fare
      trip_distance > 0,         # only trips with distances
      payment_type %in% c(0, 1), # only payment types flex fare and credit card
      duration > 0,              # only where drive time is positive
      fare_amount > 0,           # only relevant charged trips
      trip_distance < 100        # trips over 100 miles assumed input error
    )
  
  # Sampled data to run aggregate analysis
  set.seed(seed)
  taxi_sample <- taxi %>%
    dplyr::group_by(Borough_pickup) %>%
    dplyr::slice_sample(n = sample_n) %>%
    dplyr::ungroup()
  
  list(
    taxi = taxi,
    taxi_sample = taxi_sample
  )
}