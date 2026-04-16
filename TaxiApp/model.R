library(dplyr)
library(tidyr)
library(arrow)

load_data <- function(files, sample_frac_val = 0.004, seed = 123) {
  
  set.seed(seed)
  
  bind_rows(
    lapply(seq_along(files), function(i) {
      f <- files[i]
      
      month_val <- i
      
      read_parquet(f) %>%
        sample_frac(sample_frac_val, replace = FALSE) %>%
        mutate(month = month_val)
    })
  )
}

clean_data <- function(taxi) {
  
  taxi <- taxi %>%
    mutate(
      Vendor_name = case_when(
        VendorID == 1 ~ "Creative Mobile Technologies",
        VendorID == 2 ~ "Curb Mobility",
        VendorID == 6 ~ "Myle Technologies",
        VendorID == 7 ~ "Helix",
        TRUE ~ "Unknown"
      ),
      payment_type_name = case_when(
        payment_type == 0 ~ "Flex fare",
        payment_type == 1 ~ "Credit card",
        payment_type == 2 ~ "Cash",
        payment_type == 3 ~ "No charge",
        payment_type == 4 ~ "Dispute",
        payment_type == 5 ~ "Unknown",
        payment_type == 6 ~ "Voided trip",
        TRUE ~ "Unknown"
      ),
      Ratecode_name = case_when(
        RatecodeID == 1 ~ "Standard",
        RatecodeID == 2 ~ "JFK",
        RatecodeID == 3 ~ "Newark",
        RatecodeID == 4 ~ "Nassau/Westchester",
        RatecodeID == 5 ~ "Negotiated",
        RatecodeID == 6 ~ "Group ride",
        RatecodeID == 99 ~ "Unknown",
        TRUE ~ "Unknown"
      )
    ) %>%
    mutate(
      passenger_count_missing = is.na(passenger_count),
      passenger_count = ifelse(
        is.na(passenger_count) & Vendor_name != "Myle Technologies",
        1,
        passenger_count
      ),
      congestion_surcharge = replace_na(congestion_surcharge, 0),
      Airport_fee = replace_na(Airport_fee, 0)
    ) %>%
    filter(fare_amount > 0) %>%
    mutate(
      tip_percent_fare_amount = tip_amount / fare_amount
    ) %>%
    filter(payment_type_name %in% c("Flex fare", "Credit card")) %>%
    mutate(
      trip_duration_mins = as.numeric(difftime(
        tpep_dropoff_datetime,
        tpep_pickup_datetime,
        units = "mins"
      )),
      weekday = factor(
        weekdays(tpep_pickup_datetime),
        levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
      ),
      pickup_hour = as.integer(format(tpep_pickup_datetime, "%H")),
      pickup_time_of_day = case_when(
        pickup_hour >= 5  & pickup_hour < 12 ~ "Morning",
        pickup_hour >= 12 & pickup_hour < 17 ~ "Afternoon",
        pickup_hour >= 17 & pickup_hour < 21 ~ "Evening",
        TRUE ~ "Night"
      ),
      pickup_time_of_day = factor(
        pickup_time_of_day,
        levels = c("Morning", "Afternoon", "Evening", "Night")
      )
    ) %>%
    filter(
      trip_distance > 0,
      trip_duration_mins > 0,
      fare_amount > 0,
      tip_percent_fare_amount >= 0,
      tip_percent_fare_amount <= 1
    ) 
  q_dist <- quantile(taxi$trip_distance, 0.99, na.rm = TRUE)
  q_dur  <- quantile(taxi$trip_duration_mins, 0.99, na.rm = TRUE)
  q_fare <- quantile(taxi$fare_amount, 0.99, na.rm = TRUE)
  
  taxi <- taxi %>% filter(
    trip_distance <= q_dist,
    trip_duration_mins <= q_dur,
    fare_amount <= q_fare
  )
  return(taxi)
}

add_location_features <- function(taxi, borough_zone) {
  
  # Pickup
  taxi <- taxi %>%
    left_join(
      borough_zone %>% select(LocationID, Borough, Zone),
      by = c("PULocationID" = "LocationID")
    ) %>%
    mutate(
      pickup_borough = Borough,
      pickup_location = paste0(
        replace_na(Borough, "Unknown"),
        " - ",
        replace_na(Zone, "Unknown")
      )
    ) %>%
    select(-Borough, -Zone)
  
  # Dropoff
  taxi <- taxi %>%
    left_join(
      borough_zone %>% select(LocationID, Borough, Zone),
      by = c("DOLocationID" = "LocationID")
    ) %>%
    mutate(
      dropoff_borough = Borough,
      dropoff_location = paste0(
        replace_na(Borough, "Unknown"),
        " - ",
        replace_na(Zone, "Unknown")
      )
    )%>%
    select(-Borough, -Zone) %>%
    
    # Fix NA BEFORE factor
    mutate(
      pickup_borough = replace_na(pickup_borough, "Unknown"),
      dropoff_borough = replace_na(dropoff_borough, "Unknown")
    ) %>%
    
    # Then convert to factor
    mutate(
      pickup_borough = factor(pickup_borough),
      dropoff_borough = factor(dropoff_borough),
      payment_type_name = factor(payment_type_name),
      weekday = factor(
        weekday,
        levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
      ),
      pickup_time_of_day = factor(
        pickup_time_of_day,
        levels = c("Morning", "Afternoon", "Evening", "Night")
      ),
      Ratecode_name = factor(Ratecode_name)
    )
  
  return(taxi)
}

prepare_taxi_data <- function(files, borough_zone) {
  load_data(files) %>%
    clean_data() %>%
    add_location_features(borough_zone = borough_zone)
}


train_model <- function(taxi) {
  model_data <- taxi %>%
    dplyr::select(
      tip_percent_fare_amount,
      #trip_distance,
      trip_duration_mins,
      Ratecode_name
    ) %>%
    tidyr::drop_na() %>%
    droplevels()
  
  model <- lm(
    tip_percent_fare_amount ~
      #log1p(trip_distance) +
      log1p(trip_duration_mins) +
      I(log1p(trip_duration_mins)^2) +
      Ratecode_name,
    data = model_data
  )
  
  list(
    model = model,
    levels = list(
      Ratecode_name = levels(model_data$Ratecode_name)
    )
  )
}

predict_tip_percent <- function(model_obj, new_data) {
  model <- model_obj$model
  
  new_data <- new_data %>%
    mutate(
      Ratecode_name = factor(
        Ratecode_name,
        levels = model_obj$levels$Ratecode_name
      )
    )
  
  as.numeric(predict(model, newdata = new_data))
}

predict_tip <- function(model_obj, new_data) {
  stopifnot("fare_amount" %in% names(new_data))
  
  pred_pct <- predict_tip_percent(model_obj, new_data)
  pred_amt <- pred_pct * new_data$fare_amount
  
  as.numeric(pred_amt)
}
