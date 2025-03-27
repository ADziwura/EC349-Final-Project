# DATA CLEANING AND MERGING  ----------------------------------

library(dplyr)
library(readr)
library(tensorflow)
library(caTools)
library(stringr)
library(lubridate)
library(ipred)

# Loading the initial dataset
df <- read.csv("listings.csv", stringsAsFactors = FALSE)

# For computational efficiency I first remove 32368 observations of NA values for 'prices' and convert to numericals
df$price[df$price == ""] <- NA
df <- df[!is.na(df$price), ]
na_count <- sum(is.na(df$price))
cat("Number of NA values in 'price':", na_count, "\n")
str(df$price)
df$price <- as.numeric(gsub("[\\$,]", "", df$price))

# Converting binary variables
binary_columns <- c("host_is_superhost", "host_has_profile_pic", "host_identity_verified", "has_availability", "instant_bookable")
df[binary_columns] <- lapply(df[binary_columns], function(x) ifelse(x == "t", 1, 0))

# Hosts Processing  ----------------------------------

# Converting "host_response_time" to numeric
response_time_mapping <- c(
  "N/A" = 0,
  "within an hour" = 1,
  "within a few hours" = 2,
  "within a day" = 3,
  "a few days or more" = 4
)

df$host_response_time <- as.numeric(response_time_mapping[df$host_response_time])
df$host_response_rate[is.na(df$host_response_rate)] <- 0 #cover for NA values which represent 0
df$host_acceptance_rate[is.na(df$host_acceptance_rate)] <- 0

# Converting percentage based variables
df <- df %>%
  mutate(
    host_response_rate = as.numeric(gsub("%", "", host_response_rate)),
    host_acceptance_rate = as.numeric(gsub("%", "", host_acceptance_rate))
  )

# Convert "host_verifications" to numericals
df$host_verifications[df$host_verifications %in% c("[]", "None")] <- 0
df$host_verifications <- as.character(df$host_verifications)
verification_types <- c("phone", "email", "work_email")
check_verification <- function(verifications_str, verification) {
  grepl(paste0("\\b", verification, "\\b"), verifications_str, ignore.case = TRUE)
}

# Count number of verification
df$host_verifications <- rowSums(sapply(verification_types, function(verification) {
  sapply(df$host_verifications, function(x) as.integer(check_verification(x, verification)))
}))

# Property Types  ----------------------------------

# Converting neighbourhoods based on Airbnb classifications in neighbourhoods.csv
neighbourhoods <- c(
  "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
  "Camden", "City of London", "Croydon", "Ealing", "Enfield",
  "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey",
  "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington",
  "Kensington and Chelsea", "Kingston upon Thames", "Lambeth",
  "Lewisham", "Merton", "Newham", "Redbridge", "Richmond upon Thames",
  "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest",
  "Wandsworth", "Westminster"
)

df$neighbourhood_cleansed <- as.numeric(factor(df$neighbourhood_cleansed, levels = neighbourhoods))

# Categorizing property types
property_mapping <- list(
  "Apartment/Condo" = c("Entire rental unit", "Entire condo", "Room in aparthotel", 
                        "Private room in condo", "Entire serviced apartment", "Room in serviced apartment", 
                        "Shared room in condo", "Entire loft", "Private room in loft"),
  
  "House/Townhouse" = c("Entire home", "Entire townhouse", "Private room in townhouse", 
                        "Private room in home", "Private room in rental unit", 
                        "Private room in guest suite", "Shared room in townhouse"),
  
  "Guesthouse/B&B" = c("Private room in guesthouse", "Entire guesthouse", "Entire guest suite",
                       "Room in bed and breakfast", "Private room in bed and breakfast", "Shared room in guesthouse"),
  
  "Villa" = c("Entire villa", "Private room in villa", "Shared room in villa"),
  
  "Cabin/Cottage" = c("Entire cabin", "Private room in cabin", "Entire cottage", 
                      "Private room in cottage", "Private room in tiny home", "Tiny home"),
  
  "Unique Stay" = c("Houseboat", "Private room in houseboat", "Boat", "Private room in boat", 
                    "Shipping container", "Private room in shipping container", "Dome", "Hut", 
                    "Private room in hut", "Private room in shepherd's hut", "Shepherd’s hut", 
                    "Private room in treehouse", "Private room in lighthouse", "Lighthouse", 
                    "Earthen home", "Private room in earthen home"),
  
  "Farm Stay/Rural" = c("Farm stay", "Private room in farm stay", "Shared room in farm stay",
                        "Barn", "Castle", "Religious building", "Private room in religious building", 
                        "Tower", "Campsite", "Tent", "Private room in nature lodge"),
  
  "Hotel/Hostel" = c("Room in hotel", "Room in boutique hotel", "Shared room in boutique hotel", 
                     "Room in hostel", "Private room in hostel", "Shared room in hostel", 
                     "Shared room in hotel"),
  
  "Shared/Communal Stay" = c("Shared room", "Shared room in rental unit", "Shared room in serviced apartment",
                             "Shared room in vacation home", "Shared room in bed and breakfast",
                             "Shared room in guest suite", "Shared room in bungalow", "Shared room in bus"),
  
  "Traditional Homes" = c("Cycladic home", "Casa particular", "Minsu", "Private room in casa particular", 
                          "Riad", "Island", "Private room in island")
)

property_category_map <- unlist(lapply(names(property_mapping), function(category) {
  setNames(rep(category, length(property_mapping[[category]])), property_mapping[[category]])
}))
df$property_category <- property_category_map[df$property_type]
df$property_category[is.na(df$property_category)] <- "Other"
df$property_category_numeric <- as.numeric(factor(df$property_category, levels = names(property_mapping)))

# Potentially remove to safe space
# rm(property_mapping)

# Converting room types amnd reviews_per_month to numerics 
room_type_mapping <- c(
  "Entire home/apt" = 1,
  "Private room" = 2,
  "Hotel room" = 3,
  "Shared room" = 4
)

df$room_type <- as.numeric(room_type_mapping[df$room_type])
df$reviews_per_month[is.na(df$reviews_per_month)] <- 0

# Converting types of bathroom if shared or private
df <- df %>%
  mutate(shared_bathroom = ifelse(grepl("shared", tolower(bathrooms_text)), 1, 0))

# PCA analysis ----------

# Availability:
library(ggplot2)
availability_data <- df[, c("availability_30", "availability_60", "availability_90", "availability_365")]
availability_scaled <- scale(availability_data) #scale
pca_result <- prcomp(availability_scaled, center = TRUE, scale. = TRUE) # running PCA
summary(pca_result)

# Loadings (rotation matrix) Nice exercises to printing sentences in R:
print(pca_result$rotation)
pc1_loadings <- pca_result$rotation[, 1]
print(pc1_loadings)
loadings <- pca_result$rotation
for (i in 1:4) {
  pc_loadings <- loadings[, i]
  important_var <- names(which.max(abs(pc_loadings)))
  direction <- ifelse(pc_loadings[which.max(abs(pc_loadings))] > 0, "positive", "negative")
  
  print(paste("PC", i, "is most strongly associated with:", important_var, 
              "(", direction, "relationship )"))}
  
# Identyfying PC1
important_var <- names(which.max(abs(pc1_loadings)))
print(paste("Variable with highest contribution to PC1:", important_var))

# Elbow Graph
pca_var <- pca_result$sdev^2
pca_var_exp <- pca_var / sum(pca_var) * 100  # Converting to percentage
elbow_plot <- data.frame(PC = seq_along(pca_var_exp), Variance_Explained = pca_var_exp)

ggplot(elbow_plot, aes(x = PC, y = Variance_Explained)) +
  geom_point(size = 3) + 
  geom_line() +
  ggtitle("PCA Elbow Graph for 'has_availibility' variables ") +
  xlab("Principal Component") +
  ylab("Variance Explained (%)") +
  theme_minimal()

# Specific percenteges:
variance <- pca_result$sdev^2
prop_variance <- variance / sum(variance) * 100

print("Variance by each principal component:")
for (i in 1:4) {
  print(paste("PC", i, "explains", round(prop_variance[i], 2), "% of total variance"))
}

# NAs check ------------------------
# Some of the next steps are compute heavy or require external data manipulations, therefore
# for efficient computing, I remove all missing variables 

# Counting final NA values per variable
count_na <- function(df) {
  na_counts <- colSums(is.na(df))  # Count NAs in each column
  na_percent <- (na_counts / nrow(df)) * 100  # Percentages
  
  # Separete data frame for NA counts
  na_summary <- data.frame(
    Variable = names(na_counts),
    NA_Count = na_counts,
    NA_Percentage = na_percent
  )
  
  # Filter and sorting
  na_summary <- na_summary %>%
    filter(NA_Count > 0) %>%
    arrange(desc(NA_Count))
  return(na_summary)
}
na_results <- count_na(df)
print(na_results)

# There are 14528 missing reviews_ratings of listings which are dropped.
df <- df[!is.na(df$review_scores_checkin), ]

# Only after removing NA values for reviews_rating I can perform the second PCA:
# Ratings PCA:
library(ggplot2)

review_data <- df[, c("review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
                      "review_scores_communication", "review_scores_location", "review_scores_value")]

review_scaled <- scale(review_data) #scale
pca_result2 <- prcomp(review_scaled, center = TRUE, scale. = TRUE) # running PCA
summary(pca_result2)

# Loadings (rotation matrix) Nice exercises to printing sentences in R:
print(pca_result2$rotation)
pc1_loadings <- pca_result2$rotation[, 1]
print(pc1_loadings)
loadings <- pca_result2$rotation
for (i in 1:7) {
  pc_loadings <- loadings[, i]
  important_var <- names(which.max(abs(pc_loadings)))
  direction <- ifelse(pc_loadings[which.max(abs(pc_loadings))] > 0, "positive", "negative")
  
  print(paste("PC", i, "is most strongly associated with:", important_var, 
              "(", direction, "relationship )"))}

# Identyfying PC1 for reviews_rating
important_var <- names(which.max(abs(pc1_loadings)))
print(paste("Variable with highest contribution to PC1:", important_var))

# Elbow Graph
pca_var <- pca_result2$sdev^2
pca_var_exp <- pca_var / sum(pca_var) * 100  # Converting to percentage
elbow_plot <- data.frame(PC = seq_along(pca_var_exp), Variance_Explained = pca_var_exp)

ggplot(elbow_plot, aes(x = PC, y = Variance_Explained)) +
  geom_point(size = 3) + 
  geom_line() +
  ggtitle("PCA Elbow Graph for 'review_ratings' variables ") +
  xlab("Principal Component") +
  ylab("Variance Explained (%)") +
  theme_minimal()

# Specific percenteges:
variance <- pca_result2$sdev^2
prop_variance <- variance / sum(variance) * 100

print("Variance by each principal component:")
for (i in 1:7) {
  print(paste("PC", i, "explains", round(prop_variance[i], 2), "% of total variance"))
}

# Underground Stations ----------------------------------

# install.packages('geosphere') # install if needed
library(geosphere) # is this necessary in the end?

# Loading London Stations datasets 
stations <- read.csv("London Stations.csv")

# Defining arbitrary search radius in meters
radius_meters <- 500

# Counting nearby stations for a given house
count_nearby_stations <- function(latitude, longitude, stations, radius_meters) {
  distances <- distHaversine(
    matrix(c(stations$station_long, stations$station_lat), ncol = 2),
    c(longitude, latitude)
  )
  sum(distances <= radius_meters)
}

# Computing the number of stations near each house
df$num_stations_nearby <- mapply(count_nearby_stations, 
                                     df$latitude, 
                                     df$longitude, 
                                     MoreArgs = list(stations = stations, radius_meters = radius_meters))

# Cultural Sites ----------------------------------

sites <- read.csv("CIM 2023 Museums and Nightclubs (Nov 2023).csv")
radius_meters <- 1000

# Counting nearby sites based on category
count_nearby_sites <- function(latitude, longitude, sites, category, radius_meters) {
  filtered_sites <- sites[sites$category == category, ]  # Filter by category
  
  if (nrow(filtered_sites) == 0) return(0)  # If no sites of that type exist, return 0
  
  distances <- distHaversine(
    matrix(c(filtered_sites$cim_longitude, filtered_sites$cim_latitude), ncol = 2),
    c(longitude, latitude)
  )
  sum(distances <= radius_meters)
}

# Computing the number of museums and nightclubs near each house
df$num_museums <- mapply(count_nearby_sites, 
                        df$latitude, 
                        df$longitude, 
                        MoreArgs = list(sites = sites, category = "museum", radius_meters = radius_meters))

df$num_nightclubs <- mapply(count_nearby_sites, 
                           df$latitude, 
                           df$longitude, 
                           MoreArgs = list(sites = sites, category = "nightclub", radius_meters = radius_meters))

# TEXT CLEANING ------------------------
df <- df %>%
  select(-c(
    listing_url, scrape_id, last_scraped, source, picture_url, host_location, host_about, host_listings_count,
    host_url, host_id, host_thumbnail_url, host_picture_url, # Removing host-related variables
    host_neighbourhood, calendar_last_scraped, neighbourhood, # These variables are often omitted
    neighbourhood_group_cleansed, license, calendar_updated,
    number_of_reviews_ltm, number_of_reviews_l30d,
    minimum_minimum_nights,	maximum_minimum_nights,	minimum_maximum_nights,	maximum_maximum_nights,
    availability_30, availability_90, availability_365,
    host_verifications, property_type, property_category, # not useful anymore 
    latitude, longitude,  # Note: Be cautious if you need these for spatial analysis
    first_review, last_review, host_since # already covered
    ))

# Amenities ------------------------

# Original code to identify twenty most popular amenities:
  # listings$amenities <- as.character(listings$amenities)
  # amenities_list <- strsplit(listings$amenities, ",") # Split the amenities into list
  # amenities_list <- lapply(amenities_list, function(x) trimws(gsub('[\"\\[\\]]', '', x))) # Clean names
  # amenity_counts <- table(unlist(amenities_list)) %>% sort(decreasing = TRUE) # Counting and sorting occurrences of each unique amenity
  # n <- 20 # Seting the number for most common amenities
  # top_amenities <- names(amenity_counts)[1:n]
  # amenities_df <- as.data.frame(matrix(0, nrow = nrow(listings), ncol = n))
  # colnames(amenities_df) <- top_amenities # implementing into the DF
  # for (i in seq_along(amenities_list)) {
  #  amenities_df[i, top_amenities %in% amenities_list[[i]]] <- 1
  # } # Populating the dataframe
  # df <- cbind(listings, amenities_df) # combining with df

df$amenities <- as.character(df$amenities)
amenity_categories <- c("Kitchen", "Smoke alarm", "Washer", "Iron", "Hangers", 
                        "Hot water", "Carbon monoxide alarm", "Dryer", "Heating", 
                        "Wifi", "Essentials", "Bed linens", "TV", "Refrigerator", 
                        "Dishes and silverware", "Shampoo", "Cooking basics", 
                        "Microwave", "Hot water kettle", "Oven", "Parking") 
#Safety index:
safety_amenities <- c("Fire extinguisher", "Smoke alarm", "Carbon monoxide alarm")
check_amenity <- function(amenities_str, amenity) {
  grepl(paste0("\\b", amenity, "\\b"), amenities_str, ignore.case = TRUE)
}

#Kitchen index
kitchen_amenities <- c("Kitchen", "Washer", "Hot water", "Hot water kettle", "Oven",
                       "Cooking basics", "Refrigerator",  "Dishes and silverware", "Microwave")
check_amenity <- function(amenities_str, amenity) {
  grepl(paste0("\\b", amenity, "\\b"), amenities_str, ignore.case = TRUE)
}

# Essentials index
essentials_amenities <- c("Essentials", "Heating", "Bed linens", "Iron", "Shampoo", "Dryer", "Hangers")
check_amenity <- function(amenities_str, amenity) {
  grepl(paste0("\\b", amenity, "\\b"), amenities_str, ignore.case = TRUE)
}

# Binary columns for each amenity
for (amenity in amenity_categories) {
  col_name <- paste0("has_", gsub(" ", "_", tolower(amenity)))
  df[[col_name]] <- as.integer(sapply(df$amenities, function(x) check_amenity(x, amenity)))
}

#Each index is counted for how many of the safety amenities are present
# Safety index
df$safety_index <- rowSums(sapply(safety_amenities, function(amenity) {
  sapply(df$amenities, function(x) as.integer(check_amenity(x, amenity)))
}))
 
# Kitchen index
df$kitchen_index <- rowSums(sapply(kitchen_amenities, function(amenity) {
  sapply(df$amenities, function(x) as.integer(check_amenity(x, amenity)))
}))

# Essentials index
df$essentials_index <- rowSums(sapply(essentials_amenities, function(amenity) {
  sapply(df$amenities, function(x) as.integer(check_amenity(x, amenity)))
}))

# 'Description' Analysis  ----------------------------------

# Merging "name", "description", "neighborhood_overview" into single "combined_description"
df <- df %>%
  mutate(combined_description = paste(name, description, neighborhood_overview, sep = " "))

#Additionally, for storage efficiency I remove the unnecessary df's:
unnecessary_dfs <- c("availability_scaled", "availability_data", "review_scaled", "review_data", "pca_result", "pca_result2")
rm(list = intersect(ls(), unnecessary_dfs))
gc()

df <- df %>%
  select(-description, -neighborhood_overview, -name, -amenities, 
         -has_smoke_alarm, -has_carbon_monoxide_alarm, -bathrooms_text,
         -has_kitchen, -has_washer, -has_hot_water, -has_hot_water_kettle, -has_oven,
         -has_cooking_basics, -has_refrigerator, -has_dishes_and_silverware, -has_microwave, 
         -has_essentials, -has_bed_linens, -has_iron, -has_shampoo,
         -has_dryer, -has_hangers, -has_heating, -review_scores_accuracy, -review_scores_cleanliness, -review_scores_checkin,
         -review_scores_communication, -review_scores_location, -review_scores_value) #removing after PCA

# rewrite this properly using:
# df <- df %>% select(-c())
# Excluding the rest of missing cells (1000) 
df <- na.omit(df)

# The final df is saved into a second file, which is used for e.g. LLAMA text analysis
write.csv(df, "processed_data.csv", row.names = FALSE)

# I proceed to deploy text analysis using LLaMA model for host_gender and combined_description information extraction
# I run LLaMA locally in Python,
# I then load the final dataset into the analysis:
