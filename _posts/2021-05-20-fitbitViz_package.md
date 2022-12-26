---
layout: post
title: Fitbit Visualizations using the 'fitbitViz' R package
tags: [R, package, R-bloggers]
comments: true
---


This blog post explains the functionality of the [fitbitViz](https://github.com/mlampros/fitbitViz) R package. If you own any of the **Fitbit activity trackers** you can take advantage of this package to visualize your data using *‘ggplot2’*, *‘Leaflet’* and *3-dimensionsal ‘Rayshader’ Maps*. The 3-dimensional Rayshader Map requires the installation of the [CopernicusDEM](https://github.com/mlampros/CopernicusDEM) R package which includes the **30- and 90-meter elevation data**.
You can read more about the **Fitbit** [Web API](https://dev.fitbit.com/build/reference/web-api/) and how to create an application to receive a **token** and the [user-id](https://community.fitbit.com/t5/Web-API-Development/Where-do-I-find-my-User-ID/m-p/1138667/highlight/true#M4375) in the [README.md](https://github.com/mlampros/fitbitViz/blob/master/README.md) file of the package. In the **README.md** file you will find information on how to,

* [Register an Application and receive the token](https://github.com/mlampros/fitbitViz#requirements)
* [Keep track of your activities using ‘fitbitViz’, ‘Blogdown’ and ‘Github Actions’](https://github.com/mlampros/fitbitViz#keep-track-of-your-activities-using-fitbitviz-blogdown-and-github-actions) (for instance you can see [my personal setup](https://mlampros.github.io/fitbitVizBlog/) which runs on a weekly basis)

<br>

For the rest of this blog post I'll assume that the following variables are defined and that you intend to use the functions of the **fitbitViz** package from the R Console (or RStudio IDE):

<br>

```R

require(fitbitViz)

#..................
# parameter setting
#..................

USER_ID = 'My user-id'             # Specify here your 'user-id'
token = "My token"                 # Specify here your 'token'


```

<br>

**Be aware** that the **token expires after 8 hours**. If you receive a **401 HTTP error** it means that you have to **refresh your token**. You can do that using the ***refresh_token_app()*** function which requires the **client id**, **client secret** and **refresh token** of your registered Fitbit application in the following way (you can find more information on how to receive these three parameters in the [README.md](https://github.com/mlampros/fitbitViz#requirements) file):

<br>

```R

#..............................................
# Refresh token once it expires (after 8 hours)
#..............................................

client_id = 'xxxxxx'
client_secret = 'xxxxxxxxxxxxxxxxxx'
refresh_token = 'xxxxxxxxxxxxxxxxxxxxxxxx'

# refresh the token
new_token = refresh_token_app(client_id = client_id,
                              client_secret = client_secret,
                              refresh_token = refresh_token)

# a named list that includes the new 'access_token' and 'refresh_token'
str(new_token)

```

<br>

We can now continue defining the remaining variables,

<br>

```R

WEEK = 11                         # for this use case pick the 11th week of the year 2021

num_character_error = 135         # print that many character in case of an error

weeks_2021 = fitbitViz:::split_year_in_weeks(year = 2021)         # split a year in weeks

# Start the week at monday (see: https://github.com/tidyverse/lubridate/issues/509)
date_start = lubridate::floor_date(lubridate::ymd(weeks_2021[WEEK]), unit = 'weeks') + 1  

# Add 6 days to the 'date_start' variable to come to a 7-days plot
date_end = date_start + 6

sleep_time_begins = "00H 40M 0S"
sleep_time_ends = "08H 00M 0S"

VERBOSE = FALSE                       # disable verbosity

```

<br>

The previous code snippet uses one week of my personal *Fitbit* data (the *11th week of 2021*) to plot my 

* **heart rate time series**
* **heart rate heatmap**
* **heart rate variability during sleep time**
* **sleep time series**
* **GPS data of outdoor activities**
* **3-dimensional map of activities**

The pre-processed data of all these functions are also available to download by clicking on the **CSV** buttons,

<br>

![Alt text](/images/fitbitviz_images/csv_button.png)

<br>

### heart rate time series

<br>

The **heart_rate_time_series()** function takes the **user-id**, **token**, the **start-** and **end-dates**, the **start-** and **end-time**, the **detail level** (1 minute) and returns the **heart rate time series**. Each output plot (of the *multiplot*) includes in the **x-axis** the **time** and in the **y-axis** the **heart rate value**. The highest heart rate value (peak) of the day is highlighted using a vertical and horizontal **blue** line,

<br>

```R

#.......................
# heart rate time series
#.......................

heart_dat = fitbitViz::heart_rate_time_series(user_id = USER_ID,
                                              token = token,
                                              date_start = as.character(date_start),
                                              date_end = as.character(date_end),
                                              time_start = '00:00',
                                              time_end = '23:59',
                                              detail_level = '1min',
                                              ggplot_intraday = TRUE,
                                              ggplot_ncol = 2,
                                              ggplot_nrow = 4,
                                              verbose = VERBOSE,
                                              show_nchar_case_error = num_character_error)
heart_dat$plt

```

<br>

![Alt text](/images/fitbitviz_images/heart_rate_time_series.png)

<br>

### heart rate heatmap

<br>

The **heart rate heatmap** shows the **min**, **median** and **max** heart rate Levels in the **y-axis** for each day of the specified week (**x-axis**). As the legend shows, the displayed values range from 40 to 220 and higher values appear in *purple* or *orange* color,

<br>


```R

#............................
# heart rate intraday heatmap [ plot options: https://yihui.org/knitr/options/#plots ]
#............................

heart_intra = heart_dat$heart_rate_intraday

hrt_heat = fitbitViz::heart_rate_heatmap(heart_rate_intraday_data = heart_intra, 
                                         angle_x_axis = 0)
hrt_heat

```

<br>

![Alt text](/images/fitbitviz_images/heatmap.png)

<br>

### heart rate variability during sleep time

<br>

Heart Rate Variability (HRV) intraday data for a single date. HRV data applies specifically to a user's "main sleep", which is the longest single period of time asleep on a given date. It measures the HRV rate at various times and returns the *Root Mean Square of Successive Differences (rmssd)*, *Low Frequency (LF)*, *High Frequency (HF)*, and *Coverage* data for a given measurement. **Rmssd** measures short-term variability in your heart rate while asleep. **LF** and **HF** capture the power in interbeat interval fluctuations within either high frequency or low frequency bands. Finally, **coverage** refers to data completeness in terms of the number of interbeat intervals. The **fitbit_data_type_by_date()** function allows the user to also compute the 'spo2' (Blood Oxygen Saturation), 'br' (Breathing Rate), 'temp' (Temperature) and 'cardioscore' (Cardio Fitness Score or VO2 Max) by adjusting the **type** parameter.

<br>

```R

#.......................
# heart rate variability
#.......................

hrt_rt_var = fitbitViz::fitbit_data_type_by_date(user_id = USER_ID,
                                                 token = token,
                                                 date = as.character(date_start),
                                                 type = 'hrv',
                                                 plot = TRUE,
                                                 show_nchar_case_error = num_character_error)
hrt_rt_var

```

<br>

![Alt text](/images/fitbitviz_images/heart_rate_variability.png)

<br>

### sleep time series

<br>

The **sleep time series** visualization is similar to the *Fitbit Mobile* Visualization and in the **x-axis** shows the specified by the user **sleep time interval** whereas in the **y-axis** shows the **sleep Levels** (*wake*, *rem*, *light*, *deep*). Lower levels like *deep sleep* appear in dark blue whereas higher levels like *wake* appear in light blue,

<br>


```R

#.......................
# sleep data time series
#.......................

sleep_ts = fitbitViz::sleep_time_series(user_id = USER_ID,
                                        token = token,
                                        date_start = as.character(date_start),
                                        date_end = as.character(date_end),
                                        ggplot_color_palette = 'ggsci::blue_material',
                                        ggplot_ncol = 2,
                                        ggplot_nrow = 4,
                                        show_nchar_case_error = num_character_error,
                                        verbose = VERBOSE)

sleep_ts$plt_lev_segments

```

<br>

![Alt text](/images/fitbitviz_images/sleep_time_levels.png)

<br>

### GPS data of outdoor activities

<br>

To make use of the *GPS data* from the Fitbit Application we have first to extract the **log-id** for a time interval after a specified *Date*,

<br>


```R

#...................
# extract the log-id (required for the GPS data)
#...................

log_id = fitbitViz::extract_LOG_ID(user_id = USER_ID,
                                   token = token,
                                   after_Date = as.character(date_start),
                                   limit = 10,
                                   sort = 'asc',
                                   verbose = VERBOSE)
# log_id

```

<br>

Once we have the *log-id* we can define the *time zone* of the route to receive all GPS data,

<br>


```R

#....................................................
# return the gps-ctx data.table for the output log-id
#....................................................

res_tcx = fitbitViz::GPS_TCX_data(log_id = log_id,
                                  user_id = USER_ID,
                                  token = token,
                                  time_zone = 'Europe/Athens',
                                  verbose = VERBOSE)
# res_tcx

```

<br>

The following *Leaflet (Point Coordinates) Map* shows my outdoor activity during the *11th week of 2021* (the legend shows the elevation of the route),

<br>


```R

#................................
# Create the Leaflet / LeafGL Map
#................................

res_lft = fitbitViz::leafGL_point_coords(dat_gps_tcx = res_tcx,
                                         color_points_column = 'AltitudeMeters',
                                         provider = leaflet::providers$Esri.WorldImagery,
                                         option_viewer = rstudioapi::viewer,
                                         CRS = 4326)
                                         
res_lft

```

<br>

![Alt text](/images/fitbitviz_images/leafgl_map.png)

<br>

### 3-dimensional plots of activities

<br>

Another option of this package is to plot a route in 3-dimensional space. For this purpose we'll use the [rayshader](https://github.com/tylermorganwall/rayshader) package, which internally uses [rgl](https://github.com/dmurdoch/rgl) (*OpenGL*). First, we have to extend the boundaries of our route for approximately *1.000 thousand meters* (adjust this value depending on your area of interest),

<br>


```R

#...................................................
# compute the sf-object buffer and the raster-extend  (1000 meters buffer)
#...................................................

sf_rst_ext = fitbitViz::extend_AOI_buffer(dat_gps_tcx = res_tcx,
                                          buffer_in_meters = 1000,
                                          CRS = 4326,
                                          verbose = VERBOSE)
# sf_rst_ext

```

<br>

Then for the extended area we will download [Copernicus Digital Elevation Model (DEM)](https://github.com/mlampros/CopernicusDEM) data. The *Copernicus elevation data* come either in **30** or in **90** meter resolution. We will pick the *30* meter resolution product for this route. The **CopernicusDEM** is an R package, make sure that you have installed and configured the **awscli** Operating System Requirement if you intend to download and reproduce the next 3-dimensional map using the elevation data (you can find instructions in the [README.md](https://github.com/mlampros/CopernicusDEM/blob/master/README.md) file for all 3 Operating Systems),

<br>


```R

#..................................................................
# Download the Copernicus DEM 30m elevation data
# there is also the option to download the DEM 90m elevation data
# which is of lower resolution but the image size is smaller which
# means faster download
#..................................................................

dem_dir = tempdir()
# dem_dir

dem30 = CopernicusDEM::aoi_geom_save_tif_matches(sf_or_file = sf_rst_ext$sfc_obj,
                                                 dir_save_tifs = dem_dir,
                                                 resolution = 30,
                                                 crs_value = 4326,
                                                 threads = parallel::detectCores(),
                                                 verbose = VERBOSE)

TIF = list.files(dem_dir, pattern = '.tif', full.names = T)
# TIF

if (length(TIF) > 1) {

  #....................................................
  # create a .VRT file if I have more than 1 .tif files
  #....................................................

  file_out = file.path(dem_dir, 'VRT_mosaic_FILE.vrt')

  vrt_dem30 = CopernicusDEM::create_VRT_from_dir(dir_tifs = dem_dir,
                                                 output_path_VRT = file_out,
                                                 verbose = VERBOSE)
}

if (length(TIF) == 1) {

  #..................................................
  # if I have a single .tif file keep the first index
  #..................................................

  file_out = TIF[1]
}

#.......................................
# crop the elevation DEM based on the
# coordinates extent of the GPS-CTX data
#.......................................

raysh_rst = fitbitViz::crop_DEM(tif_or_vrt_dem_file = file_out,
                                sf_buffer_obj = sf_rst_ext$sfc_obj,
                                verbose = VERBOSE)
# sp::plot(raysh_rst)

```

<br>

The GPS route that I use is an *ascending & descending* route therefore we can convert the GPS (TCX) data to a spatial *LINESTRING* by using the maximum altitude as a *split point* of the route to visualize the ascending route in *blue* and the descending in *red* (there is also the alternative to specify the split point based on time using the **time_split_asc_desc** parameter),
 
<br>


```R

linestring_dat = fitbitViz::gps_lat_lon_to_LINESTRING(dat_gps_tcx = res_tcx,
                                                      CRS = 4326,
                                                      time_split_asc_desc = NULL,
                                                      verbose = VERBOSE)

```

<br>

then we create the *'elevation_sample_points' data.table parameter* for the *3-dim* plot based on the *min.*, *middle*  and *max.* altitude of the previously computed *'res_tcx'* data,

<br>

```R

idx_3m = c(which.min(res_tcx$AltitudeMeters), 
           as.integer(length(res_tcx$AltitudeMeters) / 2), 
           which.max(res_tcx$AltitudeMeters))

cols_3m = c('latitude', 'longitude', 'AltitudeMeters')
dat_3m = res_tcx[idx_3m, ..cols_3m]

```

<br>

and finally we visualize the *3-dimensional Rayshader Map*,

<br>

```R

#.....................................................
# Conversion of the 'SpatRaster' to a raster object
# because the 'rayshader' package accepts only rasters
#.....................................................

rst_obj = raster::raster(raysh_rst)
raster::projection(rst_obj) <- terra::crs(raysh_rst, proj = TRUE)


snapshot_rayshader_path = file.path(tempdir(), 'rayshader_img.png')

rgl::open3d(useNULL = TRUE)                       # this removes the second rgl-popup-window

fitbitViz::rayshader_3d_DEM(rst_buf = rst_obj,
                            rst_ext = sf_rst_ext$raster_obj_extent,
                            linestring_ASC_DESC = linestring_dat,
                            elevation_sample_points = dat_3m,
                            zoom = 0.3,
                            windowsize = c(1000, 800))

rgl::rgl.snapshot(snapshot_rayshader_path)
rgl::par3d(mouseMode = "trackball")   # options: c("trackball", "polar", "zoom", "selecting")
rgl::rglwidget()

```

<br>

![Alt text](/images/fitbitviz_images/rayshader_map.png)

<br>

In the output map we observe

* the *3 specified elevation vertical lines* (including their *altitude values* in meters) 
* in *blue* color the *ascending* route
* in *red* color the *descending* route

<br>

The attached map here is a screenshot. You can play with the map by viewing the route from a different angle in one of [my personal Fitviz-blog posts](https://mlampros.github.io/fitbitVizBlog/1/01/01/2021-04-26-to-2021-05-02-fitbit-visualizations/#anchor_3d_rayshader)

<br>

An updated version of the **fitbitViz** package can be found in my [Github repository](https://github.com/mlampros/fitbitViz) and to report bugs/issues please use the following link, [https://github.com/mlampros/fitbitViz/issues](https://github.com/mlampros/fitbitViz/issues).

<br>

### **Citation:**

If you use the **fitbitViz** R package in your paper or research please cite [https://cran.r-project.org/web/packages/fitbitViz/citation.html](https://cran.r-project.org/web/packages/fitbitViz/citation.html):

<br>

```R
@Manual{,
  title = {fitbitViz: Fitbit Visualizations},
  author = {Lampros Mouselimis},
  year = {2022},
  note = {R package version 1.0.3},
  url = {https://CRAN.R-project.org/package=fitbitViz},
}
```

<br>
