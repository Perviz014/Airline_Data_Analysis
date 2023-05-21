library(tidyverse)
library(dplyr)
library(data.table)
library(lubridate)
library(skimr)
library(timetk)
library(highcharter)
library(h2o)
library(tidymodels)
library(modeltime)
library(readr)


 df <- read.csv('daily-minimum-temperatures-in-me (1).csv')

 
 df %>% glimpse()
 
 colnames(df) <- c('Date','Temp')
 
 df$Date<-df$Date %>% as.Date('%m/%d/%Y')
 
 
 df$Temp <- parse_number(df$Temp)

 df$Temp <- df$Temp %>% as.numeric()
 
 # Replace "/" with "-"
 df$Date <- gsub("/", "-", df$Date)
 
 
 df$Date <- df$Date %>% as.Date() 
 
 
 # timetk package ----
 df %>% 
   plot_time_series(
     Date, Temp, 
     # .color_var = lubridate::year(Date),
     # .color_lab = "Year",
     .interactive = T,
     .plotly_slider = T)
 
 
 # Seasonality plots
 df %>%
   plot_seasonal_diagnostics(
     Date, Temp, .interactive = T)
 
 # Autocorrelation and Partial Autocorrelation
 df %>%
   plot_acf_diagnostics(
     Date, Temp, .lags = "1 year", .interactive = T)
 
 
 all_time_arg <- df %>% tk_augment_timeseries_signature()
 
 all_time_arg %>% skim()
 
 df <- all_time_arg %>%
   select(-contains("hour"),
          -contains("day"),
          -contains("week"),
          -minute,-second,-am.pm) %>% 
   mutate_if(is.ordered, as.character) %>% 
   mutate_if(is.character,as_factor)
 
 
 # ------------------------------------ H2O ------------------------------------
 h2o.init()    
 
 train_h2o <- df %>% filter(year < 1988) %>% as.h2o()
 test_h2o <- df %>% filter(year >= 1988) %>% as.h2o()
 
 y <- "Temp" 
 x <- df %>% select(-Temp) %>% names()
 
 model_h2o <- h2o.automl(
   x = x, y = y, 
   training_frame = train_h2o, 
   validation_frame = test_h2o,
   leaderboard_frame = test_h2o,
   stopping_metric = "RMSE",
   seed = 123, nfolds = 10,
   exclude_algos =c("DRF", "GBM","GLM",'XGBoost'), 
   max_runtime_secs = 100) 
 
 model_h2o@leaderboard %>% as.data.frame() 
 h2o_leader <- model_h2o@leader
 
 pred_h2o <- h2o_leader %>% h2o.predict(test_h2o) 
 
 h2o_leader %>% 
   h2o.rmse(train = T,
            valid = T,
            xval = T)
 
 error_tbl <- df %>% 
   filter(lubridate::year(Date) >= 1988) %>% 
   add_column(pred = pred_h2o %>% as_tibble() %>% pull(predict)) %>%
   rename(actual = Temp) %>% 
   select(Date,actual,pred)
 
 highchart() %>% 
   hc_xAxis(categories = error_tbl$Date) %>% 
   hc_add_series(data=error_tbl$actual, type='line', color='red', name='Actual') %>% 
   hc_add_series(data=error_tbl$pred, type='line', color='green', name='Predicted') %>% 
   hc_title(text='Predict')
 
 # New data (next 2 years) ----
 new_data <- seq(as.Date("1991/01/01"), as.Date("1992/12/01"), "months") %>%
   as_tibble() %>% 
   add_column(Temp=0) %>% 
   rename(Date=value) %>% 
   tk_augment_timeseries_signature() %>%
   select(-contains("hour"),
          -contains("day"),
          -contains("week"),
          -minute,-second,-am.pm) %>% 
   mutate_if(is.ordered, as.character) %>% 
   mutate_if(is.character,as_factor)
 
 
 # Forcaste ----
 new_h2o <- new_data %>% as.h2o()
 
 new_predictions <- h2o_leader %>% 
   h2o.predict(new_h2o) %>% 
   as_tibble() %>%
   add_column(Date=new_data$Date) %>% 
   select(Date,predict) %>% 
   rename(Temp=predict)
 
 df %>% 
   bind_rows(new_predictions) %>% 
   mutate(colors=c(rep('Actual',168),rep('Predicted',24))) %>% 
   hchart("line", hcaes(Date, Count, group = colors)) %>% 
   hc_title(text='Forecast') %>% 
   hc_colors(colors = c('red','green'))
 
 
 ############################
 
 train <- df %>% filter(Date < "1988-01-01")
 test <- df %>% filter(Date >= "1988-01-01")
 
 # 1.Auto ARIMA
 model_fit_arima <- arima_reg() %>%
   set_engine("auto_arima") %>%
   fit(Temp ~ Date, train)
 
 
 # 2. Prophet
 model_fit_prophet <- prophet_reg(seasonality_yearly = TRUE) %>%
   set_engine("prophet") %>%
   fit(Temp ~ Date, train)
 
 
 # Preprocessing Recipe
 recipe_spec <- recipe(Temp ~ Date, train) %>%
   step_timeseries_signature(Date) %>%
   step_fourier(Date, period = 365, K = 2) %>%
   step_dummy(all_nominal())
 
 recipe_spec %>% prep() %>% juice() %>% View()
 
 
 # Model 3: ets ----
 model_fit_ets <- exp_smoothing() %>%
   set_engine(engine = "ets") %>%
   fit(Temp ~ Date, train)
 
 
 
 
 
 
 
 
 # calibration
 calibration <- modeltime_table(
   model_fit_arima,
   model_fit_prophet,
   model_fit_ets
 ) %>%
   modeltime_calibrate(test)
 
 
 # Forecast ----
 calibration %>% 
   #filter(.model_id == 3) %>% 
   modeltime_forecast(actual_data = df) %>%
   plot_modeltime_forecast(.interactive = T,
                           .plotly_slider = T)
 
 
 # Accuracy ----
 calibration %>% modeltime_accuracy() %>% 
   table_modeltime_accuracy(.interactive = F)
 
 
 # Forecast Forward ----
 calibration %>%
   filter(.model_id %in% 1) %>% # best model
   modeltime_refit(df) %>%
   modeltime_forecast(h = "2 year", 
                      actual_data = df) %>%
   select(-contains("conf")) %>% 
   plot_modeltime_forecast(.interactive = T,
                           .plotly_slider = T,
                           .legend_show = F)
 
 