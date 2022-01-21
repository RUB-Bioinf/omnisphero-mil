# SIGMOIDAL EVALUATION OF CNN PREDICTIONS

library(drc) #R toolbox for curve fitting

###input##########################################################################################################################################

#for this script to work, two lists are neccessary: one containing response data and one the according doses for each response
#this means, that both lists need to have the same length and order (e.g. response at index 3 of the response list needs to
#aligh to dose at index 3 of the dose list)

#dose = list of dose values, one for each response value in resp
#resp = list of response values, one for each dose

#a boolean input indicates, if the data and resulting curve fit should be printed as png or not
#plot_curve = boolean parameter for png plot
#filename = full path for plot
###################################################################################################################################################

#initially setting output to overwerite it later (if curve fitting was successfull, else NaN will be put out)
final_Score <- NaN
plot_data <- NaN
estimate_data <- NaN
score_data <- NaN
upperScoreLimit <- 8 #if Scores are above this threshold, they will set to it; it serves as maximum score range for score normalization


# MODEL EVALUATION FUNCTION------------------------------------------------------------------------------------------------------------------------
#this function evaluates the fit to have a sigmidal shape
#for this purpose, the asymptotes, gradient and effect range of the curve are taken into consideration
#with these parameters, a score is calculated and returned

evaluate_fit <- function() {
  paras <- drm_object$fit$par #get parameters like asymptote position and slope for score evaluation
  #print(paste("Curve", i, ":"))
  #print(round(paras,2))

  relativeresiduals <- sum(abs(residuals(drm_object, typeRes = "working"))) / length(dose) #get residuals

  Gradient <- paras[1] / paras[4] #get curve gradient


  if (Gradient < 0) { #in case slope is positive, we make it bigger for a much worse score
    Gradient <- Gradient * 5
  }

  #lower Asymptote
  lowAsympScore <- round((0 + paras[2]), 2)
  if (lowAsympScore < -1) { #if lower asmptote is below -100% we set it there to avoid unplausible estimations way below this threshold
    lowAsympScore <- -1
  }
  #upper Asymptote
  upAsympScore <- round(abs(1 - paras[3]), 2)
  if (upAsympScore > 1) { #if upper asmptote is above 200% we set it there to avoid unplausible estimations way above this threshold
    upAsympScore <- 1
  }


  GradientScore <- tanh(round((abs(1.7 - Gradient)), 2) / 1.7) #optimal gradient should be somewhere around 5
  print(paste('Gradient:', Gradient))
  print(paste('GScore:', GradientScore))
  print(as.character(drm_object[["call"]][["fct"]]))


  ResidualScore <- round(5 * relativeresiduals, 2) #optimal residuals should be as low as possible


  #get effect size (how much curve descends) -> the more descend, the better
  dev.new()
  p <- plot(drm_object, gridsize = 100, plot = F)
  dev.off()
  effectdiffScore <- abs((1 - (p[1, "1"] - p[nrow(p), "1"])))

  rawScore <- round(sum(
    (abs(lowAsympScore) + upAsympScore) / 2,
    GradientScore,
    ResidualScore,
    effectdiffScore * 5),
                    2)


  if (rawScore > upperScoreLimit) {
    rawScore <- upperScoreLimit
  }
  #return final score as linear transformation of raw score to number between 0 and 1
  return(list(
    AsympScore = (abs(lowAsympScore) + upAsympScore / 2),
    EffectScore = effectdiffScore,
    GradientScore = GradientScore,
    ResidualScore = ResidualScore,
    rawScore = rawScore,
    finalScore = (1 - (rawScore / upperScoreLimit)) #raw score is transformed to a number between 0 and upperScoreLimit
  ))
}


# PLOT FUNCTION--------------------------------------------------------------------------------------------------------------
#this function simply plots the fit and data as png file
#the global paramater 'filename' determines the output path and filename

plot_drm_object <- function() {
  png(filename, width = 1000, height = 1000) #open plotting device with directory and pixel dimensions of plot
  par(xpd = F, #BMR lines are underneath boxplot
      mar = par()$mar + c(1, 1, 10, 1), #margins (bottom, left, top, right)
      mgp = c(3, #axis lable distance
              1, #axis numbers distance
              0)) #axis ticks distance
  plot(drm_object,
       type = "all",
       ylim = c(0, 1.5), #min and max values of y axis
       cex.main = 2, #size of title
       lwd = 2, #curve line width
       cex.lab = 2, #size of axis lable
       cex.axis = 1.5, #size of axis numbers
       pch = 19, #type of data points
       cex = 1, #size of data points
       log = '',
       main = paste(scenario, '\n',
                    "Asymp Score: ", round(1 - score_data$AsympScore, 2), '\n',
                    "Effect Score: ", round(1 - score_data$EffectScore, 2), '\n',
                    "Gradient Score: ", round(1 - score_data$GradientScore, 2), '\n',
                    "Residual Score: ", round(1 - score_data$ResidualScore, 2), '\n',
                    "final Score: ", round(score_data$finalScore, 2), '\n',
                    #"final Score: ", round(score_data$finalScore,2),
                    sep = '') #text for title
  )
  dev.off() #close plotting device
}


# CURVE GRID VALUE FUNCTION --------------------------------------------------------------------------------------------------------------
#this function simply plots the fit and data as png file
#the global paramater 'filename' determines the output path and filename

get_object_griddata <- function() {
  dev.new()
  plot_data <- plot(drm_object,
                    gridsize = 100
  )
  colnames(plot_data)[2] <- "fit_model_response"
  dev.off()
  return(plot_data)
}


#########################################################################################################################################
######Script EXECUTION###################################################################################################################
#########################################################################################################################################

#execute fit modelling function and catch error if occured
possibleError <- tryCatch( #Saving plot as png if possible
  drm_object <- drm(resp ~ dose, robust = 'mean', fct = LL.4()),
  error = function(e) e
)
if (inherits(possibleError, "error")) {
  print(paste("Curve fitting failed:", possibleError))
}
if (typeof(drm_object) != 'list') { #if it failed
  print("Curve fitting failed: No model could be fitted to the data.")
}else {
  #execute fit model evaluation function and catch error if occurred
  possibleError <- tryCatch( #Saving plot as png if possible
    score_data <- evaluate_fit(),
    error = function(e) e
  )
  if (inherits(possibleError, "error")) { #if it failed
    print(paste("Fit model evaluation failed:", possibleError))
  }else {

    final_Score <- score_data$finalScore #get final score from evaluation output

    #execute fit model plotting function and catch error if occurred
    if (plot_curve) { #if input boolean is true
      possibleError <- tryCatch( #Saving plot as png if possible
        plot_drm_object(),
        error = function(e) e
      )
      if (inherits(possibleError, "error")) { #if it failed
        print(paste("Plot failed:", possibleError))
      }
    }
  }
}


#try to get the fit model grid data from a plot
possibleError <- tryCatch(
  plot_data <- get_object_griddata(),
  error = function(e) e
)
if (inherits(possibleError, "error")) { #if it failed
  print(paste("Getting grid data from plot failed:", possibleError))
}


#try to get the model estimates for all doses from model parameter prediction
possibleError <- tryCatch(
  estimate_data <- predict(drm_object, se.fit = FALSE,
                           interval = c("prediction"),
                           level = 0.95, pava = FALSE),
  error = function(e) e
)
if (inherits(possibleError, "error")) { #if it failed
  print(paste("Getting model estimates from predict failed:", possibleError))
}



