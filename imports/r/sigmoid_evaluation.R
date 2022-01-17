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

###################################################################################################################################################

#initially setting output to overwerite it later (if curve fitting was successfull, else NaN will be put out)
finalScore <- 0 / 0 #NaN

#initially define plot function executed later in try-catch
function.plot_drm_object <- function(drm_object){

  png(filename, width = 500, height = 500) #open plotting device with directory and pixel dimensions of plot
  par(xpd = F, #BMR lines are underneath boxplot
      mar = par()$mar + c(1,1,1,1), #margins (bottom, left, top, right)
      mgp = c(3, #axis lable distance
              1, #axis numbers distance
              0)) #axis ticks distance

  plot(drm_object,
       type = "all",
       ylim = c(0, 150), #min and max values of y axis
       cex.main = 2, #size of title
       lwd = 2, #curve line width
       cex.lab = 2, #size of axis lable
       cex.axis = 1.5, #size of axis numbers
       pch = 19, #type of data points
       cex = 1, #size of data points
       main = paste(finalScore) #text for title
  )
  dev.off() #close plotting device
}







# 1) APPLY FIT MODEL-------------------------------------------------------------------------------------------------------------------------------

#create initial curve fit model
drm_object <- try(drm(resp ~ dose, robust = 'mean', fct = llogistic2()))

#figure out, if symmetrical or asymmetrical model fits better
win <- mselect(drm_object, list(LL.4(), W1.4()), linreg = F)

#now create a new drm object with the winning model
for (n in 1:nrow(win)) {
  winmodel <- rownames(win)[n]

  if (winmodel == 'LL.4') {
    drm_object <- try(drm(resp ~ dose, robust = 'mean', fct = LL.4()))
  } else if (winmodel == 'W1.4') {
    drm_object <- try(drm(resp ~ dose, robust = 'mean', fct = W1.4()))
  }

  #if the drm object is sucessfully created, break loop
  if (typeof(drm_object) == 'list') {
    break
  } else {
    next
  }
}

#if no fit model could be applied to the data, a NaN is given as output
if (typeof(drm_object) != 'list'){
  print(paste("Fitting failed for curve."))

}else{






  # 2) CALCULATE SCORE FOR FIT MODEL --------------------------------------------------------------------------------------------------------------

  paras <- drm_object$fit$par #get parameters like asymptote position and slope for score evaluation
  relativeresiduals <- sum(abs(residuals(drm_object, typeRes = "working"))) / length(dose) #get residuals
  Gradient <- paras[1] #get curve gradient
  if (Gradient < 0) { #in case slope is positive, we make it bigger for a much worse score
    Gradient <- Gradient * 5
  }


  #lower Asymptote
  lowAsympScore <- round((0 + paras[2]), 2)
  if (lowAsympScore < -100) { #if lower asmptote is below 0% we set it there to avoid unplausible estimations way below this threshold
    lowAsympScore <- 0
  }

  #upper Asymptote
  upAsympScore <- round(abs(100 - paras[3]), 2)
  if (upAsympScore > 100) { #if upper asmptote is above 200% we set it there to avoid unplausible estimations way above this threshold
    upAsympScore <- 100
  }

  GradientScore <- round((abs(1 - Gradient)^2), 2) #optimal gradient should be somewhere around 1
  ResidualScore <- round(relativeresiduals, 2) #optimal residuals should be as low as possible

  #get effect size (how much curve descends) -> the more descend, the better
  dev.new()
  p <- plot(drm_object, gridsize = 100, plot = F)
  dev.off()
  effectdiffScore <- (100 - (p[1, "1"] - p[nrow(p), "1"]))


  #final score is calculated, but should be transformed to a number between 0 and 1
  finalScore <- round(sum((abs(lowAsympScore) + upAsympScore) / 2, GradientScore, ResidualScore, effectdiffScore * 2), 2)

  #formular for linear tranormation of score to number between 0 and 1
  #tr_Score = (finalScore - min(x)) / (max(x) - min(x))
  #x needs to be figured out...





  # 3) PLOT DATA AND FIT MODEL -------------------------------------------------------------------------------------------------------------------

  if(plot_curve){ #if input boolean is true

    possibleError <- tryCatch( #Saving plot as png if possible
      plot_drm_object(drm_object),
      error=function(e) e
    )
    if(inherits(possibleError, "error")){print("Plot failed.")} #if it failed
  }
}


