# SIGMOIDAL EVALUATION OF CNN PREDICTIONS
library(drc)
library(data.table)


###############INPUTT#######################################################################
# dose <- cData[,dose]
# dose <- dose/max(dose) #normalize dose to get standarized data for slope evaluation
# resp <- cData[,resp]
############################################################################################

# initially setting the vlaue to overwrite it later
finalScore <- 0 / 0

#create curve fit model
trydrm <- try(drm(resp ~ dose, robust = 'mean', fct = llogistic2()))

#figure out, whic model fits best
win <- mselect(trydrm, list(LL.4(), W1.4()), linreg = F)

#now create a new drm object with the winning model
for (n in 1:nrow(win)) {
  winmodel <- rownames(win)[n]

  if (winmodel == 'LL.4') {
    trydrm <- try(drm(resp ~ dose, robust = 'mean', fct = LL.4()))
  } else if (winmodel == 'W1.4') {
    trydrm <- try(drm(resp ~ dose, robust = 'mean', fct = W1.4()))
  } else if (winmodel == 'EXD.3') {
    trydrm <- try(drm(resp ~ dose, robust = 'mean', fct = EXD.3()))
  }

  if (typeof(trydrm) == 'list') {
    break
  } else {
    next
  }
}
if (typeof(trydrm) != 'list') {
  print(paste("Fitting failed for curve."))
  objectList[length(objectList) + 1] <- NA
  next
}


paras <- trydrm$fit$par #get parameters like asymptote position and slope for score evaluation
parnames <- trydrm$parNames[[2]]
relativeresiduals <- sum(abs(residuals(trydrm, typeRes = "working"))) / length(dose) #get residuals
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

GradientScore <- round((abs(1 - Gradient)^2), 2) #gradient should be somewhere around 1
ResidualScore <- round(relativeresiduals, 2) #residuals shoulld be as low as possible

#get effect size (how much curve descends) -> the more descend, the better
dev.new()
p <- plot(trydrm, gridsize = 100, plot = F)
dev.off()
effectdiffScore <- (100 - (p[1, "1"] - p[nrow(p), "1"]))

#final score is calculated, but should be transformed to a number between 0 and 1
finalScore <- round(sum((abs(lowAsympScore) + upAsympScore) / 2, GradientScore, ResidualScore, effectdiffScore * 2), 2)

# Saving the plot
png(filename, width = 3000, height = 3000)
par(xpd = F, #ablines are underneath boxplot... i think
    mar = par()$mar + c(10, 12, 7, 7), #margins (bottom, left, ?, ?)
    mgp = c(3, 4, 0)) #does something with the axis ticks?
par(mfrow = c(1, 1)) #plots x rows and y columns of images into pdf

plot(trydrm,
     type = "bars", ylim = c(0, 150), cex.main = 3, lwd = 4, cex.lab = 3, cex.axis = 2, pch = 19, cex = 3,
     main = paste(finalScore)
)
dev.off()




