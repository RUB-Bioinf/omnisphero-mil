# install.packages("drc")
# library(Rserve)
# library(drc)
# Rserve()

name <- "Test"
b <- a + 2
print(a)
print(b)

print(paste(name, "says: Hello World!"))

test_funtion <- function(input) {
  print(runif(input))
  return(list(name=name,input=input))
}

test_funtion(1)

