Package based on work by S. Hundrieser, M. Klatt, and A. Munk - ''The Statistics of Circular Optimal Transport'' (2021). A short overview with illustrations is available at: [http://stochastik.math.uni-goettingen.de/cot/](http://stochastik.math.uni-goettingen.de/cot/). 

**Installation**

Download or clone the repository and install in R-Studio the file "CircularOT_0.1.1.tar.gz" with the command
```
install.packages("~/PATH-TO-File/CircularOT_0.1.1.tar.gz", repos = NULL, type = "source")
```


**Code example**
```
### Load package
library(circularOT)
library(circular)      # Package for random number generation of 
                       # von Mises distribution
                           
### Test for uniformity using COTT
set.seed(0)
cot.test_Uniformity(runif(15, 0.2, 0.8), typeOfData="UnitInt") 
#       
#          One-Sample COTT for Uniformity
# Test statistic:  0.3717834  
# P-Value:         0.046 

### Test if two samples stem from identical distribution
set.seed(5)
cot.test_Bivariate_Bootstrap(
               rvonmises(10, circular(0),     3, control.circular=list(units="radians")), 
               rvonmises(10, circular(pi), 3, control.circular=list(units="radians")), 
               typeOfData = "Radian")
#
#          Bivariate (Bootstrap) COTT for Goodness of Fit
# Test statistic:  0.4943681 
# P-Value:         0.0046
```

