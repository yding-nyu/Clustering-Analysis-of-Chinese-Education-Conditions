---
title: "Project2_Yuan Ding and Yuheng Ling"
author: "Yuan Ding"
date: "2019-01-29"
output: html_document
---

```get{r eval = TRUE}
library(MASS)
library(caret)
library(cluster)
library(NbClust)
library(klaR)
library(ggplot2)
library(ggdendro)
library(GGally)
library(e1071)
library(knitr)
library(foreign)
library(gridExtra)
library(factoextra)
library(googleVis)
set.seed(2011)
```
## Data pre-description
```{r eval = TRUE}
dta <- data.frame(read.csv("edu2013.csv",row.names = 1))
dta2013 <- dta[1:6]
View(dta2013)

#Check the bivariate plot
ggpairs(data=dta2013,columns= 1:6)

#summary data
describe(dta2013)

#histgram of feature
par(mfrow = c(2, 3)) 
par(yaxs="i",las=1)
hist(dta2013$HPTA13,breaks = 20,probability = TRUE,col = "black",border = "white",main = "High School Pupil-Teacher Ratio",xlab = "HPTA13")
lines(density(dta2013$HPTA13,na.rm=F),col="red",lwd=4)
par(yaxs="i",las=1)
hist(dta2013$GTHR13,breaks = 20,probability = TRUE,col = "black",border = "white",main = "Graduate teching High School",xlab = "GTHR13")
lines(density(dta2013$GTHR13,na.rm=F),col="red",lwd=4)
par(yaxs="i",las=1)
hist(dta2013$EER13,breaks = 20,probability = TRUE,col = "black",border = "white",main = "Education Expenditure Ratio",xlab = "EER13")
lines(density(dta2013$EER13,na.rm=F),col="red",lwd=4)
par(yaxs="i",las=1)
hist(dta2013$HLR13,breaks = 20,probability = TRUE,col = "black",border = "white",main = "High School Library Ratio",xlab = "HLR13")
lines(density(dta2013$HLR13,na.rm=F),col="red",lwd=4)
par(yaxs="i",las=1)
hist(dta2013$HSAR13,breaks = 20,probability = TRUE,col = "black",border = "white",main = "High School Sports Areas Ratio",xlab = "HSAR13")
lines(density(dta2013$HSAR13,na.rm=F),col="red",lwd=4)
par(yaxs="i",las=1)
hist(dta2013$HMCR13,breaks = 20,probability = TRUE,col = "black",border = "white",main = "High School Multimedia Classroom Ratio",xlab = "HMCR13")
lines(density(dta2013$HMCR13,na.rm=F),col="red",lwd=4)
```

## Data-preprocessing
```{r eval = TRUE}
#take log(x) to HPTA13 and GTHR 13

dta2013$HPTA13.log <- log(dta2013$HPTA13)
dta2013$GTHR13.log <- log(dta2013$GTHR13)

#histgram after log: HPTA13 and GHTR13
par(mfrow = c(2, 2)) 
par(yaxs="i",las=1)
hist(dta2013$HPTA13,breaks = 20,probability = TRUE,col = "black",border = "white",main = "High School Pupil-Teacher Ratio",xlab = "HPTA13")
lines(density(dta2013$HPTA13,na.rm=F),col="red",lwd=4)
par(yaxs="i",las=1)
hist(dta2013$HPTA13.log,breaks = 20,probability = TRUE,col = "black",border = "white",main = "High School Pupil-Teacher Ratio after log",xlab = "HPTA13.log")
lines(density(dta2013$HPTA13.log,na.rm=F),col="red",lwd=4)
par(yaxs="i",las=1)
hist(dta2013$GTHR13,breaks = 20,probability = TRUE,col = "black",border = "white",main = "Graduate teching High School",xlab = "GTHR13")
lines(density(dta2013$GTHR13,na.rm=F),col="red",lwd=4)
par(yaxs="i",las=1)
hist(dta2013$GTHR13.log,breaks = 20,probability = TRUE,col = "black",border = "white",main = "Graduate teching High School after log",xlab = "GTHR13.log")
lines(density(dta2013$GTHR13.log,na.rm=F),col="red",lwd=4)

#remove the raw HPTA13 and GTHR 13
dta2013$HPTA13 <- NULL
dta2013$GTHR13 <- NULL

#standardize all the column
dta2013.stdz <- scale(dta2013)
pairs(dta2013.stdz)
View(dta2013.stdz)
describe(dta2013.stdz)

#pca after stdz
dta2013.stdz.pr <- princomp(dta2013.stdz)
pairs(dta2013.stdz.pr$scores)
autoplot(dta2013.stdz.pr,data = dta2013.stdz ,col = 'GTHR13.log',main = 'GTHR13')
```

### Clustering
# A.Hierarchical clustering
```{r eval = TRUE}
hcl.single13 <- hclust(dist(dta2013.stdz)^2,meth='single')
plot(hcl.single13)
hcl.complete13 <- hclust(dist(dta2013.stdz)^2,meth='complete')
plot(hcl.complete13)
hcl.centroid13 <- hclust(dist(dta2013.stdz)^2,meth='centroid')
plot(hcl.centroid13)

#drawing the plots using cutree method
fviz_cluster(list(data=dta2013.stdz,cluster=cutree(hcl.complete13,3)), main="Complete Linkage")
fviz_cluster(list(data=dta2013.stdz,cluster=cutree(hcl.complete13,2)), main="Complete Linkage")

# using C(g)
NbClust(dta2013.stdz, method = "complete", index = "ch")
```

# B.K-means Clustering
```{r eval = TRUE}
#Cg
Kmeans.Cg = NbClust(data = dta2013.stdz,distance = "euclidean", method = "kmeans", index = "ch")
plot(c(2:15), Kmeans.Cg$All.index, type = "b", pch = 19, xlab = " g ", ylab = "C(g)", main = "Kmeans-C(g)", ylim = c(5, 15))
Kmeans.Cg

#silhouette
fviz_nbclust(dta2013.stdz,kmeans, method = "silhouette",k.max = 15)
km.res <- kmeans(dta2013.stdz,3)
fviz_cluster(km.res, data = dta2013.stdz,main = 'K = 3')
km.res1 <- kmeans(dta2013.stdz,2)
fviz_cluster(km.res1, data = dta2013.stdz,main = 'K = 2')
```
addmargins(xtabs(~hc.complete13+km.res$cluster))
# C.PAM & Mixed Model Clustering
```{r eval = TRUE}
#PAM
pamk.best <- pamk(dta2013.stdz)
pamk.best
clusplot(pam(dta2013.stdz, pamk.best$nc),labels=3,main = 'PAM Method')

#Mclust
m_clust <- Mclust(as.matrix(dta2013.stdz), G=3) 
summary(m_clust)
par(mfrow = c(1, 1))
plot(m_clust)
summary(m_clust,parameters = TRUE)
```

# D.Comparison
```{r eval = TRUE}
# cross-tabs
lbls.complete13 <- cutree(hcl.complete13,k=3)
xtabs(~lbls.complete13+km.res$cluster)
xtabs(~lbls.complete13+pamk.best$`pamobject`$clustering)
xtabs(~km.res$cluster+pamk.best$`pamobject`$clustering)
```

### Analysis on Additional Variables
```{r eval = TRUE}
#map
provname=c("CN-11","CN-12","CN-13","CN-14","CN-15","CN-21","CN-22","CN-23","CN-31","CN-32","CN-33","CN-34","CN-35","CN-36","CN-37","CN-41","CN-42","CN-43","CN-44","CN-45","CN-46","CN-50","CN-51","CN-52","CN-53","CN-54","CN-61","CN-62","CN-63","CN-64","CN-65");

gini13 <- (dta13$GC13)
gini13 <- 1/gini13
v <- c(3,3,2,2,2,2,2,2,3,3,3,2,3,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,1,2,2)
gdp <- c(3,3,2,1,3,3,2,2,3,3,3,1,3,1,3,1,2,2,3,1,1,2,1,1,1,1,2,1,2,2,2)

#by clustering
a<-data.frame(provname,gini13)
G2 <- gvisGeoChart(a, locationvar='provname', colorvar='gini13',options=list(region='CN',displayMode="regions",resolution="provinces",colorAxis="{colors: ['yellow','red']}" ))
plot(G2)

#by gdp
b<-data.frame(provname,gdp)
G2 <- gvisGeoChart(b, locationvar='provname', colorvar='gdp',options=list(region='CN',displayMode="regions",resolution="provinces",colorAxis="{colors: ['yellow','red']}" ))
plot(G2)
```