x=((1+sqrt(5))/2)**2
print(x)
#VECTEURS
V1=c(-1,3.2,-2,8)
V2=-2:6
V3=seq(0.05,0.2,0.05)
V4=rep(1,10)
V5=c('OUI','NON')
sort(V1)
V6=2*V2-3
print(V6)
n=length(V6)
V7=c(V6[n-2,n-1,n])
nbc=c(4138,7077,11176,6474,3735,2365,1573)
pctb=c(1.1,6.6,26.3,64.7,88.7,98,99.9)
pctbp=pctb/100
r=nbc*pctbp
result=(sum(r)/sum(nbc))*100
print(result)

#STRUCTURE DE DONNÉES
library(MASS)
?genotype
genotype[genotype$Litter=="I"|genotype$Mother=="A",]
?tapply
tapply(genotype$Wt,list(genotype$Mother,genotype$Litter),mean)

#GRAPHIQUES
plot(genotype$Litter)
Litter.table=table(genotype$Litter)
par(mfrow=c(1,2),oma=c(0,0,3,0))
barplot(Litter.table,horiz=T,col=rainbow(4))

par(mfrow=c(1,1))
plot(genotype$Wt)
hist(genotype$Wt)
paste("moy=",round(mean(genotype$Wt),2))

#IMPORT ET EXPORT DE DONNÉES
nit=read.table(file="/Users/hamzatazi/Downloads/mineral.data",sep=";",header=TRUE)


