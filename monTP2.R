rm(list=objects())
graphics.off()

#Matrices
A=rbind(c(4,-1,1),c(3,10,3))
B=cbind(2:4,4:2)

#Régression Linéaire
df=read.table("hFE.csv",sep=";",dec=",",header=TRUE)
names(df)[4]="etr"

#Estimer:calcul à la main
X=as.matrix(df)
X[,1]=1 #1re colonne de 1
Y=as.matrix(df$gain)
theta.est=solve(t(X)%*%X)%*%t(X)%*%Y
n=length(Y)
p=ncol(X)
sigma.est=sqrt(sum((X%*%theta.est-Y)^2)/(n-p))
V=solve(t(X)%*%X)*sigma.est^2 #Matrice de variance
stddev=sqrt(diag(V)) #standarddeviation i.e. ecart type
Y.est=X%*%theta.est
rep=Y.est[2]

#Estimer:fonction lm
model3=lm(gain~emitter+base+etr,data=df)
summary(model3)
names(model3)
model3$coef
estim=X%*%model3$coefficients

#Intervalle de Confiance
qt=qt(1-0.05/2,model3$df.residual) #Les deux arguments sont le niveau d'IC et le nombre de ddl
IC=data.frame(mean=theta.est,min=theta.est-qt*sqrt(diag(V)),max=theta.est+qt*sqrt(diag(V)))
predict(model3,newdata=data.frame(emitter=14.5,base=220,etr=5),interval="confidence")

#Tester et Sélectionner des variables
#1
SCRH0=sum((df$gain-mean(df$gain))^2) #Estimateur de la variance
SCRH1=sum(model3$resid^2)
F=(SCRH0-SCRH1)/3/(summary(model3)$sigma^2) #F=(SCRH0-SCRH1)/3/(SCRH1/15)

#2
anova(lm(gain~etr,data=df),model3)

#Valider le modèle:
lmf=anova(lm(gain~etr,data=df),model3)
par(mfrow=c(2,2), oma=c(0,0,2,0))
plot(df$etr,df$gain,xlab="etr",ylab="gain")
abline(lmf) #superpose droite de régression

lmf=anova(lm(gain~etr,data=df),model3)
par(mfrow=c(2,2))
plot(lmf)

#DEUXIEME PARTIE
rm(list=objects());graphics.off()
perf=read.table("perfusion.data",header=TRUE, row.names=1,skip=1)