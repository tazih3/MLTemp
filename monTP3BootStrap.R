graphics.off();rm(list=objects())

##Question 1:
#Echantillon aléatoire N(mu,sigma)
mu=1
sigma=1.5
set.seed(111)
n=100
yg=rnorm(n,mu,sigma)

#Propriétés loi empirique Fn:
m=mean(yg)
sigmab=sqrt(mean((yg-m)^2))

plot(ecdf(yg),xlim=c(-3,5))
curve(pnorm(x,mu, sigma),from=-3, to=5,add=TRUE,col=2)
#ecdf(X) donne la fonction de répartition empirique
#curve pour rajouter la courbe sur le plot

par(mfrow=c(2,2))
hist(yg,proba=TRUE,main="Echantillon Initial",xlim=c(-3,4),ylim=c(0,1))
curve(dnorm(x,mu,sigma),from=-3,to=5,add=TRUE,col=2)
#le add=TRUE c'est pour rajouter le graphe sur un autre existant
#le col=2 choisit la couleur rouge, col=4 le bleu... QUE DANS CURVE
#hist() donne des histogrammes: c'est la densité empirique

#################################################################
#################################################################
##Question 2 : Echantillon Bootstrap
plot.echant=function(yg){
  yboot=sample(yg,replace=TRUE) # on remplace les éléments de la simulation
  hist(yboot,proba=TRUE,main=paste("Echantillon Boostrap de taille",n),xlim=c(-3,4),ylim=c(0,1))
  curve(dnorm(x,mu,sigma),from=-3,to=5,add=TRUE,col=4)
}
replicate(100,plot.echant(yg))
par(mfrow=c(1,1)) #Pour tout avoir sur le meme graphe

#En le refaisant plusieurs fois: ne coincide pas du tout avec 
#la densité mais très important autour de la moyenne...

#################################################################
#################################################################
##Question 3:
#Va nous rendre B estimations bootstrap de la moyenne d'un estimateur
par(mfrow=c(2,2))
B=100
MB=replicate(B,mean(sample(yg,replace=TRUE)))
titre=paste("Loi d'Echantillon pour B=",B,"n=",n,"mu=",round(m,3),"et sigma=",round(sigmab/sqrt(n),3))
#sigma/sqrt(n) car 1/nsomme(Yi) suit N(m,sigma^2/n)
hist(MB,proba=TRUE,main=titre,xlim=c(-2,2),ylim=c(0,2))
  curve(dnorm(x,m,sigmab/sqrt(n)),col="blue",from=-2,to=2,add=TRUE)#parametre empirique
  curve(dnorm(x,mu,sigma/sqrt(n)),col="red",from=-2,to=2,add=TRUE)#parametre reel
  
#CAS DU BOOTSTRAP CENTRÉ
titre2=paste("Loi d'Echantillon Centré pour B=",B,"n=",n,"sigma=",round(sigmab/sqrt(n),2))
hist(MB-m,proba=TRUE,main=titre2,xlim=c(-2,2),ylim=c(0,2))
  curve(dnorm(x,0,sigmab/sqrt(n)),col="blue",from=-2,to=2,add=TRUE)
  curve(dnorm(x,0,sigma/sqrt(n)),col="red",from=-2,to=2,add=TRUE)
# En fait le Bootstrap est adapté à la loi centrée !!
  
#Calcul du biais:
c(original=m,biais=mean(MB)-m,sd=sd(MB))

#################################################################
#################################################################
##Question 4:
#Avec n=100 on a adéquation même si le Bootstrap n'est pas recentré !

#################################################################
#################################################################
##Question 5:
# Soit theta = m(F) le parametre d'interet
# Y~F un n-?chantillon iid
# Soit M(Y)=m(Fn) l'estimateur bas? sur la loi empirique 
# de biais E(M(Y)|F) et de variance var(M|F)
# Y*~F* un n-?chantillon tir? suivant la loi Fn o? on a fix? Y
# on calcule M*=M(Y*)

# estimateur du biais de M: b*(M*)=E*(M*)-m(Fn)
# estime lui-meme (erreur numerique) par EB= (sum M*i)/B -m(Fn)

# pour M=estimateur de la moyenne
# biais theorique sous F*=0
# biais theorique sous F= 0
K=10
Bi=c(10,20,50,100,200,300,400,500)
trajbiais=function(Bi,yg,f){
  nB=length(Bi)
  MB=replicate(Bi[nB], f(sample(yg,replace=TRUE))) # on genere le plus grand nombre de valeurs  
  m=mean(yg)
  BB=numeric(nB) #en fait numeric(nB) crée un tableau numérique de taille nB de 0, mais ça aussi
  for (ib in 1:nB) BB[ib]=mean(MB[1:Bi[ib]]) - m #On calcule le biais
  #seulement pour Bi[ib] (qui vaut 10,20,50...) échantillon, donc pour
  #sous-échantillons du max qu'on a calculé une fois pour toutes
  return(BB)
}

par(mfrow=c(2,1))

biaisB=replicate(K,trajbiais(Bi,yg,mean))

matplot(Bi,biaisB,pch=1:nB,type="b",xlab="nombre de repetitions",ylab="biais")
title("Erreur numerique du calcul du biais de l'estimateur bootstrap",cex.main=0.8)
#cex.main : taille du titre
abline(h=0.00)# biais theorique, normalement nul pour l'estimateur de la moyenne
lines(Bi,qnorm(0.975)*sigmab/sqrt((n*Bi)),lwd=2)
lines(Bi,-qnorm(0.975)*sigmab/sqrt((n*Bi)),lwd=2)
#lwd: line width, nombre entier !
# On va choisir quelque chose autour de 100

#################################################################
#################################################################
##Question 6:
# estimateur de la variance de M: Var*(M*)= E*((M*-m(Fn))^2)
# estim? (erreur num?rique) par (sum(M*i - EB)^2)/(B-1)

# pour M= estimateur de la moyenne
# variance th?orique de M sous F*= (sd^2)*((n-1)/n)/n=(sd^2)*(n-1)/n^2
# variance th?orique de M sous F: sigma^2/n

trajvar=function(Bi,yg,f){
  nB=length(Bi)
  MB=replicate(Bi[nB], f(sample(yg,replace=TRUE))) # on g?n?re le plus grand nombre de valeurs  
  BB=numeric(nB)
  for (ib in 1:nB) BB[ib]=var(MB[1:Bi[ib]])
  BB
}

# on trace la variance th?orique de l'estimateur bootstrap sous F*
v=sd(yg)^2*(n-1)/n^2
v=sigmab^2/n   # idem

VB=replicate(K,trajvar(Bi,yg,mean))

matplot(Bi,VB,pch=1:nB,type="b",xlab="nombre de r?p?titions",ylab="variance")
title("Erreur num?rique du calcul de la variance\n
      de l'estimateur bootstrap du biais de l'estimateur de l'esp?rance",cex.main=0.8)

abline(h=sd(yg)^2*(n-1)/n^2,lwd=2)# variance th?orique sous F* pour cet ?chantillon
lines(Bi,v+qnorm(0.975)*sqrt(2*v^2/Bi),lwd=2)   # fluctuations num?riques conditionnelles
lines(Bi,v-qnorm(0.975)*sqrt(2*v^2/Bi),lwd=2)   # fluctuations num?riques conditionnelles
abline(h=sigma^2/n,col="red",lwd=2) # variance th?orique sous F
vt=((2/n^3+1/n^2)/Bi +1/n^3)*2*sigma^4          # fluctuations num?riques + stat inconditionnelles
lines(Bi,sigma^2/n+qnorm(0.975)*sqrt(vt),lwd=2,col=2)
lines(Bi,sigma^2/n-qnorm(0.975)*sqrt(vt),lwd=2,col=2)

# on choisit une valeur de B autour de 100 environ 
# ce qui suffit en g?n?ral dans le cas d'une moyenne

#################################################################
#################################################################
##Question 7: Estimation d'un quantile

B=1000
MB=replicate(B,mean(sample(yg,replace=TRUE)) )

### intervalle bootstrap percentile na?f
alpha=0.05
a=c(alpha/2,1-alpha/2)
quantile(MB,a)            #IC naif par bootstrap
qnorm(a,mu,sigma/sqrt(n)) # quantiles th?oriques correspondant

### intervalle bootstrap percentile 
m- quantile(MB-m,rev(a) ) 
m- qnorm(rev(a),0,sigma/sqrt(n))  # th?orique quand on connait sigma

### intervalle bootstrap-t
# on est oblig? de calculer la moyenne et l'?cart type sur chaque ?chant bootstrap
ft=function(y,index){
  (mean(y[index])-mean(y) )/sd(y[index])
}
Mt=replicate(B,ft(yg,sample(n,replace=TRUE)) )
qt=quantile(Mt,a); qt    # estim quantile par bootstrap
qt(a,n-1)*sigma/sqrt(n)  # pour comparaison

m-sigmab*c(rev(qt))      # IC bootstrap-t

# avec approx classique gaussienne dans le monde r?el
m+c(-1,1)*qnorm(1-alpha/2) * sd(yg)/sqrt(n)
# avec loi exacte dans le monde r?el
m+c(-1,1)*qt(1-alpha/2,n-1) * sd(yg)/sqrt(n)


# rem: le code suivant donne le m?me IC (attention, partir du m?me seed)
# mais ne cherche pas le m?me quantile
ftbis=function(y,index){
  sqrt(length(y))* (mean(y[index])-mean(y) )/sd(y[index])
}

Mtbis=replicate(B,ftbis(yg,sample(n,replace=TRUE)) )
qtbis=quantile(Mtbis,a); qtbis # estimation par bootstrap
qt(a,n-1)                      # quantile exact
m-sigmab*c(rev(qtbis))/sqrt(n)

#################################################################
#################################################################
##Question 8: Avec la librairie boot
library(boot)
meanb = function(y,index) mean(y[index])

out=boot(yg,meanb,R=1000); out
boot.ci(out,type=c("basic","perc","norm"))
# out$t0= m
# out$t = l'?chantillon bootstrap de l'estimateur
# mean(out$t)-out$t0 = biais

# estimation gaussienne de boot.ci => estime le biais par bootstrap
out$t0+c(-1,1)*sd(out$t)*qnorm(1-alpha/2)+ (out$t0-mean(out$t))


##### bootstrap-t

meant=function(y,index) {
  c(mean(y[index]),  sd(y[index])/sqrt(length(y)))
}
set.seed(321)
out.t=boot(yg,meant,R=999)
res=boot.ci(out.t,type=c("norm","stud","perc"))
res

# idem
meantbis=function(y,index) {
  c(mean(y[index]),  sd(y[index]))
}
set.seed(321)
out.tbis=boot(yg,meantbis,R=999)
resbis=boot.ci(out.tbis,type=c("norm","stud","perc"))
resbis

##########################
## Q8 simulation du niveau
##########################
B=1000
n=10;r=1
mu=r

## la moyenne de 10 observations de loi expo n'est par gaussienne
MB= replicate(B,mean(rexp(n,r)))
hist(MB,proba=TRUE,main="un echantillon de l'estimateur")

ks.test((MB-mean(MB))/sd(MB),pnorm)
qqnorm((MB-mean(MB))/sd(MB))
abline(0,1)

## on tire l'?chantillon suivant la loi voulue
# Y=rnorm(n,mu)         
# Y=rqchisq(n,mu)
Y=rexp(n,r) 

hist(Y,proba=TRUE)    # ?chantillon initial

##
out.t=boot(Y,meant,R=999)
boot.ci(out.t,type=c("norm","stud"))
# avec approx gaussienne == sortie norm
out.t$t0[1]+c(-1,1)*sd(out.t$t[,1])*qnorm(1-alpha/2)+ (out.t$t0[1]-mean(out.t$t[,1]))
mean(Y)+c(-1,1)*qt(1-alpha/2,n-1)*sd(Y)/sqrt(n)

# pour comparer les IC 
compareIC=function(y,B){
  out.t=boot(y,meant,R=B)
  ci=boot.ci(out.t,type=c("norm","stud","perc"))
  c(mean=out.t$t0[1],min_n=ci$normal[2],max_n=ci$normal[3],
    min_t=ci$student[4],max_t=ci$student[5],
    min_p=ci$percent[4],max_p=ci$percent[5]
  )
}
compareIC(Y,B)

# et verifier le niveau
resIC=as.data.frame(t(replicate(100,compareIC(rnorm(n,mu,sigma),B))))

# niveau obs 
mean(resIC$min_n<mu & resIC$max_n>mu) 
mean(resIC$min_t<mu & resIC$max_t>mu) 
mean(resIC$min_p<mu & resIC$max_p>mu) 
par(mfrow=c(1,1))
plot(0,type="n",xlim=range(resIC),ylim=c(0,100))
segments(resIC$min_n,1:100,resIC$max_n,1:100)
points(resIC$mean,1:100,pch=19)
abline(v=mu,col=2)
title(paste("niveau obs=", mean(resIC$min_t<mu & resIC$max_t>mu) ))