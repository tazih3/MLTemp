rm(list=objects())
dataset=read.csv('Data.csv')

#Take care of missing data
dataset$Age=ifelse(is.na(dataset$Age),
                   ave(dataset$Age,FUN=function(x) mean(x,na.rm = TRUE)),
                   dataset$Age)
dataset$Salary=ifelse(is.na(dataset$Salary),
                   ave(dataset$Salary,FUN=function(x) mean(x,na.rm = TRUE)),
                   dataset$Salary)

#Le ave est obligé, parce qu'il y a des na et il faut préciser 
#qu'on les prend en compte dans le calcul de mean avec le na.rm