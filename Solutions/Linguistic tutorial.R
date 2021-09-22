# Data creation
pitch = c(233,204,242,130,112,142)
sex = c(rep("female",3), rep("male",3))
my.df = data.frame(sex,pitch)

# Modeling
xmdl = lm(pitch~sex, my.df)
summary(xmdl)
mean(my.df[my.df$sex=="female",]$pitch)

# Date creation
age = c(14,23,35,48,52,67)
pitch_Age = c(252,244,240,233,212,204)
my.df2 = data.frame(age, pitch_Age)
xmdl2 = lm(pitch_Age~age, my.df2)
summary(xmdl2)
plot(my.df2, ylim = c(180,280), xlim = c(0,80))
abline(xmdl2)

my.df2$age.c = my.df2$age-mean(my.df2$age)
xmdl3 = lm(pitch~age.c, my.df2)
summary(xmdl3)
my.df2$age.c

