
rm(list = ls())
setwd("~/TEPLS/case_study/")
packages <- c("raster", "rgdal", "plotly", "Thermimage")
#lapply(packages, install.packages, dependencies = TRUE)
lapply(packages, require, character.only = TRUE)
train_label <- read.csv("train_label.csv",header = T)
# select group blur = 1, stain = 1
F1_W1_label = train_label[train_label$blur == 1 & train_label$stain == 1,]
#write.csv(F1_W1_label, "F1_W1_label.csv")
# load image data
setwd("~/TEPLS/case_study/train")
image<-array(NA,c(520,696,dim(F1_W1_label)[1]))
for (i in 1:dim(F1_W1_label)[1]){
  image[,,i]<-(1/255)*as.array(raster(as.character(F1_W1_label$image_name[i])))
}

# train test split
smp_siz = floor(0.75*400) 
set.seed(123) 
train_ind = sample(seq_len(400),size = smp_siz)  
#F1_W1_label[,-train_ind]
#write.csv(train_ind, "F1W1_train_idx.csv")
train_image = image[,,train_ind]
test_image = image[,,-train_ind]

# mPCA for dimension reduction
library(rTensor)
image_mean=apply(train_image,c(1,2),mean)
fnorm_den=rep(NA,300)
for(j in 1:300){
  tmp=train_image[,,j]-image_mean
  fnorm_den[j]=sum(tmp^2)
}

# 520 * 696 * 300
p0 = 10
q0 = 20
mpca_x=mpca(as.tensor(train_image),ranks=c(p0,q0),max_iter = 1000, tol=1e-1)
mpca_x$conv
attributes(mpca_x)
temp_A=mpca_x$U[[1]]
temp_B=mpca_x$U[[2]]

#temp_U = NULL
#temp_u_vec=NULL
fnorm_num_MPCA=rep(NA,300)
for (i in 1:300){
  
  # tempU = t(temp_A)%*%attributes(mpca_x$est)$data[,,i]%*%temp_B # the same as 
  
  tempU = t(temp_A)%*%train_image[,,i]%*%temp_B		
  #temp_U = rbind(temp_U,as.matrix(tempU))
  
  #temp_u_vec = rbind(temp_u_vec,as.vector(tempU))
  
  fnorm_num_MPCA[i]=sum((attributes(mpca_x$est)$data[,,i]-image_mean)^2)
  
}
# percentage of variation explained by MPCA
sum(fnorm_num_MPCA)/sum(fnorm_den)

# write out the estimated U for training data
U_p0_q0_train = array(0,dim = c(p0,q0,300))
vec_U_train = matrix(0,nrow = 300,ncol = p0*q0)
for (i in 1:300){
  U_p0_q0_train[,,i] = t(temp_A)%*%train_image[,,i]%*%temp_B
  vec_U_train[i,] = as.vector(U_p0_q0_train[,,i])
}
y_train = F1_W1_label$count[train_ind]
output_train = cbind(y_train,vec_U_train)
write.csv(output_train,"mpca_F1_W1_train.csv")

# reduce test data
U_p0_q0_test = array(0,dim = c(p0,q0,100))
vec_U_test = matrix(0,100,p0*q0)
for (i in 1:100){
  U_p0_q0_test[,,i] = t(temp_A)%*%test_image[,,i]%*%temp_B
  vec_U_test[i,] = as.vector(U_p0_q0_test[,,i])
}
y_test = F1_W1_label$count[-train_ind]
output_test = cbind(y_test,vec_U_test)
write.csv(output_test,"mpca_F1_W1_test.csv")
