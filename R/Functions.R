#' Plot the optimal densities for \beta and \theta
#'
#' @param mod an output of \code{VBmlr}
#' @param param the parameter to be plotted: either "beta" or "theta"
#' @param index the index of the parameter, i.e., the entry in the corresponding matrix \beta or \theta

#'
#' @return  95% HPD interval and the density plot
#' @export
qDensity = function(mod, param, index) {
  require(ggplot2)
  require(TeachingDemos)
  
  j = index[1]
  k = index[2]
  jk = paste(j,",",k,sep="")
  
  d = nrow(mod$Mu_q_theta)
  d_ = ncol(mod$Mu_q_theta)

  if (param == 'beta') {
    if ((j > 1)&(j < (d+1))&(k < j)) {
      HPD = hpd(function(x) qnorm(x, mod$Mu_q_beta[j,k], sqrt(mod$Sigma_q_beta[k,k,j])))
      fHPD = dnorm(HPD[1], mod$Mu_q_beta[j,k], sqrt(mod$Sigma_q_beta[k,k,j]))
      
      p = ggplot() + 
        geom_segment(aes(x=HPD[1],xend=HPD[1],y=0,yend=fHPD),size=1.2,col="red",alpha=0.5) +
        geom_segment(aes(x=HPD[2],xend=HPD[2],y=0,yend=fHPD),size=1.2,col="red",alpha=0.5) +
        stat_function(fun = function(x) dnorm(x, mod$Mu_q_beta[j,k], sqrt(mod$Sigma_q_beta[k,k,j])),
                                   size = 1.2) + theme_minimal() +
        scale_x_continuous(breaks=HPD,limits=c(mod$Mu_q_beta[j,k]-4*sqrt(mod$Sigma_q_beta[k,k,j]), mod$Mu_q_beta[j,k]+4*sqrt(mod$Sigma_q_beta[k,k,j]))) +
        ggtitle(bquote(paste(paste(mu," = ",.(round(mod$Mu_q_beta[j,k],3)),sep=""),
                             paste("   and   "),
                             paste(sigma," = ",.(round(sqrt(mod$Sigma_q_beta[k,k,j]),3)),sep="")))) +
        xlab(bquote(beta[.(jk)])) + ylab("") +
        theme(plot.title = element_text(hjust = 0.5),
              text = element_text(size=15), 
              axis.title.x = element_text(size=20))
      
    } else {
      print("Wrong Indexes provided")
    }
  }
  
  if (param == 'theta') {
    if ((j > 0)&(j < (d+1))&(k > 0)&(k < d_+1)) {
      HPD = hpd(function(x) qnorm(x, mod$Mu_q_theta[j,k], sqrt(mod$Sigma_q_theta[k,k,j])))
      fHPD = dnorm(HPD[1], mod$Mu_q_theta[j,k], sqrt(mod$Sigma_q_theta[k,k,j]))
      
      Stdev = sqrt(mod$Sigma_q_theta[k,k,j])
      p = ggplot() + geom_segment(aes(x=HPD[1],xend=HPD[1],y=0,yend=fHPD),size=1.2,col="red",alpha=0.5) +
        geom_segment(aes(x=HPD[2],xend=HPD[2],y=0,yend=fHPD),size=1.2,col="red",alpha=0.5) +
        stat_function(fun = function(x) dnorm(x, mod$Mu_q_theta[j,k], Stdev),
                                   size = 1.2) + theme_minimal() +
        scale_x_continuous(breaks=HPD,limits=c(mod$Mu_q_theta[j,k]-4*Stdev, mod$Mu_q_theta[j,k]+4*Stdev)) +
        ggtitle(bquote(paste(paste(mu," = ",.(round(mod$Mu_q_theta[j,k],3)),sep=""),
                             paste("   and   "),
                             paste(sigma," = ",.(round(Stdev,3)),sep="")))) +
        xlab(bquote(theta[.(jk)])) + ylab("") +
        theme(plot.title = element_text(hjust = 0.5),
              text = element_text(size=15), 
              axis.title.x = element_text(size=20))
    } else {
      print("Wrong Indexes provided")
    }
  }
  
  if (param == 'nu') {
    if ((j > 0)&(j < (d+1))&(k > 0)&(k < (d+1)&(j == k))) {
      HPD = hpd(function(x) qgamma(x, mod$a_q_nu[j], mod$b_q_nu[j]))
      fHPD = dgamma(HPD[1], mod$a_q_nu[j], mod$b_q_nu[j])
      
      mu = mod$mu_q_nu[j]
      s  = sqrt(mod$a_q_nu[j]/mod$b_q_nu[j]^2)
      p = ggplot() + 
        geom_segment(aes(x=HPD[1],xend=HPD[1],y=0,yend=fHPD),size=1.2,col="red",alpha=0.5) +
        geom_segment(aes(x=HPD[2],xend=HPD[2],y=0,yend=fHPD),size=1.2,col="red",alpha=0.5) +
        stat_function(fun = function(x) dgamma(x, mod$a_q_nu[j], mod$b_q_nu[j]),
                                   size = 1.2) + theme_minimal() +
        scale_x_continuous(breaks=HPD,limits=c(mu-4*s, mu+4*s)) +
        ggtitle(bquote(paste(paste(mu," = ",.(round(mu,3)),sep=""),
                             paste("   and   "),
                             paste(sigma," = ",.(round(s,3)),sep="")))) +
        xlab(bquote(nu[.(as.character(j))])) + ylab("") +
        theme(plot.title = element_text(hjust = 0.5),
              text = element_text(size=15), 
              axis.title.x = element_text(size=20))
    } else {
      print("Wrong Indexes provided")
    }
  }
  
  HPD = round(HPD,4)
  print(paste0("HPD 95% interval: [",HPD[1]," ; ",HPD[2],"]"))
  p
}

#' SAVS operator to sparsify posterior estimates (Ray and Bhattacharya, 2018: Signal Adaptive Variable Selector for the Horseshoe Prior)
#'
#' @param Par_hat the regression coefficient matrix to be sparsified
#' @param Z the corresponding covariates

#'
#' @return  Sparse estimated regression coefficient matrix
#' @export
SAVS = function(Par_hat,Z) {
  Phi_sp = matrix(0,nrow(Par_hat),ncol(Par_hat))
  
  Znorm = apply(Z,1,function(x) sum(x^2))
  Mu = 1/abs(Par_hat)^2
  
  for (j in 1:nrow(Par_hat)) {
    logicVec = abs(Par_hat[j,])*Znorm > Mu[j,]
    Phi_sp[j,] = Par_hat[j,]
    Phi_sp[j,!logicVec] = 0
  }
  
  Par_sp = Phi_sp
  Par_sp
}

#' Plot a matrix as an heatmap
#'
#' @param m1 the matrix to be plotted
#' @param m2 (optional) second matrix to be plotted next to \code{m1}. Can be used to compare estimated vs true or to plot separately endogenous and exogenous covariates effects

#'
#' @export
matrixplot = function(m1, m2 = NULL) {
  require(ggplot2)
  require(reshape2)
  
  if (is.null(m2)) {
    
    d = nrow(m1)
    p = ncol(m1)
    
    m.mat = melt(t(m1)[,d:1])
    names(m.mat)[3] = "Value"

    if ((!is.null(rownames(m1))) & (!is.null(colnames(m2)))) {
      nx = colnames(m1)
      ny = rownames(m1)[d:1]
    } else {
      nx = 1:p
      ny = d:1
    }
    
    pl = ggplot() + 
      geom_tile(data = m.mat, aes(x=as.numeric(Var1), y=as.numeric(Var2), fill=Value)) + ylab('') + xlab('') +
      scale_fill_gradient2(low = "red", high = "blue", mid="white",
                           midpoint = 0) +
      geom_rect(aes(ymin=0.5,ymax=d+0.5,xmin=0.5,xmax=p+0.5),col="black",fill=NA,linetype='dashed') +
      theme(panel.grid = element_blank(), panel.background = element_rect(fill='white'),
            plot.background = element_rect(color=NA), axis.title.x=element_blank(), axis.title.y=element_blank(),
            axis.text.x = element_text(angle=90, vjust=0.4), text = element_text(size=15),
            legend.position = "right") +
      scale_x_continuous(1:p, breaks=1:p, labels = nx) + 
      scale_y_continuous(1:d, breaks=1:d, labels = ny) 
  }
  
  if (!is.null(m2)) {
    require(ggnewscale)
    d = nrow(m1)
    p1 = ncol(m1)
    p2 = ncol(m2)
    
    m.mat1 = melt(t(m1)[,d:1])
    m.mat2 = melt(t(m2)[,d:1])
    
    names(m.mat1)[3] = "Value1"
    names(m.mat2)[3] = "Value2"
    
    if ((!is.null(rownames(m1))) & 
        (!is.null(colnames(m1))) & 
        (!is.null(colnames(m2)))) {
      nx = c(colnames(m1),colnames(m2))
      ny = rownames(m1)[d:1]
    } else {
      nx = c(1:p1,1:p2)
      ny = d:1
    }
    
    pl = ggplot() + 
      geom_tile(data = m.mat1, aes(x=as.numeric(Var1), y=as.numeric(Var2), fill=Value1)) + ylab('') + xlab('') +
      scale_fill_gradient2(low = "red", high = "blue", mid="white",
                           midpoint = 0) +
      new_scale_fill() +
      geom_tile(data = m.mat2, aes(x=as.numeric(Var1)+p1+1, y=as.numeric(Var2), fill=Value2)) + ylab('') + xlab('') +
      scale_fill_gradient2(low = "red", high = "blue", mid="white",
                           midpoint = 0) +
      geom_rect(aes(ymin=0.5,ymax=d+0.5,xmin=0.5,xmax=p1+0.5),col="black",fill=NA,linetype='dashed') +
      geom_rect(aes(ymin=0.5,ymax=d+0.5,xmin=p1+1.5,xmax=p1+p2+1+0.5),col="black",fill=NA,linetype='dashed') +
      scale_x_continuous(1:(p1+p2+1), breaks=(1:(p1+p2+1))[-(p1+1)], labels = nx) + 
      scale_y_continuous(1:d, breaks=1:d, labels = ny) +
      theme(panel.grid = element_blank(), panel.background = element_rect(fill='white'),
            plot.background = element_rect(color=NA), axis.title.x=element_blank(), axis.title.y=element_blank(),
            axis.text.x = element_text(angle=90, vjust=0.4), text = element_text(size=15),
            legend.position = "right")
  }
  
  plot(pl)
}


#' Density approximation for Omega (see the paper for more details). Only for \code{SV=FALSE}.
#'
#' @param mod an output of \code{VBmlr} with \code{SV=FALSE}

#'
#' @return  \code{df}: degrees of freedom of the Wishart approximation
#' @return  \code{V}: scale matrix of the Wishart approximation
#' @export
qOmega = function(mod) {
  d = nrow(mod$Mu_q_theta)
  
  Om = mod$Omega_hat
  mu_lnu = psigamma(mod$a_q_nu)-log(mod$b_q_nu)
  
  df = uniroot(function(x) d*log(2)-d*log(x)+log(det(Om))+sum(psigamma(1/2*(x+1-seq(1,d))))-sum(mu_lnu), 
               interval=c(d-1+1e-5,1e5))$root
  V = Om/df
  
  list(df=df,V=V)
}

#' Exact variational predictive distribution (see the paper for more details). Only for \code{SV=FALSE}.
#'
#' @param mod an output of \code{VBmlr} with \code{SV=FALSE}
#' @param qOm an output of \code{qOmega}
#' @param z the vector of new observed covariates for prediction
#' @param R the number of samples from the variational predictive posterior

#'
#' @return  Samples from the variational predictive posterior \code{d x R}
#' @export
qExactPredictive = function(mod,qOm,z,R=5000) {
  require(mvnfast)
  require(Matrix)
  
  d = nrow(mod$Mu_q_theta)
  d_ = ncol(mod$Mu_q_theta)
  
  Z = kronecker(diag(1,d),t(z))
  
  mu_th = as.vector(mod$Mu_q_theta)
  Sigma_th = as.matrix(bdiag(lapply(seq(dim(mod$Sigma_q_theta)[3]), function(x) mod$Sigma_q_theta[,,x])))
  
  nu = qOm$df-d+1
  S = 1/nu*solve(qOm$V)
  
  Ysim = matrix(NA,d,R)
  for (r in 1:R) {
    Th_sim = t(rmvn(1,mu_th,Sigma_th))
    Ysim[,r] = t(mvnfast::rmvt(1,Z%*%Th_sim,S,nu))
  }
  
  Ysim
}

#' Gaussian approximation to the variational predictive distribution (see the paper for more details). Only for \code{SV=FALSE}.
#'
#' @param mod an output of \code{VBmlr} with \code{SV=FALSE}
#' @param qOm an output of \code{qOmega}
#' @param z the vector of new observed covariates for prediction

#'
#' @return  \code{mu}: the vector of means of the approximated Gaussian predictive distribution 
#' @return  \code{Sigma}: the variance matrix of the approximated Gaussian predictive distribution 
#' @export
qApproxPredictive = function(mod,qOm,z) {
  library(Matrix)
  d = nrow(mod$Mu_q_theta)
  d_ = ncol(mod$Mu_q_theta)
  
  Z = kronecker(diag(1,d),t(z))
  
  mu_th = as.vector(mod$Mu_q_theta)
  Sigma_th = bdiag(lapply(seq(dim(mod$Sigma_q_theta)[3]), function(x) mod$Sigma_q_theta[,,x]))
  
  nu = qOm$df-d+1
  S = 1/nu*solve(qOm$V)
  
  R = (nu-2)/nu*solve(S)
  Stilde = solve(solve(Sigma_th)+t(Z)%*%R%*%Z)
  
  Sy = solve(R-R%*%Z%*%Stilde%*%t(Z)%*%R)
  Muy = Sy%*%(R%*%Z%*%Stilde%*%solve(Sigma_th)%*%mu_th)
  
  list(mu=as.vector(Muy),
       Sigma=as.matrix(Sy))
}



