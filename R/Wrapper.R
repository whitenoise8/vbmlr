#' Variational Bayes inference for multivariate linear regressions with autoregressive effects and stochastic volatility (Bernardi, Bianchi, and Bianco, 2024: Variational Inference for Large Bayesian Vector Autoregressions)
#'
#' @param Y is a \code{d} times \code{T} matrix of observations
#' @param X is a \code{p} times \code{T} matrix of exogenous covariates. If not specified (i.e, \code{X=NULL}) only an intercept is included
#' @param AR if \code{TRUE} an autoregressive effect of order 1 is included (VAR(1))
#' @param hyper is a list of algorithm hyperparameters depending on \code{prior}
#' @param prior can be "normal" (gaussian prior), "lasso" (adaptive Bayesian lasso), "ng" (normal-gamma), "hs" (horseshoe)
#' @param SV if \code{TRUE} stochastic volatility is added to the model
#' @param maxIter maximum number of iterations in the CAVI algorithm
#' @param Tol_ELBO tolerance for relative variation in ELBO for convergence 
#' @param Tol_Par tolerance for relative variation in variational parameters for convergence 
#' @param Trace if 1 display the progress of the algorithm

#'
#' @return  \code{Omega_hat}: Estimated precision matrix (posterior mean of the approximation), \code{d x d} if the \code{SV=FALSE} and \code{d x d x n} if the \code{SV=TRUE}
#' @return  \code{a_q_nu}: Shape parameter of the posterior gamma approximation for \nu (only for \code{SV=FALSE})
#' @return  \code{b_q_nu}: Rate parameter of the posterior gamma approximation for \nu (only for \code{SV=FALSE})
#' @return  \code{mu_q_nu}: Posterior mean of the gamma approximation for \nu 
#' @return  \code{Mu_q_beta}: Posterior mean of the normal approximation for \beta (\code{d x d}) 
#' @return  \code{Sigma_q_beta}: Posterior variances of the normal approximation for \beta (\code{d x d x d}) 
#' @return  \code{Mu_q_theta}: Posterior mean of the normal approximation for \theta (\code{d x (d+p+1)}). It includes autoregressive matrix if \code{AR=TRUE}, intercept and regression coefficient matrix. See the paper for details. 
#' @return  \code{Sigma_q_theta}: Posterior variances of the normal approximation for \theta (\code{(d+p+1) x (d+p+1) x d}). It includes autoregressive matrix if \code{AR=TRUE}, intercept and regression coefficient matrix. See the paper for details. 
#' @return  \code{lowerBoundIter}: collection of ELBO until convergence
#' @return  \code{lowerBound}: value of ELBO at convergence
#' @return  \code{convergence}: check: if 1 successfully converged
#' @export

VBmlr = function(Y,X=NULL,AR=TRUE,
                 hyper = list(a_nu = 0.1,
                              b_nu = 0.1,
                              tau = 1,
                              ups = 1),
                 prior = "normal",
                 SV = FALSE,
                 maxIter = 500, 
                 Tol_ELBO = 1e-2, 
                 Tol_Par = 1e-2, 
                 Trace = 0) {
  
  n = ncol(Y)
  if (!is.null(X)) Z = rbind(matrix(1,1,n),X)
  if (is.null(X)) Z = matrix(1,1,n)
  X = Z
  if (AR==TRUE) X = rbind(Y,X)
  
  if (!SV) {
    if (prior == "normal") {
      out = fVBmlr(Y,X,
                   hyper,
                   maxIter = maxIter,
                   Tol_ELBO = Tol_ELBO, 
                   Tol_Par = Tol_Par, 
                   Trace = Trace)
    }
    
    if (prior == "lasso") {
      out = fVBmlrL(Y,X,
                    hyper,
                    maxIter = maxIter,
                    Tol_ELBO = Tol_ELBO, 
                    Tol_Par = Tol_Par, 
                    Trace = Trace)
    }
    
    if (prior == "ng") {
      out = fVBmlrNG(Y,X,
                     hyper,
                     maxIter = maxIter,
                     Tol_ELBO = Tol_ELBO, 
                     Tol_Par = Tol_Par, 
                     Trace = Trace)
    }
    
    if (prior == "hs") {
      out = fVBmlrHS(Y,X,
                     hyper,
                     maxIter = maxIter,
                     Tol_ELBO = Tol_ELBO, 
                     Tol_Par = Tol_Par, 
                     Trace = Trace)
    }
  }
  
  if (SV) {
    if (prior == "normal") {
      out = fVBmlrSV(Y,X,
                   hyper,
                   maxIter = maxIter,
                   Tol_ELBO = Tol_ELBO, 
                   Tol_Par = Tol_Par, 
                   Trace = Trace)
    }
    
    if (prior == "lasso") {
      out = fVBmlrSVL(Y,X,
                    hyper,
                    maxIter = maxIter,
                    Tol_ELBO = Tol_ELBO, 
                    Tol_Par = Tol_Par, 
                    Trace = Trace)
    }
    
    if (prior == "ng") {
      out = fVBmlrSVNG(Y,X,
                     hyper,
                     maxIter = maxIter,
                     Tol_ELBO = Tol_ELBO, 
                     Tol_Par = Tol_Par, 
                     Trace = Trace)
    }
    
    if (prior == "hs") {
      out = fVBmlrSVHS(Y,X,
                     hyper,
                     maxIter = maxIter,
                     Tol_ELBO = Tol_ELBO, 
                     Tol_Par = Tol_Par, 
                     Trace = Trace)
    }
  }
  
  
  out
}
