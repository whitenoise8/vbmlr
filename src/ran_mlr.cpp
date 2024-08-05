// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace arma;
using namespace Rcpp;

void set_seed(unsigned int seed) {
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);  
}

//' Generate samples from the model
//'
//' @param n number of samples
//' @param params list of parameters: 1) \code{Omega}: the precision matrix (\code{d x d}), 2) \code{mu}: the intercept vector (\code{d x 1}), 3) \code{Phi}: the autoregressive matrix (\code{d x d}), 4) \code{Gamma}: the regression coefficients matrix (\code{d x p})
//' @param nburn number of observations to discard before sampling
//' @param seed_set set the seed for replicability (if 0, no set seed)
//' @return \code{Y} the matrix of observations (\code{d x n})
//' @return \code{X} the matrix of exogenous covariates (independent normally distributed) (\code{p x n})
//' @export
// [[Rcpp::export]]
Rcpp::List ran_mlr(int n, Rcpp::List params, int nburn = 1000, int seed_set = 0) {
  if (seed_set > 0) set_seed(seed_set);
  
  Rcpp::List out;
  
    arma::mat Omega = params["Omega"];
    arma::vec mu = params["mu"];
    arma::mat Phi = params["Phi"];
    arma::mat Gamma = params["Gamma"];
    
    int d = Omega.n_cols;
    int p = Gamma.n_cols;
    
    arma::mat Sigma = inv_sympd(Omega);
    
    arma::mat Y = zeros(d, n+nburn);
    Y.col(0) = mvnrnd(mu, Sigma, 1);
    
    arma::mat X = zeros(p, n+nburn);
    X.col(0) = mvnrnd(zeros(p), eye(p,p), 1);
    
    for (int t=1; t < n+nburn; t++) {
      X.col(t) = mvnrnd(zeros(p), eye(p,p), 1);
      Y.col(t) = mvnrnd(mu + Phi*Y.col(t-1) + Gamma*X.col(t-1), Sigma, 1);
    }
    
    Y.shed_cols(0,nburn-1);
    X.shed_cols(0,nburn-1);
  
  out["Y"] = Y;
  out["X"] = X;
  
  return out;
}

