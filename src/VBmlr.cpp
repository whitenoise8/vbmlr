// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <math.h>  
#include <RcppNumerical.h>
#include <cmath>
#include <iostream>
#include <boost/math/common_factor.hpp> 
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/digamma.hpp>

using namespace Numer;
using namespace std;
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(BH)]]

/* Define Log-Bessel function */
double f_logbess(double nu, double a)
{
  double res;
  res = boost::math::cyl_bessel_k(nu,a);
  return log(res);
}

/* Define a function to compute the gradient of
 * the Log-Bessel */
double grad_cpp(double nu0, double a0) {
  
  Rcpp::Environment pkg = Rcpp::Environment::namespace_env("numDeriv");
  Rcpp::Function grad = pkg["grad"];
  
  Rcpp::List results = grad(Rcpp::InternalFunction(f_logbess), 
                            Rcpp::_["x"] = nu0,
                            Rcpp::_["a"] = a0);
  
  return results[0];
}

/* Define the optimal density of Eta and Xi */
class myfun_cost: public Func
{
private:
  double d;
  double taulam;
  double logtau;
  double loglam;
  double e3;
public:
  myfun_cost(double d_, double taulam_, double logtau_, double loglam_, double e3_) : 
  d(d_), taulam(taulam_), logtau(logtau_), loglam(loglam_), e3(e3_) {}
  
  double operator()(const double& x) const
  {
    return exp(d*x*log(x)-d*lgamma(x)-x*(0.5*taulam+e3-logtau-loglam+d*log(2)));
  }
};

class myfun_mu: public Func
{
private:
  double k;
  double d;
  double taulam;
  double logtau;
  double loglam;
  double e3;
public:
  myfun_mu(double k_, double d_, double taulam_, double logtau_, double loglam_, double e3_) : 
  k(k_), d(d_), taulam(taulam_), logtau(logtau_), loglam(loglam_), e3(e3_) {}
  
  double operator()(const double& x) const
  {
    return x*exp(-log(k)+d*x*log(x)-d*lgamma(x)-x*(0.5*taulam+e3-logtau-loglam+d*log(2)));
  }
};


/* Define a function to obtain normalizing constant and
 * expectation of Eta and Xi optimal */
Rcpp::List integrate_latent_factor(const double d, const double taulam, const double logtau, const double loglam, const double e3)
{
  const double lower = 0, upper = 10;
  
  myfun_cost f_tilde(d, taulam, logtau, loglam, e3);
  double err_est;
  int err_code;
  const double res_cost = integrate(f_tilde, lower, upper, err_est, err_code);
  
  myfun_mu f(res_cost, d, taulam, logtau, loglam, e3);
  const double res = integrate(f, lower, upper, err_est, err_code);
  
  return Rcpp::List::create(
    Rcpp::Named("value") = res,
    Rcpp::Named("const_int") = res_cost,
    Rcpp::Named("error_estimate") = err_est,
    Rcpp::Named("error_code") = err_code
  );
}

// Get the precision matrix Q
mat getQ(double n, double k0) {
  mat Q = zeros(n,n);
  
  Q(0,0) = 1+1/k0;
  for(int i = 1; i < n; i++) {
    Q(i,i) = 2;
    Q(i,i-1) = -1;
    Q(i-1,i) = -1;
  }
  Q(n-1,n-1) = 1;
  
  return Q;
}

// Invert a tridiagonal matrix
mat invtridiag(mat A) {
  int n = A.n_cols;
  
  vec alpha = zeros(n);
  vec gamma = zeros(n);
  
  alpha(0) = A(0,0);
  
  double a21 = A(1,0);
  for (int i = 1; i < n; i++) {
    gamma(i) = a21/alpha(i-1);
    alpha(i) = A(i,i)-a21*gamma(i);
  }
  
  mat C = zeros(n,n);
  C(n-1,n-1) = 1/alpha(n-1);  
  for (int j = n-2; j > -1; j--) {
    C(n-1,j) = -a21/alpha(j)*C(n-1,j+1);
    C(j,n-1) = C(n-1,j);
  }
  for (int i = n-2; i > 0; i--) for (int j = i-1; j > -1; j--) {
    C(i,i) = 1/alpha(i)+pow(a21/alpha(i),2)*C(i+1,i+1);
    C(i,j) = -a21/alpha(j)*C(i,j+1);
    C(j,i) = C(i,j);
  }
  C(0,0) = 1/alpha(0)+pow(a21/alpha(0),2)*C(1,1);
  
  return C;
}

// Run the update of Wand 2014 to approximate SV process
Rcpp::List wand2014(arma::rowvec fold1,
                    arma::mat Sold,
                    arma::rowvec s2old1,
                    arma::rowvec s1,
                    double mu_q_s2eta_inv,
                    double Tol = 1e-3) {
  
  arma::vec fold = fold1.t();
  arma::vec s2old = s2old1.t();
  arma::vec s = s1.t();
  
  double n = s.n_elem;
  arma::vec i = ones(n+1);
  i(0) = 0;
  arma::vec splus = zeros(n+1);
  splus(span(1,n)) = s;
  int p = n+1;
  
  double k0 = 100;
  arma::mat Q = getQ(n+1,k0);
  
  int conv = 1;
  int nIt = 0;
  
  while (conv == 1) {
    nIt = nIt + 1;
    
    arma::vec dvec = splus%exp(-fold+0.5*s2old);
    arma::vec ups = -0.5*i +0.5*dvec -mu_q_s2eta_inv*Q*fold;
    
    arma::mat invSnew = -0.5*diagmat(dvec) -mu_q_s2eta_inv*Q;
    arma::mat Snew = -invtridiag(invSnew);
    arma::mat fnew = fold + Snew*ups;
    
    double delta = max(abs(fold-fnew));
    
    if (delta < Tol) conv = 0;
    
    fold = fnew;
    Sold = Snew;
    s2old = Snew.diag();
  }
  
  return Rcpp::List::create(
    Rcpp::Named("f") = fold,
    Rcpp::Named("S") = Sold,
    Rcpp::Named("s2") = s2old,
    Rcpp::Named("iter") = nIt,
    Rcpp::Named("convergence") = conv
  );
}


// [[Rcpp::export]]
Rcpp::List fVBmlr(arma::mat D, 
                  arma::mat X,
                  Rcpp::List hyper, 
                  int maxIter = 500, 
                  double Tol_ELBO = 1e-2, 
                  double Tol_Par = 1e-2, 
                  int Trace = 0) {
  
  Rcpp::List out;
  /* Get dimensions */
  double d = D.n_rows;
  double n = D.n_cols-1;
  double d_ = X.n_rows;
  
  /* Non sparse prior */
  
  /* Get Hyperparameters */
  double a_nu = hyper["a_nu"]; 
  double b_nu = hyper["b_nu"]; 
  
  double tau = hyper["tau"];
  double ups = hyper["ups"]; 
  
  /* Initialize */
  arma::cube Sigma_q_phi = zeros(d_,d_,d);
  for (int i = 0; i < d; i++) {
    Sigma_q_phi.slice(i) = eye(d_,d_);
  }
  arma::mat Mu_q_phi = zeros(d,d_);
  
  arma::mat Mu_q_beta = zeros(d,d);
  arma::cube Sigma_q_beta = zeros(d,d,d);
  arma::mat Omega_hat = eye(d,d);
  
  arma::vec mu_q_nu = ones(d);
  arma::vec a_q_nu = (n/2+a_nu)*ones(d);
  arma::vec b_q_nu = ones(d);
  
  /* Get useful quantities */
  arma::mat Y = D.cols(1,n);
  arma::vec y = vectorise(Y.t());
  arma::mat Z = X.cols(0,n-1);
  arma::mat ZZ_t = Z*Z.t();
  arma::mat Id = eye(d,d);
  
  /* Set convergence criterion */
  int converged = 0;
  int itNum = 0;
  double eps_det = 1e-8;
  
  arma::vec lowerBoundIter = zeros(maxIter+1);
  arma::vec VecOld = join_vert(join_vert(mu_q_nu,vectorise(Mu_q_phi)),
                               vectorise(Mu_q_beta));
  
  /* Start Updating */
  while(converged == 0) {
    /* Update iteration and initialize ELBO (with only constants) */
    itNum = itNum + 1;
    double lowerBound = d*(-0.5*n*log(2*datum::pi) +a_nu*log(b_nu) -lgamma(a_nu)) +
      +0.5*d*(d-1)*0.5*(1-log(tau)) +0.5*d*d_*(1-log(ups));
      
      /* Update NU_1 */
      arma::rowvec Q = Y.row(0) - Mu_q_phi.row(0)*Z;
      arma::mat W = Q*Q.t() + trace(ZZ_t*Sigma_q_phi.slice(0));
      b_q_nu(0) = b_nu + W(0,0)/2;
      mu_q_nu(0) = a_q_nu(0)/b_q_nu(0);
      /* Update ELBO */
      lowerBound = lowerBound - (a_q_nu(0)*log(b_q_nu(0))-lgamma(a_q_nu(0)));
      
      /* Update J-th regression */
      for (int j = 1; j < d; j++) {
        
        /* Get matrix K and vector k */
        arma::mat K_phi = zeros(j,j);
        for (int i = 0; i < j; i++) {
          K_phi(i,i) = trace(Sigma_q_phi.slice(i)*ZZ_t);
        }
        
        /* Update BETA_j */
        arma::mat Q = Y.rows(0,j-1) - Mu_q_phi.rows(0,j-1)*Z;
        arma::mat Sigma_beta_w = inv_sympd(mu_q_nu(j)*(Q*Q.t() + K_phi) + 1/tau*eye(j,j));
        Sigma_q_beta.slice(j).submat(0,0,j-1,j-1) = Sigma_beta_w;
        arma::vec mu_beta_w = Sigma_beta_w*(mu_q_nu(j)*(Q*(Y.row(j)-Mu_q_phi.row(j)*Z).t()));
        Mu_q_beta.submat(j,0,j,j-1) = mu_beta_w.t();
        arma::vec mu_q_betasq = Sigma_beta_w.diag() + pow(mu_beta_w,2);
        /* Update ELBO */
        lowerBound = lowerBound + 0.5*(log(det(Sigma_beta_w)+eps_det)-sum(mu_q_betasq)/tau);
        
        /* Update NU_j */
        arma::rowvec Yj = Y.row(j);
        arma::mat Sigma_phi_j = Sigma_q_phi.slice(j);
        arma::rowvec S = Yj - mu_beta_w.t()*Q - Mu_q_phi.row(j)*Z;
        arma::mat W = S*S.t() + trace(ZZ_t*Sigma_phi_j) + mu_beta_w.t()*K_phi*mu_beta_w + trace(Sigma_beta_w*Q*Q.t()) + 
          trace(Sigma_beta_w*K_phi);
        b_q_nu(j) = b_nu + W(0,0)/2;
        mu_q_nu(j) = a_q_nu(j)/b_q_nu(j);
        /* Update ELBO */
        lowerBound = lowerBound - (a_q_nu(j)*log(b_q_nu(j))-lgamma(a_q_nu(j)));
      }
      
      /* Get matrix C */
      arma::mat C = zeros(d,d);
      for (int i0 = 0; i0 < d; i0++) C = C + mu_q_nu(i0)*Sigma_q_beta.slice(i0);
      arma::mat Mu_omega = (Id - Mu_q_beta).t()*diagmat(mu_q_nu)*(Id - Mu_q_beta);
      Omega_hat = Mu_omega + C;
      
      arma::vec Mu = kron(Omega_hat,Z)*y;
      
      /* Update PHI_j */
      for (int j = 0; j < d; j++) {
        Sigma_q_phi.slice(j) = inv_sympd(Omega_hat(j,j)*ZZ_t + 1/ups*eye(d_,d_));
        
        arma::rowvec Om_jbarj = Omega_hat.row(j);
        Om_jbarj.shed_col(j);
        
        arma::mat Mu_phi_w = Mu_q_phi;
        Mu_phi_w.shed_row(j);
        arma::vec mu_phi_jbar = vectorise(Mu_phi_w.t());
        
        arma::vec Mu_j = Mu.subvec(j*d_,j*d_+d_-1);
        
        arma::vec mu_q_phi_j = Sigma_q_phi.slice(j)*(Mu_j-kron(Om_jbarj,ZZ_t)*mu_phi_jbar);
        Mu_q_phi.row(j) = mu_q_phi_j.t();
        arma::vec mu_q_phisq = Sigma_q_phi.slice(j).diag() + pow(mu_q_phi_j,2);
        /* Update ELBO */
        lowerBound = lowerBound + 0.5*(log(det(Sigma_q_phi.slice(j))+eps_det)-sum(mu_q_phisq)/ups);
      }
      
      /* Store iteration results */
      lowerBoundIter(itNum) = lowerBound;
      arma::vec VecNew = join_vert(join_vert(mu_q_nu,vectorise(Mu_q_phi)),
                                   vectorise(Mu_q_beta));
      
      /* Check convergence */
      if (itNum > 1) {
        double lowerBoundOld = lowerBoundIter(itNum-2);
        double delta_Par = max(abs((VecNew - VecOld)/VecOld));
        double delta_Elbo = abs(lowerBound - lowerBoundOld);
        
        if (delta_Par < Tol_Par)  if (delta_Elbo < Tol_ELBO) converged = 1;
        if (Trace == 1) {
          cout << "Iteration number:" << itNum << "; Parameter R.E:" << delta_Par << "| ELBO R.I:" << delta_Elbo << endl;
        }
      }
      
      if (itNum == maxIter) converged = 1;
      VecOld = VecNew;
  }
  
  /* Return results */
  out["Omega_hat"] = Omega_hat;
  
  out["a_q_nu"] = a_q_nu;
  out["b_q_nu"] = b_q_nu;
  out["mu_q_nu"] = mu_q_nu;
  
  out["Mu_q_beta"] = Mu_q_beta;
  out["Sigma_q_beta"] = Sigma_q_beta;
  
  out["Mu_q_theta"] = Mu_q_phi;
  out["Sigma_q_theta"] = Sigma_q_phi;
  
  out["lowerBoundIter"] = lowerBoundIter.subvec(1,itNum);
  out["lowerBound"] = lowerBoundIter(itNum);
  out["convergence"] = converged;
  
  return out;
}



// [[Rcpp::export]]
Rcpp::List fVBmlrL(arma::mat D, 
                   arma::mat X, 
                   Rcpp::List hyper, 
                   int maxIter = 500, 
                   double Tol_ELBO = 1e-2, 
                   double Tol_Par = 1e-2, 
                   int Trace = 0) {
  
  Rcpp::List out;
  /* Get dimensions */
  double d = D.n_rows;
  double n = D.n_cols-1;
  double d_ = X.n_rows;
  
  /* Get Hyperparameters */
  double a_nu = hyper["a_nu"]; 
  double b_nu = hyper["b_nu"]; 
  
  double e1 = hyper["e1"];    
  double e2 = hyper["e2"];
  
  double h1 = hyper["h1"];     
  double h2 = hyper["h2"]; 
  
  /* Initialize optimal quantities */
  arma::cube Sigma_q_phi = zeros(d_,d_,d);
  for (int i = 0; i < d; i++) {
    Sigma_q_phi.slice(i) = eye(d_,d_);
  }
  arma::mat Mu_q_phi = zeros(d,d_);
  
  arma::mat Mu_q_beta = zeros(d,d);
  arma::cube Sigma_q_beta = zeros(d,d,d);
  arma::mat Omega_hat = eye(d,d);
  
  arma::vec mu_q_nu = ones(d);
  arma::vec a_q_nu = (n/2+a_nu)*ones(d);
  arma::vec b_q_nu = ones(d);
  
  arma::mat Mu_q_tau = 0.1*trimatl(ones(d,d));
  Mu_q_tau.diag() = zeros(d);
  arma::mat Mu_q_recip_tau = Mu_q_tau;
  
  arma::mat Mu_q_lam = Mu_q_tau;
  
  arma::mat Mu_q_recip_ups = ones(d,d_);
  arma::mat Mu_q_ups = ones(d,d_);
  
  arma::mat Mu_q_kappa = ones(d,d_);
  
  /* Define useful quantities */
  arma::mat Y = D.cols(1,n);
  arma::vec y = vectorise(Y.t());
  arma::mat Z = X.cols(0,n-1);
  arma::mat ZZ_t = Z*Z.t();
  arma::mat Id = eye(d,d);
  
  /* Convergence parameters */
  int converged = 0;
  int itNum = 0;
  double eps_det = 1e-8;
  
  arma::vec VecOld = join_vert(join_vert(mu_q_nu,vectorise(Mu_q_phi)),
                               vectorise(Mu_q_beta));
  arma::vec lowerBoundIter = ones(maxIter+1);
  
  /* Start Updating */
  while(converged == 0) {
    /* Update iteration and initialize ELBO (with only constants) */
    itNum = itNum + 1;
    double lowerBound = d*(-0.5*n*log(2*datum::pi) +a_nu*log(b_nu) -lgamma(a_nu)) +
      0.5*d*(d-1)*0.5 +0.5*d*d +
      0.5*d*(d-1)*(e1*log(e2) - lgamma(e1)) +
      d*d*(h1*log(h2) - lgamma(h1));
    
    arma::mat Yj = Y.row(0);
    /* Update NU_1 */
    arma::rowvec Q = Yj - Mu_q_phi.row(0)*Z;
    arma::mat W = Q*Q.t() + trace(ZZ_t*Sigma_q_phi.slice(0));
    b_q_nu(0) = b_nu + W(0,0)/2;
    mu_q_nu(0) = a_q_nu(0)/b_q_nu(0);
    /* Update ELBO */
    lowerBound = lowerBound - (a_q_nu(0)*log(b_q_nu(0))-lgamma(a_q_nu(0)));
    
    /* Update J-th regression */
    for (int j = 1; j < d; j++) {
      
      /* Get matrix K and vector k */
      arma::mat K_phi = zeros(j,j);
      for (int i = 0; i < j; i++) {
        K_phi(i,i) = trace(Sigma_q_phi.slice(i)*ZZ_t);
      }
      
      /* Update BETA_j */
      arma::mat Q = Y.rows(0,j-1) - Mu_q_phi.rows(0,j-1)*Z;
      arma::mat Sigma_beta_w = inv_sympd(mu_q_nu(j)*(Q*Q.t() + K_phi) + 
        diagmat(Mu_q_recip_tau.submat(j,0,j,j-1)));
      Sigma_q_beta.slice(j).submat(0,0,j-1,j-1) = Sigma_beta_w;
      arma::vec mu_beta_w = Sigma_beta_w*(mu_q_nu(j)*(Q*(Y.row(j) - Mu_q_phi.row(j)*Z).t()));
      Mu_q_beta.submat(j,0,j,j-1) = mu_beta_w.t();
      arma::vec mu_q_betasq = Sigma_beta_w.diag() + pow(mu_beta_w,2);
      /* Update ELBO */
      lowerBound = lowerBound + 0.5*log(det(Sigma_beta_w)+eps_det);
      
      arma::mat Yj = Y.row(j);
      /* Update NU_j */
      arma::mat Sigma_phi_j = Sigma_q_phi.slice(j);
      arma::rowvec S = Yj - mu_beta_w.t()*Q - Mu_q_phi.row(j)*Z;
      arma::mat W = S*S.t() + trace(ZZ_t*Sigma_phi_j) + mu_beta_w.t()*K_phi*mu_beta_w + trace(Sigma_beta_w*Q*Q.t()) + 
        trace(Sigma_beta_w*K_phi);
      b_q_nu(j) = b_nu + W(0,0)/2;
      mu_q_nu(j) = a_q_nu(j)/b_q_nu(j);
      /* Update ELBO */
      lowerBound = lowerBound - (a_q_nu(j)*log(b_q_nu(j))-lgamma(a_q_nu(j)));
      
      /* Update TAU_j */
      for (int k = 0; k < j; k++) { 
        Mu_q_recip_tau(j,k) = sqrt(Mu_q_lam(j,k)/mu_q_betasq(k));
        Mu_q_tau(j,k) = sqrt(mu_q_betasq(k)/Mu_q_lam(j,k))+1/Mu_q_lam(j,k);
        /* Update ELBO */
        lowerBound = lowerBound - (0.25*log(Mu_q_lam(j,k)/mu_q_betasq(k))-log(boost::math::cyl_bessel_k(0.5, sqrt(Mu_q_lam(j,k)*mu_q_betasq(k)))));
        
        /* Update LAM_j */
        Mu_q_lam(j,k) = (1+e1)/(Mu_q_tau(j,k)/2+e2);
        /* Update ELBO */
        lowerBound = lowerBound - (1+e1)*log(Mu_q_tau(j,k)/2+e2) +lgamma(1+e1);
        lowerBound = lowerBound + 0.5*Mu_q_tau(j,k)*Mu_q_lam(j,k);
      }
    }
    
    /* Get matrix C */
    arma::mat C = zeros(d,d);
    for (int i0 = 0; i0 < d; i0++) C = C + mu_q_nu(i0)*Sigma_q_beta.slice(i0);
    arma::mat Mu_omega = (Id - Mu_q_beta).t()*diagmat(mu_q_nu)*(Id - Mu_q_beta);
    Omega_hat = Mu_omega + C;
    
    arma::vec Mu = kron(Omega_hat,Z)*y;
    
    /* Update PHI_j */
    for (int j = 0; j < d; j++) {
      Sigma_q_phi.slice(j) = inv_sympd(Omega_hat(j,j)*ZZ_t + diagmat(Mu_q_recip_ups.row(j)));
      
      arma::rowvec Om_jbarj = Omega_hat.row(j);
      Om_jbarj.shed_col(j);
      
      arma::mat Mu_phi_w = Mu_q_phi;
      Mu_phi_w.shed_row(j);
      arma::vec mu_phi_jbar = vectorise(Mu_phi_w.t());
      
      arma::vec Mu_j = Mu.subvec(j*d_,j*d_+d_-1);
      
      arma::vec mu_q_phi_j = Sigma_q_phi.slice(j)*(Mu_j-kron(Om_jbarj,ZZ_t)*mu_phi_jbar);
      Mu_q_phi.row(j) = mu_q_phi_j.t();
      arma::vec mu_q_phisq = Sigma_q_phi.slice(j).diag() + pow(mu_q_phi_j,2);
      /* Update ELBO */
      lowerBound = lowerBound + 0.5*log(det(Sigma_q_phi.slice(j))+eps_det);
      
      /* Update UPS_j */
      Mu_q_recip_ups.row(j) = sqrt(Mu_q_kappa.row(j)/mu_q_phisq.t());
      Mu_q_ups.row(j) = sqrt(mu_q_phisq.t()/Mu_q_kappa.row(j))+1/Mu_q_kappa.row(j);
      /* Update ELBO */
      
      
      /* Update KAPPA */
      Mu_q_kappa.row(j) = (1+h1)/(0.5*Mu_q_ups.row(j)+h2);
      /* Update ELBO */
      lowerBound = lowerBound - (1+h1)*sum(log(Mu_q_ups.row(j)/2+h2)) +d_*lgamma(1+h1);
      
    }
    
    /* Store iteration results */
    lowerBoundIter(itNum) = lowerBound;
    arma::vec VecNew = join_vert(join_vert(mu_q_nu,vectorise(Mu_q_phi)),
                                 vectorise(Mu_q_beta));
    
    /* Check convergence */
    if (itNum > 1) {
      double lowerBoundOld = lowerBoundIter(itNum-2);
      double delta_Par = max(abs((VecNew - VecOld)/VecOld));
      double delta_Elbo = abs((lowerBound - lowerBoundOld)/lowerBoundOld);
      
      if (delta_Par < Tol_Par)  if (delta_Elbo < Tol_ELBO) converged = 1;
      if (Trace == 1) {
        cout << "Iteration number:" << itNum << "; Parameter R.E:" << delta_Par << "| ELBO R.I:" << delta_Elbo << endl;
      }
    }
    
    if (itNum == maxIter) converged = 1;
    VecOld = VecNew;
  }
  
  /* Return results */
  out["Omega_hat"] = Omega_hat;
  
  out["a_q_nu"] = a_q_nu;
  out["b_q_nu"] = b_q_nu;
  out["mu_q_nu"] = mu_q_nu;
  
  out["Mu_q_beta"] = Mu_q_beta;
  out["Sigma_q_beta"] = Sigma_q_beta;
  
  out["Mu_q_theta"] = Mu_q_phi;
  out["Sigma_q_theta"] = Sigma_q_phi;
  
  out["lowerBoundIter"] = lowerBoundIter.subvec(1,itNum);
  out["lowerBound"] = lowerBoundIter(itNum);
  out["convergence"] = converged;
  
  return out;
}



// [[Rcpp::export]]
Rcpp::List fVBmlrNG(arma::mat D, 
                    arma::mat X, 
                    Rcpp::List hyper,
                    int maxIter = 500, 
                    double Tol_ELBO = 1e-2, 
                    double Tol_Par = 1e-2, 
                    int Trace = 0) {
  
  Rcpp::List out;
  /* Get dimensions */
  double d = D.n_rows;
  double n = D.n_cols-1;
  double d_ = X.n_rows;
  
  /* Get Hyperparameters */
  double a_nu = hyper["a_nu"]; 
  double b_nu = hyper["b_nu"]; 
  
  double e1 = hyper["e1"];    
  double e2 = hyper["e2"]; 
  double e3 = hyper["e3"];  
  
  double h1 = hyper["h1"];     
  double h2 = hyper["h2"]; 
  double h3 = hyper["h3"]; 
  
  /* Initialize optimal quantities */
  arma::cube Sigma_q_phi = zeros(d_,d_,d);
  for (int i = 0; i < d; i++) {
    Sigma_q_phi.slice(i) = eye(d_,d_);
  }
  arma::mat Mu_q_phi = zeros(d,d_);
  
  arma::mat Mu_q_beta = zeros(d,d);
  arma::cube Sigma_q_beta = zeros(d,d,d);
  arma::mat Omega_hat = eye(d,d);
  
  arma::vec mu_q_nu = ones(d);
  arma::vec a_q_nu = (n/2+a_nu)*ones(d);
  arma::vec b_q_nu = ones(d);
  
  arma::mat Mu_q_tau = 0.1*trimatl(ones(d,d));
  Mu_q_tau.diag() = zeros(d);
  arma::mat Mu_q_recip_tau = 1/Mu_q_tau;
  arma::mat Mu_q_logtau = log(Mu_q_tau);
  
  arma::mat Mu_q_lam = trimatl(ones(d,d));
  Mu_q_lam.diag() = zeros(d);    
  arma::mat Mu_q_loglam = Mu_q_lam;
  
  arma::vec mu_q_eta = 0.8*ones(d);
  
  arma::mat Mu_q_ups = 0.1*ones(d,d_);
  arma::mat Mu_q_recip_ups = Mu_q_ups;
  arma::mat Mu_q_logups = log(Mu_q_ups);
  
  arma::mat Mu_q_kappa = ones(d,d_);
  arma::mat Mu_q_logkappa = ones(d,d_);
  
  arma::vec mu_q_xi = 0.8*ones(d);
  
  /* Define useful quantities */
  arma::mat Y = D.cols(1,n);
  arma::vec y = vectorise(Y.t());
  arma::mat Z = X.cols(0,n-1);
  arma::mat ZZ_t = Z*Z.t();
  arma::mat Id = eye(d,d);
  
  /* Convergence parameters */
  int converged = 0;
  int itNum = 0;
  double eps_det = 1e-8;
  
  arma::vec VecOld = join_vert(join_vert(mu_q_nu,vectorise(Mu_q_phi)),
                               vectorise(Mu_q_beta));
  arma::vec lowerBoundIter = ones(maxIter+1);
  
  /* Start Updating */
  while(converged == 0) {
    /* Update iteration and initialize ELBO (with only constants) */
    itNum = itNum + 1;
    double lowerBound = d*(-0.5*n*log(2*datum::pi) +a_nu*log(b_nu) -lgamma(a_nu)) +
      0.5*d*(d-1)*0.5 +0.5*d*d +
      0.5*d*(d-1)*(e1*log(e2) - lgamma(e1) + log(e3)) +
      d*d*(h1*log(h2) - lgamma(h1) + log(h3));
    
    arma::mat Yj = Y.row(0);
    /* Update NU_1 */
    arma::rowvec Q = Yj - Mu_q_phi.row(0)*Z;
    arma::mat W = Q*Q.t() + trace(ZZ_t*Sigma_q_phi.slice(0));
    b_q_nu(0) = b_nu + W(0,0)/2;
    mu_q_nu(0) = a_q_nu(0)/b_q_nu(0);
    /* Update ELBO */
    lowerBound = lowerBound - (a_q_nu(0)*log(b_q_nu(0))-lgamma(a_q_nu(0)));
    
    /* Update J-th regression */
    for (int j = 1; j < d; j++) {
      
      /* Get matrix K and vector k */
      arma::mat K_phi = zeros(j,j);
      for (int i = 0; i < j; i++) {
        K_phi(i,i) = trace(Sigma_q_phi.slice(i)*ZZ_t);
      }
      
      /* Update BETA_j */
      arma::mat Q = Y.rows(0,j-1) - Mu_q_phi.rows(0,j-1)*Z;
      arma::mat Sigma_beta_w = inv_sympd(mu_q_nu(j)*(Q*Q.t() + K_phi) + 
        diagmat(Mu_q_recip_tau.submat(j,0,j,j-1)));
      Sigma_q_beta.slice(j).submat(0,0,j-1,j-1) = Sigma_beta_w;
      arma::vec mu_beta_w = Sigma_beta_w*(mu_q_nu(j)*(Q*(Y.row(j) - Mu_q_phi.row(j)*Z).t()));
      Mu_q_beta.submat(j,0,j,j-1) = mu_beta_w.t();
      arma::vec mu_q_betasq = Sigma_beta_w.diag() + pow(mu_beta_w,2);
      /* Update ELBO */
      lowerBound = lowerBound + 0.5*log(det(Sigma_beta_w)+eps_det);
      
      arma::mat Yj = Y.row(j);
      /* Update NU_j */
      arma::mat Sigma_phi_j = Sigma_q_phi.slice(j);
      arma::rowvec S = Yj - mu_beta_w.t()*Q - Mu_q_phi.row(j)*Z;
      arma::mat W = S*S.t() + trace(ZZ_t*Sigma_phi_j) + mu_beta_w.t()*K_phi*mu_beta_w + trace(Sigma_beta_w*Q*Q.t()) + 
        trace(Sigma_beta_w*K_phi);
      b_q_nu(j) = b_nu + W(0,0)/2;
      mu_q_nu(j) = a_q_nu(j)/b_q_nu(j);
      /* Update ELBO */
      lowerBound = lowerBound - (a_q_nu(j)*log(b_q_nu(j))-lgamma(a_q_nu(j)));
      
      /* Update TAU_j */
      double zeta_q_tau = mu_q_eta(j)-0.5;
      arma::rowvec a_q_tau = mu_q_eta(j)*Mu_q_lam.row(j)+1e-4;
      arma::vec b_q_tau = mu_q_betasq;
      
      for (int k = 0; k < j; k++) { 
        Mu_q_tau(j,k) = sqrt(b_q_tau(k)/a_q_tau(k)) *
          boost::math::cyl_bessel_k(zeta_q_tau+1.0, sqrt(a_q_tau(k)*b_q_tau(k))) /
            boost::math::cyl_bessel_k(zeta_q_tau, sqrt(a_q_tau(k)*b_q_tau(k)));
        Mu_q_recip_tau(j,k) = sqrt(a_q_tau(k)/b_q_tau(k)) * 
          boost::math::cyl_bessel_k(zeta_q_tau+1.0, sqrt(a_q_tau(k)*b_q_tau(k))) /
            boost::math::cyl_bessel_k(zeta_q_tau, sqrt(a_q_tau(k)*b_q_tau(k))) - 
              2*zeta_q_tau/b_q_tau(k);
        Mu_q_logtau(j,k) = log(sqrt(b_q_tau(k)/a_q_tau(k))) + grad_cpp(zeta_q_tau, sqrt(a_q_tau(k)*b_q_tau(k)));
        
        /* Update ELBO */
        lowerBound = lowerBound - zeta_q_tau/2*log(a_q_tau(k)/b_q_tau(k)) + 
        log(2*boost::math::cyl_bessel_k(zeta_q_tau, sqrt(a_q_tau(k)*b_q_tau(k))));
        
        /* Update LAM_j */
        Mu_q_lam(j,k) = (mu_q_eta(j) + e1)/(0.5*mu_q_eta(j)*Mu_q_tau(j,k) + e2);
        Mu_q_loglam(j,k) = -log(0.5*mu_q_eta(j)*Mu_q_tau(j,k) + e2) + boost::math::digamma(mu_q_eta(j) + e1);
        /* Update ELBO */
        lowerBound = lowerBound - (mu_q_eta(j) + e1)*log(0.5*mu_q_eta(j)*Mu_q_tau(j,k) + e2) +lgamma(mu_q_eta(j) + e1);
      }
      
      /* Update ETA_j */
      Rcpp::List out_eta = integrate_latent_factor(j,
                                                   sum(Mu_q_tau(j,span(0,j-1))%Mu_q_lam(j,span(0,j-1))),
                                                   sum(Mu_q_logtau(j,span(0,j-1))),
                                                   sum(Mu_q_loglam(j,span(0,j-1))),e3);
      mu_q_eta(j) = out_eta["value"];
      double c_eta = out_eta["const_int"];
      /* Update ELBO */
      lowerBound = lowerBound +log(c_eta) +mu_q_eta(j)*sum(Mu_q_lam(j,span(0,j-1))%Mu_q_tau(j,span(0,j-1))-(Mu_q_loglam(j,span(0,j-1))+Mu_q_logtau(j,span(0,j-1))));
      
    }
    
    /* Get matrix C */
    arma::mat C = zeros(d,d);
    for (int i0 = 0; i0 < d; i0++) C = C + mu_q_nu(i0)*Sigma_q_beta.slice(i0);
    arma::mat Mu_omega = (Id - Mu_q_beta).t()*diagmat(mu_q_nu)*(Id - Mu_q_beta);
    Omega_hat = Mu_omega + C;
    
    arma::vec Mu = kron(Omega_hat,Z)*y;
    
    /* Update PHI_j */
    for (int j = 0; j < d; j++) {
      Sigma_q_phi.slice(j) = inv_sympd(Omega_hat(j,j)*ZZ_t + diagmat(Mu_q_recip_ups.row(j)));
      
      arma::rowvec Om_jbarj = Omega_hat.row(j);
      Om_jbarj.shed_col(j);
      
      arma::mat Mu_phi_w = Mu_q_phi;
      Mu_phi_w.shed_row(j);
      arma::vec mu_phi_jbar = vectorise(Mu_phi_w.t());
      
      arma::vec Mu_j = Mu.subvec(j*d_,j*d_+d_-1);
      
      arma::vec mu_q_phi_j = Sigma_q_phi.slice(j)*(Mu_j-kron(Om_jbarj,ZZ_t)*mu_phi_jbar);
      Mu_q_phi.row(j) = mu_q_phi_j.t();
      arma::vec mu_q_phisq = Sigma_q_phi.slice(j).diag() + pow(mu_q_phi_j,2);
      /* Update ELBO */
      lowerBound = lowerBound + 0.5*log(det(Sigma_q_phi.slice(j))+eps_det);
      
      /* Update UPS */
      double zeta_q_ups = mu_q_xi(j) - 0.5;
      arma::rowvec a_q_ups = mu_q_xi(j)*Mu_q_kappa.row(j)+1e-4;
      arma::vec b_q_ups = mu_q_phisq;
      
      for (int j0 = 0; j0 < d_; j0++) {
        Mu_q_ups(j,j0) = sqrt(b_q_ups(j0)/a_q_ups(j0)) *
          boost::math::cyl_bessel_k(zeta_q_ups+1, sqrt(a_q_ups(j0)*b_q_ups(j0))) /
            boost::math::cyl_bessel_k(zeta_q_ups, sqrt(a_q_ups(j0)*b_q_ups(j0)));
        Mu_q_recip_ups(j,j0) = sqrt(a_q_ups(j0)/b_q_ups(j0)) * 
          boost::math::cyl_bessel_k(zeta_q_ups+1, sqrt(a_q_ups(j0)*b_q_ups(j0))) /
            boost::math::cyl_bessel_k(zeta_q_ups, sqrt(a_q_ups(j0)*b_q_ups(j0))) - 
              2*zeta_q_ups/b_q_ups(j0);
        Mu_q_logups(j,j0) = log(sqrt(b_q_ups(j0)/a_q_ups(j0))) + grad_cpp(zeta_q_ups, sqrt(a_q_ups(j0)*b_q_ups(j0)));
        
        /* Update ELBO */
        lowerBound = lowerBound - zeta_q_ups/2*log(a_q_ups(j0)/b_q_ups(j0)) +
        log(2*boost::math::cyl_bessel_k(zeta_q_ups, sqrt(a_q_ups(j0)*b_q_ups(j0))));
        
        /* Update KAPPA */
        double a_q_kappa = mu_q_xi(j) + h1;
        double b_q_kappa = 0.5*mu_q_xi(j)*Mu_q_ups(j,j0) + h2;
        
        Mu_q_kappa(j,j0) = a_q_kappa/b_q_kappa;
        Mu_q_logkappa(j,j0) = -log(b_q_kappa) + boost::math::digamma(a_q_kappa);
        
        /* Update ELBO */
        lowerBound = lowerBound + (-a_q_kappa*log(b_q_kappa) + lgamma(a_q_kappa)); 
      }
      
      /* Update XI */
      Rcpp::List out_xi = integrate_latent_factor(d_,sum(Mu_q_ups.row(j)%Mu_q_kappa.row(j)),
                                                  sum(Mu_q_logups.row(j)),sum(Mu_q_logkappa.row(j)),h3);
      mu_q_xi(j) = out_xi["value"];
      double c_xi = out_xi["const_int"];
      
      /* Update ELBO */
      lowerBound = lowerBound + log(c_xi) +
      mu_q_xi(j)*(sum(Mu_q_ups.row(j)%Mu_q_kappa.row(j)) - (sum(Mu_q_logups.row(j))+sum(Mu_q_logkappa.row(j)))); 
    }
    
    /* Store iteration results */
    lowerBoundIter(itNum) = lowerBound;
    arma::vec VecNew = join_vert(join_vert(mu_q_nu,vectorise(Mu_q_phi)),
                                 vectorise(Mu_q_beta));
    
    /* Check convergence */
    if (itNum > 1) {
      double lowerBoundOld = lowerBoundIter(itNum-2);
      double delta_Par = max(abs((VecNew - VecOld)/VecOld));
      double delta_Elbo = abs((lowerBound - lowerBoundOld)/lowerBoundOld);
      
      if (delta_Par < Tol_Par)  if (delta_Elbo < Tol_ELBO) converged = 1;
      if (Trace == 1) {
        cout << "Iteration number:" << itNum << "; Parameter R.E:" << delta_Par << "| ELBO R.I:" << delta_Elbo << endl;
      }
    }
    
    if (itNum == maxIter) converged = 1;
    VecOld = VecNew;
  }
  
  /* Return results */
  out["Omega_hat"] = Omega_hat;
  
  out["a_q_nu"] = a_q_nu;
  out["b_q_nu"] = b_q_nu;
  out["mu_q_nu"] = mu_q_nu;
  
  out["Mu_q_beta"] = Mu_q_beta;
  out["Sigma_q_beta"] = Sigma_q_beta;
  
  out["Mu_q_theta"] = Mu_q_phi;
  out["Sigma_q_theta"] = Sigma_q_phi;
  
  out["lowerBoundIter"] = lowerBoundIter.subvec(1,itNum);
  out["lowerBound"] = lowerBoundIter(itNum);
  out["convergence"] = converged;
  
  return out;
}



// [[Rcpp::export]]
Rcpp::List fVBmlrHS(arma::mat D, 
                    arma::mat X,
                    Rcpp::List hyper,
                    int maxIter = 500, 
                    double Tol_ELBO = 1e-2, 
                    double Tol_Par = 1e-2, 
                    int Trace = 0) {
  
  Rcpp::List out;
  /* Get dimensions */
  double d = D.n_rows;
  double n = D.n_cols-1;
  double d_ = X.n_rows;
  
  /* Get Hyperparameters */
  double a_nu = hyper["a_nu"]; 
  double b_nu = hyper["b_nu"]; 
  
  /* Initialize optimal quantities */
  arma::cube Sigma_q_phi = zeros(d_,d_,d);
  for (int i = 0; i < d; i++) {
    Sigma_q_phi.slice(i) = eye(d_,d_);
  }
  arma::mat Mu_q_phi = zeros(d,d_);
  
  arma::mat Mu_q_beta = zeros(d,d);
  arma::cube Sigma_q_beta = zeros(d,d,d);
  arma::mat Omega_hat = eye(d,d);
  
  arma::vec mu_q_nu = ones(d);
  arma::vec a_q_nu = (n/2+a_nu)*ones(d);
  arma::vec b_q_nu = ones(d);
  
  arma::mat Mu_q_recip_tau = trimatl(ones(d,d));
  Mu_q_recip_tau.diag() = zeros(d);
  double mu_q_recip_gamma = 1;
  arma::mat Mu_q_recip_lam = trimatl(ones(d,d));
  Mu_q_recip_lam.diag() = zeros(d);
  double mu_q_recip_eta = 1;
  
  arma::mat Mu_q_recip_ups = ones(d,d_);
  arma::vec mu_q_recip_delta = ones(d);
  arma::mat Mu_q_recip_kappa = ones(d,d_);
  arma::vec mu_q_recip_xi = ones(d);
  
  /* Define useful quantities */
  arma::mat Y = D.cols(1,n);
  arma::vec y = vectorise(Y.t());
  arma::mat Z = X.cols(0,n-1);
  arma::mat ZZ_t = Z*Z.t();
  arma::mat Id = eye(d,d);
  
  /* Convergence parameters */
  int converged = 0;
  int itNum = 0;
  double eps_det = 1e-8;
  
  arma::vec VecOld = join_vert(join_vert(mu_q_nu,vectorise(Mu_q_phi)),
                               vectorise(Mu_q_beta));
  arma::vec lowerBoundIter = ones(maxIter+1);
  
  /* Start Updating */
  while(converged == 0) {
    /* Update iteration and initialize ELBO (with only constants) */
    itNum = itNum + 1;
    double lowerBound = 0;
    
    arma::mat Yj = Y.row(0);
    /* Update NU_1 */
    arma::rowvec Q = Yj - Mu_q_phi.row(0)*Z;
    arma::mat W = Q*Q.t() + trace(ZZ_t*Sigma_q_phi.slice(0));
    b_q_nu(0) = b_nu + W(0,0)/2;
    mu_q_nu(0) = a_q_nu(0)/b_q_nu(0);
    /* Update ELBO */
    lowerBound = lowerBound - (a_q_nu(0)*log(b_q_nu(0))-lgamma(a_q_nu(0)));
    
    /* Update J-th regression */
    double sum_gamma = 0;
    for (int j = 1; j < d; j++) {
      
      /* Get matrix K and vector k */
      arma::mat K_phi = zeros(j,j);
      for (int i = 0; i < j; i++) {
        K_phi(i,i) = trace(Sigma_q_phi.slice(i)*ZZ_t);
      }
      
      /* Update BETA_j */
      arma::mat Q = Y.rows(0,j-1) - Mu_q_phi.rows(0,j-1)*Z;
      arma::mat Sigma_beta_w = inv_sympd(mu_q_nu(j)*(Q*Q.t() + K_phi) + 
        mu_q_recip_gamma*diagmat(Mu_q_recip_tau.submat(j,0,j,j-1)));
      Sigma_q_beta.slice(j).submat(0,0,j-1,j-1) = Sigma_beta_w;
      arma::vec mu_beta_w = Sigma_beta_w*(mu_q_nu(j)*(Q*(Y.row(j) - Mu_q_phi.row(j)*Z).t()));
      Mu_q_beta.submat(j,0,j,j-1) = mu_beta_w.t();
      arma::vec mu_q_betasq = Sigma_beta_w.diag() + pow(mu_beta_w,2);
      /* Update ELBO */
      lowerBound = lowerBound + 0.5*log(det(Sigma_beta_w)+eps_det);
      
      arma::mat Yj = Y.row(j);
      /* Update NU_j */
      arma::mat Sigma_phi_j = Sigma_q_phi.slice(j);
      arma::rowvec S = Yj - mu_beta_w.t()*Q - Mu_q_phi.row(j)*Z;
      arma::mat W = S*S.t() + trace(ZZ_t*Sigma_phi_j) + mu_beta_w.t()*K_phi*mu_beta_w + trace(Sigma_beta_w*Q*Q.t()) + 
        trace(Sigma_beta_w*K_phi);
      b_q_nu(j) = b_nu + W(0,0)/2;
      mu_q_nu(j) = a_q_nu(j)/b_q_nu(j);
      /* Update ELBO */
      lowerBound = lowerBound - (a_q_nu(j)*log(b_q_nu(j))-lgamma(a_q_nu(j)));
      
      /* Update TAU_j */
      for (int k = 0; k < j; k++) { 
        Mu_q_recip_tau(j,k) = 1/(0.5*mu_q_betasq(k)*mu_q_recip_gamma+Mu_q_recip_lam(j,k));
        
        /* Update LAM_j */
        Mu_q_recip_lam(j,k) = 1/(1+Mu_q_recip_tau(j,k));
        
        /* Update ELBO */
        lowerBound = lowerBound + Mu_q_recip_tau(j,k)*Mu_q_recip_lam(j,k) -
        (log(1+Mu_q_recip_tau(j,k)) +log(0.5*mu_q_betasq(k)*mu_q_recip_gamma+Mu_q_recip_lam(j,k)) +log(datum::pi));
      }
      
      sum_gamma = sum_gamma + sum(mu_q_betasq.t()%Mu_q_recip_tau.submat(j,0,j,j-1));
    }
    
    /* Update GAMMA */
    mu_q_recip_gamma = 0.5*(0.5*d*(d-1)+1)/(0.5*sum_gamma+mu_q_recip_eta);
    
    /* Update ETA_j */
    mu_q_recip_eta = 1/(1+mu_q_recip_gamma);
    
    /* Update ELBO */
    lowerBound = lowerBound + mu_q_recip_gamma*(mu_q_recip_eta+sum_gamma) -
    (0.5*(0.5*d*(d-1)+1)*log(0.5*sum_gamma+mu_q_recip_eta) +log(1+mu_q_recip_gamma) +log(datum::pi));
    
    /* Get matrix C */
    arma::mat C = zeros(d,d);
    for (int i0 = 0; i0 < d; i0++) C = C + mu_q_nu(i0)*Sigma_q_beta.slice(i0);
    arma::mat Mu_omega = (Id - Mu_q_beta).t()*diagmat(mu_q_nu)*(Id - Mu_q_beta);
    Omega_hat = Mu_omega + C;
    
    arma::vec Mu = kron(Omega_hat,Z)*y;
    
    /* Update PHI_j */
    for (int j = 0; j < d; j++) {
      Sigma_q_phi.slice(j) = inv_sympd(Omega_hat(j,j)*ZZ_t + diagmat(mu_q_recip_delta(j)*Mu_q_recip_ups.row(j)));
      
      arma::rowvec Om_jbarj = Omega_hat.row(j);
      Om_jbarj.shed_col(j);
      
      arma::mat Mu_phi_w = Mu_q_phi;
      Mu_phi_w.shed_row(j);
      arma::vec mu_phi_jbar = vectorise(Mu_phi_w.t());
      
      arma::vec Mu_j = Mu.subvec(j*d_,j*d_+d_-1);
      
      arma::vec mu_q_phi_j = Sigma_q_phi.slice(j)*(Mu_j-kron(Om_jbarj,ZZ_t)*mu_phi_jbar);
      Mu_q_phi.row(j) = mu_q_phi_j.t();
      arma::vec mu_q_phisq = Sigma_q_phi.slice(j).diag() + pow(mu_q_phi_j,2);
      /* Update ELBO */
      lowerBound = lowerBound + 0.5*log(det(Sigma_q_phi.slice(j))+eps_det);
      
      /* Update UPS_j */
      Mu_q_recip_ups.row(j) = 1/(0.5*mu_q_phisq.t()*mu_q_recip_delta(j)+Mu_q_recip_kappa.row(j));
      /* Update ELBO */
      
      /* Update KAPPA_j */
      Mu_q_recip_kappa.row(j) = 1/(1+Mu_q_recip_ups.row(j));
      /* Update ELBO */
      
      
      /* Update DELTA */
      mu_q_recip_delta(j) = (0.5*(d+1))/(0.5*sum(mu_q_phisq.t()%Mu_q_recip_ups.row(j))+mu_q_recip_xi(j));
      /* Update ELBO */
      
      
      /* Update XI_j */
      mu_q_recip_xi(j) = 1/(1+mu_q_recip_delta(j));
      /* Update ELBO */
    }
    
    /* Store iteration results */
    lowerBoundIter(itNum) = lowerBound;
    arma::vec VecNew = join_vert(join_vert(mu_q_nu,vectorise(Mu_q_phi)),
                                 vectorise(Mu_q_beta));
    
    /* Check convergence */
    if (itNum > 1) {
      double lowerBoundOld = lowerBoundIter(itNum-2);
      double delta_Par = max(abs((VecNew - VecOld)/VecOld));
      double delta_Elbo = abs((lowerBound - lowerBoundOld)/lowerBoundOld);
      
      if (delta_Par < Tol_Par)  if (delta_Elbo < Tol_ELBO) converged = 1;
      if (Trace == 1) {
        cout << "Iteration number:" << itNum << "; Parameter R.E:" << delta_Par << "| ELBO R.I:" << delta_Elbo << endl;
      }
    }
    
    if (itNum == maxIter) converged = 1;
    VecOld = VecNew;
  }
  
  /* Return results */
  out["Omega_hat"] = Omega_hat;
  
  out["a_q_nu"] = a_q_nu;
  out["b_q_nu"] = b_q_nu;
  out["mu_q_nu"] = mu_q_nu;
  
  out["Mu_q_beta"] = Mu_q_beta;
  out["Sigma_q_beta"] = Sigma_q_beta;
  
  out["Mu_q_theta"] = Mu_q_phi;
  out["Sigma_q_theta"] = Sigma_q_phi;
  
  out["lowerBoundIter"] = lowerBoundIter.subvec(1,itNum);
  out["lowerBound"] = lowerBoundIter(itNum);
  out["convergence"] = converged;
  
  return out;
}



// [[Rcpp::export]]
Rcpp::List fVBmlrSV(arma::mat D, 
                    arma::mat X,
                    Rcpp::List hyper, 
                    int maxIter = 500, 
                    double Tol_ELBO = 1e-2, 
                    double Tol_Par = 1e-2, 
                    int Trace = 0) {
  
  Rcpp::List out;
  /* Get dimensions */
  double d = D.n_rows;
  double n = D.n_cols-1;
  double d_ = X.n_rows;
  
  /* Non sparse prior */
  
  /* Get Hyperparameters */
  double a_om = hyper["a_om"]; 
  double b_om = hyper["b_om"]; 
  
  double tau = hyper["tau"];
  double ups = hyper["ups"]; 
  
  /* Initialize */
  arma::cube Sigma_q_phi = zeros(d_,d_,d);
  for (int i = 0; i < d; i++) {
    Sigma_q_phi.slice(i) = eye(d_,d_);
  }
  arma::mat Mu_q_phi = zeros(d,d_);
  
  arma::mat Mu_q_beta = zeros(d,d);
  arma::cube Sigma_q_beta = zeros(d,d,d);
  
  arma::cube Omega_hat = zeros(d,d,n);
  arma::cube Sigma_hat = zeros(d,d,n);
  
  arma::mat mu_q_nu = ones(d,n);
  arma::mat mu_q_h = ones(d,n+1);
  arma::cube Sigma_q_h = zeros(n+1,n+1,d);
  arma::mat sigma2_q_h = ones(d,n+1);
  
  arma::vec mu_q_om2inv = 3*ones(d);
  
  /* Get useful quantities */
  arma::mat Y = D.cols(1,n);
  arma::vec y = vectorise(Y.t());
  arma::mat Z = X.cols(0,n-1);
  arma::cube ZZ_t = zeros(d_,d_,n);
  for (int t = 0; t < n; t++) ZZ_t.slice(t) = Z.col(t)*Z.col(t).t();
  arma::rowvec ones_row = ones(n).t();
  arma::vec ones_col = ones(n);
  arma::mat In = eye(n,n);
  arma::mat Id = eye(d,d);
  arma::mat mS = zeros(d,n);
  
  /* Set convergence criterion */
  int converged = 0;
  int itNum = 0;
  double eps_det = 1e-8;
  
  arma::vec lowerBoundIter = zeros(maxIter+1);
  arma::vec VecOld = join_vert(vectorise(Mu_q_phi),
                               vectorise(Mu_q_beta));
  
  /* Start Updating */
  while(converged == 0) {
    /* Update iteration and initialize ELBO (with only constants) */
    itNum = itNum + 1;
    double lowerBound = 0;
    
    /* Update H_1 */
    mS.row(0) = (Y.row(0)-Mu_q_phi.row(0)*Z)%(Y.row(0)-Mu_q_phi.row(0)*Z);
    for (int t = 0; t < n; t++) {
      mS(0,t) = mS(0,t) + trace(ZZ_t.slice(t)*Sigma_q_phi.slice(0));
    }
    Rcpp::List opt_out0 = wand2014(mu_q_h.row(0),
                                   Sigma_q_h.slice(0),
                                   sigma2_q_h.row(0),
                                   mS.row(0),
                                   mu_q_om2inv(0));
    arma::vec mh0 = opt_out0["f"];
    arma::mat Sh0 = opt_out0["S"];
    arma::vec sh0 = opt_out0["s2"];
    
    mu_q_h.row(0) = mh0.t();
    Sigma_q_h.slice(0) = Sh0;
    sigma2_q_h.row(0) = sh0.t();
    
    /* Update NU_1 */
    mu_q_nu.row(0) = exp(-mh0(span(1,n)).t()+0.5*sh0(span(1,n)).t());
    
    /* Update OMEGA^2_1 */
    double A_q_om0 = a_om + 0.5*(n+1);
    double B_q_om0 = b_om + 0.5*as_scalar(pow(mh0(0),2)+pow(mh0(n),2)+2*sum(pow(mh0(span(1,n-1)),2)) -
                                          2*sum((mh0(span(0,n-1)))%(mh0(span(1,n)))) +
                                          (sh0(0)+sh0(n)+2*sum(sh0(span(1,n-1)))-2*sum(Sh0.diag(1))));
    mu_q_om2inv(0) = A_q_om0/B_q_om0;
    
    /* Update J-th regression */
    for (int j = 1; j < d; j++) {
      
      arma::rowvec Yj = Y.row(j);
      arma::mat Sigma_phi_j = Sigma_q_phi.slice(j);
      
      /* Get matrix K */
      arma::cube K_phi_nu = zeros(j,j,n);
      arma::mat K_phi_sum = zeros(j,j);
      for (int t = 0; t < n; t++) {
        for (int i = 0; i < j; i++) {
          K_phi_nu(i,i,t) = mu_q_nu(j,t)*trace(Sigma_q_phi.slice(i)*ZZ_t.slice(t));
        }
        K_phi_sum = K_phi_sum + K_phi_nu.slice(t);
      }
      
      /* Update BETA_j */
      arma::mat Q = Y.rows(0,j-1) - Mu_q_phi.rows(0,j-1)*Z;
      arma::mat Sigma_beta_w = inv_sympd(Q*diagmat(mu_q_nu.row(j))*Q.t() + K_phi_sum + 1/tau*eye(j,j));
      Sigma_q_beta.slice(j).submat(0,0,j-1,j-1) = Sigma_beta_w;
      arma::vec mu_beta_w = Sigma_beta_w*(Q*diagmat(mu_q_nu.row(j))*(Y.row(j)-Mu_q_phi.row(j)*Z).t());
      Mu_q_beta.submat(j,0,j,j-1) = mu_beta_w.t();
      arma::vec mu_q_betasq = Sigma_beta_w.diag() + pow(mu_beta_w,2);
      /* Update ELBO */
      
      /* Update H_1 */
      mS.row(j) = (Yj-mu_beta_w.t()*Q-Mu_q_phi.row(j)*Z)%(Yj-mu_beta_w.t()*Q-Mu_q_phi.row(j)*Z);
      for (int t = 0; t < n; t++) {
        mS(j,t) = mS(j,t) + 
          trace(ZZ_t.slice(t)*Sigma_phi_j) + 
          as_scalar(mu_beta_w.t()*K_phi_nu.slice(t)*mu_beta_w/mu_q_nu(j,t)) + 
          trace(Sigma_beta_w*Q.col(t)*Q.col(t).t()) + 
          trace(Sigma_beta_w*K_phi_nu.slice(t)/mu_q_nu(j,t));
      }
      Rcpp::List opt_out = wand2014(mu_q_h.row(j),
                                    Sigma_q_h.slice(j),
                                    sigma2_q_h.row(j),
                                    mS.row(j),
                                    mu_q_om2inv(j));
      arma::vec mh = opt_out["f"];
      arma::mat Sh = opt_out["S"];
      arma::vec sh = opt_out["s2"];
      
      mu_q_h.row(j) = mh.t();
      Sigma_q_h.slice(j) = Sh;
      sigma2_q_h.row(j) = sh.t();
      
      /* Update NU_1 */
      mu_q_nu.row(j) = exp(-mh(span(1,n)).t()+0.5*sh(span(1,n)).t());
      
      /* Update OMEGA^2_1 */
      double A_q_om = a_om + 0.5*(n+1);
      double B_q_om = b_om + 0.5*as_scalar(pow(mh(0),2)+pow(mh(n),2)+2*sum(pow(mh(span(1,n-1)),2)) -
                                           2*sum((mh(span(0,n-1)))%(mh(span(1,n)))) +
                                           (sh(0)+sh(n)+2*sum(sh(span(1,n-1)))-2*sum(Sh.diag(1))));
      mu_q_om2inv(j) = A_q_om/B_q_om;
      
    }
    
    
    /* Get matrix C and E(Om) */
    arma::cube C = zeros(d,d,n);
    for (int t = 0; t < n; t++) {
      for (int i0 = 0; i0 < d; i0++) C.slice(t) = C.slice(t) + mu_q_nu(i0,t)*Sigma_q_beta.slice(i0);
      Omega_hat.slice(t) = (Id - Mu_q_beta).t()*diagmat(mu_q_nu.col(t))*(Id - Mu_q_beta) + C.slice(t);
      Sigma_hat.slice(t) = inv_sympd(Omega_hat.slice(t));
    }
    
    /* Update PHI_j */
    for (int j = 0; j < d; j++) {
      
      arma::mat OZZ = zeros(d_,d_);
      arma::vec OZY = zeros(d_);
      arma::vec OZT = zeros(d_);
      for (int t = 0; t < n; t++) {
        OZZ = OZZ + Omega_hat(j,j,t)*ZZ_t.slice(t); 
        OZY = OZY + kron(Omega_hat.slice(t).row(j),Z.col(t))*Y.col(t);
        
        arma::rowvec Om_j = Omega_hat.slice(t).row(j);
        Om_j.shed_col(j);
        arma::mat Mu_phi_w = Mu_q_phi;
        Mu_phi_w.shed_row(j);
        OZT = OZT + kron(Om_j,ZZ_t.slice(t))*vectorise(Mu_phi_w.t());
      }
      
      Sigma_q_phi.slice(j) = inv_sympd(OZZ + 1/ups*eye(d_,d_));
      
      arma::vec mu_q_phi_j = Sigma_q_phi.slice(j)*(OZY-OZT);
      Mu_q_phi.row(j) = mu_q_phi_j.t();
      arma::vec mu_q_phisq = Sigma_q_phi.slice(j).diag() + pow(mu_q_phi_j,2);
      /* Update ELBO */
      
    }
    
    /* Store iteration results */
    lowerBoundIter(itNum) = lowerBound;
    arma::vec VecNew = join_vert(vectorise(Mu_q_phi),
                                 vectorise(Mu_q_beta));
    
    /* Check convergence */
    if (itNum > 1) {
      double lowerBoundOld = lowerBoundIter(itNum-2);
      double delta_Par = max(abs((VecNew - VecOld)/VecOld));
      double delta_Elbo = abs(lowerBound - lowerBoundOld);
      
      if (delta_Par < Tol_Par)  if (delta_Elbo < Tol_ELBO) converged = 1;
      if (Trace == 1) {
        cout << "Iteration number:" << itNum << "; Parameter R.E:" << delta_Par << "| ELBO R.I:" << delta_Elbo << endl;
      }
    }
    
    if (itNum == maxIter) converged = 1;
    VecOld = VecNew;
  }
  
  /* Return results */
  out["Omega_hat"] = Omega_hat;
  out["Sigma_hat"] = Sigma_hat;
  
  out["mu_q_nu"] = mu_q_nu;
  
  out["Mu_q_beta"] = Mu_q_beta;
  out["Sigma_q_beta"] = Sigma_q_beta;
  
  out["Mu_q_theta"] = Mu_q_phi;
  out["Sigma_q_theta"] = Sigma_q_phi;
  
  out["lowerBoundIter"] = lowerBoundIter.subvec(1,itNum);
  out["lowerBound"] = lowerBoundIter(itNum);
  out["convergence"] = converged;
  
  
  return out;
}


// [[Rcpp::export]]
Rcpp::List fVBmlrSVL(arma::mat D, 
                     arma::mat X,
                     Rcpp::List hyper, 
                     int maxIter = 500, 
                     double Tol_ELBO = 1e-2, 
                     double Tol_Par = 1e-2, 
                     int Trace = 0) {
  
  Rcpp::List out;
  /* Get dimensions */
  double d = D.n_rows;
  double n = D.n_cols-1;
  double d_ = X.n_rows;
  
  /* Bayesian Lasso */
  
  /* Get Hyperparameters */
  double a_om = hyper["a_om"]; 
  double b_om = hyper["b_om"]; 
  
  double tau = hyper["tau"];
  
  double h1 = hyper["h1"];     
  double h2 = hyper["h2"];  
  
  /* Initialize */
  arma::cube Sigma_q_phi = zeros(d_,d_,d);
  for (int i = 0; i < d; i++) {
    Sigma_q_phi.slice(i) = eye(d_,d_);
  }
  arma::mat Mu_q_phi = zeros(d,d_);
  
  arma::mat Mu_q_beta = zeros(d,d);
  arma::cube Sigma_q_beta = zeros(d,d,d);
  
  arma::cube Omega_hat = zeros(d,d,n);
  arma::cube Sigma_hat = zeros(d,d,n);
  
  arma::mat mu_q_nu = ones(d,n);
  arma::mat mu_q_h = ones(d,n+1);
  arma::cube Sigma_q_h = zeros(n+1,n+1,d);
  arma::mat sigma2_q_h = ones(d,n+1);
  
  arma::vec mu_q_om2inv = 3*ones(d);
  
  arma::mat Mu_q_recip_ups = ones(d,d_);
  arma::mat Mu_q_ups = ones(d,d_);
  
  arma::mat Mu_q_kappa = ones(d,d_);
  
  /* Get useful quantities */
  arma::mat Y = D.cols(1,n);
  arma::vec y = vectorise(Y.t());
  arma::mat Z = X.cols(0,n-1);
  arma::cube ZZ_t = zeros(d_,d_,n);
  for (int t = 0; t < n; t++) ZZ_t.slice(t) = Z.col(t)*Z.col(t).t();
  arma::rowvec ones_row = ones(n).t();
  arma::vec ones_col = ones(n);
  arma::mat In = eye(n,n);
  arma::mat Id = eye(d,d);
  arma::mat mS = zeros(d,n);
  
  /* Set convergence criterion */
  int converged = 0;
  int itNum = 0;
  double eps_det = 1e-8;
  
  arma::vec lowerBoundIter = zeros(maxIter+1);
  arma::vec VecOld = join_vert(vectorise(Mu_q_phi),
                               vectorise(Mu_q_beta));
  
  /* Start Updating */
  while(converged == 0) {
    /* Update iteration and initialize ELBO (with only constants) */
    itNum = itNum + 1;
    double lowerBound = 0;
    
    /* Update H_1 */
    mS.row(0) = (Y.row(0)-Mu_q_phi.row(0)*Z)%(Y.row(0)-Mu_q_phi.row(0)*Z);
    for (int t = 0; t < n; t++) {
      mS(0,t) = mS(0,t) + trace(ZZ_t.slice(t)*Sigma_q_phi.slice(0));
    }
    Rcpp::List opt_out0 = wand2014(mu_q_h.row(0),
                                   Sigma_q_h.slice(0),
                                   sigma2_q_h.row(0),
                                   mS.row(0),
                                   mu_q_om2inv(0));
    arma::vec mh0 = opt_out0["f"];
    arma::mat Sh0 = opt_out0["S"];
    arma::vec sh0 = opt_out0["s2"];
    
    mu_q_h.row(0) = mh0.t();
    Sigma_q_h.slice(0) = Sh0;
    sigma2_q_h.row(0) = sh0.t();
    
    /* Update NU_1 */
    mu_q_nu.row(0) = exp(-mh0(span(1,n)).t()+0.5*sh0(span(1,n)).t());
    
    /* Update OMEGA^2_1 */
    double A_q_om0 = a_om + 0.5*(n+1);
    double B_q_om0 = b_om + 0.5*as_scalar(pow(mh0(0),2)+pow(mh0(n),2)+2*sum(pow(mh0(span(1,n-1)),2)) -
                                          2*sum((mh0(span(0,n-1)))%(mh0(span(1,n)))) +
                                          (sh0(0)+sh0(n)+2*sum(sh0(span(1,n-1)))-2*sum(Sh0.diag(1))));
    mu_q_om2inv(0) = A_q_om0/B_q_om0;
    
    /* Update J-th regression */
    for (int j = 1; j < d; j++) {
      arma::rowvec Yj = Y.row(j);
      arma::mat Sigma_phi_j = Sigma_q_phi.slice(j);
      
      /* Get matrix K */
      arma::cube K_phi_nu = zeros(j,j,n);
      arma::mat K_phi_sum = zeros(j,j);
      for (int t = 0; t < n; t++) {
        for (int i = 0; i < j; i++) {
          K_phi_nu(i,i,t) = mu_q_nu(j,t)*trace(Sigma_q_phi.slice(i)*ZZ_t.slice(t));
        }
        K_phi_sum = K_phi_sum + K_phi_nu.slice(t);
      }
      
      /* Update BETA_j */
      arma::mat Q = Y.rows(0,j-1) - Mu_q_phi.rows(0,j-1)*Z;
      arma::mat Sigma_beta_w = inv_sympd(Q*diagmat(mu_q_nu.row(j))*Q.t() + K_phi_sum + 1/tau*eye(j,j));
      Sigma_q_beta.slice(j).submat(0,0,j-1,j-1) = Sigma_beta_w;
      arma::vec mu_beta_w = Sigma_beta_w*(Q*diagmat(mu_q_nu.row(j))*(Y.row(j)-Mu_q_phi.row(j)*Z).t());
      Mu_q_beta.submat(j,0,j,j-1) = mu_beta_w.t();
      arma::vec mu_q_betasq = Sigma_beta_w.diag() + pow(mu_beta_w,2);
      /* Update ELBO */
      
      /* Update H_1 */
      mS.row(j) = (Yj-mu_beta_w.t()*Q-Mu_q_phi.row(j)*Z)%(Yj-mu_beta_w.t()*Q-Mu_q_phi.row(j)*Z);
      for (int t = 0; t < n; t++) {
        mS(j,t) = mS(j,t) + 
          trace(ZZ_t.slice(t)*Sigma_phi_j) + 
          as_scalar(mu_beta_w.t()*K_phi_nu.slice(t)*mu_beta_w/mu_q_nu(j,t)) + 
          trace(Sigma_beta_w*Q.col(t)*Q.col(t).t()) + 
          trace(Sigma_beta_w*K_phi_nu.slice(t)/mu_q_nu(j,t));
      }
      Rcpp::List opt_out = wand2014(mu_q_h.row(j),
                                    Sigma_q_h.slice(j),
                                    sigma2_q_h.row(j),
                                    mS.row(j),
                                    mu_q_om2inv(j));
      arma::vec mh = opt_out["f"];
      arma::mat Sh = opt_out["S"];
      arma::vec sh = opt_out["s2"];
      
      mu_q_h.row(j) = mh.t();
      Sigma_q_h.slice(j) = Sh;
      sigma2_q_h.row(j) = sh.t();
      
      /* Update NU_1 */
      mu_q_nu.row(j) = exp(-mh(span(1,n)).t()+0.5*sh(span(1,n)).t());
      
      /* Update OMEGA^2_1 */
      double A_q_om = a_om + 0.5*(n+1);
      double B_q_om = b_om + 0.5*as_scalar(pow(mh(0),2)+pow(mh(n),2)+2*sum(pow(mh(span(1,n-1)),2)) -
                                           2*sum((mh(span(0,n-1)))%(mh(span(1,n)))) +
                                           (sh(0)+sh(n)+2*sum(sh(span(1,n-1)))-2*sum(Sh.diag(1))));
      mu_q_om2inv(j) = A_q_om/B_q_om;
    }
    
    
    /* Get matrix C and E(Om) */
    arma::cube C = zeros(d,d,n);
    for (int t = 0; t < n; t++) {
      for (int i0 = 0; i0 < d; i0++) C.slice(t) = C.slice(t) + mu_q_nu(i0,t)*Sigma_q_beta.slice(i0);
      Omega_hat.slice(t) = (Id - Mu_q_beta).t()*diagmat(mu_q_nu.col(t))*(Id - Mu_q_beta) + C.slice(t);
      Sigma_hat.slice(t) = inv_sympd(Omega_hat.slice(t));
    }
    
    /* Update PHI_j */
    for (int j = 0; j < d; j++) {
      
      arma::mat OZZ = zeros(d_,d_);
      arma::vec OZY = zeros(d_);
      arma::vec OZT = zeros(d_);
      for (int t = 0; t < n; t++) {
        OZZ = OZZ + Omega_hat(j,j,t)*ZZ_t.slice(t); 
        OZY = OZY + kron(Omega_hat.slice(t).row(j),Z.col(t))*Y.col(t);
        arma::rowvec Om_j = Omega_hat.slice(t).row(j);
        Om_j.shed_col(j);
        arma::mat Mu_phi_w = Mu_q_phi;
        Mu_phi_w.shed_row(j);
        OZT = OZT + kron(Om_j,ZZ_t.slice(t))*vectorise(Mu_phi_w.t());
      }
      
      Sigma_q_phi.slice(j) = inv_sympd(OZZ + diagmat(Mu_q_recip_ups.row(j)));
      
      arma::vec mu_q_phi_j = Sigma_q_phi.slice(j)*(OZY-OZT);
      Mu_q_phi.row(j) = mu_q_phi_j.t();
      arma::vec mu_q_phisq = Sigma_q_phi.slice(j).diag() + pow(mu_q_phi_j,2);
      /* Update ELBO */
      
      /* Update UPS_j */
      Mu_q_recip_ups.row(j) = sqrt(Mu_q_kappa.row(j)/mu_q_phisq.t());
      Mu_q_ups.row(j) = sqrt(mu_q_phisq.t()/Mu_q_kappa.row(j))+1/Mu_q_kappa.row(j);
      /* Update ELBO */
      
      
      /* Update KAPPA */
      Mu_q_kappa.row(j) = (1+h1)/(0.5*Mu_q_ups.row(j)+h2);
      /* Update ELBO */
    }
    
    /* Store iteration results */
    lowerBoundIter(itNum) = lowerBound;
    arma::vec VecNew = join_vert(vectorise(Mu_q_phi),
                                 vectorise(Mu_q_beta));
    
    /* Check convergence */
    if (itNum > 1) {
      double lowerBoundOld = lowerBoundIter(itNum-2);
      double delta_Par = max(abs((VecNew - VecOld)/VecOld));
      double delta_Elbo = abs(lowerBound - lowerBoundOld);
      
      if (delta_Par < Tol_Par)  if (delta_Elbo < Tol_ELBO) converged = 1;
      if (Trace == 1) {
        cout << "Iteration number:" << itNum << "; Parameter R.E:" << delta_Par << "| ELBO R.I:" << delta_Elbo << endl;
      }
    }
    
    if (itNum == maxIter) converged = 1;
    VecOld = VecNew;
  }
  
  /* Return results */
  out["Omega_hat"] = Omega_hat;
  out["Sigma_hat"] = Sigma_hat;
  
  out["mu_q_nu"] = mu_q_nu;
  
  out["Mu_q_beta"] = Mu_q_beta;
  out["Sigma_q_beta"] = Sigma_q_beta;
  
  out["Mu_q_theta"] = Mu_q_phi;
  out["Sigma_q_theta"] = Sigma_q_phi;
  
  out["lowerBoundIter"] = lowerBoundIter.subvec(1,itNum);
  out["lowerBound"] = lowerBoundIter(itNum);
  out["convergence"] = converged;
  
  
  return out;
}



// [[Rcpp::export]]
Rcpp::List fVBmlrSVNG(arma::mat D, 
                      arma::mat X,
                      Rcpp::List hyper, 
                      int maxIter = 500, 
                      double Tol_ELBO = 1e-2, 
                      double Tol_Par = 1e-2, 
                      int Trace = 0) {
  
  Rcpp::List out;
  /* Get dimensions */
  double d = D.n_rows;
  double n = D.n_cols-1;
  double d_ = X.n_rows;
  
  /* Normal-Gamma */
  
  /* Get Hyperparameters */
  double a_om = hyper["a_om"]; 
  double b_om = hyper["b_om"]; 
  
  double tau = hyper["tau"];
  
  double h1 = hyper["h1"];     
  double h2 = hyper["h2"]; 
  double h3 = hyper["h3"]; 
  
  /* Initialize */
  arma::cube Sigma_q_phi = zeros(d_,d_,d);
  for (int i = 0; i < d; i++) {
    Sigma_q_phi.slice(i) = eye(d_,d_);
  }
  arma::mat Mu_q_phi = zeros(d,d_);
  
  arma::mat Mu_q_beta = zeros(d,d);
  arma::cube Sigma_q_beta = zeros(d,d,d);
  
  arma::cube Omega_hat = zeros(d,d,n);
  arma::cube Sigma_hat = zeros(d,d,n);
  
  arma::mat mu_q_nu = ones(d,n);
  arma::mat mu_q_h = ones(d,n+1);
  arma::cube Sigma_q_h = zeros(n+1,n+1,d);
  arma::mat sigma2_q_h = ones(d,n+1);
  
  arma::vec mu_q_om2inv = 3*ones(d);
  
  arma::mat Mu_q_ups = 0.1*ones(d,d_);
  arma::mat Mu_q_recip_ups = Mu_q_ups;
  arma::mat Mu_q_logups = log(Mu_q_ups);
  
  arma::mat Mu_q_kappa = ones(d,d_);
  arma::mat Mu_q_logkappa = ones(d,d_);
  
  arma::vec mu_q_xi = ones(d);
  
  /* Get useful quantities */
  arma::mat Y = D.cols(1,n);
  arma::vec y = vectorise(Y.t());
  arma::mat Z = X.cols(0,n-1);
  arma::cube ZZ_t = zeros(d_,d_,n);
  for (int t = 0; t < n; t++) ZZ_t.slice(t) = Z.col(t)*Z.col(t).t();
  arma::rowvec ones_row = ones(n).t();
  arma::vec ones_col = ones(n);
  arma::mat In = eye(n,n);
  arma::mat Id = eye(d,d);
  arma::mat mS = zeros(d,n);
  
  /* Set convergence criterion */
  int converged = 0;
  int itNum = 0;
  double eps_det = 1e-8;
  
  arma::vec lowerBoundIter = zeros(maxIter+1);
  arma::vec VecOld = join_vert(vectorise(Mu_q_phi),
                               vectorise(Mu_q_beta));
  
  /* Start Updating */
  while(converged == 0) {
    /* Update iteration and initialize ELBO (with only constants) */
    itNum = itNum + 1;
    double lowerBound = 0;
    
    /* Update H_1 */
    mS.row(0) = (Y.row(0)-Mu_q_phi.row(0)*Z)%(Y.row(0)-Mu_q_phi.row(0)*Z);
    for (int t = 0; t < n; t++) {
      mS(0,t) = mS(0,t) + trace(ZZ_t.slice(t)*Sigma_q_phi.slice(0));
    }
    Rcpp::List opt_out0 = wand2014(mu_q_h.row(0),
                                   Sigma_q_h.slice(0),
                                   sigma2_q_h.row(0),
                                   mS.row(0),
                                   mu_q_om2inv(0));
    arma::vec mh0 = opt_out0["f"];
    arma::mat Sh0 = opt_out0["S"];
    arma::vec sh0 = opt_out0["s2"];
    
    mu_q_h.row(0) = mh0.t();
    Sigma_q_h.slice(0) = Sh0;
    sigma2_q_h.row(0) = sh0.t();
    
    /* Update NU_1 */
    mu_q_nu.row(0) = exp(-mh0(span(1,n)).t()+0.5*sh0(span(1,n)).t());
    
    /* Update OMEGA^2_1 */
    double A_q_om0 = a_om + 0.5*(n+1);
    double B_q_om0 = b_om + 0.5*as_scalar(pow(mh0(0),2)+pow(mh0(n),2)+2*sum(pow(mh0(span(1,n-1)),2)) -
                                          2*sum((mh0(span(0,n-1)))%(mh0(span(1,n)))) +
                                          (sh0(0)+sh0(n)+2*sum(sh0(span(1,n-1)))-2*sum(Sh0.diag(1))));
    mu_q_om2inv(0) = A_q_om0/B_q_om0;
    
    /* Update J-th regression */
    for (int j = 1; j < d; j++) {
      arma::rowvec Yj = Y.row(j);
      arma::mat Sigma_phi_j = Sigma_q_phi.slice(j);
      
      /* Get matrix K */
      arma::cube K_phi_nu = zeros(j,j,n);
      arma::mat K_phi_sum = zeros(j,j);
      for (int t = 0; t < n; t++) {
        for (int i = 0; i < j; i++) {
          K_phi_nu(i,i,t) = mu_q_nu(j,t)*trace(Sigma_q_phi.slice(i)*ZZ_t.slice(t));
        }
        K_phi_sum = K_phi_sum + K_phi_nu.slice(t);
      }
      
      /* Update BETA_j */
      arma::mat Q = Y.rows(0,j-1) - Mu_q_phi.rows(0,j-1)*Z;
      arma::mat Sigma_beta_w = inv_sympd(Q*diagmat(mu_q_nu.row(j))*Q.t() + K_phi_sum + 1/tau*eye(j,j));
      Sigma_q_beta.slice(j).submat(0,0,j-1,j-1) = Sigma_beta_w;
      arma::vec mu_beta_w = Sigma_beta_w*(Q*diagmat(mu_q_nu.row(j))*(Y.row(j)-Mu_q_phi.row(j)*Z).t());
      Mu_q_beta.submat(j,0,j,j-1) = mu_beta_w.t();
      arma::vec mu_q_betasq = Sigma_beta_w.diag() + pow(mu_beta_w,2);
      /* Update ELBO */
      
      /* Update H_1 */
      mS.row(j) = (Yj-mu_beta_w.t()*Q-Mu_q_phi.row(j)*Z)%(Yj-mu_beta_w.t()*Q-Mu_q_phi.row(j)*Z);
      for (int t = 0; t < n; t++) {
        mS(j,t) = mS(j,t) + 
          trace(ZZ_t.slice(t)*Sigma_phi_j) + 
          as_scalar(mu_beta_w.t()*K_phi_nu.slice(t)*mu_beta_w/mu_q_nu(j,t)) + 
          trace(Sigma_beta_w*Q.col(t)*Q.col(t).t()) + 
          trace(Sigma_beta_w*K_phi_nu.slice(t)/mu_q_nu(j,t));
      }
      Rcpp::List opt_out = wand2014(mu_q_h.row(j),
                                    Sigma_q_h.slice(j),
                                    sigma2_q_h.row(j),
                                    mS.row(j),
                                    mu_q_om2inv(j));
      arma::vec mh = opt_out["f"];
      arma::mat Sh = opt_out["S"];
      arma::vec sh = opt_out["s2"];
      
      mu_q_h.row(j) = mh.t();
      Sigma_q_h.slice(j) = Sh;
      sigma2_q_h.row(j) = sh.t();
      
      /* Update NU_1 */
      mu_q_nu.row(j) = exp(-mh(span(1,n)).t()+0.5*sh(span(1,n)).t());
      
      /* Update OMEGA^2_1 */
      double A_q_om = a_om + 0.5*(n+1);
      double B_q_om = b_om + 0.5*as_scalar(pow(mh(0),2)+pow(mh(n),2)+2*sum(pow(mh(span(1,n-1)),2)) -
                                           2*sum((mh(span(0,n-1)))%(mh(span(1,n)))) +
                                           (sh(0)+sh(n)+2*sum(sh(span(1,n-1)))-2*sum(Sh.diag(1))));
      mu_q_om2inv(j) = A_q_om/B_q_om;
    }
    
    
    /* Get matrix C and E(Om) and ...*/
    arma::cube C = zeros(d,d,n);
    for (int t = 0; t < n; t++) {
      for (int i0 = 0; i0 < d; i0++) C.slice(t) = C.slice(t) + mu_q_nu(i0,t)*Sigma_q_beta.slice(i0);
      Omega_hat.slice(t) = (Id - Mu_q_beta).t()*diagmat(mu_q_nu.col(t))*(Id - Mu_q_beta) + C.slice(t);
      Sigma_hat.slice(t) = inv_sympd(Omega_hat.slice(t));
    }
    
    /* Update PHI_j */
    for (int j = 0; j < d; j++) {
      
      arma::mat OZZ = zeros(d_,d_);
      arma::vec OZY = zeros(d_);
      arma::vec OZT = zeros(d_);
      for (int t = 0; t < n; t++) {
        OZZ = OZZ + Omega_hat(j,j,t)*ZZ_t.slice(t); 
        OZY = OZY + kron(Omega_hat.slice(t).row(j),Z.col(t))*Y.col(t);
        arma::rowvec Om_j = Omega_hat.slice(t).row(j);
        Om_j.shed_col(j);
        arma::mat Mu_phi_w = Mu_q_phi;
        Mu_phi_w.shed_row(j);
        OZT = OZT + kron(Om_j,ZZ_t.slice(t))*vectorise(Mu_phi_w.t());
      }
      
      Sigma_q_phi.slice(j) = inv_sympd(OZZ + diagmat(Mu_q_recip_ups.row(j)));
      
      arma::vec mu_q_phi_j = Sigma_q_phi.slice(j)*(OZY-OZT);
      Mu_q_phi.row(j) = mu_q_phi_j.t();
      arma::vec mu_q_phisq = Sigma_q_phi.slice(j).diag() + pow(mu_q_phi_j,2);
      /* Update ELBO */
      
      /* Update UPS */
      double zeta_q_ups = mu_q_xi(j) - 0.5;
      arma::rowvec a_q_ups = mu_q_xi(j)*Mu_q_kappa.row(j)+1e-4;
      arma::vec b_q_ups = mu_q_phisq;
      
      for (int j0 = 0; j0 < d_; j0++) {
        Mu_q_ups(j,j0) = sqrt(b_q_ups(j0)/a_q_ups(j0)) *
          boost::math::cyl_bessel_k(zeta_q_ups+1, sqrt(a_q_ups(j0)*b_q_ups(j0))) /
            boost::math::cyl_bessel_k(zeta_q_ups, sqrt(a_q_ups(j0)*b_q_ups(j0)));
        Mu_q_recip_ups(j,j0) = sqrt(a_q_ups(j0)/b_q_ups(j0)) * 
          boost::math::cyl_bessel_k(zeta_q_ups+1, sqrt(a_q_ups(j0)*b_q_ups(j0))) /
            boost::math::cyl_bessel_k(zeta_q_ups, sqrt(a_q_ups(j0)*b_q_ups(j0))) - 
              2*zeta_q_ups/b_q_ups(j0);
        Mu_q_logups(j,j0) = log(sqrt(b_q_ups(j0)/a_q_ups(j0))); // + grad_cpp(zeta_q_ups, sqrt(a_q_ups(j0)*b_q_ups(j0)));
        
        /* Update ELBO */
        
        /* Update KAPPA */
        double a_q_kappa = mu_q_xi(j) + h1;
        double b_q_kappa = 0.5*mu_q_xi(j)*Mu_q_ups(j,j0) + h2;
        
        Mu_q_kappa(j,j0) = a_q_kappa/b_q_kappa;
        Mu_q_logkappa(j,j0) = -log(b_q_kappa) + boost::math::digamma(a_q_kappa);
        
        /* Update ELBO */
      }
      
      /* Update XI */
      Rcpp::List out_xi = integrate_latent_factor(d_,sum(Mu_q_ups.row(j)%Mu_q_kappa.row(j)),
                                                  sum(Mu_q_logups.row(j)),sum(Mu_q_logkappa.row(j)),h3);
      mu_q_xi(j) = out_xi["value"];
      double c_xi = out_xi["const_int"];
      
      /* Update ELBO */
    }
    
    /* Store iteration results */
    lowerBoundIter(itNum) = lowerBound;
    arma::vec VecNew = join_vert(vectorise(Mu_q_phi),
                                 vectorise(Mu_q_beta));
    
    /* Check convergence */
    if (itNum > 1) {
      double lowerBoundOld = lowerBoundIter(itNum-2);
      double delta_Par = max(abs((VecNew - VecOld)/VecOld));
      double delta_Elbo = abs(lowerBound - lowerBoundOld);
      
      if (delta_Par < Tol_Par)  if (delta_Elbo < Tol_ELBO) converged = 1;
      if (Trace == 1) {
        cout << "Iteration number:" << itNum << "; Parameter R.E:" << delta_Par << "| ELBO R.I:" << delta_Elbo << endl;
      }
    }
    
    if (itNum == maxIter) converged = 1;
    VecOld = VecNew;
  }
  
  /* Return results */
  out["Omega_hat"] = Omega_hat;
  out["Sigma_hat"] = Sigma_hat;
  
  out["mu_q_nu"] = mu_q_nu;
  
  out["Mu_q_beta"] = Mu_q_beta;
  out["Sigma_q_beta"] = Sigma_q_beta;
  
  out["Mu_q_theta"] = Mu_q_phi;
  out["Sigma_q_theta"] = Sigma_q_phi;
  
  out["lowerBoundIter"] = lowerBoundIter.subvec(1,itNum);
  out["lowerBound"] = lowerBoundIter(itNum);
  out["convergence"] = converged;
  
  
  return out;
}


// [[Rcpp::export]]
Rcpp::List fVBmlrSVHS(arma::mat D, 
                      arma::mat X,
                      Rcpp::List hyper, 
                      int maxIter = 500, 
                      double Tol_ELBO = 1e-2, 
                      double Tol_Par = 1e-2, 
                      int Trace = 0) {
  
  Rcpp::List out;
  /* Get dimensions */
  double d = D.n_rows;
  double n = D.n_cols-1;
  double d_ = X.n_rows;
  
  /* Horseshoe prior */
  
  /* Get Hyperparameters */
  double a_om = hyper["a_om"]; 
  double b_om = hyper["b_om"]; 
  
  double tau = hyper["tau"];
  
  /* Initialize */
  arma::cube Sigma_q_phi = zeros(d_,d_,d);
  for (int i = 0; i < d; i++) {
    Sigma_q_phi.slice(i) = eye(d_,d_);
  }
  arma::mat Mu_q_phi = zeros(d,d_);
  
  arma::mat Mu_q_beta = zeros(d,d);
  arma::cube Sigma_q_beta = zeros(d,d,d);
  
  arma::cube Omega_hat = zeros(d,d,n);
  arma::cube Sigma_hat = zeros(d,d,n);
  
  arma::mat mu_q_nu = ones(d,n);
  arma::mat mu_q_h = ones(d,n+1);
  arma::cube Sigma_q_h = zeros(n+1,n+1,d);
  arma::mat sigma2_q_h = ones(d,n+1);
  
  arma::vec mu_q_om2inv = 3*ones(d);
  
  arma::mat Mu_q_recip_ups = ones(d,d_);
  arma::vec mu_q_recip_delta = ones(d);
  arma::mat Mu_q_recip_kappa = ones(d,d_);
  arma::vec mu_q_recip_xi = ones(d);
  
  /* Get useful quantities */
  arma::mat Y = D.cols(1,n);
  arma::vec y = vectorise(Y.t());
  arma::mat Z = X.cols(0,n-1);
  arma::cube ZZ_t = zeros(d_,d_,n);
  for (int t = 0; t < n; t++) ZZ_t.slice(t) = Z.col(t)*Z.col(t).t();
  arma::rowvec ones_row = ones(n).t();
  arma::vec ones_col = ones(n);
  arma::mat In = eye(n,n);
  arma::mat Id = eye(d,d);
  arma::mat mS = zeros(d,n);
  
  /* Set convergence criterion */
  int converged = 0;
  int itNum = 0;
  double eps_det = 1e-8;
  
  arma::vec lowerBoundIter = zeros(maxIter+1);
  arma::vec VecOld = join_vert(vectorise(Mu_q_phi),
                               vectorise(Mu_q_beta));
  
  /* Start Updating */
  while(converged == 0) {
    /* Update iteration and initialize ELBO (with only constants) */
    itNum = itNum + 1;
    double lowerBound = 0;
    
    /* Update H_1 */
    mS.row(0) = (Y.row(0)-Mu_q_phi.row(0)*Z)%(Y.row(0)-Mu_q_phi.row(0)*Z);
    for (int t = 0; t < n; t++) {
      mS(0,t) = mS(0,t) + trace(ZZ_t.slice(t)*Sigma_q_phi.slice(0));
    }
    Rcpp::List opt_out0 = wand2014(mu_q_h.row(0),
                                   Sigma_q_h.slice(0),
                                   sigma2_q_h.row(0),
                                   mS.row(0),
                                   mu_q_om2inv(0));
    arma::vec mh0 = opt_out0["f"];
    arma::mat Sh0 = opt_out0["S"];
    arma::vec sh0 = opt_out0["s2"];
    
    mu_q_h.row(0) = mh0.t();
    Sigma_q_h.slice(0) = Sh0;
    sigma2_q_h.row(0) = sh0.t();
    
    /* Update NU_1 */
    mu_q_nu.row(0) = exp(-mh0(span(1,n)).t()+0.5*sh0(span(1,n)).t());
    
    /* Update OMEGA^2_1 */
    double A_q_om0 = a_om + 0.5*(n+1);
    double B_q_om0 = b_om + 0.5*as_scalar(pow(mh0(0),2)+pow(mh0(n),2)+2*sum(pow(mh0(span(1,n-1)),2)) -
                                          2*sum((mh0(span(0,n-1)))%(mh0(span(1,n)))) +
                                          (sh0(0)+sh0(n)+2*sum(sh0(span(1,n-1)))-2*sum(Sh0.diag(1))));
    mu_q_om2inv(0) = A_q_om0/B_q_om0;
    
    /* Update J-th regression */
    for (int j = 1; j < d; j++) {
      arma::rowvec Yj = Y.row(j);
      arma::mat Sigma_phi_j = Sigma_q_phi.slice(j);
      
      /* Get matrix K */
      arma::cube K_phi_nu = zeros(j,j,n);
      arma::mat K_phi_sum = zeros(j,j);
      for (int t = 0; t < n; t++) {
        for (int i = 0; i < j; i++) {
          K_phi_nu(i,i,t) = mu_q_nu(j,t)*trace(Sigma_q_phi.slice(i)*ZZ_t.slice(t));
        }
        K_phi_sum = K_phi_sum + K_phi_nu.slice(t);
      }
      
      /* Update BETA_j */
      arma::mat Q = Y.rows(0,j-1) - Mu_q_phi.rows(0,j-1)*Z;
      arma::mat Sigma_beta_w = inv_sympd(Q*diagmat(mu_q_nu.row(j))*Q.t() + K_phi_sum + 1/tau*eye(j,j));
      Sigma_q_beta.slice(j).submat(0,0,j-1,j-1) = Sigma_beta_w;
      arma::vec mu_beta_w = Sigma_beta_w*(Q*diagmat(mu_q_nu.row(j))*(Y.row(j)-Mu_q_phi.row(j)*Z).t());
      Mu_q_beta.submat(j,0,j,j-1) = mu_beta_w.t();
      arma::vec mu_q_betasq = Sigma_beta_w.diag() + pow(mu_beta_w,2);
      /* Update ELBO */
      
      /* Update H_1 */
      mS.row(j) = (Yj-mu_beta_w.t()*Q-Mu_q_phi.row(j)*Z)%(Yj-mu_beta_w.t()*Q-Mu_q_phi.row(j)*Z);
      for (int t = 0; t < n; t++) {
        mS(j,t) = mS(j,t) + 
          trace(ZZ_t.slice(t)*Sigma_phi_j) + 
          as_scalar(mu_beta_w.t()*K_phi_nu.slice(t)*mu_beta_w/mu_q_nu(j,t)) + 
          trace(Sigma_beta_w*Q.col(t)*Q.col(t).t()) + 
          trace(Sigma_beta_w*K_phi_nu.slice(t)/mu_q_nu(j,t));
      }
      Rcpp::List opt_out = wand2014(mu_q_h.row(j),
                                    Sigma_q_h.slice(j),
                                    sigma2_q_h.row(j),
                                    mS.row(j),
                                    mu_q_om2inv(j));
      arma::vec mh = opt_out["f"];
      arma::mat Sh = opt_out["S"];
      arma::vec sh = opt_out["s2"];
      
      mu_q_h.row(j) = mh.t();
      Sigma_q_h.slice(j) = Sh;
      sigma2_q_h.row(j) = sh.t();
      
      /* Update NU_1 */
      mu_q_nu.row(j) = exp(-mh(span(1,n)).t()+0.5*sh(span(1,n)).t());
      
      /* Update OMEGA^2_1 */
      double A_q_om = a_om + 0.5*(n+1);
      double B_q_om = b_om + 0.5*as_scalar(pow(mh(0),2)+pow(mh(n),2)+2*sum(pow(mh(span(1,n-1)),2)) -
                                           2*sum((mh(span(0,n-1)))%(mh(span(1,n)))) +
                                           (sh(0)+sh(n)+2*sum(sh(span(1,n-1)))-2*sum(Sh.diag(1))));
      mu_q_om2inv(j) = A_q_om/B_q_om;
    }
    
    
    /* Get matrix C and E(Om) and ...*/
    arma::cube C = zeros(d,d,n);
    for (int t = 0; t < n; t++) {
      for (int i0 = 0; i0 < d; i0++) C.slice(t) = C.slice(t) + mu_q_nu(i0,t)*Sigma_q_beta.slice(i0);
      Omega_hat.slice(t) = (Id - Mu_q_beta).t()*diagmat(mu_q_nu.col(t))*(Id - Mu_q_beta) + C.slice(t);
      Sigma_hat.slice(t) = inv_sympd(Omega_hat.slice(t));
    }
    
    /* Update PHI_j */
    for (int j = 0; j < d; j++) {
      
      arma::mat OZZ = zeros(d_,d_);
      arma::vec OZY = zeros(d_);
      arma::vec OZT = zeros(d_);
      for (int t = 0; t < n; t++) {
        OZZ = OZZ + Omega_hat(j,j,t)*ZZ_t.slice(t); 
        OZY = OZY + kron(Omega_hat.slice(t).row(j),Z.col(t))*Y.col(t);
        arma::rowvec Om_j = Omega_hat.slice(t).row(j);
        Om_j.shed_col(j);
        arma::mat Mu_phi_w = Mu_q_phi;
        Mu_phi_w.shed_row(j);
        OZT = OZT + kron(Om_j,ZZ_t.slice(t))*vectorise(Mu_phi_w.t());
      }
      
      Sigma_q_phi.slice(j) = inv_sympd(OZZ + diagmat(mu_q_recip_delta(j)*Mu_q_recip_ups.row(j)));
      
      arma::vec mu_q_phi_j = Sigma_q_phi.slice(j)*(OZY-OZT);
      Mu_q_phi.row(j) = mu_q_phi_j.t();
      arma::vec mu_q_phisq = Sigma_q_phi.slice(j).diag() + pow(mu_q_phi_j,2);
      /* Update ELBO */
      
      /* Update UPS_j */
      Mu_q_recip_ups.row(j) = 1/(0.5*mu_q_phisq.t()*mu_q_recip_delta(j)+Mu_q_recip_kappa.row(j));
      /* Update ELBO */
      
      /* Update KAPPA_j */
      Mu_q_recip_kappa.row(j) = 1/(1+Mu_q_recip_ups.row(j));
      /* Update ELBO */
      
      
      /* Update DELTA */
      mu_q_recip_delta(j) = (0.5*(d+1))/(0.5*sum(mu_q_phisq.t()%Mu_q_recip_ups.row(j))+mu_q_recip_xi(j));
      /* Update ELBO */
      
      
      /* Update XI_j */
      mu_q_recip_xi(j) = 1/(1+mu_q_recip_delta(j));
      /* Update ELBO */
    }
    
    /* Store iteration results */
    lowerBoundIter(itNum) = lowerBound;
    arma::vec VecNew = join_vert(vectorise(Mu_q_phi),
                                 vectorise(Mu_q_beta));
    
    /* Check convergence */
    if (itNum > 1) {
      double lowerBoundOld = lowerBoundIter(itNum-2);
      double delta_Par = max(abs((VecNew - VecOld)/VecOld));
      double delta_Elbo = abs(lowerBound - lowerBoundOld);
      
      if (delta_Par < Tol_Par)  if (delta_Elbo < Tol_ELBO) converged = 1;
      if (Trace == 1) {
        cout << "Iteration number:" << itNum << "; Parameter R.E:" << delta_Par << "| ELBO R.I:" << delta_Elbo << endl;
      }
    }
    
    if (itNum == maxIter) converged = 1;
    VecOld = VecNew;
  }
  
  /* Return results */
  out["Omega_hat"] = Omega_hat;
  out["Sigma_hat"] = Sigma_hat;
  
  out["mu_q_nu"] = mu_q_nu;
  
  out["Mu_q_beta"] = Mu_q_beta;
  out["Sigma_q_beta"] = Sigma_q_beta;
  
  out["Mu_q_theta"] = Mu_q_phi;
  out["Sigma_q_theta"] = Sigma_q_phi;
  
  out["lowerBoundIter"] = lowerBoundIter.subvec(1,itNum);
  out["lowerBound"] = lowerBoundIter(itNum);
  out["convergence"] = converged;
  
  
  return out;
}

