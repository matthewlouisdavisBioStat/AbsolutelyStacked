#' Fit generalized linear models and conditional mixed-effects models using constrained maximum likelihood estimation.
#' @param link_function character, what link function is being used to model the GLM. Options include: linear, logit, log, probit, weibull, softmax, or inverse
#' @param X a list of matrices for softmax regression, a matrix for all other link functions, corresponding to fixed effects (the effect subject to constraint)
#' @param Y a matrix for softmax regression, a vector for all other link functions, corresponding to the outcome of interest
#' @param Z a list of matrices for softmax regression, a matrix for all other link functions, corresponding to random effects to be estimated
#' @param tau exponential dispersion parameter, fixed if est_tau == F
#' @param est_tau T or F, should the exponential dispersion parameter be estimated?
#' @param sigmas vector of numeric elements corresponding to intial values for the variance components associated with random effects.
#' @param beta_update a vector corresponding to the initial values of the fixed effects of interest.
#' @param u_update a vector corresponding to the intial values of the random effects of interest
#' @param lambda manually-specified l2-norm penalty: this appears in the score and information, but is excluded when computing the log-likelihood. Defaults to 0, but set to 0.000001 for make_glmerStackedModel().
#' @param sigma_list a list named by each random-effect variance parameter, each element of which is a vector containing the column indices of the Z matrix that correspond to each random-effect variance component.
#' @param use_quadprog T or F, should sequential quadratic programming using the quadprog package (T) or a general non-linear programmming algorithm using the Rsolnp package (F) be used to find MLEs under constraint. Defaults to F.
#' @param sum_constraint real number referring to the sum-constraint on fixed effect coefficients.
#' @param lower_bound_constraint real number (or vector with length equal to the length beta_update AND the number of columns of X) defining the lower bound constraint on fixed effect coefficients.
#'
#' @details First, generalized linear mixed effects models are fit using unconstrained maximum likelihood estimation via a damped Newton-Raphson algorithm. Coefficients are then updated to reflect the desired constraints, with initial values supplied to the functions based off of the unconstrained MLEs.
#'
#' @return A list of elements including constrained and unconstrained estimates of MLE weights.
#'
#' @importFrom quadprog solve.QP
#' @importFrom Rsolnp solnp
#' @importFrom Matrix nearPD
#'
#' @seealso \code{\link[quadprog]{solve.QP}}, \code{\link[Rsolnpl]{solnp}}
#'
#'
#' @examples
#' \donttest{
#'
#' ## Simulate Data
#' library(magrittr)
#' N <- 1000
#' Xt <- cbind((runif(N)),(runif(N)),(runif(N)),(runif(N)))
#' Bt <- 2*c(-.25,0,0.25,0.75)
#' Zt <- kronk(lapply(1:10,function(i){
#'   cbind(rep(1,N/10))
#' }))
#' Ut <- rnorm(10,0,2)
#' ## sse estimate of sigma^2 for observed error term for intializing (N-qr(Xt)$rank))
#' Mut <- Xt %*% Bt + Zt %*% Ut
#' Yt <- sapply(Mut,function(mu)rnorm(1,mu))
#' tau <- sd(Yt) ## for intializing only.....
#'
#'
#' ## setup for random efx
#' Z <- Zt
#' sigma_list <- list("sigma_1" = 1:(ncol(Z)))
#' names(sigma_list) <- paste0("sigma_",1:length(sigma_list))
#' for(i in 1:length(sigma_list)){
#'   names(sigma_list[[i]]) <- rep(names(sigma_list)[i],
#'                                 length(sigma_list[[i]]))
#' }
#' sigmas <- 1 ## for initializing only
#' names(sigmas) <- names(sigma_list)
#'
#' ## Constrained Maximum Likelihood-Based stacking
#' stacked_output_solnp <- glmer_constrained(
#'   link_function = 'linear',
#'   X = Xt,
#'   Y = Yt,
#'   Z = Z,
#'   tau = 1,
#'   est_tau = T,
#'   sigmas = sigmas,
#'   beta_update = rep(1/ncol(Xt),ncol(Xt)),
#'   u_update = rep(0,ncol(Z)),
#'   lambda = 0,
#'   sigma_list = sigma_list,
#'   use_quadprog = F,
#'   sum_constraint = sum(Bt),
#'   lower_bound_constraint = c(-.5,0,0,0)
#' )
#' stacked_output_qp <- glmer_constrained(
#'   link_function = 'linear',
#'   X = Xt,
#'   Y = Yt,
#'   Z = Z,
#'   tau = 1,
#'   est_tau = T,
#'   sigmas = sigmas,
#'   beta_update = rep(1/ncol(Xt),ncol(Xt)),
#'   u_update = rep(0,ncol(Z)),
#'   lambda = 0,
#'   sigma_list = sigma_list,
#'   use_quadprog = T,
#'   sum_constraint = sum(Bt),
#'   lower_bound_constraint = c(-.5,0,0,0)
#' )
#'
#' ## GLM (requires installation of lme4 for testing purposes)
#' Z2 <- apply(Z,1,function(z){
#'   sum(sapply(1:10,function(i){
#'     i*(z[i]==1)
#'   }))
#' })
#' temp_data <- as.data.frame(cbind(Yt,Xt,Z2))
#' colnames(temp_data) <- c(
#'   "Y",
#'   paste0("X",1:ncol(Xt)),
#'   "Z"
#' )
#' fitLMERraw <- lme4::lmer(as.formula(
#'   "Y ~ 0 + X1 + X2 + X3 + X4 + (1|Z)"
#' ),data = temp_data)
#' fitLMER <- fitLMERraw %>%
#'   summary
#' u_hat <- (fitLMERraw %>% coef)[[1]][,1]
#' mle <- c(fitLMER$coefficients[,1])
#'
#'
#' res <- data.frame(c(Bt,Ut),
#'                   round(c(stacked_output_solnp$beta_update,stacked_output_solnp$u_update),4),
#'                   round(c(stacked_output_qp$beta_update,stacked_output_qp$u_update),4),
#'                   round(c(stacked_output_solnp$beta_mle,stacked_output_solnp$u_update),4),
#'                   round(c(mle,u_hat),4))
#' colnames(res) <- c("True Betas",
#'                    "Constrained MLE Rsolnp",
#'                    "Constrained MLE quadprog",
#'                    "Unconstrained MLE glmer_constrained",
#'                    "Unconstrained MLE lmer")
#' rownames(res) <- c("Beta1", "Beta2" ,"Beta3","Beta4",paste0("RandInt",1:10))
#' print(res)
#' }
#' @export
glmer_constrained <- function(link_function,
                         X,
                         Y,
                         Z,
                         tau,
                         est_tau = F,
                         sigmas = 1,
                         beta_update,
                         u_update,
                         lambda = 0,
                         sigma_list,
                         use_quadprog = F,
                         sum_constraint = 1,
                         lower_bound_constraint = 0){
  #print("Loading it all Up")
  probit <- F
  softmax <- F
  survival <- F
  b_names <- names(beta_update)
  u_names <- names(u_update)
  d_censored <- NULL
  ## establish link function components
  if (link_function == "linear"){

    ## link function
    link <- function(mu)mu

    ## derivative of link function
    link_prime <- function(mu)1

    ## cumulant generating function
    cumgenfunc <- function(eta)0.5*eta^2

    ## offset function
    C <- function(Y, tau)sum(-0.5*log(2*pi) + -0.5*log(tau^2) + -0.5/tau^2 * Y^2)

    ## the first deriv of cum gen func, expectation of suff. stat, should be inverse of link function
    mu_of_eta <- function(eta)eta

    ## the second deriv of cum gen func, the negative hessian
    v_of_mu <- function(mu)1

  } else if (link_function == "logit") {

    ## link function
    link <- function(mu)log(mu / (1 - mu))

    ## derivative of link function
    link_prime <- function(mu)1 / (mu * (1 - mu))

    ## cumulant generating function
    cumgenfunc <- function(eta)log(1 + exp(eta))

    ## offset function
    C <- function(Y, tau)log(1)

    ## the first deriv of cum gen func, expectation of suff. stat, should be inverse of link function
    mu_of_eta <- function(eta)exp(eta) / (1 + exp(eta))

    ## the second deriv of cum gen func, the negative hessian
    v_of_mu <- function(mu)mu * (1 - mu)

  } else if (link_function == "inverse") {

    ## link function
    link <- function(mu)-(1/mu)

    ## derivative of link function
    link_prime <- function(mu)1/mu^2

    ## cumulant generating function
    cumgenfunc <- function(eta)log(-1/eta)

    ## offset function (describes the weibull distribution offset term)
    C <- function(Y, tau)sum(log(tau^2 / Y))

    ## the first deriv of cum gen func, expectation of suff. stat, should be inverse of link function
    mu_of_eta <- function(eta)-1/eta

    ## the negative second deriv of cum gen func, the negative hessian
    v_of_mu <- function(mu)mu^2

  } else if (link_function == "log") {

    ## link function
    link <- function(mu)log(mu)

    ## derivative of link function
    link_prime <- function(mu)1 / mu

    ## cumulant generating function
    cumgenfunc <- function(eta)exp(eta)

    ## offset function
    C <- function(Y, tau) log(1) # sum(log(sapply(Y, factorial)))

    ## the first deriv of cum gen func, expectation of suff. stat, should be inverse of link function
    mu_of_eta <- function(eta)exp(eta)

    ## the second deriv of cum gen func, the negative hessian
    v_of_mu <- function(mu)mu

  } else if (link_function == "probit") {
    probit <- T

    ## link function
    link <- function(mu)qnorm(mu)

    ## derivative of link function
    link_prime <- function(mu)1 / (dnorm(qnorm(mu)))

    ## cumulant generating function NA for probit....?
    cumgenfunc <- function(eta)NA

    ## offset function
    C <- function(Y, tau) log(1)

    ## the first deriv of cum gen func, expectation of suff. stat, should be inverse of link function
    mu_of_eta <- function(eta)pnorm(eta)

    ## the second deriv of cum gen func, the negative hessian
    v_of_mu <- function(mu)mu*(1-mu)

  } else if (link_function == "softmax") {
    softmax <- T

    ## link function
    link <- function(mu)log(mu / (1 - sum(mu)))

    ## derivative of link function
    link_prime <- function(mu){
      mat <- outer(mu,-mu)
      diag(mat) <- mu*(1-mu)
      solve(mat)
    }

    ## cumulant generating function
    cumgenfunc <- function(eta)log(1 + sum(exp(eta)))

    ## offset function
    C <- function(Y, tau)log(1)

    ## the first deriv of cum gen func, expectation of suff. stat, should be inverse of link function
    mu_of_eta <- function(eta)exp(eta) / (1 + sum(exp(eta)))

    ## the second deriv of cum gen func, the negative hessian
    v_of_mu <- function(mu){
      mat <- outer(mu,-mu)
      diag(mat) <- mu*(1-mu)
      mat
    }

  } else if(link_function == "weibull"){
    survival <- T

    ## censorship
    d_censored <- !grepl("[+]",as.character(Y))

    ## log-transformed time
    Y <- log(sapply(Y,function(y){
      y <- as.character(y)
      if(grepl("[+]",y)) y <- substr(y,1,nchar(y)-1)
      as.numeric(y)
    }))[1:length(Y)]


    ## link function
    link <- function(mu)log(mu)

    ## get from the linear-link function to the estimate of the mean
    mu_of_eta <- function(eta,t)exp((Y-as.numeric(eta))/t)

    ## The below doesn't matter!! just held in for compatibility sake
    ## derivative of link function
    link_prime <- function(mu)1 / mu

    ## cumulant generating function
    cumgenfunc <- function(eta)exp(eta)

    ## offset function
    C <- function(Y, tau) log(1) # sum(log(sapply(Y, factorial)))

    ## the second deriv of cum gen func, the negative hessian
    v_of_mu <- function(mu)mu

  }

  #print("Likelihood, Score, and Information")
  ## compute log likelihood of glmmix REF model
  compute_loglik <- function(Y_function,
                             X_function,
                             Z_function,
                             B,
                             U,
                             tau_function,
                             cumgenfunc,
                             C,
                             sigmas_function,
                             sigma_list_function,
                             probit_function = F,
                             softmax_function = F,
                             D =  d_censored,
                             surv = survival){

    if(!probit_function & !softmax_function & !surv){

      ## canonical REF form
      Eta <- X_function %*% B + Z_function %*% U
      as.numeric( (sum(Y_function*Eta) - sum(sapply(Eta,cumgenfunc)))/tau_function^2 + C(Y_function,tau_function)) +
        sum((sapply(sigma_list_function,function(inds){
          Z_function_u <- Z_function[,inds]
          sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
          U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
          fix_inf(as.numeric(sigma_u > .00001),(
            -0.5*ncol(Z_function_u)*log(2*pi) +
              -ncol(Z_function_u)*log(sigma_u) +
              -0.5*sum(U_group^2)/sigma_u^2)
          )
        })))

    } else if(probit_function){

      ## for probit_function
      Eta <- X_function %*% B + Z_function %*% U
      as.numeric(sum(Y_function*pnorm(Eta) + (1-Y_function)*(1-pnorm(Eta)))) +
        sum((sapply(sigma_list_function,function(inds){
          Z_function_u <- Z_function[,inds]
          sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
          U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
          fix_inf(as.numeric(sigma_u > .00001),(
            -0.5*ncol(Z_function_u)*log(2*pi) +
              -ncol(Z_function_u)*log(sigma_u) +
              -0.5*sum(U_group^2)/sigma_u^2)
          )
        })))
    } else if(softmax_function){
      ## for multinomial regression

      ## adjust indices to include all random intercepts
      sigma_list_function <- lapply(sigma_list_function,function(s){
        s <- c(s,s+length(s))
      })
      ## canonical REF form but generalized for higher-dimensional arrays
      Eta <- X_function %**% B + Z_function %**% U
      Mu <- t(apply(Eta,1,mu_of_eta))
      sum(cbind(Y_function,1-rowSums(Y_function)) * cbind(log(Mu),log(1-rowSums(Mu)))) +
        sum((sapply(sigma_list_function,function(inds){
          sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
          U_group <- c(unlist(lapply(U,diag))[inds])*as.numeric(sigma_u > 0.00001)
          fix_inf(as.numeric(sigma_u > .00001),(
            -0.5*length(U_group)*log(2*pi) +
              -length(U_group)*log(sigma_u) +
              -0.5*sum(U_group^2)/sigma_u^2)
          )
        })))
    } else if(surv){

      ## Weibull AFT model
      Eta <- X_function %*% B + Z_function %*% U
      Mu <- mu_of_eta(Eta,tau_function)
      sum(D * (log(1/tau_function) + log(Mu))) +
      sum(-Mu) +
        sum((sapply(sigma_list_function,function(inds){
          Z_function_u <- Z_function[,inds]
          sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
          U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
          fix_inf(as.numeric(sigma_u > .00001),(
            -0.5*ncol(Z_function_u)*log(2*pi) +
              -ncol(Z_function_u)*log(sigma_u) +
              -0.5*sum(U_group^2)/sigma_u^2)
          )
        })))
    }
  }


  ## function of mu to get a weight
  w <- function(mu){
    if(softmax){
      (solve(as.matrix((v_of_mu(mu)%*%(link_prime(mu))^2))))
    } else{
      1/(v_of_mu(mu)*(link_prime(mu))^2)
    }
  }

  ## score as a function of Beta
  score <- function(tau_function,
                    X_function,
                    Y_function,
                    Z_function,
                    B,
                    U,
                    sigmas_function,
                    sigma_list_function,
                    lambda_upd = lambda,
                    softmax_function,
                    surv = survival,
                    D = d_censored){

    if(!softmax_function & !surv){
      Eta <- (X_function %*% B + Z_function %*% U)
      Mu <- sapply(Eta,mu_of_eta)
      prod <- sapply(Mu,function(mu){
        w(mu)*link_prime(mu)
      })
      prod[is.na(prod)] <- 0
      X_functionWD <- t(apply(X_function,2,function(x_i)x_i*prod))
      Z_functionWD <- t(apply(Z_function,2,function(x_i)x_i*prod))
      score_u <- cbind(c(unlist(sapply(sigma_list_function,function(inds){
        sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
        U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
        (as.numeric(sigma_u > .00001)*((Z_functionWD[inds,] %*% ((Y_function - Mu))/tau_function^2 - U_group/sigma_u^2)))
      }))))
      score_sigmas_function <- cbind(c(unlist(sapply(sigma_list_function,function(inds){
        sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
        U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
        (as.numeric(sigma_u > .00001)*(-0.5*length(U_group)/(sigma_u^2) +
                                         0.5*sum(U_group^2)/(sigma_u^4)))
      }))))
      mat <- rbind(
        cbind( (X_functionWD %*% (Y_function - Mu))) / tau_function^2,
        score_u,
        score_sigmas_function
      )
      mat - cbind(rep(lambda_upd,nrow(mat))) # light penalty to ensure p.d. of info
    } else if(softmax_function) {
      ## adjust indices to include all random intercepts
      sigma_list_function <- lapply(sigma_list_function,function(s){
        s <- c(s,s+length(s))
      })
      Eta <- (X_function %**% B + Z_function %**% U)
      Mu <- t(apply(Eta,1,mu_of_eta))
      score_b <- X_function %*t1*% lapply(1:ncol(Y_function),
                                function(j){
                                  Y_function[,j] - Mu[,j]
                                })
      score_u <- Z_function %*t1*% lapply(1:ncol(Y_function),
                                function(j){
                                  Y_function[,j] - Mu[,j]
                                })
      score_sigmas_function <- matrix(c(c(c(unlist(sapply(sigma_list_function,
                                                          function(inds){
        sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
        U_group <- c(unlist(lapply(U,diag))[inds])*as.numeric(sigma_u > 0.00001)
        k <- (as.numeric(sigma_u > .00001)*(-0.5*length(U_group)/(sigma_u^2) +
                                              0.5*sum(U_group^2)/(sigma_u^4)))
        c(diag(k,ncol(Y_function)))
      }))))),ncol = ncol(Y_function),byrow = T)

      mat <- rbind(
        score_b / tau_function^2,
        score_u / tau_function^2,
        score_sigmas_function
      )
      mat - sapply(1:ncol(mat),function(i)
        cbind(rep(lambda_upd,nrow(mat)))) # light penalty to ensure p.d. of info
    } else if(surv){
      Eta <- X_function %*% B + Z_function %*% U
      Mu <- mu_of_eta(Eta,tau_function)
      W <- diag(Mu)
      score_sigmas_function <- cbind(c(unlist(sapply(sigma_list_function,
                                                     function(inds){
        sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
        U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
        (as.numeric(sigma_u > .00001)*(-0.5*length(U_group)/(sigma_u^2) +
                                         0.5*sum(U_group^2)/(sigma_u^4)))
      }))))
      mat <- rbind(cbind(t(X_function) %*% (Mu-D) / tau_function),
            cbind(t(Z_function) %*% (Mu-D) / tau_function),
            score_sigmas_function)
      mat - sapply(1:ncol(mat),function(i){
        cbind(rep(lambda_upd,nrow(mat)))}) # light penalty to ensure p.d. of info
    }
  }

  ## Observed info as a function of Beta
  info <- function(tau_function,
                   X_function,
                   Z_function,
                   B,
                   U,
                   sigmas_function,
                   sigma_list_function,
                   fisher = T, # argument no longer matters in current implementation, ignore
                   softmax_function = F,
                   surv = survival,
                   D = d_censored,
                   lambda_upd = lambda){
    if(!softmax_function & !surv){
      Eta <- (X_function %*% B + Z_function %*% U)
      prod <- sapply(Eta,function(eta)sqrt(w(mu_of_eta(eta))))
      prod[is.na(prod)] <- 0
      X_functionW <- t(apply(X_function,2,function(x_i)x_i*prod))
      Z_functionW <- t(apply(Z_function,2,function(x_i)x_i*prod))
      mat1 <- X_functionW %*% t(X_functionW)
      mat2 <- (X_functionW %*% t(Z_functionW))
      mat3 <- t(mat2)

      mat4 <- matrix(0,nrow = ncol(Z_function),ncol = ncol(Z_function))
      for(i in 1:length(sigma_list_function)){
        inds <- sigma_list_function[[i]]
        Z_function_u <- Z_function[,inds]
        sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
        U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
        mat4[inds,inds] <- as.numeric(sigma_u > .00001)*(Z_functionW[inds,] %*% t(Z_functionW[inds,]) +
                                                           diag(as.numeric(tau_function^2)/sigma_u^2 ,ncol(Z_function_u)))
      }

      mat_null1 <- matrix(0,nrow = nrow(mat1),ncol = length(sigma_list_function))
      if(fisher){
        mat_cov1 <-  matrix(0,nrow = length(U),ncol = length(sigma_list_function))
      } else{
        mat_cov1 <- matrix(0,nrow = length(U),ncol = length(sigma_list_function))
        for(i in 1:length(sigma_list_function)){
          inds <- sigma_list_function[[i]]
          sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
          U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
          mat_cov1[inds,i] <- -cbind(U_group)/(sigma_u^4)
        }
      }
      mat_null2 <- t(mat_null1)
      mat_cov2 <- t(mat_cov1)
      if(fisher){
        mat5 <- diag(sapply(sigma_list_function,function(inds){
          sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
          U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
          as(
            (0.5*length(U_group)/(sigma_u^4)),
            'matrix')
        }),length(sigma_list_function))

      } else {
        mat5 <- diag(sapply(sigma_list_function,function(inds){
          sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
          U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
          as.numeric(sigma_u > .00001)*as(
            -( 0.5*length(U_group)/(sigma_u^4) - sum(U_group^2)/(sigma_u^6) ),
            'matrix')
        }),length(sigma_list_function))
      }
      tau_function <- as.numeric(abs(tau_function))

      ## final constructed information matrix
      mat <- rbind(cbind(mat1,mat2,mat_null1),
                   cbind(mat3,mat4,mat_cov1*tau_function^2),
                   cbind(mat_null2,mat_cov2*tau_function^2,mat5*tau_function^2)) / tau_function^2

      mat + diag(lambda_upd,ncol(mat)) ## light penalty to ensure p.d
    } else if(softmax_function){

        ## first, generate the canonical link-function parameter and transformed expectation
      Eta <- (X_function %**% B + Z_function %**% U)
      Mu <- t(apply(Eta,1,mu_of_eta))

        ## for general matrices of the form A B B C = A B^2 C
        ## where the information = B^2
        ## operations can be sped up greatly using the SVD decomposition
      W <- lapply(1:nrow(Mu),function(i){
        v <- v_of_mu(Mu[i,])
        sv <- svd(v)
        sv$u %*% diag(sqrt(sv$d))
      })

        ## these operations are the equivalent of standard REF form
        ## but utilizes custom 'tensorial' operations
      X_functionW <- X_function %*t2*% W
      Z_functionW <- Z_function %*t2*% W
      mat1 <- X_functionW %*t3*% X_functionW
      mat2 <- X_functionW %*t3*% Z_functionW
      mat3 <- t(mat2)
      mat4 <- Z_functionW %*t3*% Z_functionW +

        ## the variance of random effects are no longer single scalar variables,
        ## but K-1 by K-1 diagonal matrices, which must be kronecker-producted
        kronk(lapply(sigma_list_function,function(inds){
        sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
        diag(1/sigma_u^2,length(inds))
      }))

        ## now we have the fixed and random effects components of the information down,
        ## we compute the piece of the information relating to variance components
      mat_null1 <- matrix(0,nrow = nrow(mat1),ncol = length(sigma_list_function)*ncol(cbind(Y)))
      mat_cov1 <-  matrix(0,nrow = length(U),ncol = length(sigma_list_function)*ncol(cbind(Y)))
      mat_null2 <- t(mat_null1)
      mat_cov2 <- t(mat_cov1)

          ## again, the variance components are no longer scalars but K-1 by K-1 matrices
      mat5 <- kronk(lapply(sigma_list_function,function(inds){
        sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
        diag(
          (0.5*length(U)/(sigma_u^4)),ncol(cbind(Y)))
      }))

          ## for compatibility with dispersion models
      tau_function <- as.numeric(abs(tau_function))

        ## final constructed information matrix
      mat <- rbind(cbind(mat1,mat2,mat_null1),
                   cbind(mat3,mat4,mat_cov1*tau_function^2),
                   cbind(mat_null2,mat_cov2*tau_function^2,mat5*tau_function^2)) / tau_function^2

        ## adding a custom light l2 penalty helps ensure p.d.
      mat + diag(lambda_upd,ncol(mat))

    } else if(surv){

      Eta <- X_function %*% B + Z_function %*% U
      Mu <- mu_of_eta(Eta,tau_function)
      prod <- sqrt(Mu)
      sigMat <- kronk(lapply(sigma_list_function,function(inds){
        sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
        diag(1/sigma_u^2,length(inds))
      }))

      X_functionW <- t(apply(X_function,2,function(x_i)x_i*prod))
      Z_functionW <- t(apply(Z_function,2,function(x_i)x_i*prod))
      mat1 <- X_functionW %*% t(X_functionW)
      mat2 <- (X_functionW %*% t(Z_functionW))
      mat3 <- t(mat2)
      mat4 <- matrix(0,nrow = ncol(Z_function),ncol = ncol(Z_function))
      for(i in 1:length(sigma_list_function)){
        inds <- sigma_list_function[[i]]
        Z_function_u <- Z_function[,inds]
        sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
        U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
        mat4[inds,inds] <- as.numeric(sigma_u > .00001)*(Z_functionW[inds,] %*% t(Z_functionW[inds,]) +
                                                           diag(as.numeric(tau_function^2)/sigma_u^2 ,ncol(Z_function_u)))
      }

      mat_null1 <- matrix(0,nrow = nrow(mat1),ncol = length(sigma_list_function))
      if(fisher){
        mat_cov1 <-  matrix(0,nrow = length(U),ncol = length(sigma_list_function))
      } else{
        mat_cov1 <- matrix(0,nrow = length(U),ncol = length(sigma_list_function))
        for(i in 1:length(sigma_list_function)){
          inds <- sigma_list_function[[i]]
          sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
          U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
          mat_cov1[inds,i] <- -cbind(U_group)/(sigma_u^4)
        }
      }
      mat_null2 <- t(mat_null1)
      mat_cov2 <- t(mat_cov1)
      if(fisher){
        mat5 <- diag(sapply(sigma_list_function,function(inds){
          sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
          U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
          as(
            (0.5*length(U_group)/(sigma_u^4)),
            'matrix')
        }),length(sigma_list_function))

      } else {
        mat5 <- diag(sapply(sigma_list_function,function(inds){
          sigma_u <- max(sigmas_function[unique(names(inds))],0.00001)
          U_group <- c(U[inds])*as.numeric(sigma_u > 0.00001)
          as.numeric(sigma_u > .00001)*as(
            -( 0.5*length(U_group)/(sigma_u^4) - sum(U_group^2)/(sigma_u^6) ),
            'matrix')
        }),length(sigma_list_function))
      }
      tau_function <- as.numeric(abs(tau_function))

      ## final constructed information matrix
      mat <- rbind(cbind(mat1 / tau_function ^ 2,mat2 / tau_function ^ 2, mat_null1),
                   cbind(mat3 / tau_function ^ 2,mat4 / tau_function ^ 2, mat_cov1),
                   cbind(mat_null2,mat_cov2,mat5))

      mat + diag(lambda_upd,ncol(mat)) ## light penalty to ensure p.d if desired
    }

  }

  ## Criterion for convergence
  eps <- 100
  sc <- 100

  ## Run iteraton starting at beta hat
  if(length(beta_update)== 1){
    beta_update <- as(beta_update,'matrix')
  }

  ## ## ## ## Find unrestricted mles ## ## ## ##
  if(softmax){
    theta_update <- rbind(do.call('rbind',lapply(beta_update,function(x)rep(x,ncol(Y)))),
                          u_update,
                          matrix(t(sapply(list(sigmas),
                                          function(s){
                                            diag(rep(s,ncol(cbind(Y))))
                                          })),ncol = ncol(cbind(Y)))
    )
    beta_inds <- 1:length(beta_update)
    u_inds <- length(beta_update) + 1:nrow(u_update)
    sig_inds <- (1:nrow(theta_update))[-c(1:(length(beta_update) + nrow(u_update)))]
    theta_update <- theta_update[-sig_inds,]
  } else {
    theta_update <- c(beta_update,
                      u_update,
                      sigmas^2)
    sig_inds <- (1:length(theta_update))[-c(1:(length(beta_update) + length(u_update)))]
    theta_update <- theta_update[-c(sig_inds)]
  }
  eps <- 100
  count <- 0
  max_count <- ifelse2(softmax,25,25)
  sigs <- sigmas
  names(sigs) <- names(sigma_list)
  sigs <- ifelse(is.nan(sigs),0,sigs)
  #print("Starting Fisher-Scoring Algorithm")
  while(eps > 1e-5 & count < max_count){

    theta <- cbind(theta_update)
    theta_update <- cbind(theta_update)
    info_update <- info(tau,
                        X,
                        Z,
                        ifelse2(softmax,
                                lapply(beta_inds,function(r)diag(cbind(theta)[r,])),
                                unlist(lapply(1:length(c(beta_update)),function(r)c(cbind(c(theta))[r,])))),
                        ifelse2(softmax,
                                lapply(u_inds,function(r)diag(cbind(theta)[r,])),
                                unlist(lapply(length(c(beta_update))+1:length(c(u_update)),function(r)c(cbind(c(theta))[r,])))),
                        sigs,
                        sigma_list,
                        softmax_function = softmax)[-c(sig_inds),-c(sig_inds)]

    info_update[is.na(info_update) | is.nan(info_update) | !is.finite(info_update)] <- 0
    info_inv <- quick_inv(info_update)

    score_update <- score(tau,
                          X,
                          Y,
                          Z,
                          ifelse2(softmax,
                                  lapply(beta_inds,function(r)diag(cbind(theta)[r,])),
                                  unlist(lapply(1:length(c(beta_update)),function(r)c(cbind(c(theta))[r,])))),
                          ifelse2(softmax,
                                  lapply(u_inds,function(r)diag(cbind(theta)[r,])),
                                  unlist(lapply(length(c(beta_update))+1:length(c(u_update)),function(r)c(cbind(c(theta))[r,])))),
                          sigs,
                          sigma_list,
                          softmax_function = softmax)[-c(sig_inds),]
    score_update[is.na(score_update) | is.nan(score_update) | !is.finite(score_update)] <- 0


    if(count > 1){
      prev_curr_lik <- curr_lik
    }
    curr_lik <- compute_loglik(Y,X,Z,
                               ifelse2(softmax,
                                       lapply(beta_inds,function(r)diag(cbind(theta)[r,])),
                                       unlist(lapply(1:length(c(beta_update)),function(r)c(cbind(c(theta))[r,])))),
                               ifelse2(softmax,
                                       lapply(u_inds,function(r)diag(cbind(theta)[r,])),
                                       unlist(lapply(length(c(beta_update))+1:length(c(u_update)),function(r)c(cbind(c(theta))[r,])))),
                               tau,
                               cumgenfunc,
                               C,
                               sigs,
                               sigma_list  ,
                               probit_function = probit,
                               softmax_function = softmax)
    if(count > 1){
      if(prev_curr_lik > curr_lik){
        #print("Previous Likelihood Was Greater")
        break
      }
    }
    upd_lik <- curr_lik - 1
    big_step <- info_inv %*% score_update
    tau_upd <- tau

    ## start damp stepping
    damp_step_attempts <- 0
    while((curr_lik > upd_lik + 0.00001) & damp_step_attempts < 15){

      ## damp the newton step
      damp <- 1/(2^damp_step_attempts) # 1/2, 1/8, etc.
      damp_step_attempts <- damp_step_attempts + 1

      ## update fixed effects and random effects via a damped fisher-scoring step
      theta_update <-
        theta + damp*(big_step)

      ## if a survival outcome, half-step the Weibull scale parameter
      if(survival & est_tau){
        Eta <- X %*% cbind(theta_update)[1:length(beta_update),] +
               Z %*% cbind(theta_update)[length(beta_update) + 1:length(u_update),]
        Mu <- mu_of_eta(Eta,tau)
        tau_upd <- 0.5*tau + 0.5*as.numeric(
                                            -crossprod((Y - as.numeric(Eta)),
                                            (d_censored-Mu)/sum(d_censored)))
      }
      ## if a poisson over-dispersed model, half-step the dispersion parameter
      if(link_function == 'log' & est_tau){
        Eta <- X %*% cbind(theta_update)[1:length(beta_update),] +
               Z %*% cbind(theta_update)[length(beta_update) + 1:length(u_update),]
        Mu <- mu_of_eta(Eta)
        tau_upd <- 0.5*tau + 0.5*sqrt(mean((Y - Mu)^2/Mu))*(length(Y)/(length(Y)-length(beta_update)))
      }
      ## if a linear model, half-step the error variance
      if(link_function == 'linear' & est_tau){
        Eta <- X %*% cbind(theta_update)[1:length(beta_update),] +
          Z %*% cbind(theta_update)[length(beta_update) + 1:length(u_update),]
        Mu <- mu_of_eta(Eta)
        tau_upd <- 0.5*tau + 0.5*sqrt(mean((Y - Mu)^2))*(length(Y)/(length(Y)-length(beta_update)))
      }

      ## take two steps before adjusting the random effects variance components
      if(count > 0){

        ## update random effects variance components
        u_update_temp <- ifelse2(!softmax,c(
          cbind(theta_update)[length(beta_update) + 1:length(u_update),]),
          cbind(theta_update)[u_inds,])
        sigs <- ((c(sapply(sigma_list,function(inds){
          sqrt(mean((cbind(u_update_temp)[inds,])^2))
        }))))
        names(sigs) <- names(sigma_list)
        sigs <- ifelse(is.nan(sigs),0,sigs)
      }

      ## compute log-likelihood
      upd_lik <- try({compute_loglik(Y,X,Z,
                                ifelse2(softmax,
                                        lapply(beta_inds,function(r)diag(cbind(theta_update)[r,])),
                                        unlist(lapply(1:length(c(beta_update)),function(r)c(cbind(c(theta_update))[r,])))),
                                ifelse2(softmax,
                                        lapply(u_inds,function(r)diag(cbind(theta_update)[r,])),
                                        unlist(lapply(length(c(beta_update))+1:length(c(u_update)),function(r)c(cbind(c(theta_update))[r,])))),
                                tau_upd,
                                cumgenfunc,
                                C,
                                sigs,
                                sigma_list  ,
                                probit_function = probit,
                                softmax_function = softmax,
                                surv = )},silent = T)

      if(length(upd_lik) == 0){
        upd_lik <- curr_lik
        damp_step_attempts <- 15
        warning('Error in Computing Likelihood in Fisher-Scoring: Check for Computational Issues!')
      }
      if(class(upd_lik) == 'try-error' | is.na(upd_lik) | is.nan(upd_lik) | is.null(upd_lik) | !is.finite(upd_lik)){
        upd_lik <- curr_lik
        damp_step_attempts <- 15
        warning('Error in Computing Likelihood in Fisher-Scoring: Check for Computational Issues!')
      }
    }


    eps <- norm(cbind(theta_update)-cbind(theta),'2') +
      sqrt((tau-tau_upd)^2)
    tau <- tau_upd
    count <- count + 1
  }

  ## final updated MLEs
  u_update <- ifelse2(!softmax,c(
    cbind(theta_update)[length(beta_update) + 1:length(u_update),]),
    cbind(theta_update)[u_inds,])
  beta_update <- ifelse2(!softmax,c(
    cbind(theta_update)[1:length(beta_update),]),
    cbind(theta_update)[beta_inds,])
  sigmas <-
    c(sapply(sigma_list,function(inds){
      sqrt(mean((cbind(u_update)[inds,])^2))
    }))
  names(sigmas) <- names(sigma_list)
  if(softmax){
    beta_update <- lapply(1:nrow(beta_update),function(i){
      diag(beta_update[i,])
    })
    u_update <- lapply(1:nrow(u_update),function(i){
      diag(u_update[i,])
    })
  }

  newton_iter <- count

    ## final information matrix is evaluated at MLEs
    info_update <-
      info(tau,
           X,
           Z,
           beta_update,
           u_update,
           sigmas, sigma_list,softmax_function = softmax,
           fisher = F)
    info_update[is.na(info_update) | is.nan(info_update) | !is.finite(info_update)] <- 0

    ## final score evaluated at MLEs
    score_update <- score(tau,
                          X,
                          Y,
                          Z,
                          beta_update,
                          u_update,
                          sigmas,sigma_list,lambda,softmax_function = softmax)
    score_update[is.na(score_update) | is.nan(score_update) | !is.finite(score_update)] <- 0

    sc_old <- sc
    sc <- norm(info_update, '2') # for numerical stability
    eps <- abs(sc_old - sc)
    beta_mle <- beta_update
    #print("Starting solving for MLEs under constraint")
    #if(!(link_function == 'weibull')){

    ## Run sequential quadratic programming
    if(use_quadprog){
      count <- 0
      eps <- 100
      weights <- rep(0,length(beta_update))
      while(eps > 0.0001 & count < 100){
      #print(eps)
      prev_weights <- weights
      weights <- solve.QP(
        as(nearPD(info_update[1:length(beta_update),1:length(beta_update)])[[1]],'matrix')/sc,
        dvec =
          cbind(

            ## for multinom where k > 2,
            ## rowMeans() projects a k-dimensional space onto 1-dimensional space
            ## = the matrix operation (1/k,1/k....)'  %*% [-db + .5b'Db] %*% (1/k,1/k.....)
            ## projects onto 1 dimensional space
            rowMeans(cbind(
              cbind(score_update)[1:length(beta_update),] + #
                info_update[1:length(beta_update),1:length(beta_update)] %*%
                cbind(theta_update)[1:length(beta_update),]))
          ) / sc,
        Amat = cbind(1,diag(length(beta_update))),
        bvec = c(sum_constraint,ifelse2(length(lower_bound_constraint) == length(beta_update),
                                        lower_bound_constraint,
                                        rep(lower_bound_constraint,length(beta_update)))),
        meq = 1
      )$solution
      prev_eps <- eps
      eps <- sqrt(sum((prev_weights - weights)^2))

      ## discontinue if the error starts to increase, or isn't changing between iterations
      if(prev_eps <= eps & count >= 5){
        eps <- 0
      }
      count <- count + 1
      if(!softmax){
        (beta_update <- weights[1:length(beta_update)])
      } else {
        (beta_update <- lapply(weights,function(weight)diag(weight,ncol(cbind(Y)))))
      }
      names(beta_update) <- b_names
      names(u_update) <- u_names
      info_update <-
        info(tau,
             X,
             Z,
             beta_update,
             u_update,
             sigmas, sigma_list,softmax_function = softmax,
             fisher = F)
      info_update[is.na(info_update) | is.nan(info_update) | !is.finite(info_update)] <- 0

      score_update <- score(tau,
                            X,
                            Y,
                            Z,
                            beta_update,
                            u_update,
                            sigmas,sigma_list,lambda,softmax_function = softmax)
      score_update[is.na(score_update) | is.nan(score_update) | !is.finite(score_update)] <- 0
      sc <- norm(info_update, '2')
      }

    ## Run non-linear programming
    } else {
        beta_update <- sapply(rowMeans(cbind(cbind(theta_update)[1:length(beta_update),])),
                              function(b)max(b,0))
        beta_update <- beta_update/sum(beta_update)
        LB <- ifelse2(length(lower_bound_constraint) == length(beta_update),
                      lower_bound_constraint,
                      rep(lower_bound_constraint,length(beta_update)))
        weights <- try({solnp( beta_update,
                          function(weights) -compute_loglik(Y,
                                                            X,
                                                            Z,
                                                            ifelse2(softmax,
                                                                    lapply(weights,function(weight)diag(weight,ncol(cbind(Y)))),
                                                                    weights),#beta_update,
                                                            u_update,
                                                            tau,
                                                            cumgenfunc,
                                                            C,
                                                            sigmas,
                                                            sigma_list,
                                                            probit,
                                                            softmax_function = softmax),
                          eqfun = function(x) sum(x), eqB = sum_constraint,
                          LB = LB,
                          control = list(trace = FALSE))$pars},silent = T)
        if(class(weights) == 'try-error'){

          ## if initial values don't work, go back to 1/length(beta_update)
          weights <- try({solnp( rep(1/length(beta_update),
                                     length(beta_update)),
                                 function(weights) -compute_loglik(Y,
                                                                   X,
                                                                   Z,
                                                                   ifelse2(softmax,
                                                                           lapply(weights,function(weight)diag(weight,ncol(cbind(Y)))),
                                                                           weights),#beta_update,
                                                                   u_update,
                                                                   tau,
                                                                   cumgenfunc,
                                                                   C,
                                                                   sigmas,
                                                                   sigma_list,
                                                                   probit,
                                                                   softmax_function = softmax),
                                 eqfun = function(x) sum(x), eqB = sum_constraint,
                                 LB = LB,
                                 control = list(trace = FALSE))$pars},silent = T)

          ## still didn't work? weights are just 1/length(beta_update) unweighted average
          if(class(weights) == 'try-error'){
            weights <- rep(1/length(beta_update), length(beta_update))
            warning(paste0("solnp() failed: all weights = 1/",length(beta_update)))
          }
        }
        count <- count + 1
        if(!softmax){
          (beta_update <- weights[1:length(beta_update)])
        } else {
          (beta_update <- lapply(weights,function(weight)diag(weight,ncol(cbind(Y)))))
        }
        names(beta_update) <- b_names
        names(u_update) <- u_names
    }
    #print("Finished Fitting Under Constraint")
  loglik <- compute_loglik(Y,X,Z,beta_update,u_update,tau,cumgenfunc,C,sigmas,sigma_list,probit,softmax_function = softmax)
  n <- length(Y)
  k <- length(weights[round(weights,4) > 0.0001])
  AIC <- ifelse2(link_function != 'linear',
                 as.numeric(-2*loglik + 2*k),
                 as.numeric(-2*loglik + 2*k*n/(n-k-1))) ## use corrected AICc when possible
  #print("Returning Fit")
  return(list("weights" = weights,
              "beta_update" = beta_update,
              "u_update" = u_update,
              'loglik' = loglik,
              'AIC' = AIC,
              'quadprog_iterations' = ifelse2(use_quadprog,count,NA),
              'vcov' = try({quick_inv(info_update)},
                         silent = T),
              'sigmas' = sigmas,
              'sigma_list' = sigma_list,
              'tau' = tau,
              'newton_iter' = newton_iter,
              'beta_mle' = beta_mle,
              'damp_step_attempts' = damp_step_attempts,
              'loglik_function' = compute_loglik,
              'Y' = Y,
              'X' = X,
              'Z' = Z,
              'cumgenfunc' = cumgenfunc,
              'C' = C,
              'probit' = probit,
              'softmax' = softmax))
}


