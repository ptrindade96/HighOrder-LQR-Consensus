% Use a truncated Newton method to solve the LQR consensus problem with
% symmetry on the Laplacian
%
%      minimize    \bar{J}(\gamma \kron E*Diag(w)*E')
%
% Syntax:
% [L_opt,gamma_opt,J_opt] = lqrc_newtonCG_sym(M,Z,Q,R,L0,gamma_0,tolerance);
%
% Inputs: 
% problem data: {M,Z,Q,R},
% initial conditions: {L0,gamma_0},
% tolerance for the stopping criterion: tolerance.
%
% Outputs: 
% minimizers {L_opt,gamma_opt} and minimum value J_opt.

function [L_opt,gamma_opt,J_opt] = lqrc_newtonCG_sym(Z,Q,R,L0,gamma_0,tolerance)

    % Generate orthonormal basis for off-consensus subspace
    M = length(gamma_0);
    n = length(L0(1,:));
    S = diag(n:-1:1)-tril(ones(n,n));
    S = S(:,1:end-1)./sqrt((n:-1:2).*(n-1:-1:1));
    S_M = kron(eye(M),S);

    % Define some useful operators
    I = eye(n);
    e = @(i) I(:,i);

    % Retrieve the incidence matrix and the vector of weights from L0
    p = nnz(triu(L0,1));
    w0 = zeros(p,1);
    E = zeros(n,p);
    p_ = 0;
    for i=1:n
        for j=i+1:n
            if abs(L0(i,j))>1e-8
                p_ = p_ + 1;
                w0(p_) = -L0(i,j);
                E(:,p_) = e(i)-e(j);
            end
            if p_ == p
                break
            end
        end
    end

    % Create function to compute the closed-loop matrix for each pair (gamma, w)
    A_M = diag(ones(M-1,1),1);
    AA = kron(A_M(1:end-1,:),eye(n-1));
    A_CL = @(gamma,w) [AA;-kron(gamma',S'*E*diag(w)*E'*S)];

    % Set the variables using the initial conditions
    K = kron(gamma_0',E*diag(w0)*E');
    w = w0;
    gamma = gamma_0;
    
    % Compute the objective
    Z_til = S_M'*Z*S_M;     % To save computation
    Q_til = S_M'*Q*S_M;     % To save computation
    X = S_M*lyap(A_CL(gamma,w),Z_til)*S_M';   
    J = trace((Q + K'*R*K) * X);

    % Iterate
    max_iter = 300;
    disp_iter = 1;
    alpha = 0.3;        % Variable related to the Armijo rule
    beta  = 0.5;        % Variable related to the Armijo rule
    for iter = 1:max_iter
        L = E*diag(w)*E';
        
        % Compute the gradients over gamma and L 
        P = S_M*lyap(A_CL(gamma,w)', Q_til + kron(gamma*gamma',S'*L'*R*L*S))*S_M';
        grad_K = 2*(R*K - P((M-1)*n+1:end,:))*X; 
        grad_gamma = reshape(grad_K,n^2,M)'*L(:);
        grad_w = diag(E'*reshape(sum(reshape(grad_K.*kron(gamma',ones(n,n)),n^2,M),2),n,n)*E);
    
        % Stopping criterion
        norm_grad = sqrt(grad_w'*grad_w + grad_gamma'*grad_gamma);
        if disp_iter
            disp(['Iteration ',num2str(iter),char(9),'Norm of Gradient: ',num2str(norm_grad,'%6.3E')])
        end
        if norm_grad < tolerance
            break;
        end
    
        % Compute the Newton direction using the conjugate gradient scheme
        [w_Newton,gamma_Newton] = lqrc_CG_sym(E,w,gamma,P,X,R,grad_K,grad_w,grad_gamma);
    
        % Line Search
        stepsize = 1;
        while 1
            w_next = w + stepsize*w_Newton;
            gamma_next = gamma + stepsize*gamma_Newton;
            L_next = E*diag(w_next)*E';
            maxEigAcl = max(real(eig(A_CL(gamma_next,w_next))));
            if maxEigAcl >= 0
                J_next = Inf;
            else
                X_next = S_M*lyap(A_CL(gamma_next,w_next),Z_til)*S_M';
                QQ_next = Q + kron(gamma_next*gamma_next',L_next'*R*L_next);
                J_next = X_next(:)'*QQ_next(:);
            end
    
            % Armijo rule
            if  J - J_next > -stepsize*alpha*(w_Newton'*grad_w+gamma_Newton'*grad_gamma)
                break;
            end
    
            stepsize = beta*stepsize;
            if stepsize < 1.e-16            
                disp('Extremely small stepsize. Stopping method.');
                disp('Returning latest value.');
                gamma_opt = gamma;
                L_opt = E*diag(w)*E';
                J_opt = J;
                return
            end
    
        end
        % update current step
        w = w + stepsize*w_Newton;
        gamma = gamma + stepsize*gamma_Newton;
        K = kron(gamma',E*diag(w)*E');
        X = X_next;
        J = J_next;
    end
    
    if iter == max_iter
        disp('Maximum number of iterations reached.')
        disp(['The norm of the gradient is ', num2str(norm_grad), '.'])
    end    
    
    gamma_opt = gamma;
    L_opt = E*diag(w)*E';
    J_opt = J;

end


%% Conjugate gradient method to compute Newton direction
% Inputs: 
% Current variables: w and gamma
% Solutions of the Lyapunov equations associated with the gradient: P, X
% Weigthing matrix associated with the input vector: R
% Gradients: grad_K, grad_w and grad_gamma
%
% Output: 
% Newton directions w_Newton and gamma_Newton.

function [w_Newton,gamma_Newton] = lqrc_CG_sym(E,w,gamma,P,X,R,grad_K,grad_w,grad_gamma)
    
    %%% Define the vec operator
    vec = @(X) X(:);
    
    %%% Construct some needed matrices
    M = length(gamma);
    n = length(E(:,1));
    K = kron(gamma',E*diag(w)*E');
    A = kron(diag(ones(M-1,1),1),eye(n));
    B = kron([zeros(M-1,1);1],eye(n));
    S = diag(n:-1:1)-tril(ones(n,n));
    S = S(:,1:end-1)./sqrt((n:-1:2).*(n-1:-1:1));
    S_M = kron(eye(M),S);
    A_CL = S_M'*(A - B*K)*S_M;  % To save computation
    F = R*K - B'*P;             % To save computation
    G1 = zeros(M*(n-1),M*(n-1));

    %%% Initialization of the conjugate gradient scheme
    w_tilde = zeros(length(w),1);
    gamma_tilde = zeros(M,1);
    r_w = grad_w;
    r_gamma = grad_gamma;
    p_w = -r_w;
    p_gamma = -r_gamma;
    
    %%% Stopping criterion
    norm_grad_gamma_w = norm([grad_gamma;grad_w]);
    epsilon = min(0.5,sqrt(norm_grad_gamma_w))*norm_grad_gamma_w;      
    
    %%% Iterations of the conjugate gradient scheme
    q = M+length(w);
    disp_msgs = 0;      % flag to control warning messages
    for k = 0:q
        % Start by computing H in the direction kron(gamma,E*diag(p_w)*E') + kron(p_gamma,E*diag(w)*E')
        P_K = kron(gamma',E*diag(p_w)*E') + kron(p_gamma',E*diag(w)*E');
        G1(end-n+2:end,:) = S'*P_K*X*S_M;
        G2 = -F'*P_K;
        Xtilde = S_M*lyap(A_CL,-(G1 + G1'))*S_M';
        Ptilde = S_M*lyap(A_CL',-S_M'*(G2 + G2')*S_M)*S_M';    
        H = 2*((R*P_K - B'*Ptilde)*X + F*Xtilde);
        h_w = diag(E'*(reshape(sum(reshape(H.*kron(gamma',ones(n,n)),n^2,M),2),n,n)+reshape(sum(reshape(grad_K.*kron(p_gamma',ones(n,n)),n^2,M),2),n,n))*E);
        h_gamma = reshape(H,n^2,M)'*vec(E*diag(w)*E') + reshape(grad_K,n^2,M)'*vec(E*diag(p_w)*E');

        % Negative Curvature Test
        inner = h_w(:)'*p_w(:) + p_gamma'*h_gamma;
        if (inner <= 0) && (k == 0)
            gamma_Newton = -grad_gamma;
            w_Newton = -grad_w;
            if disp_msgs
                disp('Negative curvature. Stopping CG.')
            end
            break;
        elseif (inner <= 0) && (k > 0)
            gamma_Newton = gamma_tilde;
            w_Newton = w_tilde;
            if disp_msgs
                disp('Negative curvature. Stopping CG.')
            end
            break;
        end
        
        % Update L_tilde, g_tilde, and the residuals
        r_norm_squared = r_w(:)'*r_w(:) + r_gamma'*r_gamma;
        alpha = r_norm_squared/inner;
        w_tilde = w_tilde + alpha*p_w;
        gamma_tilde = gamma_tilde + alpha*p_gamma;
        r_w_next = r_w + alpha*h_w;
        r_gamma_next = r_gamma + alpha*h_gamma;
        
        % Stopping Criterion
        r_norm_next = sqrt(r_w_next(:)'*r_w_next(:) + r_gamma_next'*r_gamma_next);
        if r_norm_next < epsilon
            gamma_Newton = gamma_tilde;
            w_Newton = w_tilde;
            break
        end

        % Update directions
        beta = r_norm_next^2/r_norm_squared;
        p_w = -r_w_next + beta*p_w;
        p_gamma = -r_gamma_next + beta*p_gamma;
        r_w = r_w_next;
        r_gamma = r_gamma_next;
    end
    
    if k == q
        gamma_Newton = gamma_tilde;
        w_Newton = w_tilde;
        disp('Maximum number of conjugate gradient method iterations reached!')
        disp(['Norm of the residual is ',num2str( r_norm_next, '%10.2E' ),'.'])        
    end

end


