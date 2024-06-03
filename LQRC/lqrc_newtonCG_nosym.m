%% Truncated Newton method for the non-symmetric case
% 
% Use a truncated Newton method to solve the LQR consensus problem without 
% symmetry:
%
%      minimize    J(\gamma \otimes L)
%      subject to  L \in structure
%
% Syntax:
% [L_opt,gamma_opt,J_opt] = lqrc_newtonCG_nosym(M,Z,Q,R,ES,L0,gamma_0,tolerance);
%
% Inputs: 
% problem data: {M,Z,Q,R},
% initial conditions: {L0,gamma_0},
% tolerance for the stopping criterion: tolerance.
%
% Outputs: 
% minimizers {L_opt,gamma_opt} and minimum value J_opt.

function [L_opt,gamma_opt,J_opt] = lqrc_newtonCG_nosym(Z,Q,R,L0,gamma_0,tolerance)

    % Generate orthonormal basis for off-consensus subspace
    M = length(gamma_0);
    n = length(L0(1,:));
    S = diag(n:-1:1)-tril(ones(n,n));
    S = S(:,1:end-1)./sqrt((n:-1:2).*(n-1:-1:1));
    S_M = kron(eye(M),S);

    % Create function to compute the closed-loop matrix for each pair (gamma, L)
    A_M = diag(ones(M-1,1),1);
    AA = kron(A_M(1:end-1,:),eye(n-1));
    A_CL = @(gamma,L) [AA;-kron(gamma',S'*L*S)];

    % Set the variables using the initial conditions
    L = L0;
    gamma = gamma_0;
    K = kron(gamma',L);

    % Contruct the projectors for the Laplacian matrices
    I = eye(n);
    e = @(i)I(:,i);
    o = ones(n,1);
    Projectors = zeros(n,n,n);
    ES = abs(L0)>1e-8;
    for i=1:n
        ESS = find(not(ES(i,1:n)));
        T = [(o-sum(e(ESS),2))/sqrt(n-length(ESS)),e(ESS)];
        Projectors(:,:,i) = eye(n) - T*T';
    end
    Proj = @(K) reshape(pagemtimes(Projectors,reshape(K',n,1,n)),n,n)';
    
    % Compute the objective
    Z_til = S_M'*Z*S_M;     % To save computation
    Q_til = S_M'*Q*S_M;     % To save computation
    X = S_M*lyap(A_CL(gamma,L),Z_til)*S_M';   
    J = trace((Q + K'*R*K) * X);
   
    % Iterate
    disp_iter = 1;
    max_iter = 300;
    alpha = 0.3;        % Variable related to the Armijo rule
    beta  = 0.5;        % Variable related to the Armijo rule
    for iter = 1:max_iter
        
        % Compute the gradients over g and L 
        P = S_M*lyap(A_CL(gamma,L)', Q_til + kron(gamma*gamma',S'*L'*R*L*S))*S_M';
        grad_K = 2*(R*K - P((M-1)*n+1:end,:))*X; 
        grad_L = Proj(reshape(sum(reshape(grad_K.*kron(gamma',ones(n,n)),n^2,M),2),n,n));
        grad_gamma = reshape(grad_K,n^2,M)'*L(:);
    
        % Stopping criterion
        norm_grad = sqrt(norm(grad_L,'fro')^2 + grad_gamma'*grad_gamma);
        if disp_iter
            disp(['Iteration ',num2str(iter),char(9),'Norm of Gradient: ',num2str(norm_grad,'%6.3E')])
        end
        if norm_grad < tolerance
            break;
        end
    
        % Compute the Newton direction using the conjugate gradient scheme
        [L_Newton,gamma_Newton] = lqrc_CG_nosym(L,gamma,P,X,R,grad_K,grad_L,grad_gamma,Proj);
    
        % Line Search
        stepsize = 1;
        while 1
            L_next = L + stepsize*L_Newton;
            gamma_next = gamma + stepsize*gamma_Newton;
            maxEigAcl = max(real(eig(A_CL(gamma_next,L_next))));
            if maxEigAcl >= 0
                J_next = Inf;
            else
                X_next = S_M*lyap(A_CL(gamma_next,L_next),Z_til)*S_M';  
                QQ_next = Q + kron(gamma_next*gamma_next',L_next'*R*L_next);
                J_next = X_next(:)'*QQ_next(:);
            end
    
            % Armijo rule
            if  J - J_next > -stepsize*alpha*(L_Newton(:)'*grad_L(:)+gamma_Newton'*grad_gamma)
                break;
            end
    
            stepsize = beta*stepsize;
            if stepsize < 1.e-16            
                disp('Extremely small stepsize. Stopping method.'); 
                disp('Returning latest value.');
                gamma_opt = gamma;
                L_opt = L;
                J_opt = J;
                return
            end
    
        end
        % update current step
        L = L + stepsize*L_Newton;
        gamma = gamma + stepsize*gamma_Newton;
        K = kron(gamma',L);
        X = X_next;
        J = J_next;
    end
    
    if iter == max_iter
        disp('Maximum number of iterations reached.')
        disp(['The norm of the gradient is ', num2str(norm_grad), '.'])
    end    
    
    gamma_opt = gamma;
    L_opt = L;
    J_opt = J;

end


%% Conjugate gradient method to compute Newton direction
% Inputs: 
% Current variables: L and gamma
% Solutions of the Lyapunov equations associated with the gradient: P, X
% Weigthing matrix associated with the input vector: R
% Gradients: grad_K and grad_gamma
% Ortogonal projector operator onto the constraint space: Proj
%
% Output: 
% Newton directions w_Newton and gamma_Newton.

function [L_Newton,gamma_Newton] = lqrc_CG_nosym(L,gamma,P,X,R,grad_K,grad_L,grad_gamma,Proj)
    
    %%% Construct some needed matrices
    M = length(gamma);
    n = length(L(:,1));
    K = kron(gamma',L);
    A = kron(diag(ones(M-1,1),1),eye(n));
    B = kron([zeros(M-1,1);1],eye(n));
    S = diag(n:-1:1)-tril(ones(n,n));
    S = S(:,1:end-1)./sqrt((n:-1:2).*(n-1:-1:1));
    S_M = kron(eye(M),S);
    A_CL = S_M'*(A - B*K)*S_M;   % To save computation
    F = R*K - B'*P;              % To save computation
    G1 = zeros(M*(n-1),M*(n-1));

    %%% Initialization of the conjugate gradient scheme
    L_tilde = zeros(n,n);
    gamma_tilde = zeros(M,1);
    R_L = grad_L;
    r_gamma = grad_gamma;
    P_L = -R_L;
    p_gamma = -r_gamma;

    %%% Stopping criterion
    norm_grad_gammaL = sqrt(norm(grad_L,'fro')^2 + grad_gamma'*grad_gamma);
    epsilon = min(0.5,sqrt(norm_grad_gammaL)) * norm_grad_gammaL;
   
    %%% Iterations of the conjugate gradient scheme
    q = n^2-n+M;
    disp_msgs = 0;      % flag to control warning messages
    for k = 0:q
        % Start by computing H in the direction kron(gamma',P_L) + kron(p_gamma',L)
        P_K = kron(gamma',P_L) + kron(p_gamma',L);
        G1(end-n+2:end,:) = S'*P_K*X*S_M;
        G2 = -F'*P_K;
        Xtilde = S_M*lyap(A_CL,-(G1 + G1'))*S_M';
        Ptilde = S_M*lyap(A_CL',-S_M'*(G2 + G2')*S_M)*S_M';    
        H = 2*((R*P_K - B'*Ptilde)*X + F*Xtilde); 
        H_L = Proj(reshape(sum(reshape(H.*kron(gamma',ones(n,n)),n^2,M),2),n,n)+reshape(sum(reshape(grad_K.*kron(p_gamma',ones(n,n)),n^2,M),2),n,n));
        h_gamma = reshape(H,n^2,M)'*L(:) + reshape(grad_K,n^2,M)'*P_L(:);

        % Negative Curvature Test
        inner = H_L(:)'*P_L(:) + p_gamma'*h_gamma;
        if (inner <= 0) && (k == 0)
            gamma_Newton = -grad_gamma;
            L_Newton = -grad_L;
            if disp_msgs
                disp('Negative curvature. Stopping CG.')
            end
            break;
        elseif (inner <= 0) && (k > 0)
            gamma_Newton = gamma_tilde;
            L_Newton = L_tilde;
            if disp_msgs
                disp('Negative curvature. Stopping CG.')
            end
            break;
        end
        
        % Update L_tilde, g_tilde, and the residuals
        r_norm_squared = R_L(:)'*R_L(:) + r_gamma'*r_gamma;
        alpha = r_norm_squared/inner;
        L_tilde = L_tilde + alpha*P_L;
        gamma_tilde = gamma_tilde + alpha*p_gamma;
        R_L_next = R_L + alpha*H_L;
        r_gamma_next = r_gamma + alpha*h_gamma;
        
        % Stopping Criterion
        r_norm_next = sqrt(R_L_next(:)'*R_L_next(:) + r_gamma_next'*r_gamma_next);
        if r_norm_next < epsilon
            gamma_Newton = gamma_tilde;
            L_Newton = L_tilde;
            break
        end

        % Update directions
        beta = r_norm_next^2/r_norm_squared;
        P_L = -R_L_next + beta*P_L;
        p_gamma = -r_gamma_next + beta*p_gamma;
        R_L = R_L_next;
        r_gamma = r_gamma_next;
    end
    
    if k == q
        gamma_Newton = gamma_tilde;
        L_Newton = L_tilde;
        disp('Maximum number of conjugate gradient method iterations reached!')
        disp(['Norm of residual is ',num2str( r_norm_next, '%10.2E' ),'.'])         
    end

end


