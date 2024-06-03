% Use Newton's method in conjunction with the conjugate gradient scheme
% to solve the generalized LQR consensus problem
%
%      minimize    \bar{J}(K)
%      subject to   K*(I_M \kron 1_n) = 0
%                   K \hadamard (1_M^T \kron ES) = 0
%
% Syntax:
% [K_opt,J_opt] = glqrc(Z,Q,R,ES,K0,tolerance)
%
% Inputs: 
% state-space representation data: {Z,Q,R},
% sparsity constraint matrix: ES,
% initial condition: K0,
% stopping criterion tolerance: tolerance.
%
% Outputs: 
% minimizer K_opt and minimum value Jopt.
function [K_opt,J_opt] = glqrc(Z,Q,R,ES,K0,tolerance)

    % Check if the dimensions of the input matrices are correct
    n = length(K0(:,1));
    M = length(K0(1,:))/n;
    if M-floor(M) > 0
        error('K0 has incompatible dimensions')
    end
    if length(ES(:,1))~=n || length(ES(1,:))~=n
        error('ES has the wrong dimensions')
    end
    if norm(K0.*kron(ones(1,M),ES),'fro')>1e-10
        error('K0 does not fullfil the sparsity imposed by ES')
    end
    if length(Z(:,1))~=n*M || length(Z(1,:))~=n*M
        error('Z has incompatible dimensions')
    end
    if length(Q(:,1))~=n*M || length(Q(1,:))~=n*M
        error('Q has incompatible dimensions')
    end
    if length(R(:,1))~=n || length(R(1,:))~=n
        error('R has incompatible dimensions')
    end
    
    % Generate orthonormal basis for off-consensus subspace
    S = diag(n:-1:1)-tril(ones(n,n));
    S = S(:,1:end-1)./sqrt((n:-1:2).*(n-1:-1:1));
    S_M = kron(eye(M),S);

    % Create function to compute the closed-loop matrix associated to a gain
    A_M = diag(ones(M-1,1),1);
    AA = kron(A_M(1:end-1,:),eye(n-1));
    A_CL = @(K) [AA;-S'*K*S_M];

    % Check if the initial condition leads to consensus before using it
    if max(real(eig(A_CL(K0)))) >= 0
        error('The initial gain K0 does not lead to consensus!')
    end
    K = K0;
    
    % Contruct the projector of the gain K onto the feasible set
    I = eye(n);
    e = @(i) I(:,i);
    o = ones(n,1);
    Projectors = zeros(n,n,n);
    for i=1:n
        ESS = find(ES(i,:));
        T = [(o-sum(e(ESS),2))/sqrt(n-length(ESS)),e(ESS)];
        Projectors(:,:,i) = eye(n) - T*T';
    end
    Proj = @(K) reshape(pagemtimes(Projectors,reshape(K',n,M,n)),M*n,n)';

    % Compute the objective
    Z_til = S_M'*Z*S_M;     % To save computation
    Q_til = S_M'*Q*S_M;     % To save computation
    X = S_M*lyap(A_CL(K),Z_til)*S_M';   
    J = trace((Q + K'*R*K)*X);

    % Iterate
    max_iter = 300;
    disp_iter = 1;
    alpha = 0.3;        % Variable related to the Armijo rule
    beta  = 0.5;        % Variable related to the Armijo rule
    for iter=1:max_iter
        % Compute the gradient 
        P = S_M*lyap(A_CL(K)', Q_til + S_M'*K'*R*K*S_M)*S_M';
        grad_K = 2*Proj((R*K - P(end-n+1:end,:))*X);

        % Check stopping criterion
        norm_grad = norm(grad_K,'fro');
        if disp_iter
            disp(['Iteration ',num2str(iter),char(9),'Norm of Gradient: ',num2str(norm_grad,'%6.3E')])
        end
        if norm_grad < tolerance
            break;
        end

        % Compute the Newton direction using the conjugate gradient scheme
        K_Newton = glqrc_CG(K,P,X,R,grad_K,ES,S_M,Proj);

        % Line search
        stepsize = 1;
        while 1
            K_next = K + stepsize*K_Newton;
            maxEigAcl = max(real(eig(A_CL(K_next))));
            if maxEigAcl >= 0
                J_next = Inf;
            else
                X_next = S_M*lyap(A_CL(K_next),Z_til)*S_M';
                QQ_next = Q + K_next'*R*K_next;
                J_next = X_next(:)'*QQ_next(:);
            end

            % Armijo rule
            if  J - J_next > -stepsize*alpha*K_Newton(:)'*grad_K(:)
                break;
            end

            stepsize = beta*stepsize;
            if stepsize < 1.e-16            
                error('Extremely small stepsize. Stopping method.');            
            end

        end

        % update current step
        K = K_next;
        X = X_next;
        J = J_next;
    end
    
    if iter == max_iter
        disp('Maximum number of iterations reached.')
        disp(['The norm of the gradient is ', num2str(norm_grad), '.'])
    end    
    
    K_opt = K;
    J_opt = J;
end
