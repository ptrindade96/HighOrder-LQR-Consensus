% Uses the Newton method to solve the Fixed Laplacian LQR consensus problem
%
%      minimize    \bar{J}_L(\gamma) = \bar{J}(\gamma' \kron L)
%
% Syntax:
% [gamma_opt,J_opt] = fllqrc(L,Z,Q,R,gamma_0,tolerance);
%
% Inputs: 
% problem data: {L,Z,Q,R},
% initial condition gamma_0,
% tolerance for the stopping criterion: tolerance.
%
% Outputs: 
% minimizer gamma_opt and minimum value J_opt.
function [gamma_opt,J_opt] = fllqrc(L,Z,Q,R,gamma_0,tolerance)

    % Check if the dimensions of the input matrices are correct
    M = length(gamma_0);
    n = length(L(1,:));
    if length(L(:,1))~=n
        error('L is not square!');
    end
    if ~all(size(Z)==[n*M,n*M])
        error('Z has incompatible dimensions')
    end
    if ~all(size(Q)==[n*M,n*M])
        error('Q has incompatible dimensions')
    end
    if ~all(size(R)==[n,n])
        error('R has incompatible dimensions')
    end

    % Generate an orthonormal basis for off-consensus subspace
    o = ones(n,1);
    S = diag(n:-1:1)-tril(ones(n,n));
    S = S(:,1:end-1)./sqrt((n:-1:2).*(n-1:-1:1));
    S_M = kron(eye(M),S);

    % Create modified variables and save computations
    L_til = S'*L*S;
    Z_til = S_M'*Z*S_M;
    Q_til = S_M'*Q*S_M;
    S_R = S+(o*o'*L*S/n)/L_til;
    R_til = S_R'*R*S_R;
    LRL_til = L_til'*R_til*L_til;
    G1 = zeros(M*(n-1),M*(n-1));

    % Create function to compute the closed-loop matrix associated to a gain
    A_CL = @(g) [kron([zeros(M-1,1),eye(M-1)],eye(n-1));-kron(g',L_til)];

    % Check if the initial condition leads to consensus before using it
    if max(real(eig(A_CL(gamma_0)))) >= -1e-8
        error('The initial conditions L0 and g0 do not lead to consensus!')
    end
    k = log(gamma_0);
    
    % Compute the objective
    X = lyap(A_CL(exp(k)),Z_til);   
    J = trace((Q_til + kron(exp(k)*exp(k)',LRL_til)) * X);

    % Iterate
    max_iter = 100;
    disp_iter = 1;
    alpha = 0.3;        % Variable related to the Armijo rule
    beta  = 0.5;        % Variable related to the Armijo rule
    for iter = 1:max_iter
        
        % Compute the gradient
        P = lyap(A_CL(exp(k))', Q_til + kron(exp(k)*exp(k)',LRL_til));
        grad_K = 2*(R_til*kron(exp(k)',L_til) - P(end-n+2:end,:))*X;
        grad_g = reshape(grad_K,(n-1)^2,M)'*L_til(:);
        grad_k = exp(k).*grad_g;
        
        % Stopping criterion
        norm_grad = norm(grad_g);
        if disp_iter
            disp(['Iteration ',num2str(iter),char(9),'Norm of Gradient: ',num2str(norm_grad,'%6.3E')])
        end
        if norm_grad < tolerance
            break;
        end

        % Compute the gradient
        H_g = zeros(M,M);
        e = eye(M);
        Z_ = R_til*kron(exp(k)',L_til) - P(end-n+2:end,:);
        for i=1:M
            G1(end-n+2:end,:) = kron(e(:,i)',L_til)*X;
            G2 = -Z_'*kron(e(:,i)',L_til);
            Xtilde = lyap(A_CL(exp(k)),-(G1 + G1'));
            Ptilde = lyap(A_CL(exp(k))',-(G2 + G2'));    
            H = 2*((R_til*kron(e(:,i)',L_til) - Ptilde(end-n+2:end,:))*X + Z_*Xtilde);
            H_g(:,i) = reshape(H,(n-1)^2,M)'*L_til(:);
        end
        H_k = exp(k)'.*H_g.*exp(k)+diag(exp(k).*grad_g);
       
        % Compute the Newton direction
        [V,D] = eig(H_k);
        if all(diag(D)>0)
            k_Newton = -H_k\grad_k;
        else
            disp("Negative Curvature Detected");
            % Search direction with no component along the negative curvature directions
            D = diag(D);
            D(D>0) = 1./D(D>0);
            D(D<0) = 0;
            D = diag(D);
            pinv_H = V*D*V';
            k_Newton = -pinv_H*grad_k;
        end

        % Line Search
        stepsize = 1;
        while 1
            k_next = k + stepsize*k_Newton;
            maxEigAcl = max(real(eig(A_CL(exp(k_next)))));
            if maxEigAcl >= 0
                J_next = Inf;
            else
                X_next = lyap(A_CL(exp(k_next)),Z_til);
                QQ_next = Q_til + kron(exp(k_next)*exp(k_next)',LRL_til);
                J_next = X_next(:)'*QQ_next(:);
            end
    
            % Armijo rule
            if  J - J_next > -stepsize*alpha*grad_k'*k_Newton
                break;
            end
            stepsize = beta*stepsize;
            
            if stepsize < 1.e-16
                disp('Extremely small stepsize. Stopping method.');
                disp('Returning latest value.');
                gamma_opt = exp(k);
                J_opt = J;
                return
            end
    
        end
        
        % update current step
        k = k + stepsize*k_Newton;
        X = X_next;
        J = J_next;
    end

    if iter == max_iter
        disp('Maximum number of iterations reached!')
        disp(['The norm of gradient is ', num2str(norm_grad), '.'])
    end    
    
    gamma_opt = exp(k);
    J_opt = J;

end
