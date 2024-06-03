%% Conjugate gradient method to compute newton direction
function K_Newton = glqrc_CG(K,P,X,R,grad_K,ES,S_M,Proj)
    
    % Obtain the dimensions
    n = length(K(:,1));
    M = length(K(1,:))/n;

    % Some needed matrices, to store computations
    A = kron(diag(ones(M-1,1),1),eye(n));
    B = kron([zeros(M-1,1);1],eye(n));
    A_CL = S_M'*(A - B*K)*S_M;  % closed-loop matrix
    F = R*K - B'*P; 

    % Stopping criterion
    norm_grad_K = norm(grad_K,'fro');
    epsilon = min(0.5,sqrt(norm_grad_K)) * norm_grad_K; 
    
    % Initialization
    K_tilde = zeros(size(K));
    R_K = grad_K;
    P_K = -grad_K;
    
    % Iterate
    q = M*(nnz(not(ES))-n);     % number of "free variables"
    disp_msgs = 0;              % flag to control warning messages
    for k = 0:q

        % Compute H(K,P_K) (Hessian at K along direction P_K)
        G1 = B*P_K*X;
        G2 = -F'*P_K;
        Xtilde = S_M*lyap(A_CL,-S_M'*(G1 + G1')*S_M)*S_M';
        Ptilde = S_M*lyap(A_CL',-S_M'*(G2 + G2')*S_M)*S_M';
        H = 2*Proj((R*P_K - B'*Ptilde)*X + F*Xtilde);
        
        % Negative Curvature Test
        inner = H(:)'*P_K(:);
        if (inner <= 0) && (k == 0)
            K_Newton = -grad_L;
            if disp_msgs
                disp('Negative curvature. Stopping CG.')
            end
            break;
        elseif (inner <= 0) && (k > 0)
            K_Newton = K_tilde;
            if disp_msgs
                disp('Negative curvature. Stopping CG.')
            end
            break;
        end
        
        % Update K_tilde and the residual
        r_norm_squared = R_K(:)'*R_K(:);
        alpha = r_norm_squared/inner;
        K_tilde = K_tilde + alpha*P_K;
        R_K_next = R_K + alpha*H;
        
        % Check stopping criterion
        r_norm_next = norm(R_K_next,'fro');
        if r_norm_next < epsilon
            K_Newton = K_tilde;
            break
        end

        % Update directions
        beta = r_norm_next^2/r_norm_squared;
        P_K = -R_K_next + beta*P_K;
        R_K = R_K_next;
    end
    
    if k == q
        K_Newton = K_tilde;
        if disp_msgs
            disp('Maximum number of conjugate gradient method iterations reached!')
            disp(['Norm of the residual is ',num2str( r_norm_next, '%10.2E' ),'.'])
        end            
    end
        
end
