% Use a truncated Newton method to solve the LQR consensus problem
%
%      minimize    \bar{J}(\gamma \kron L)
%      subject to   L*1_n = 0
%                   L \hadamard ES = 0
%
% Syntax:
% [L_opt,gamma_opt,J_opt] = lqrc_newtonCG(M,Z,Q,R,ES,L0,gamma_0,tolerance,symmetry);
%
% Inputs: 
% problem data: {M,Z,Q,R},
% initial conditions: {L0,gamma_0} (sparsity pattern of L0 is considered),
% tolerance for the stopping criterion: tolerance,
% impose symmetry on the resulting Laplacian: symmetry [true/false] (when
% ommited, defaults to false).
%
% Outputs: 
% minimizers {L_opt,gamma_opt} and minimum value J_opt.
function [L_opt,gamma_opt,J_opt] = lqrc(Z,Q,R,L0,gamma_0,tolerance,symmetry)

    % Check if the dimensions of the input matrices are correct
    M = length(gamma_0);
    n = length(L0(1,:));
    if length(L0(:,1))~=n
        error('L0 is not square!');
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

    % Check whether symmetry variable was passed
    if ~exist('symmetry','var')
        symmetry = false;
    end
    if symmetry && norm(L0-L0')>1e-10
        error('"symmetry" is set to "true", but L0 is not symetric')
    end

    % Generate orthonormal basis for the off-consensus subspace
    S = diag(n:-1:1)-tril(ones(n,n));
    S = S(:,1:end-1)./sqrt((n:-1:2).*(n-1:-1:1));

    % Check if the initial condition leads to consensus before proceeding
    A_M = diag(ones(M-1,1),1);
    AA = kron(A_M(1:end-1,:),eye(n-1));
    if max(real(eig([AA;-kron(gamma_0',S'*L0*S)]))) >= 0
        error('The initial conditions L0 and g0 do not lead to consensus!')
    end

    % Call the functions considering the symmetry variable
    if symmetry == true
        [L_opt,gamma_opt,J_opt] = lqrc_newtonCG_sym(Z,Q,R,L0,gamma_0,tolerance);
    else
        [L_opt,gamma_opt,J_opt] = lqrc_newtonCG_nosym(Z,Q,R,L0,gamma_0,tolerance);
    end

end
