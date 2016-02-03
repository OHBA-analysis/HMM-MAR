function [pdc, dtf] = subrutpdc(S,numIterations,tol)
% obtains approximated pdc and dc from cross-spectral information

Nf = size(S,1); N = size(S,2);
% Wilson factorization
H = wilsonfact(permute(S,[2 3 1]),numIterations,tol);
% Computing the PDC
G = zeros(size(H));
pdc = zeros(size(H));
dtf = zeros(size(H));
for f=1:Nf
    G(:,:,f) = inv(H(:,:,f));
    for i=1:N
        for j=1:N
            if i~=j
                pdc(i,j,f) = abs(G(i,j,f)) / sqrt(sum( abs(G(:,j,f)).^2 )) ;
                dtf(i,j,f) = abs(H(i,j,f)) / sqrt(sum( abs(H(i,:,f)).^2 )) ;  % note different normalisation
            end
        end
    end
end
pdc = permute(pdc,[3 1 2]);
dtf = permute(dtf,[3 1 2]);
end

%-------------------------------------------------------------------
function [H, Z, S] = wilsonfact(S,Niterations,tol)
%
% This function is an implemention of Wilson's algorithm (Eq. 3.1)
% for spectral matrix factorization
%
% Inputs : S (1-sided, 3D-spectral matrix in the form of Channel x Channel x frequency)
%        : fs (sampling frequency in Hz)
%        : freq (a vector of frequencies) at which S is given
% Outputs: H (transfer function)
%        : Z (noise covariance)
%        : psi (left spectral factor)
%
% Ref: G.T. Wilson,"The Factorization of Matricial Spectral Densities,"
% SIAM J. Appl. Math.23,420-426(1972).
% Modification over the function sfactorization_wilson (fieldtrip),
% implemented by  M. Dhamala & G. Rangarajan, UF, Aug 3-4, 2006.

% number of channels
m   = size(S,1);
N   = size(S,3)-1;
N2  = 2*N;

% preallocate memory for efficiency
Sarr   = zeros(m,m,N2) + 1i.*zeros(m,m,N2);
gam    = zeros(m,m,N2);
gamtmp = zeros(m,m,N2);
psi    = zeros(m,m,N2);
I      = eye(m); % Defining m x m identity matrix

%Step 1: Forming 2-sided spectral densities for ifft routine in matlab
for f_ind = 1:N+1
    Sarr(:,:,f_ind) = S(:,:,f_ind);
    if(f_ind>1)
        Sarr(:,:,2*N+2-f_ind) = S(:,:,f_ind).';
    end
end

%Step 2: Computing covariance matrices
for k1 = 1:m
    for k2 = 1:m
        gam(k1,k2,:) = real(ifft(squeeze(Sarr(k1,k2,:))));
    end
end

%Step 3: Initializing for iterations
gam0 = gam(:,:,1);
[h, dum] = chol(gam0);
if dum
    warning('initialization for iterations did not work well, using arbitrary starting condition');
    h = rand(m,m); h = triu(h); %arbitrary initial condition
end

for ind = 1:N2
    psi(:,:,ind) = h;
end

%Step 4: Iterating to get spectral factors
for iter = 1:Niterations
    for ind = 1:N2
        invpsi     = inv(psi(:,:,ind));
        g(:,:,ind) = invpsi*Sarr(:,:,ind)*invpsi'+I;%Eq 3.1
    end
    gp = PlusOperator(g,m,N+1); %gp constitutes positive and half of zero lags
    psi_old = psi;
    leaveloop = 0;
    for k = 1:N2
        psi(:,:,k) = psi(:,:,k)*gp(:,:,k);
        %if isnan(rcond(psi(:,:,k))),
        %   leaveloop=1;
        %   break
        %end
        psierr(k)  = norm(psi(:,:,k)-psi_old(:,:,k),1);
    end
    %if leaveloop,
    %    psi = psi_old;
    %    break
    %end
    psierrf = mean(psierr);
    if(psierrf<tol),
        break;
    end; % checking convergence
end

%Step 5: Getting covariance matrix from spectral factors
for k1 = 1:m
    for k2 = 1:m
        gamtmp(k1,k2,:) = real(ifft(squeeze(psi(k1,k2,:))));
    end
end

%Step 6: Getting noise covariance & transfer function (see Example pp. 424)
A0    = gamtmp(:,:,1);
A0inv = inv(A0);
Z     = A0*A0.'; %Noise covariance matrix not multiplied by sampling frequency

H = zeros(m,m,N+1) + 1i*zeros(m,m,N+1);
for k = 1:N+1
    H(:,:,k) = psi(:,:,k)*A0inv;       %Transfer function
    %S(:,:,k) = psi(:,:,k)*psi(:,:,k)'; %Updated cross-spectral density
end
end

%---------------------------------------------------------------------
function gp = PlusOperator(g,nchan,nfreq)

g   = transpose(reshape(g, [nchan^2 2*(nfreq-1)]));
gam = ifft(g);

% taking only the positive lags and half of the zero lag
gamp  = gam;
beta0 = 0.5*gam(1,:);

gamp(1,          :) = reshape(triu(reshape(beta0, [nchan nchan])),[1 nchan^2]);
gamp(nfreq+1:end,:) = 0;

% reconstituting
gp = fft(gamp);
gp = reshape(transpose(gp), [nchan nchan 2*(nfreq-1)]);
end
