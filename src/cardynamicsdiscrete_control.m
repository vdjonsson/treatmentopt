clear 
close all 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code related to treatment optimization of cellular therapies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Illustrate simple discrete dynamics  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Final simulation time [days]
tf = 16;

% Euler approx parameter
tau = .1;

% Evaluate at different switching intervals 
si = [tf/16,tf/8,tf/4,tf/2,tf];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial condition 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pat = 'MGH26';
x0all = csvread('../sims/MGH26.csv')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Molecular kinetics/Hill function drugs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = csvread('../sims/simsHillK.csv');

global kp;
kp = 2; % the hill coefficient 

drugs = {'IL13Ra2 CAR T', 'HER2 CAR T','EGFRvIII CAR T', 'IL13Ra2+EGFRvIII CAR T', 'IL13Ra2+HER2 CAR T', 'EGFRvIII+HER2 CAR T'}
states = { 'IL13Ra2','ERBB2', 'EGFR', 'IL13Ra2/EGFR','IL13Ra2/ERBB2','EGFR/ERBB2'}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Toxicity constraint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global ydose;
global ymax;

ydose = 0.3;
ymax = 0.5; 

leninit = length(x0all);

k=1;

for k=1:k+1:leninit
  
    x0 = x0all(k, :)';
  
    for i=1:i+1:length(si)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % System parameters
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % We re-evaluate the theraphy every switchinginterval [days]
        switchinginterval = si(i) 
        strout = [pat,'-', int2str(switchinginterval)]
        titlestr = join(strout,"-")

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Import cell and drug dynamics
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Cell dynamics 
        R = csvread('../sims/growthRates.csv');
        A = diag(R); 

        % CAR T dynamics expansion matrix rows= CART columns=cancer cells 
        E = csvread('../sims/expansionRates.csv'); 
        P = csvread('../sims/degradationRates.csv');
        P = -P';
        
        nstates = length(A);
        ndrugs = length(P);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Initial conditions: Simulation
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Solve for initial controller
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        y0 = zeros(ndrugs,1);
        u = zeros(ndrugs,1);             
        [index, u, feasible] =  solve_switchcontroller(x0,y0, K, kp, A, E, P,switchinginterval, tau);
        
        yin = u*ydose;  % new y input
        x = x0;  % Cells IC 
        y = y0 + yin;  % CAR T IC
       
    
        % plotting variables 
        xplot = x0;
        yplot = y0; 
        tplot = 0;    
        uplot = index;
        
        for t = tau:tau:tf
        
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Solve for the controller at the appropriate time interval
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if mod(t/tau,switchinginterval/tau)==0
                u = zeros(ndrugs,1);             
                [index, u, feasible] =  solve_switchcontroller(x,y, K,kp, A, E, P,switchinginterval, tau)
                
                yin = u*ydose;
                y = y + yin; 
            end
            
            D = construct_drug_matrix(x,y, kp, K);
              
            yactive =  y > 0;
            x = x + tau*(A-sum(D.*reshape(yactive,1,1,numel(yactive)),3))*x;
            y = y + tau*(diag(yactive)*(E*x+diag(P)))*y; 
             
            xplot = [xplot x];
            yplot = [yplot y];
            tplot = [tplot t];
            uplot = [uplot index];
            
        end
        
        graph_output(tplot, xplot, yplot, uplot, ymax, drugs, switchinginterval);

    end
end

function [xsim, ysim]=  simulate_dynamics(x,y, drug, K, kp, A, E, P, switchinginterval, tp ) 
        
        global ydose; 
        nstates = length(x); 
        ndrugs = length(y);

        tau = tp; 

        % Initialize the x and y vectors with the last
        % timepoint
        xsim = zeros(ndrugs, (switchinginterval/tau));
        ysim = zeros(ndrugs, (switchinginterval/tau));
        ysim(:,1) = y + ydose*full(sparse(1,[drug],1,1,6))';
        xsim(:,1) = x;

        % Now run through the dynamics for the switching
        % interval            
        for tsim = 2:(switchinginterval/tau)
            Dsim = zeros(nstates, nstates, ndrugs);
            for j = 1:ndrugs
               Dsim(:,:,j) = diag(ysim(j,tsim-1)^kp./(ysim(j,tsim-1)^kp + K(:,j).^kp));
            end                 
            yactivesim =  ysim(:,tsim-1) > 0;
            xsim(:,tsim)= xsim(:,tsim-1) + tau*(A-sum(Dsim.*reshape(yactivesim,1,1,numel(yactivesim)),3))*xsim(:,tsim-1);
            ysim(:,tsim) = ysim(:,tsim-1) + tau*(diag(yactivesim)*(E*xsim(:,tsim-1)+diag(P)))*ysim(:,tsim-1);
        end
    end

function [inddrug, uout , feasible] = run_MILP(X,L, ymax) 

    ndrugs = length(L);

    % MILP formulation: 
    % minimize (Xu) 
    % subject to sum(Lu)< b; drug constraint 
    % sum(u) = 1; one choise active 

    % Equality constraint: only one drug should be active at once
    Leq = ones(1,ndrugs);
    beq = 1;

    % bounds on controller       
    lb = zeros(ndrugs,1);
    ub = ones(ndrugs,1); % 

    [uout,fval,exitflag,output] = intlinprog(X,ndrugs,L,ymax,Leq,beq, lb, ub )

    feasible = 1; 
    if output.numfeaspoints == 0
        'No feasible solution' 
        uout = NaN;
        feasible = 0;
    end
    
    inddrug = find(abs(uout -1) < 1e-3); 

end

function [inddrug, uout , feasible] = run_MILP_linear(X,E,A,x,y)

    global ydose;
    global ymax; 
    
    ndrugs = length(E);

    % MILP formulation: 
    % minimize (Xu) 
    % subject to sum(Lu)< b; drug constraint 
    % sum(u) = 1; one choice active
    
    % Find the bound for the drug over time interval [t,t+tau]
    % Ax(t) is an upper bound for x during time interval [t,t+tau]
    % EAx(t)ydose is an upper bound for x during time interval [t,t+tau]
    % for current drugs 
    % EAx(t)y is an upper bound for x during time interval [t,t+tau] for
    % current drugs 
    
    L = (E*A*x.*y + E*A*x*ydose + y + ones(ndrugs,1)*ydose)';
    
    % Equality constraint: only one drug should be active at once
    Leq = ones(1,ndrugs);
    beq = 1;
    
    % TODO:Enforce binary constraints, write out as problem 

    % bounds on controller       
    lb = zeros(ndrugs,1);
    ub = ones(ndrugs,1); % 

    [uout,fval,exitflag,output] = intlinprog(X,[1:ndrugs],L,ymax,Leq,beq,lb, ub )

    feasible = 1;
    if output.numfeaspoints == 0
        'No feasible solution' 
        uout = NaN;
        feasible = 0;
    end
    
    is_binary = sum(abs(uout-1) < 1e-3) == 1;
    inddrug = find(abs(uout-1) < 1e-3); 
    
end

function [inddrug, uout , feasible] = solve_switchcontroller(x,y, K, kp, A, E, P,switchinginterval, tp)

    global ymax;
    global ydose;
   
    nstates = length(x);
    ndrugs = length(y);
    ymtd = ydose*ones(ndrugs,1);
    
    L = zeros(1, ndrugs);
    X = zeros(1, ndrugs);
    
    for j = 1:ndrugs
        D(:,:,j) = diag(ydose^kp./(ydose^kp + K(:,j).^kp));
    end

    Dv = zeros(1, ndrugs);
    for indx = 1:ndrugs
        X(indx) = ones(1,nstates)*expm((A-D(:,:,indx))*switchinginterval)*x;
    end
      
    [inddrug, uout , feasible] = run_MILP_linear(X,E,A, x,y) 

    if feasible == 0 
        'NO FEASIBLE SOLUTION, L, X' 
        L
        X
        switchinginterval
    end

end

function D = construct_drug_matrix(x, y, kp, K)

    ndrugs = length(y);
    nstates = length(x); 
    D = zeros(nstates, nstates, ndrugs);
    for j = 1:ndrugs
       D(:,:,j) = diag(y(j)^kp./(y(j)^kp + K(:,j).^kp));   
    end
    
end

function graph_output(tplot, xplot, yplot, uplot, ymax, drugs,switchinginterval)

    lw = 1;
    figure()
    set(gca,'FontSize',12)  

    sumstates = sum(xplot,1);
    nsubfigs = 3;

    % State plot 
    subplot(nsubfigs, 1,1)
    plot(tplot,sumstates,'-ro', 'LineWidth',lw,'MarkerSize',3)
    hold on; 
    plot(tplot,xplot,'LineWidth',lw)
    hold off;

    title('Total cancer cells');
    legend('Total');
    ylabel('Cells');

    % Subpopulation plot 
    subplot(nsubfigs, 1,2)
    sumcars = sum(yplot,1);

    plot(tplot,sumcars,'-om', 'LineWidth',lw, 'MarkerSize',3)
    hold on; 
    plot(tplot,yplot, 'LineWidth',lw)
    hline = refline([0 ymax]);
    hline.Color = 'r';
    hold off; 
    ylim([0,0.6])
    title('CAR T cell dynamics');
    ylabel('Cells');
    legend('sum CAR')

    % Treatment strategy plot
    titlestr = ['CAR strategy ','switch interval=', num2str(switchinginterval), ' week']
    subplot(nsubfigs,1,3);    
    hold on; 
    plot(tplot,uplot, 'k', 'LineWidth', lw)
    hold off; 
    ylim([0.9,6.1])
    yticks([1,2,3, 4, 5, 6])
    yticklabels(drugs)
    title(titlestr) 
    
    fname = ['../paper/figs/simulation_',  num2str(switchinginterval), '_week']; 

    xlabel('Week');
    print(gcf,fname,'-dpng','-r300');


end


           