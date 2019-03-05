function [ output_args ] = testSimpleFit()

%% 

rng(2);

xMin = 0;
xMax = 20;
nrIntervals = 1000;
dx = (xMax-xMin)/nrIntervals;
xs = (xMin:dx:xMax)';

%% dx/dt = a*x+beta => see if the recovered function is exponential x(t) = x(0)*exp(at)
% a = 1;
% b = 0;
% dXdT = a*xs + b + normrnd(0,1,size(xs));
% dXdT_fitObj = fit(xs, dXdT, 'poly1');
% h = figure('position', [500 100 400 300]);
% plot(dXdT_fitObj, xs, dXdT);
% ylabel('dx/dt');
% 
% ys = dXdT_fitObj(xs);
% integrationIndices = find(ys > 0.1);
% integrationIndices = integrationIndices(2:end);
% %t = cumsum( (xs(2:end) - xs(1:(end-1))) ./ ys(2:end)); % when integrating make sure ys is not zero or very close to zero
% t = cumsum( (xs(integrationIndices ) - xs(integrationIndices -1)) ./ mean([ys(integrationIndices), ys(integrationIndices-1)], 2)); % when integrating make sure ys is not zero or very close to zero
% 
% true_xs = xs(integrationIndices(1)) * exp(a*t);
% 
% 
% h2 = figure('position', [100 100 400 300]);
% plot(t, xs(integrationIndices -1));
% hold on
% plot(t, true_xs);
% xlabel('t');
% ylabel('x');
% legend('fitted', 'true');

%% dx/dt = a*x^2 + b*x + c => fit with gaussian process
a = -0.05;
b = 1;
c = 1;

dXdT = a*xs.^2 + b*xs + c + normrnd(0,1,size(xs));
h3 = figure('position', [500 100 500 400]);
scatter(xs, dXdT, 6);
ylabel('dx/dt');

gprMdl = fitrgp(xs,dXdT,'Basis','linear',...
  'FitMethod','exact','PredictMethod','exact');
ypred = resubPredict(gprMdl);

y_true = a*xs.^2 + b*xs + c;
plot(xs,y_true,'b.');
hold on;
plot(xs,ypred,'r','LineWidth',1.5);
xlabel('x');
ylabel('dx/dt');
legend('true','GPR prediction');
hold off

end

