data1=load('ex1data1.txt');

X = data1(:,1);
y = data1(:,2);
m = length(y);

X = [ones(m, 1), data1(:,1)]; % Add a column of ones to X

disp('X, y and m set for data1');
