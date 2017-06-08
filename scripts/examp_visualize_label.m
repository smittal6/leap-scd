

file = load('FTBW0_X85-FTLG0_I1743.mat');
b = file.labels;
[y,Fs] = audioread('../../wav/train/FTBW0_X85-FTLG0_I1743.wav');

hop = fix(Fs*10e-3);
chop = fix(Fs*200e-3);
temp = ones(1,hop);
indx = 1;
for i = 1:length(b)
    x(indx:indx+hop-1) = double(b(i))*temp;
    indx = indx+hop;
end
if length(x)>length(y)
x = x(1:length(y));
else
y = y(1:length(x));
end    
    
    
figure;
plot(y./max(abs(y))); hold on;
plot(x);

    