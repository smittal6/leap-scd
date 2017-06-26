
% obtaining mel spectrum

poolsize = 4;
parpool(poolsize);

path = '/home/user/work/scd/wav/test/'; %Path of wav files
savedir = '/home/user/work/scd/feats/gamma/test/'; %The save directory
filename = dir([path '*.wav']);
len = length(filename);
Fs_proc = 16e3;
get_mel = 1;

parfor i =1:len

display([i len])
[sig,Fs] = audioread(strcat(path,filename(i).name));
%display(length(sig))

if Fs<Fs_proc
   sig = resample(sig,Fs_proc,Fs);
end
Fs = Fs_proc;
%sig = [diff(sig);0]; % pre-emphasis
if get_mel
    % Here, change the parameters according to the needs
    wmsec = 0.025; hop = .010; nfilts = 64;
    fmin = 20; fmax = Fs/2;
    [STFTmag,F2] = gammatonegram(sig,Fs,wmsec,hop,nfilts,fmin,fmax,0);
    STFTmag=STFTmag';
    disp(size(STFTmag))
else
    [STFTmag] = get_spectgm(sig,Fs);
    STFTmag=STFTmag';
end

writehtk(strcat(savedir,filename(i).name(1:end-4),'.htk'),STFTmag,0.01,9);
%clearvars STFTmag

end
