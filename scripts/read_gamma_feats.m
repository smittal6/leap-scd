
% obtaining mel spectrum
%f=fopen('/home/neerajs/work/blurp_universe/TIMIT/list_of_over_files_revv3_noise_full_db.list');
%f=textscan(f,'%s');
%len=cellfun('length',f)
%poolsize = 8;
%parpool(poolsize);

path = '/home/neerajs/work/NEW_REGIME/SID/WAV/val/'; %Path of wav files
savedir = ''
filename = dir([path '*.wav']);
len = length(filename);
Fs_proc = 16e3;
get_mel = 1;

for i =1:1%len
display([i len])
[sig,Fs] = audioread(strcat(path,filename(i).name));

if Fs<Fs_proc
   sig = resample(sig,Fs_proc,Fs);
end
Fs = Fs_proc;
sig = [diff(sig);0]; % pre-emphasis
if get_mel
    wmsec = 0.025; hop = .010; nfilts = 64;
    fmin = 50; fmax = Fs/4;
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


