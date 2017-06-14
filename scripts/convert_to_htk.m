clear all; clc;
poolsize = 8;
parpool(poolsize);
%Par parameters are for parallel processing of the files
% ----- update the path to respective directories
label_train_addr = '/home/neerajs/work/NEW_REGIME/SID/OVERLAP/val/';
mfcc_feats_addr = '/home/neerajs/work/NEW_REGIME/SID/FEATS/mfcc_after/train/';
kurt_feats_addr = '/home/neerajs/work/NEW_REGIME/SID/FEATS/kurt_after/val/';
sfm_feats_addr = '/home/neerajs/work/NEW_REGIME/SID/FEATS/sfm_after/val/';
mel_feats_addr = '/home/neerajs/work/NEW_REGIME/SID/FEATS/mel_after/train_4Khz/'
linear_feats_addr='/home/neerajs/work/NEW_REGIME/SID/FEATS/linear_after/train/'
EXTRA='/home/neerajs/work/NEW_REGIME/SID/FEATS/energy/val/'
context_addr = '/home/neerajs/work/NEW_REGIME/SID/FEATS/context/';
% ----- list of files
f=fopen('/home/neerajs/work/NEW_REGIME/SID/LIST/val.list');
f=textscan(f,'%s');
len=cellfun('length',f)
type = 'EXTRA';
context_size = 5;

parfor i = 1:len
%for i = 1:len
	[i len]

	% read the label file
	vad = load([label_train_addr f{1}{i}]);

	% check the feature type
	switch(type)
		case 'KURT'
			data_kurt = load([kurt_feats_addr f{1}{i} '.mat']);
			data = data_kurt.normdata';
                        op_path = 'kurt';
		case 'SFM'
			data_sfm = load([sfm_feats_addr f{1}{i} '.mat']);
			data = data_sfm.normdata';
			op_path = 'sfm';
		case 'MFCC'
			[data_mfcc,a,b,c,d] = readhtk([mfcc_feats_addr f{1}{i} '.htk']);
			data = data_mfcc;
			op_path = 'mfcc';
                case 'MEL'
			[data_mel,a,b,c,d] = readhtk([mel_feats_addr f{1}{i} '.htk']);
			data = data_mel;
			op_path = 'mel';
		case 'LINEAR'
			[data_linear,a,b,c,d]=readhtk([linear_feats_addr f{1}{i} '.htk']);
			data=data_linear;
			op_path='linear';
		case 'EXTRA'
			[data_extra,a,b,c,d]=readhtk([EXTRA f{1}{i} '.htk']);
			data=data_extra';
			op_path='energy'
	
	end
	size(data)
	f{1}{i}
	% read each feature file and make context feature file
 	nframes = min(size(data,2),length(vad.labels));
	data_write = zeros(nframes,(2*context_size+1)*size(data,1));
	label_write = zeros(nframes,1);
	ssegs = 1;
	for index = 1:nframes
		display([index nframes])
	        % make left/right context
        	if (index<(context_size+1))
               	temp_l = [];
	        temp_r = data(:,index+1:index+context_size);
              	label_check = vad.labels(index:index+context_size);
                elseif (nframes-index+1)<(context_size+1)
              	temp_l = data(:,index-context_size:index-1);
               	temp_r = [];
		label_check = vad.labels(index-context_size:index);
               	else
              	temp_r = data(:,index+1:index+context_size);
               	temp_l = data(:,index-context_size:index-1);
		label_check = vad.labels(index-context_size:index+context_size);
                end

                % fill left/right context if empty
                if isempty(temp_l)
                temp_l=fliplr(temp_r);
                elseif isempty(temp_r)
                temp_r =fliplr(temp_l);
                end

                % complete the context frame with the center frame
		label_write(ssegs) = vad.labels(index);
                data_temp = [temp_l data(:,index) temp_r];
		data_write(ssegs,:) = data_temp(:)';
		ssegs = ssegs + 1;
		%if 1 %(sum(label_check)==0 || sum(label_check)>5)
		%	if sum(label_check)==0
                %	label_write(ssegs) = 0;
		%	else
		%	label_write(ssegs) = 1;
		%	end
                %	data_temp = [temp_l data(:,index) temp_r];
		%	data_write(ssegs,:) = data_temp(:)';
  		%	ssegs = ssegs +1;
		%end
	end
%	data_write = data_write(1:ssegs-1,:);	
%	label_write = label_write(1:ssegs-1,:);
 	% save them
	writehtk([context_addr op_path '/val/' f{1}{i} '.htk'],data_write,0.11,9); 
	writehtk([context_addr 'labels/val/' f{1}{i} '.htk'],label_write,0.11,9);
end
