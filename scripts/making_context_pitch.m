

clear all; clc;
poolsize = 8;
parpool(poolsize);

% ----- update the path to respective directories
dataset_type = 'train/'
label_addr = '/home/siddharthm/scd/vad/10/';
gamma_feats_addr = '/home/siddharthm/scd/feats/pitch/';
context_addr = '/home/siddharthm/scd/context/600/';

% ----- list of files
f=fopen('/home/siddharthm/scd/lists/rawtrainfiles.list');
f=textscan(f,'%s');
len=cellfun('length',f)
type = 'pitch';
context_size = 30; 

%This into 10msec is the one sided context

parfor i = 1:len
%for i = 1:%len
    [i len]

    % read the label file
    vad = readhtk([label_addr dataset_type f{1}{i} '.htk']);

    % check the feature type
    switch(type)
         case 'pitch'
              all_matrix = load(strcat(gamma_feats_addr,dataset_type,f{1}{i},'.Pitch'));
              no_zero=find(all_matrix(:,2));
              inter_matrix=all_matrix(no_zero,:);
              x_new=all_matrix(:,1);
              y_val=inter_matrix(:,2);
              x_val=inter_matrix(:,1);
              y_val = [y_val;y_val(end)];
              y_val(y_val>400) = 350;
              x_val = [x_val; x_new(end)];
              y_new=interp1(x_val,y_val,x_new,'linear', 'extrap');
              K=30;
              dct_y_new = dct(y_new);
              dct_y_new(K+1:end) = 0;
              y_new_approx = idct(dct_y_new);
              data=y_new_approx./max(abs(y_new_approx));
              data=data';
              %size(data)
              op_path='pitch';
              op_path2='pitch_all';
         case 'GAMMA'
              [data_extra,a,b,c,d]=readhtk([gamma_feats_addr dataset_type f{1}{i} '.htk']);
              data=data_extra';
              data=flipud(data);
              op_path='gamma'
    end
    size(data)
    f{1}{i}

    % read each feature file and make context feature file
    nframes = min(size(data,2),length(vad));
    data_write = zeros(nframes,1);
    label_write = zeros(nframes,1);
    ssegs = 1;
    for index = 1:nframes
        %display([index nframes]);
        % make left/right context
        if (index<(context_size+1))
            temp_l = [];
            temp_r = data(:,index+1:index+context_size);
            label_l = [];
            label_r = vad(index+1:index+context_size);
        elseif (nframes-index+1)<(context_size+1)
            temp_l = data(:,index-context_size:index-1);
            temp_r = [];
            
            label_l = vad(index-context_size:index-1);
            label_r = [];
        else
            temp_r = data(:,index+1:index+context_size);
            temp_l = data(:,index-context_size:index-1);
            
            label_l = vad(index-context_size:index-1);
            label_r = vad(index+1:index+context_size);
        end

        % fill the left/right context, if empty
        if isempty(temp_l)
            temp_l = fliplr(temp_r);
            label_l = fliplr(label_r);
        elseif isempty(temp_r)
            temp_r = fliplr(temp_l);
            label_r = fliplr(label_l); 
        end

        % complete the context frame with the center frame
        data_temp = [temp_l data(:,index) temp_r];
        standard_dev = std(data_temp);
        data_write(ssegs,:) = standard_dev';
        %data_write(ssegs,:) = data_temp(:)';
        
        %size(label_l)
        %size(label_r)
        % create the label for the context
        label_temp = [label_l vad(index) label_r];
        %size(label_temp)
        %%%%%% write the percentage stuff here
        p0 = length(find(label_temp==0))/(2*context_size+1);
        p1 = length(find(label_temp==1))/(2*context_size+1);
        p2 = length(find(label_temp==2))/(2*context_size+1);

        if p0>0.7
            label_write(ssegs) = 0;
        else
            if min(p1,p2)>0.1
                label_write(ssegs) = 2;
            else
                label_write(ssegs) = 1;
            end
        end   
        ssegs = ssegs+1;
    end
    % save them
    %size(data_write);
    %size(label_write);
    writehtk([context_addr op_path '/'  dataset_type f{1}{i} '.htk'],[data_write label_write],0.11,9); 
    writehtk([context_addr op_path2 '/'  dataset_type f{1}{i} '_all.htk'],[data_write y_new data' label_write],0.11,9); 
    
end
poolobj = gcp('nocreate'); delete(poolobj);
