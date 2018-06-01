%% step 0-0: config
clear all;clc;
disp('step 0: config...');
% /home/sist312/software/caffe/matlab
%addpath(genpath('/home/guangcong/projects/caffe/matlab'));
addpath(genpath('/home/wanggc/projects/caffe20180417b/caffe/matlab'));
% addpath(genpath('/home/sist312/software/caffe/matlab'));
config.solver_prototxt_path='solver.prototxt';
%config.batchsize=50; 
config.batchsize=100; 
config.resize_img_h=32;
config.resize_img_w=32;
config.crop_h=24;
config.crop_w=24;
config.model_path='my_net.caffemodel';

config.max_iter=20000; 
%add_layer_examples_num=20000;
confidence_tre=0.97;%这个似乎可以小的吧？  0.999 change to 0.99 ,12.22.09
%emphasize_factor=1;%


few_label_num=4000;
eval_label_num=10000;
all_always_train_num=50;  %add
test_ind_v=50001:60000;
    
    
err=0.001;
add_err=0.005;
addedNum=20000;
max_precision=0;
evaluation_error_seq=[];
open_switch=0;

%% step 0-1: how to write a .prototxt
init1_;
global opts;
networks_;
global train_type;
global train_para;
global test_type;
global test_para;

layer_num=1;
to_layer=4;
one_layer_times=10;
add_data_times=3;
begin_layer=29;
%end_layer=17;
end_layer=33;
init_layer=7;
step = 7;

type_train=[train_type(1:init_layer),train_type(begin_layer:end_layer)];
para_train=[train_para(1:init_layer),train_para(begin_layer:end_layer)];
type_test=[test_type(1:init_layer),test_type(begin_layer:end_layer)];
para_test=[test_para(1:init_layer),test_para(begin_layer:end_layer)];

grow_net_train.type=type_train;
grow_net_train.para=para_train;
grow_net_train.name='grow_net_train.prototxt';
layer(grow_net_train,'train');

grow_net_test.type=type_test;
grow_net_test.para=para_test;
grow_net_test.name='grow_net_test.prototxt';
layer(grow_net_test,'test');

%% step 1:  initize a solver
disp('step 1:  initize a solver...');
caffe.set_device(7);
caffe.set_mode_gpu();
% caffe.set_device(1);
caffe_solver = caffe.Solver(config.solver_prototxt_path);   % add
caffe_solver.net.save(config.model_path);

%% step 2: prepare data
path_mat='/home/wanggc/datasets/cifar-10-batches-mat/data_batch_1.mat';
load(path_mat);
all_imgs=data;
lbls=double(labels);

path_mat='/home/wanggc/datasets/cifar-10-batches-mat/data_batch_2.mat';
load(path_mat);
all_imgs=[all_imgs;data];
lbls=[lbls;double(labels)];
path_mat='/home/wanggc/datasets/cifar-10-batches-mat/data_batch_3.mat';
load(path_mat);
all_imgs=[all_imgs;data];
lbls=[lbls;double(labels)];
path_mat='/home/wanggc/datasets/cifar-10-batches-mat/data_batch_4.mat';
load(path_mat);
all_imgs=[all_imgs;data];
lbls=[lbls;double(labels)];
path_mat='/home/wanggc/datasets/cifar-10-batches-mat/data_batch_5.mat';
load(path_mat);
all_imgs=[all_imgs;data];
lbls=[lbls;double(labels)];

path_mat='/home/wanggc/datasets/cifar-10-batches-mat/test_batch.mat';
load(path_mat);
all_imgs=[all_imgs;data];
%test_labels=labels;
all_labels=[lbls;double(labels)];

all_imgs=all_imgs';
labels=lbls;
test_num=size(data,1);
  

data_num=length(labels);
% fig_cells=cell(10,1);
fig_mat=zeros(10,few_label_num/10);
for i=0:9
    one_fig_index=find(labels==i);
    index_index=randperm(length(one_fig_index),few_label_num/10);
    %fig_cells{i+1,1}=one_fig_index(index_index);
    fig_mat(i+1,:)=one_fig_index(index_index);
end
%label_ind_v,eval_label_ind_v,no_label_ind_v,test_label_ind_v
%updata_labels,eval_labels
label_ind_v=reshape(fig_mat,[few_label_num,1]);
all_ind=1:data_num;  % exclude test num
temp_ind_v=setdiff(all_ind,label_ind_v);
eval_label_ind_ind=randperm(length(temp_ind_v),eval_label_num);
eval_label_ind_v=temp_ind_v(eval_label_ind_ind);
eval_labels=labels(eval_label_ind_v);

split_no_label_ind_v=setdiff(temp_ind_v,eval_label_ind_v);

%no_label_ind_v=setdiff(temp_ind_v,eval_label_ind_v);   % the other indexes
%no_label_ind_v=temp_ind_v;   % the other indexes
updata_labels=-ones(data_num+test_num,1);
updata_labels(label_ind_v,1)=labels(label_ind_v,1);
no_label_ind_v=[temp_ind_v,test_ind_v];         %%%%%%%%%%%%add,122911

%test_label_ind_v                                       %%%%%%%%%%%%add,17,02,20,14
test_label_ind_v=data_num+1:data_num+test_num;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% step 3 : data layer and solver->step(1)
disp('step 3 : data layer and solver->step(1)...');
%逐渐增加网络深度
%cur_num=few_label_num;
pre_num=0;
new_label_ind_v=label_ind_v';
is_updata=1;
precision=0;
diff_precision=1;
debug_file_name=strcat(datestr(now,29),'---',datestr(now,13),'.log');
while(1)
    %if is_updata
     %em_label_data_ind=repmat(label_ind_v,[emphasize_factor,1]);
     %cur_label_data_ind=[em_label_data_ind;new_label_ind_v']; 
     cur_label_data_ind=[label_ind_v;new_label_ind_v']; 
     pre_num=length(label_ind_v)+length(new_label_ind_v);
     is_updata=0;
    %end
    %迭代若干次就评估一次
    for k=1:config.max_iter       
        data=zeros(config.crop_h,config.crop_w,3,config.batchsize);
        local_ind_v=randi(length(new_label_ind_v'),[1,config.batchsize-all_always_train_num]);
        global_ind_v=new_label_ind_v(local_ind_v);
        all_always_train_mat=zeros(10,all_always_train_num/10);
        for i=0:9
            index_index=randperm(few_label_num/10,all_always_train_num/10);
            all_always_train_mat(i+1,:)=fig_mat(i+1,index_index);
        end
        all_always_train_v=reshape(all_always_train_mat,[all_always_train_num,1]);   
        global_ind_v=[global_ind_v';all_always_train_v];

        batch_imgs=all_imgs(:,global_ind_v);
        batch_labels=updata_labels(global_ind_v,1);            %%%%%%%%%%%%%%%%%%%
    
        %加载data
        for i=1:config.batchsize 
            im_data=batch_imgs(:,i);
            im_data=reshape(im_data,[32,32,3]);
            %im_data=im_data(:,:,[3,2,1]);
            %save im_data im_data;
            %im_data = permute(im_data, [2, 1]);
            im_data = single(im_data);
            mirror_flag=randperm(2,1);
            if mirror_flag==1
                im_data=flipdim(im_data,1);
            end
            %1.minus mean_data?
            im_data=im_data-128;
            %2.resize
            %rand_factor=1+0.15*rand(1,1);
            %3.crop
            im_data=crop(im_data,config.crop_h,config.crop_w,'train');
            data(:,:,:,i)=im_data;
        end  
        %label
        batch_labels=reshape(batch_labels,[1,1,1,config.batchsize]);

        caffe_solver.net.blobs('data').set_data(data);
        caffe_solver.net.blobs('labels').set_data(batch_labels);
        caffe_solver.step(1);
        if(mod(k,500)==0)
            disp(['*****',num2str(layer_num),'*******crop***9.3***iters:',num2str(k)]);
        end  
    end
    caffe_solver.net.save(config.model_path);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %测试一下少量样本or上一次得到的所有label?  上一次得到的所有label  
    %eval_data=all_imgs(:,eval_label_ind_v);
    all_data_num=size(all_imgs,2);
    caffe_solver.test_nets.copy_from(config.model_path);
    test_batch_size=100;
    crop_times=16;
    all_predit_ind=zeros(1,all_data_num);
    all_max_scores=zeros(1,all_data_num);
    
    for i=1:test_batch_size:all_data_num
        accumulate_scores=zeros(10,config.batchsize);
        for cr=1:crop_times
            test_data=test_blob(all_imgs(:,i:i+test_batch_size-1),config.resize_img_h,config.resize_img_w,config.crop_h,config.crop_w,'train');
            %test_data=test_blob(eval_data(:,i:i+test_batch_size-1),config.resize_img_h,config.resize_img_w,config.crop_h,config.crop_w,'test');
            scores=caffe_solver.test_nets.forward({test_data});
            scores=scores{1,1};
            scores=reshape(scores,[10,config.batchsize]);
            accumulate_scores=accumulate_scores+scores;
        end
        accumulate_scores=accumulate_scores/crop_times;
        [max_pro,max_ind]=max(accumulate_scores,[],1);

        all_predit_ind(1,i:i+test_batch_size-1)=max_ind;
        all_max_scores(1,i:i+test_batch_size-1)=max_pro;
    end
    diff_num_logit=(all_predit_ind'-1)-all_labels(:,1);
    % errors
    evaluation_error=1-sum(logical(diff_num_logit(eval_label_ind_v)==0))/length(eval_label_ind_v);
    training_error_su=1-sum(logical(diff_num_logit(label_ind_v)==0))/length(label_ind_v);
    training_error_un=1-sum(logical(diff_num_logit(split_no_label_ind_v)==0))/length(split_no_label_ind_v);
    training_error_all=1-(sum(logical(diff_num_logit(split_no_label_ind_v)==0))+...
        sum(logical(diff_num_logit(label_ind_v)==0)))/(length(label_ind_v)+length(split_no_label_ind_v));
    test_error=1-sum(logical(diff_num_logit(test_ind_v)==0))/length(test_ind_v);

    evaluation_error_seq=[evaluation_error_seq,evaluation_error];
    %once open switch, do not close at this layer even if add_data_flag==0
    add_data_flag=convergence(evaluation_error_seq,add_data_times,add_err);
    if add_data_flag==1
        open_switch=1;
    end
    convergence_flag=convergence(evaluation_error_seq,one_layer_times,err);

    if open_switch==1
        no_label_max_scores=all_max_scores(no_label_ind_v);
        confidence_ind=find(no_label_max_scores>confidence_tre); 
        new_label_ind_v=no_label_ind_v(confidence_ind);
        updata_labels(new_label_ind_v,1)=all_predit_ind(1,new_label_ind_v)-1;
    end

    psedu_labeled_num=length(new_label_ind_v);
    psedu_labeled_error=1-sum(logical(diff_num_logit(new_label_ind_v)==0))/length(new_label_ind_v);
    %layer_num

    

    fid = fopen(debug_file_name,'a');
    fprintf(fid,[num2str(layer_num),',',num2str(training_error_su),',',num2str(training_error_un),',',...
        num2str(training_error_all),',',num2str(evaluation_error),',',num2str(test_error),',',...
        num2str(psedu_labeled_error),',',num2str(psedu_labeled_num),'\n']);
    fclose(fid); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    %若满足增加层的条件，则增加
    
    if (convergence_flag==1 && layer_num<to_layer)
        evaluation_error_seq=[];
        open_switch=0;
        caffe.reset_all();
        layer_num=layer_num+1;
        type_train=[train_type(1:layer_num*step),train_type(begin_layer:end_layer)];
        para_train=[train_para(1:layer_num*step),train_para(begin_layer:end_layer)];
        type_test=[test_type(1:layer_num*step),test_type(begin_layer:end_layer)];
        para_test=[test_para(1:layer_num*step),test_para(begin_layer:end_layer)];

        %trans the last layer names
        for ii=layer_num*step+1:length(para_train)
            para_train{ii}.name=strcat(para_train{ii}.name,'-',num2str(layer_num));
            para_test{ii}.name=strcat(para_test{ii}.name,'-',num2str(layer_num));
        end

        grow_net_train.type=type_train;
        grow_net_train.para=para_train;
        layer(grow_net_train,'train');

        grow_net_test.type=type_test;
        grow_net_test.para=para_test;
        layer(grow_net_test,'test');     
        caffe_solver = caffe.Solver(config.solver_prototxt_path);  %add
        caffe_solver.net.copy_from(config.model_path);  %add170118

    end   
end




