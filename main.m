clear all;
close all;
clc;
init1_;
global opts;
addpath(genpath('/home/wang/matlab_dir/bmvc2016/caffe-temporal-pool-lstm-fc/matlab/ijcai2017-semi'));
%% type
type{1}='conv';
type{2}='conv';
type{3}='pooling';
type{4}='interproduct';
type{5}='norm';
type{6}='loss';
% layer(type);

%% para
conv1.Coutput=32;
conv1.Ckernel=5;

conv2.Coutput=64;
conv2.Ckernel=5;

para{1}=conv1;
para{2}=conv2;
para{3}=[];
para{4}=[];
para{5}=[];
para{6}=[];

opts.lr_on=0;

net.type=type;
net.para=para;
pro_name='hello';
layer(net,pro_name,'train');
