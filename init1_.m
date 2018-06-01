% All path and parameters initializing are contained here, 
% Note that check them before every exam!!!

global opts;

opts.name='VGG_ILSVRC_16_layer';
opts.lr_on=1;
%data_dim
opts.dim(1,1)=100 ;
opts.dim(1,2)=3 ;
opts.dim(1,3)=24 ;
opts.dim(1,4)=24 ;
%labels_dim
opts.dim(2,1)=100 ;
opts.dim(2,2)=1;
opts.dim(2,3)=1;
opts.dim(2,4)=1;

opts.flag=1;%if 1 have relu if 0 without

%parameter of conv
opts.convPara.Coutput=32;
opts.convPara.Cstride=1;
opts.convPara.Cpad=1;
opts.convPara.Ckernel=3;
opts.convPara.Cweighttype='gaussian';
opts.convPara.Cweightstd=0.01;
opts.convPara.Cbiastype='constant';
opts.convPara.Cbiasvalue=0;
opts.convPara.pro_lr(1)=1;
opts.convPara.pro_decay(1)=1;
opts.convPara.pro_lr(2)=2;
opts.convPara.pro_decay(2)=0;

%parameter of pool
opts.poolPara.poolsize='MAX';
%opts.poolPara.poolsize='AVE';
opts.poolPara.poolkernel=2;
opts.poolPara.stride=2;

%parameter of InnerProduct
opts.fcPara.pro_lr(1)=1;
opts.fcPara.pro_decay(1)=1;
opts.fcPara.pro_lr(2)=2;
opts.fcPara.pro_decay(2)=0;

opts.fcPara.Poutput=400;
opts.fcPara.Pweighttype='gaussian';
opts.fcPara.Pweightstd=0.05;
opts.fcPara.Pbiastype='constant';
opts.fcPara.Pbiasvalue=0;

opts.dropPara.dropratio=0.5;
