function layer(net,phase)    
% (type,name,phase,conv_ke,conv_output,fc_output,poolke,lr_on)
% type: layer type
% name: layer name
% phase: train or test
% conv_ke: conv kernel
% conv_output: output channels
% fc_output: fully connection output channels
% poolke: pool kernel
% lr_on: learning rate on or off, 1 for on, while 0 for off
type=net.type;
para=net.para;
name=net.name;

global opts;
% name=input('input the layer name:','s');
% fid1=[name,'.prototxt'];   %�����µ�txt�ļ�
fid1=name;
c=fopen(fid1,'wt');   %��txt�ļ�
%for i=1:3
%      str=['D:/Resized/' scenes(i,1).name];
layername=['name:"',opts.name,'"']; 
fprintf(c,'%s\n\n',layername);        %����д��txt�ļ���%sΪ�����ʽ��strΪд����ݡ�����

temp='input: "data"';
fprintf(c,'%s\n',temp);  

if ~strcmp(phase,'test')
    sentence=['input_dim: ',num2str(opts.dim(1,1))]; 
else
    sentence='input_dim: 100';
end
fprintf(c,'%s\n',sentence);  
sentence=['input_dim: ',num2str(opts.dim(1,2))]; 
fprintf(c,'%s\n',sentence); 
sentence=['input_dim: ',num2str(opts.dim(1,3))]; 
fprintf(c,'%s\n',sentence); 
sentence=['input_dim: ',num2str(opts.dim(1,4))]; 
fprintf(c,'%s\n',sentence); 

fprintf(c,'\n%s'); 

if ~strcmp(phase,'test')
    temp='input: "labels"'; 
    fprintf(c,'%s\n',temp);  
    for i=1:4
    sentence=['input_dim: ',num2str(opts.dim(2,i))]; 
    fprintf(c,'%s\n',sentence);  
    end 
    fprintf(c,'\n%s');
end

conv_num=1;
pool_num=1;
product_num=1;
norm_num=1;
drop_num=1;
bn_num=1;
scale_num=1;
soft_num=1;
relu_num=1;
bottom='data';
for m=1:length(type)
T=type{m};
switch T
    case 'conv'
    [sen top]= conv(bottom,conv_num,para{m});     
    for i=1:length(sen)
    fprintf(c,'\n%s',sen{i});  
    end
    fprintf(c,'\n%s'); 
    conv_num=conv_num+1;

    clear sen;

    case 'pooling'
    [sen top]=pool(bottom,pool_num,para{m});
    for i=1:length(sen)
    fprintf(c,'\n%s',sen{i});  
    end
    fprintf(c,'\n%s'); 
    pool_num=pool_num+1;
    clear sen;

    case 'interproduct'
     [sen top]=interproduct(bottom,product_num,para{m});
     for i=1:length(sen)
     fprintf(c,'\n%s',sen{i});  
     end
     fprintf(c,'\n%s'); 
     product_num=product_num+1;
     clear sen;
     
     case 'norm'
     [sen top]=norm(bottom,norm_num,para{m});
     for i=1:length(sen)
     fprintf(c,'\n%s',sen{i});  
     end
     fprintf(c,'\n%s'); 
     norm_num=norm_num+1;
     clear sen;
    
     case 'loss'
     [sen top]=loss(bottom,para{m});
     for i=1:length(sen)
     fprintf(c,'\n%s',sen{i});  
     end
     fprintf(c,'\n%s'); 
     clear sen;
     
    case 'drop'
     [sen top]=dropout(bottom,drop_num,para{m});
     for i=1:length(sen)
     fprintf(c,'\n%s',sen{i});  
     end
     fprintf(c,'\n%s'); 
     drop_num=drop_num+1;
     clear sen;
 
    case 'batchnorm'
     [sen top]=batchnorm(bottom,bn_num,para{m});
     for i=1:length(sen)
     fprintf(c,'\n%s',sen{i});  
     end
     fprintf(c,'\n%s'); 
     bn_num=bn_num+1;
     clear sen;
 
    case 'scale'
     [sen top]=scale(bottom,scale_num,para{m});
     for i=1:length(sen)
     fprintf(c,'\n%s',sen{i});  
     end
     fprintf(c,'\n%s'); 
     scale_num=scale_num+1;
     clear sen;
     
    case 'relu'
     [sen top]=relu(bottom,relu_num,para{m});
     for i=1:length(sen)
     fprintf(c,'\n%s',sen{i});  
     end
     fprintf(c,'\n%s'); 
     relu_num=relu_num+1;
     clear sen;
    
    case 'softmax'
    [sen top]=softmax(bottom,soft_num,para{m}); 
     for i=1:length(sen)
     fprintf(c,'\n%s',sen{i});  
     end
     fprintf(c,'\n%s'); 
     soft_num=soft_num+1;
     clear sen;     
end
bottom=[];
bottom=top;
end
%nd
 fclose(c);    %�ر�txt�ļ� 
end
 
%% conv
function [sen top]= conv(bottom,num,para)            
global opts;
if ~isempty(para)
    conv_ke=para.Ckernel;
    conv_output=para.Coutput;
    conv_stride=para.Cstride;
    conv_pad=para.Cpad;
    conv_std=para.Cweightstd;
else
    conv_ke=opts.convPara.Ckernel;
    conv_output=opts.convPara.Coutput;
    conv_stride=opts.convPara.Cstride;
    conv_pad=opts.convPara.Cpad; 
    conv_std=opts.convPara.Cweightstd;     
end
flag=opts.flag;
opts.count=1;
sen{opts.count}='layer {';add();
sen{opts.count}=[' bottom: "',bottom,'"'];add()
sen{opts.count}=[' top: "conv',num2str(num),'"'];add()
top=['conv',num2str(num)];
%sen{opts.count}=[' name: "conv',num2str(num),'"'];add()
sen{opts.count}=[' name: "',para.name,'"'];add()
sen{opts.count}=' type: "Convolution"';add()
if opts.lr_on
   par_1r(1)= opts.convPara.pro_lr(1);
   par_decay(1)= opts.convPara.pro_decay(1);
   par_1r(2)= opts.convPara.pro_lr(2);
   par_decay(2)= opts.convPara.pro_decay(2);
else
   par_1r(1)= 0;
   par_decay(1)= 0;
   par_1r(2)= 0;
   par_decay(2)= 0;
end
for i=1:2
sen{opts.count}='   param {';add();
sen{opts.count}=[ '    lr_mult: ',num2str(par_1r(i))];add();
sen{opts.count}=[ '    decay_mult: ',num2str(par_decay(i))];add();
sen{opts.count}='    }';add();
end
%parameter section  
sen{opts.count}=' convolution_param {';add();
% sen{opts.count}=[ '  num_output: ',num2str(opts.Coutput)];add();
sen{opts.count}=[ '  num_output: ',num2str(conv_output)];add();
sen{opts.count}=[ '  stride: ',num2str(conv_stride)];add();
sen{opts.count}=[ '  pad: ',num2str(conv_pad)];add();
% sen{opts.count}=[ '  kernel_size: ',num2str(opts.Ckernel)];add();
sen{opts.count}=[ '  kernel_size: ',num2str(conv_ke)];add();
sen{opts.count}='  weight_filler {';add();
sen{opts.count}=[ '   type: "',opts.convPara.Cweighttype,'"'];add();
sen{opts.count}=[ '   std: ',num2str(conv_std)];add();
sen{opts.count}='   }';add();
sen{opts.count}='   bias_filler {';add();
sen{opts.count}=[ '   type: "',opts.convPara.Cbiastype,'"'];add();
sen{opts.count}=[ '   value: ',num2str(opts.convPara.Cbiasvalue)];add();
sen{opts.count}='   }';add();
sen{opts.count}='  }';add();

sen{opts.count}=' }';add()
sen{opts.count}=' ';add()
% if flag==1
% sen{opts.count}=' layer {';add()
% sen{opts.count}=['  bottom: "conv',num2str(num),'"'];add()
% sen{opts.count}=['  top: "conv',num2str(num),'"'];add()
% sen{opts.count}=['  name: "relu',num2str(num),'"'];add()
% sen{opts.count}='  type: "ReLU"';add()
% sen{opts.count}=' }';add()
% else
% end
end
 
%% pool
function [sen top]=pool(bottom,num,para)
global opts;
if ~isempty(para)
    poolkernel=para.poolkernel;
    poolstride=para.poolstride;
else
    poolkernel=opts.poolPara.poolkernel;
    poolstride=opts.poolPara.stride;
end

opts.count=1;
sen{opts.count}='layer {';add();
sen{opts.count}=[' bottom: "',bottom,'"'];add();
sen{opts.count}=[' top: "pool',num2str(num),'"'];add();
top=['pool',num2str(num)];
%sen{opts.count}=[' name: "pool',num2str(num),'"'];add();
sen{opts.count}=[' name: "',para.name,'"'];add()
sen{opts.count}=' type: "Pooling"';add();
sen{opts.count}=' pooling_param {';add();
sen{opts.count}=[ '  pool:',opts.poolPara.poolsize];add();
% sen{opts.count}=[ '  kernel_size:',num2str(opts.poolkernel)];add();

if ~isempty(para)
    if para.global==1
        sen{opts.count}=[ '  global_pooling:true'];add();
    else
        sen{opts.count}=[ '  kernel_size:',num2str(poolkernel)];add();
        sen{opts.count}=[ '  stride:',num2str(poolstride)];add();
    end
end



sen{opts.count}='  }';add();
sen{opts.count}=' }';add();
 end

 %% fc
function [sen top]=interproduct(bottom,num,para)              
global opts;
if ~isempty(para)
    fc_output=para.Poutput;
    fc_std=para.Pweightstd;
else
    fc_output=opts.fcPara.Poutput;
    fc_std=opts.fcPara.Pweightstd;
end

opts.count=1;
sen{opts.count}='layer {';add();
sen{opts.count}=[' bottom: "',bottom,'"'];add();
sen{opts.count}=[' top: "fc',num2str(num),'"'];add();
top=['fc',num2str(num)];
%sen{opts.count}=[' name: "fc',num2str(num),'"'];add();
sen{opts.count}=[' name: "',para.name,'"'];add()
sen{opts.count}=' type: "InnerProduct"';add();
if opts.lr_on
   par_1r(1)= opts.fcPara.pro_lr(1);
   par_decay(1)= opts.fcPara.pro_decay(1);
   par_1r(2)= opts.fcPara.pro_lr(2);
   par_decay(2)= opts.fcPara.pro_decay(2);
else
   par_1r(1)= 0;
   par_decay(1)= 0;
   par_1r(2)= 0;
   par_decay(2)= 0;
end
for i=1:2
sen{opts.count}='   param {';add();
sen{opts.count}=[ '    lr_mult: ',num2str(par_1r(i))];add();
sen{opts.count}=[ '    decay_mult: ',num2str(par_decay(i))];add();
sen{opts.count}='    }';add();
end
%parameter section
sen{opts.count}='   inner_product_param {';add();
% sen{opts.count}=[ '   num_output: ',num2str(opts.Poutput)];add();
sen{opts.count}=[ '   num_output: ',num2str(fc_output)];add();
sen{opts.count}='   weight_filler {';add();
sen{opts.count}=[ '    type: "',opts.fcPara.Pweighttype,'"'];add();
sen{opts.count}=[ '    std: ',num2str(fc_std)];add();
sen{opts.count}='    }';add();
sen{opts.count}='    bias_filler {';add();
sen{opts.count}=[ '    type: "',opts.fcPara.Pbiastype,'"'];add();
sen{opts.count}=[ '    value: ',num2str(opts.fcPara.Pbiasvalue)];add();
sen{opts.count}='   }';add();
sen{opts.count}='  }';add();

sen{opts.count}=' }';add();
 end
 
%% norm 
function [sen top]=norm(bottom,num,para)
global opts;
opts.count=1;
sen{opts.count}='layer {';add();
sen{opts.count}=[' bottom: "',bottom,'"'];add();
sen{opts.count}=[' top: "norm"'];add();
top='norm';
%sen{opts.count}=[' name: "wangl2_',num2str(num),'"'];add();
sen{opts.count}=[' name: "',para.name,'"'];add()
sen{opts.count}=' type: "WangL2"';add();
sen{opts.count}='  }';add();
end

%% loss
function [sen top]=loss(bottom,para) 
global opts;
opts.count=1;
sen{opts.count}='layer {';add();
sen{opts.count}=[' bottom: "',bottom,'"'];add();
sen{opts.count}=' bottom: "labels"';add();
sen{opts.count}=[' top: "loss"'];add();
top='loss';
%sen{opts.count}=[' name: "loss"'];add();
sen{opts.count}=[' name: "',para.name,'"'];add()
sen{opts.count}=' type: "SoftmaxWithLoss"';add();
sen{opts.count}='  }';add();
end
function [sen top]=dropout(bottom,num,para) 
global opts;
opts.count=1;
sen{opts.count}='layer {';add();
sen{opts.count}=[' bottom: "',bottom,'"'];add();
sen{opts.count}=[' top: "',bottom,'"'];add();
top=bottom;
%sen{opts.count}=[' name: "drop',num2str(num),'"'];add();
sen{opts.count}=[' name: "',para.name,'"'];add()
sen{opts.count}=' type: "Dropout"';add();
sen{opts.count}=' dropout_param {';add();
sen{opts.count}=[ '    dropout_ratio: ',num2str(opts.dropPara.dropratio)];add();
sen{opts.count}='   }';add();
sen{opts.count}='  }';add();
end

function [sen top]=batchnorm(bottom,num,para) 
global opts;
opts.count=1;
sen{opts.count}='layer {';add();
sen{opts.count}=[' bottom: "',bottom,'"'];add();
sen{opts.count}=[' top: "',bottom,'"'];add();
top=bottom;
%sen{opts.count}=[' name: "batchnorm',num2str(num),'"'];add();
sen{opts.count}=[' name: "',para.name,'"'];add()
sen{opts.count}=' type: "BatchNorm"';add();
% sen{opts.count}=' batch_norm_param {';add();
% sen{opts.count}='    use_global_stats: false ';add();
% sen{opts.count}='   }';add();
sen{opts.count}='  }';add();
end

function [sen top]=scale(bottom,num,para) 
global opts;
opts.count=1;
sen{opts.count}='layer {';add();
sen{opts.count}=[' bottom: "',bottom,'"'];add();
sen{opts.count}=[' top: "',bottom,'"'];add();
top=bottom;
%sen{opts.count}=[' name: "scale',num2str(num),'"'];add();
sen{opts.count}=[' name: "',para.name,'"'];add()
sen{opts.count}=' type: "Scale"';add();
sen{opts.count}=' scale_param {';add();
sen{opts.count}='    bias_term: true ';add();
sen{opts.count}='   }';add();
sen{opts.count}='  }';add();
end

function [sen top]=relu(bottom,num,para) 
global opts;
opts.count=1;
sen{opts.count}='layer {';add();
sen{opts.count}=[' bottom: "',bottom,'"'];add();
sen{opts.count}=[' top: "',bottom,'"'];add();
top=bottom;
%sen{opts.count}=[' name: "relu',num2str(num),'"'];add();
sen{opts.count}=[' name: "',para.name,'"'];add()
sen{opts.count}=' type: "ReLU"';add();
sen{opts.count}='  }';add();
end


function [sen top]=softmax(bottom,num,para) 
global opts;
opts.count=1;
sen{opts.count}='layer {';add();
sen{opts.count}=[' bottom: "',bottom,'"'];add();
sen{opts.count}=[' top: "prop',num2str(num),'"'];add();
top=['prop',num2str(num)];
%sen{opts.count}=[' name: "prop',num2str(num),'"'];add();
sen{opts.count}=[' name: "',para.name,'"'];add()
sen{opts.count}=' type: "Softmax"';add();
sen{opts.count}='  }';add();
end

 function count=add()
 global opts;
 opts.count= opts.count+1;
 conut=opts.count;
 end