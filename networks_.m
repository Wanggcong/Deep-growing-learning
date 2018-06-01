global train_type;
global train_para;
global test_type;
global test_para;



convo1_1.Coutput=32;
convo1_1.Ckernel=3;
convo1_1.Cstride=1;
convo1_1.Cpad=1;   % 0 change to 1,12.22.09
convo1_1.Cweightstd=0.0001;
convo1_1.name='convo1_1';

convo1_2.Coutput=32;
convo1_2.Ckernel=3;
convo1_2.Cstride=1;
convo1_2.Cpad=1;   % 0 change to 1,12.22.09
convo1_2.Cweightstd=0.0001;
convo1_2.name='convo1_2';

convo2_1.Coutput=64;
convo2_1.Ckernel=3;
convo2_1.Cstride=1;
convo2_1.Cpad=1;   % 0 change to 1,12.22.09
convo2_1.Cweightstd=0.01;
convo2_1.name='convo2_1';

convo2_2.Coutput=64;
convo2_2.Ckernel=3;
convo2_2.Cstride=1;
convo2_2.Cpad=1;   % 0 change to 1,12.22.09
convo2_2.Cweightstd=0.01;
convo2_2.name='convo2_2';

convo3_1.Coutput=128;
convo3_1.Ckernel=3;
convo3_1.Cstride=1;
convo3_1.Cpad=1;  % 0 change to 1,12.22.09
convo3_1.Cweightstd=0.01;
convo3_1.name='convo3_1';

convo3_2.Coutput=128;
convo3_2.Ckernel=3;
convo3_2.Cstride=1;
convo3_2.Cpad=1;  % 0 change to 1,12.22.09
convo3_2.Cweightstd=0.01;
convo3_2.name='convo3_2';


convo4_1.Coutput=128;
convo4_1.Ckernel=2;
convo4_1.Cstride=1;
convo4_1.Cpad=1;  % 0 change to 1,12.22.09
convo4_1.Cweightstd=0.01;
convo4_1.name='convo4_1';

convo4_2.Coutput=128;
convo4_2.Ckernel=2;
convo4_2.Cstride=1;
convo4_2.Cpad=0;  % 0 change to 1,12.22.09
convo4_2.Cweightstd=0.01;
convo4_2.name='convo4_2';


pool1.poolkernel=2;
pool1.poolstride=2;
pool1.global=0;
pool1.name='pool1';

pool2.poolkernel=2;
pool2.poolstride=2;
pool2.global=0;
pool2.name='pool2';


% pool3.poolkernel=3;
% pool3.poolstride=2;
% pool3.global=0;
% pool3.name='pool3';

pool3.poolkernel=2;
pool3.poolstride=2;
pool3.global=0;
pool3.name='pool3';

pool4.poolkernel=2;
pool4.poolstride=1
pool4.global=0;
pool4.name='pool4';



fc1.Poutput=64;
fc1.Pweightstd=0.1;
fc1.name='fc1';

fc2.Poutput=10;
fc2.Pweightstd=0.1;
fc2.name='fc2';


relu1_1.name='relu1_1';
relu1_2.name='relu1_2';


relu2_1.name='relu2_1';
relu2_2.name='relu2_2';


relu3_1.name='relu3_1';
relu3_2.name='relu3_2';


relu4_1.name='relu4_1';
relu4_2.name='relu4_2';

%relu5.name='relu5';
relu6.name='relu6';


bn1_1.name='bn1_1';
bn1_2.name='bn1_2';


bn2_1.name='bn2_1';
bn2_2.name='bn2_2';


bn3_1.name='bn3_1';
bn3_2.name='bn3_2';


bn4_1.name='bn4_1';
bn4_2.name='bn4_2';


%bn5.name='bn5';


drop1.name='drop1';
loss.name='loss';
scores.name='scores';


train_type={'conv','relu','batchnorm',...
            'conv','relu','batchnorm','pooling',...
            'conv','relu','batchnorm',...
            'conv','relu','batchnorm','pooling',...
            'conv','relu','batchnorm',...
            'conv','relu','batchnorm','pooling',...
            'conv','relu','batchnorm',...
            'conv','relu','batchnorm','pooling',...                                          
            'interproduct','relu','drop','interproduct','loss'};
train_para={convo1_1,relu1_1,bn1_1,...
            convo1_2,relu1_2,bn1_2,pool1,...
            convo2_1,relu2_1,bn2_1,...
            convo2_2,relu2_2,bn2_2,pool2,...
            convo3_1,relu3_1,bn3_1,... 
            convo3_2,relu3_2,bn3_2,pool3,... 
            convo4_1,relu4_1,bn4_1,...                 
            convo4_2,relu4_2,bn4_2,pool4,...                 
            fc1,relu6,drop1,fc2,loss};

test_type= {'conv','relu','batchnorm',...
            'conv','relu','batchnorm','pooling',...
            'conv','relu','batchnorm',...
            'conv','relu','batchnorm','pooling',...
            'conv','relu','batchnorm',...
            'conv','relu','batchnorm','pooling',...
            'conv','relu','batchnorm',...
            'conv','relu','batchnorm','pooling',...                      
            'interproduct','relu','drop','interproduct','softmax'};
test_para= {convo1_1,relu1_1,bn1_1,...
            convo1_2,relu1_2,bn1_2,pool1,...
            convo2_1,relu2_1,bn2_1,...
            convo2_2,relu2_2,bn2_2,pool2,...
            convo3_1,relu3_1,bn3_1,... 
            convo3_2,relu3_2,bn3_2,pool3,... 
            convo4_1,relu4_1,bn4_1,...                 
            convo4_2,relu4_2,bn4_2,pool4,...          
            fc1,relu6,drop1,fc2,scores};          

