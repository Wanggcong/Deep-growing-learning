function [ conv_or_not ] = convergence( percision_seq,ref_num,err )
%CONVERGENCE Summary of this function goes here
%   Detailed explanation goes here
%input: 
%percision_seq: each epoch precision

%output:
%convergence or not
    conv_or_not=0;
    seq_len=length(percision_seq);
    if seq_len>ref_num
        mean_pre=mean(percision_seq(seq_len-ref_num:seq_len-1));
        cur_pre=percision_seq(seq_len);
        if abs(cur_pre-mean_pre)<err
            conv_or_not=1;
        end
    end
end

