function [ data ] = test_blob( imgs,resize_img_h, resize_img_w,crop_h,crop_w,phase)
    %TEST_NET Summary of this function goes here
    % input:
    % net_model:.prototxt
    % net_weights: .caffemodel
    % imgs:784*10000
    % labels:10000*1

    % resize_img_h=32;
    % resize_img_w=32;
    % crop_h=28;
    % crop_w=28;

    %resize_scales=25:31;                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    data_num=size(imgs,2);
    %% step 3 : set data and net.forward
    data=zeros(crop_h,crop_w,3,data_num);
    for k=1:data_num       
        im_data=imgs(:,k);
        im_data=reshape(im_data,[resize_img_h,resize_img_w,3]);
        %im_data=im_data(:,:,[3,2,1]);
        %im_data = permute(im_data, [2, 1]);
        % convert from uint8 to single
        im_data = single(im_data);     

        %bound=0.5;
        %im_data=single(im_data(:,:)>bound); 
        %1.minus mean_data?
        im_data=im_data-128;
        %2.resize
        % scale_index=randperm(length(resize_scales),1);
        % one_scale=resize_scales(scale_index);
        % im_data=imresize(im_data,[one_scale,one_scale]);

        %im_data = imresize(im_data, [resize_img_h, resize_img_w]); % resize using Matlab's imresize
        %3.crop
        %im_data=crop(im_data,crop_h,crop_w,'test');
        im_data=crop(im_data,crop_h,crop_w,phase);
        %data(:,:,:,i)=im_data_crop*0.1; 
        %如果是单通道，复制成多通道
        %data(:,:,:,i)=im_data_crop;
        data(:,:,:,k)=im_data;

        %label
        %batch_labels=reshape(batch_labels,[1,1,1,batchsize]);
    end
end

