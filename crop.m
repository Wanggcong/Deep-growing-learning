function [ cropped_img ] = crop( img,crop_h,crop_w,phase )
%CROP Summary of this function goes here
%   randomly crop for train, while center crop for test
    [img_h,img_w,~]=size(img);
    if(img_h<crop_h || img_w<crop_w)
        error('crop dim should be not more than image dim!');
    end
    if(strcmp(phase,'train'))
        h_off=randi(img_h-crop_h+1,1,1);
        w_off=randi(img_w-crop_w+1,1,1);
    else
        h_off=floor((img_h-crop_h)/2)+1;
        w_off=floor((img_w-crop_w)/2)+1;        
    end
    cropped_img=img(h_off:crop_h+h_off-1,w_off:crop_w+w_off-1,:);
end

