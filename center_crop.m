function [ crops_data ] = center_crop( im )
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
% d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
% mean_data = d.mean_data;
mean_data=128;
IMAGE_DIM_H = 100;
IMAGE_DIM_W = 250;
CROPPED_DIM_H = 80;
CROPPED_DIM_W = 230;
% CROPPED_DIM=0;
% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM_H IMAGE_DIM_W], 'bilinear');  % resize im_data
im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)
im_data=im_data*0.1;

% oversample (4 corners, center, and their x-axis flips)

%{
crops_data = zeros(CROPPED_DIM_H, CROPPED_DIM_W, 3, 10, 'single');
indices_h = [0 IMAGE_DIM_H-CROPPED_DIM_H] + 1;
indices_w = [0 IMAGE_DIM_W-CROPPED_DIM_W] + 1;

% indices=0;
n = 1;
for i = indices_h
  for j = indices_w
    crops_data(:, :, :, n) = im_data(i:i+CROPPED_DIM_H-1, j:j+CROPPED_DIM_W-1, :);
    crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
    n = n + 1;
  end
end
center_h = floor(indices_h(2) / 2) + 1;
center_w = floor(indices_w(2) / 2) + 1;

crops_data(:,:,:,5) = ...
  im_data(center_h:center_h+CROPPED_DIM_H-1,center_w:center_w+CROPPED_DIM_W-1,:);
crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);
%}


crops_data = zeros(CROPPED_DIM_H, CROPPED_DIM_W, 3, 1, 'single');
indices_h = [0 IMAGE_DIM_H-CROPPED_DIM_H] + 1;
indices_w = [0 IMAGE_DIM_W-CROPPED_DIM_W] + 1;
center_h = floor(indices_h(2) / 2) + 1;
center_w = floor(indices_w(2) / 2) + 1;
crops_data(:,:,:,:)=im_data(center_h:center_h+CROPPED_DIM_H-1,center_w:center_w+CROPPED_DIM_W-1,:);

end

