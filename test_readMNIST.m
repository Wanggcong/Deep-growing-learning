clear all;clc;
path_data='/home/wang/Downloads/dataset/MNIST/stl_exercise/mnist/t10k-images.idx3-ubyte';
path_label='/home/wang/Downloads/dataset/MNIST/stl_exercise/mnist/t10k-labels.idx1-ubyte';
imgs = loadMNISTImages(path_data);
labels = loadMNISTLabels(path_label);
few_label_num=1000;
data_num=length(labels);
fig_cells=cell(10,1);
for i=0:9
    one_fig_index=find(labels==i);
    index_index=randperm(length(one_fig_index),few_label_num/10);
    fig_cells{i+1,1}=one_fig_index(index_index);
end

%% visualize debug
for i=1:few_label_num/10
    img=imgs(:,fig_cells{9,1}(i));
    I=reshape(img,[28,28]);
    imshow(I);
end
