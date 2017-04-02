#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  // negative_slope [默认值 0] : 当输入为x负数时，指定输出为negative_slope * x；默认值为0.   
  // 当输入为x正数时，指定输出为x；  
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  //PReLu
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

/*[转]来自百度的一个描述 
caffe中怎么固定前面的网络参数，训练后面层的参数？ 
这里面就用到了propagate_down， 
有两种情况：比如有4个全连接层A->B->C->D 
    a. 你希望C层的参数不会改变，C前面的AB层的参数也不会改变，这种情况也就是D层的梯度不往前反向传播到D层的输入blob（也就是C层的输出blob 没有得到梯度），你可以通过设置D层的propagate_down为false来做到。 
         propagate_down的数量与输入blob的数量相同，假如你某个层有2个输入blob，那么你应该在该layer的Param里面写上两行： 
         propagate_down : 0    # 第1个输入blob不会得到反向传播的梯度 
         propagate_down : 0    # 第2个输入blob不会得到反向传播的梯度 
         这样的话，你这个layer的梯度就不会反向传播啦，前面的所有layer的参数也就不会改变了 
    b. 你希望C层的参数不会改变，但是C前面的AB层的参数会改变，这种情况，只是固定了C层的参数，C层得到的梯度依然会反向传播给前面的B层。只需要将对应的参数blob的学习率调整为0： 
你在layer里面加上param { lr_mult: 0 }就可以了，比如全连接层里面： 
layer { 
    type: "InnerProduct" 
    param { # 对应第1个参数blob的配置，也就是全连接层的参数矩阵的配置 
         lr_mult: 0 # 学习率为0，其他参数可以看caffe.proto里面的ParamSpec这个类型 
    } 
    param { # 对应第2个参数blob的配置，也就是全连接层的偏置项的配置 
        lr_mult: 0 # 学习率为0 
    } 
} 
*/  

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      // 当输入bottom_data大于０，则bottom_diff[i] = top_diff[i]  
      // 当输入bottom_data小于等于０，则bottom_diff[i] = top_diff[i] * negative_slope  
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
