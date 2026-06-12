1. minisgl服务会起几个进程？1 + tp_size + 1 + num_tokenizer

2. 为什么tokenizer设置参数num_tokenizer，而detokennizer固定只有1个？

3. 什么是ZMQ？

4. 为什么要设置cuda graph最大的batch size？

1. 为什么不同stage，比如prefill和decode需要用不同的attention backend？

2. 为什么`max-prefill-length`这个参数不建议设置得太小呢？

3. 对NCCL和ZMQ的基本理解：

4. scheduler和engine作用的区别？

topk_softmax kernel的优化