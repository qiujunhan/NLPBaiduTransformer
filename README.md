# 2022年国际大数据竞赛

## 1 环境与安装
### 1.1 系统环境
Centos 7  
Python 3.6.2+ (64 bit)  
pip 或 pip3 版本 20.2.2或更高版本 (64 bit)

### 1.2 安装飞桨（PaddlePaddle）
建议安装paddlepaddle-gpu==2.2.2，方式参照飞桨官网（https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html）

### 1.3 代码准备
获取PaddleNLP（https://github.com/PaddlePaddle/PaddleNLP） 2.2.6版本源码，参考如下操作：  
```
git clone https://github.com/PaddlePaddle/PaddleNLP.git  
cd PaddleNLP && git checkout v2.2.6  
```
获取源码后，将PaddleNLP/examples/machine_translation/transformer/reader.py，替换为本项目中的reader.py，以读取示例格式的数据。

### 1.4 环境变量配置
请确保CUDA Driver API、CUDA runtime、CUDNN、NCCL2（单机多卡训练时需要）等环境变量配置正确
export LD_LIBRARY_PATH=/自行配置

> 可参考PaddlePaddle安装文档中的验证命令，确认PaddlePaddle环境安装正确  
python3 -c 'import paddle;paddle.utils.run_check()' 

> 配置paddlenlp源码所在路径，即“代码准备”中PaddleNLP的绝对路径  
export PNLP_ROOT=/path/to/PaddleNLP

> 配置PYTHONPATH，引入paddlenlp模块  
export PYTHONPATH=$PNLP_ROOT

## 2 数据处理
### 2.1 数据集与读数文件下载 
https://dataset-bj.cdn.bcebos.com/qianyan/datasets.tar.gz

### 2.2 预处理与Tokenize
对数据进行分词等预处理，可参考使用下面的工具
中文： jieba0.42.1 （https://github.com/fxsjy/jieba）  
法文： wmt-13a （https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_13a.py）  
俄文： wmt-13a （https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_13a.py）  
泰文： pythainlp （https://github.com/PyThaiNLP/pythainlp）  

将预处理后的数据转化为token形式，可使用开源的tokenizer，如：
Huggingface tokenizers （https://github.com/huggingface/tokenizers）  

Tokenizer使用示例：
```python
from tokenizers import Tokenizer 
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
#输入已预处理文件（file.txt），训练tokenizer并保存（tokenizer.json）
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["<s>", "<e>", "<unk>"])
tokenizer.train(files=["file.txt"], trainer=trainer)
tokenizer.save("tokenizer.json") 
#Tokenize
output = tokenizer.encode("我们 都 有 这样 那样 的 问题 。")
print(output.tokens)
# ▁ 我 们 ▁ 都 ▁ 有 ▁这样 ▁那样 ▁ 的 ▁问题 ▁ 。
#Detokenize
tokenizer = Tokenizer.from_file("tokenizer.json")
token_list = "▁ 我 们 ▁ 都 ▁ 有 ▁这样 ▁那样 ▁ 的 ▁问题 ▁ 。".split(" ")
id_list = [tokenizer.token_to_id(token) for token in token_list]
output = tokenizer.decode(id_list) 
print(output)
#我们 都 有 这样 那样 的 问题 。
print(output.replace(" ", ""))
#我们都有这样那样的问题。
```

### 2.3 数据集格式
数据需要组织成训练集、开发集和测试集，其中每行数据为一对源语言和目标语言的token序列，通过转义字符“\t”分隔。以中法数据为例：
```
▁ 他 ▁提出 ▁ 了 ▁一个 ▁基 于 ▁ 线 性 ▁回 归 ▁ 的 ▁ 数 学 模 型 ▁ 。 ▁Il ▁pro pose ▁un ▁modèle ▁ma th é m atique ▁bas é ▁sur ▁une ▁ré g re s s ion ▁linéaire ▁ .
```
数据集以外，代码运行还需一份词典，其中含有源语言与目标语言token，以及一些特殊```token（<s>, <e> ,<unk>）```。

## 3 训练与预测
### 3.1 单机单卡训练
> 指定GPU设备  
```javascript
export CUDA_VISIBLE_DEVICES=0
```
>进入工作路径
```bash
cd $PNLP_ROOT/examples/machine_translation/transformer
```
>#执行训练脚本
```bash
python3 train.py --config ./configs/transformer.base.yaml \
	--train_file data.train \ #训练集路径
	--dev_file data.dev \ #开发集路径
	--vocab_file vocab \ #词表路径
	--unk_token "<unk>" \ 
	--bos_token "<s>" \
	--eos_token "<e>"
```

### 3.2 单机多卡训练
> #指定GPU设备
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```
> 进入工作路径
```bash
cd $PNLP_ROOT/examples/machine_translation/transformer
```
> 执行训练脚本
```bash
python3 -u -m paddle.distributed.launch --gpus '0,1,2,3'  \
	train.py --config ./configs/transformer.base.yaml \
	--train_file data.train \ #训练集路径
	--dev_file data.dev \ #开发集路径
	--vocab_file vocab \ #词表路径
	--unk_token "<unk>" \ 
	--bos_token "<s>" \
	--eos_token "<e>"
```

### 3.3 预测
将配置文件transformer.base.yaml中的“init_from_params”项配置为您的模型路径。
> 指定GPU设备
```bash
export CUDA_VISIBLE_DEVICES=0
```
> 进入工作路径
```bash
cd $PNLP_ROOT/examples/machine_translation/transformer
```
> 执行测试脚本
```bash
python3 predict.py --config ./configs/transformer.base.yaml \
    --test_file data.test \
    --vocab_file vocab \
    --unk_token "<unk>" \
    --bos_token "<s>" \
    --eos_token "<e>" \
    --without_ft
```
