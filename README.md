# 又出新模型了？本文教你以不变应万变
## 频率派的最大似然估计和贝叶斯派的最大后验估计
（摘自软绵绵的小熊猫）
它们都是参数估计的方法，都是对模型的参数进行求解。
* 频率派把未知参数看成普通变量，把样本看成随机变量。
* 概率就是某一个随机试验进行无穷次重复，然后统计发生的频率。样本空间可以无限大，可以有放回无限反复抽取。

* 预测一个小时后上海会不会下雨，对于统计学比较拉胯的情况，贝叶斯派优势体现出来。

* 频率派和贝叶斯派最大的区别就在于对于随机变量如何看待。

频率派认为随机变量是一个固定的值（就是一个普通的变量）
贝叶斯派认为变量一开始即服从某一分布，我们观测到的值就是分布的不断叠加，这样的叠加就意味着对参数分布的估计会产生变化

机器学习案例当中的做法：
不管用什么方法，最终都要落实到模型的求解上来，也就是给我一些数据，我们来求解模型和参数。
以扔硬币为例：
抛一枚硬币，十次，9次正面，1次反面。有的人认为硬币就是均匀的。这里的先验知识是什么呢？最大后验又是什么。


#### 一）频率派求解模型参数的做法：
三步走——1）设计模型，2）设计损失函数，3）具体的参数求解（牛顿法、梯度下降、直接给出解析解）![16273742517978](https://i.loli.net/2021/07/27/hQFNJ79dxnc21SE.jpg)
公式中，theta是未知常量，也就是我们要求的硬币正面朝上的概率，X是一个随机变量，即扔硬币试验的结果，它们是独立无关的。
我们要做的就是通过X反过来求theta。又因为在若干次实验中，事件是真实发生了，那么这些事件同时发生的概率该如何计算呢？当然是直接相乘。公式中P代表二项分布。argmax是似然函数，我们要求得它最大同时得到theta的估计，这个叫做极大似然估计。那我们对它求极值就可以得到我们需要的theta，也就是我们这里硬币正面朝上的几率。有的时候为了方便求解，会加上log函数，叫做log似然函数。这就是频率派求解模型参数的做法，就是上面说过的三步走——1）设计模型，2）设计损失函数，3）具体的参数求解（牛顿法、梯度下降、直接给出解析解）
这是所有的频率派极大似然估计的套路，这类模型本质是统计学问题，目的是做参数估计。

#### 二）贝叶斯学派求解模型参数的做法：
问题在于theta不是一个固定的值，它是一个分布。这里面假设它是一个分布，叫做p。我们要求的什么呢？我们在已有数据条件下，让后验概率最大，那我们借助贝叶斯公式来求它。px是整个样本空间，我们关心的是左边最大值时候的theta，所以我们要求的是参数，和分母无关，所以可以理解成它的最大值其实正比于整个分子。

![16273645899149](https://i.loli.net/2021/07/27/UPvVzgZapRwcWoK.jpg)就是这样。这是最大后验估计得出来的结果。
后验概率最大，关心的是左边最大值时的theta，我们要求的是参数和分母无关。

最大后验估计只是贝叶斯派的一种参数估计的方法，因为应用了贝叶斯公式是为了求theta而不是完全的贝叶斯派。
什么是完全的贝叶斯派？其实就是贝叶斯估计或者叫贝叶斯预测，我们要求的就是完整的后验概率的计算，就是要把刚才和theta无关的分母求出来。把整个后验概率求出来之后，有什么用处呢？求出后验概率之后，就可以借助theta来完成对一个新样本的预测。![16273744428310](https://i.loli.net/2021/07/27/vI8bMShKQtk2p43.jpg)
这样就可以对x0这样的新样本来预测它的结果。换句话说，不止关心模型的参数theta，更加关心的是一个端到端的问题。从所有已知的数据当中求解新样本的概率。这一部分有点边缘概率的意思，我们计算得到新样本的预测结果，这是真正的贝叶斯预测。

真正的贝叶斯预测是非常难做的，因为我们在刚才的最大后验估计中，忽略的分母部分其实需要完整的把它算出来，然后做一个对参数空间的积分。那这里面这个积分其实是非常难求的，甚至很多时候解析解找不到，只能用一些数值积分的方法，比如说一些采样的方法能够把它解出来，后面会在概率图模型当中详细讲解，这里先跳过。现在只需要知道完整的贝叶斯预测求解非常困难，需要用到一些数值积分的方法就可以。
![16273745212871](https://i.loli.net/2021/07/27/dTsEIlFP9LYyuOf.jpg)
贝叶斯派的最大后验估计，仔细看就会发现其中有一部分就是前面提到的极大似然估计，另外的部分其实就是我们对参数的默认分布的假设，这里也叫先验。

# 谁是对的？下面的推导会证明，频率派极大似然估计加上L2正则化之后，和贝叶斯派最大后验估计，结果一致！
回到核心问题上，频率派 vs 贝叶斯派，极大似然估计 vs 最大后验估计。只要先验不总是为1，那么这两个式子估计出来的结果就不一样。那么谁是对的呢？回到上面抛硬币的例子，抛了10次有9次正面朝上，有的人觉得是0.9，完全按前面统计结果推算————就是上面MLE公式；有的人完全按照自己的直觉，按照自己多年来对硬币的了解，按照自己的先验知识，觉得应该在0.9基础上进行修正————就是下面MAP公式。

到这里，看起来好像还是贝叶斯派好。频率派看起来做到了尊重事实，但是很容易收到样本数量不足的影响，这也就是机器学习频率派统计学习方法中的过拟合的表现。

**那面对过拟合怎么办呢？样本数量没法增加了————上正则化！**那上正则化会变成什么样子，下面来研究一个线性回归的例子。
#### 一）频率派的做法
![16273590440433](https://i.loli.net/2021/07/27/egVHdhG3aYT8qFs.jpg)![16273590991203](https://i.loli.net/2021/07/27/sEcDiZrVwU5g84k.jpg)**这是在线性回归问题中，假设误差服从高斯分布的，频率派极大似然估计，为了防止过拟合加上L2正则化，得到的参数解。**

#### 二）贝叶斯派的做法
既然是贝叶斯派，那我们就需要假定有一个W的先验，这里我们先假设w服从一个正态分布， ![草稿本-199](https://i.loli.net/2021/07/27/nFINTKO5CXolDyQ.jpg)      

####三）结论：
对于线性回归，模型的参数（先验）服从高斯分布的情况下，**最大后验估计的计算结果**与**加上L2正则化的极大似然估计的估算结果**是一致的。
PS:如果模型的参数服从的不是高斯分布，如果是拉普拉斯分布，那么最大后验估计的结果与L1正则化的极大似然估计的结果又会保持一致。

面对样本过少带来的过拟合，频率派通常会使用正则化的技巧，来使模型变得更稳定，泛化能力更强。那贝叶斯派的最大后验估计，则对参数使用不同的先验分布来进行校正。

1. 为什么在推导过程中假设误差服从正态分布
2. 为什么L2正则化对应的先验是正态分布，L1正则化对应的是拉普拉斯分布；这两者对应的背后有什么样的数学解释，揭示了什么样的本质
                                                                                                                                                                                                               

下一期讲你见过的机器学习模型中，哪些模型是频率派做法，哪些模型又是贝叶斯派做法。这些频率派模型当中，它们的损失函数为什么要这么设计；贝叶斯派做法当中，它们又是如何引入先验分布的，它们为什么引入这样的先验分布。    

https://confluence.cec.lab.emc.com/display/FOIT/Sprint+Goals
https://eos2git.cec.lab.emc.com/Sylvia-Wang4/SemanticParsing
https://eos2git.cec.lab.emc.com/Clarity360 No public
https://eos2git.cec.lab.emc.com/Nick-Qiu/Detection_model_as_service
ssh root@10.199.196.34  Password123!
https://zhuanlan.zhihu.com/p/45025702    
https://digitalcloud.dell.com/#/home   GPU resource


https://pandoacademies.com/UI/Learner/LearnerILPDashboard.aspx#
https://jira.cec.lab.emc.com/secure/RapidBoard.jspa?rapidView=7661&projectKey=KARMAN






相关的论文内容学习

OPEN QUESTION ANSWERING OVER TABLE AND TEXT【ICLR，2021】
HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data【EMNLP，2020】
TAT-QA:A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance【ACL，(2021)】
Reasoning over Hybrid Chain for Table-and-Text Open Domain QA【Preprint】
Parameter-Effificient Abstractive Question Answering over Tables or Text【】（pre-train）
TAPAS: Weakly Supervised Table Parsing via Pre-training【ACL,2018】
Dual Reader-Parser on Hybrid Textual and Tabular Evidence for Open Domain Question Answering【ACL.2021】
TABERT: Pretraining for Joint Understanding of Textual and Tabular Data【ACL，2020】
TABBIE: Pretrained Representations of Tabular Data【NAACL，2021】
TabNet: Attentive Interpretable Tabular Learning【AAAI Conference on Artificial Intelligence，2021】
Open Domain Question Answering over Tables via Dense Retrieval【NAACL-HLT，2021】
TSQA: Tabular Scenario Based Question Answering 【AAAI Conference on Artificial Intelligence，2021】
CFGNN: Cross Flow Graph Neural Networks for Question Answering on Complex Tables【AAAI Conference on Artifificial Intelligence ，2020】
Retrieval & Interaction Machine for Tabular Data Prediction【KDD，2021】
UnitedQA: A Hybrid Approach for Open Domain Question Answering【ACL，2021】
T-RAG: End-to-End Table Question Answering via Retrieval-Augmented 【PPrint】


Thinking
大家都是在努力的构造各个领域，或者公共的dataset，发布出来。自己提出一个效果还不错的model出来，作为baseline，然后鼓励大家在他的数据集上搞一搞，这样搞得效果好像还不错，都能发一些比较好的会议。HybridQA，OTT-QA——（HybridQA的进化版，open版），TAT-QA

相关的数据集以及论文分类

大致可分为：

需要同时在表格和文本上推理的数据集（混合语境）
HybridQA
OTT-QA（HybridQA的进化版，更适合于开放域问题）
TAT-QA：text+table，同时还赋予模型数字推理能力（计算），还能预估数字的范围比例。
纯表格数据集
纯文本数据集




2、HybridQA
应该是混合QA问题上提出来的第一个大规模的同时基于表格数据和相关的段落数据，并把两者结合给出答案的的公共数据集了。

OTT-QA（HybridQA的进化版，open版）
他把HybridQA这个数据集进行了一些修改，把一些问题重新注释了，即所谓的”去语境化“，这样使得OTT-QA这个数据集更适合于开放域问题的QA。同时他们还往里面补充了一些新的问题，以消除原本的数据集里面潜在的偏见问题。贡献二：采用了两种新的技术（”早融合“和”跨块阅读器“技术）来增强了数据检索和汇聚的效果，增加模型的准确率。原文中是这样说的：Combining these two techniques improves the score significantly, to above 27%.

3、TAT-QA
主要就包含两个方面

1、数据集的构建（表格及其相关文本，它们是从真实世界的金融报告中提取出来的）

问题的构建（一个混合语境产生至少6个问题【提取问题和计算问题】）提取问题的答案可以是单一的span或者是来自表格或段落的多个span。计算问题的答案需要进行数字推理产生答案。

答案的构建

2、模型的构建

工作原理：该模型应用序列TAGging从表格中提取相关单元，从段落中提取文本跨度（span），对它们进行用一组聚合运算器（一共有10种）对它们进行符号推理，得出最终的答案。这个模型能够对表格数据和段落进行推理，还能在上面进行符号推理（做算数运算）

它用Sequence Tagging 的方式来解决表格数据和段落之间的相关性查找问题。这样的方式准确率貌似不是很高，作者在原文中随机选取了100个实例来进行错误分析，发现84%的错误都是提取的支持性证据的不足或者缺乏而造成的。

这就让人忍不住遐想，有没有别的方法来提高一下子表格数据和相关段落之间的相关性查找？？？

4、Reasoning over Hybrid Chain（预训练框架CAPR）
他在OTT-QA的数据集上接着搞，引入了混合推理链，来提高QA问题的正确率。在推理链条中首次引入了图的概念。（CARP fifirst formulates a heterogeneous graph, whose nodes are information pieces in the relevant table and passages, to represent the interaction residing in hybrid knowledge.）然后使用了一个 基于transformer的提取器，在图上确定最佳推理路径。

主要就包含两个方面：

1、提出了一个ChAin-centric的推理和预训练框架(CARP)

这个框架能够对推理过程中的可解释性做出贡献。

最主要的贡献就是在a retriever和a reader之间，加入了一个a chain extractor。（见下图）。

2、一种新的以链为中心的预训练方法，以加强预训练模型在分辨交叉模态推理过程、缓解数据稀疏问题的能力

这种预训练的方法，呢能够构成大规模的伪推理路径，并反过来生成问题。但是需要合成一个大规模的、高复杂的推理语料库。




知识检索的时候，他也采用了early-fusion机制，同时还采用了BLINK (Ledell et al., 2020) 的方法，把表格和相关的表格单元格连接起来。最后在检索知识的时候，使用了a shared RoBERTa-encoder (Liu et al., 2019)方法。在训练检索器模型的时候，借鉴了Karpukhin et al. (2020)的方法。

5、Parameter-Effificient Abstractive Question Answering over Tables or Text
研究预训练模型结构的，对其中的结构进行修改，减少训练成本的。

主要是在预训练过程中，对参数的有效性进行了探索和研究，分别对表格式数据集和文本数据集两种进行独立的分析，对其中的编码器和解码器的adapter layers进行消减（ablate ）分析。使用adapter-tuning，并证明将额外的可训练参数减少到0.7%-1.0%，在表格式数据集上效果要好于目前最先进的模型，在表格式数据集上可以达到差不多的效果，但是可训练的参数却要大大的少于fine-tuning.

另一个方面，对预训练模型的结构进行了研究，ablate adapter layers in both encoder and decoder modules to study their impact and show that beginning layers from both encoder and decoder can be eliminated without significant drop in performance. We also demonstrate that last encoder adapter layers are indispensable and have greater contribution than decoder layers at the same level.





6、TAPAS: Weakly Supervised Table Parsing via Pre-training
这篇文章主要讲了两个方面：

提出了TAPAS，用于在表格上进行QA。使用了弱监督训练，不产生logical forms。
介绍了TAPAS的预训练方法：从维基百科上抓的表格和相关的文本段落作为一个有效的联合预训练的初始化设置，采用拓展的BERT编码了表格序列作为训练的输入。
回答表格上的自然语言问题通常被看作是一项语义解析任务。为了减轻完整逻辑形式的收集成本，最近一种流行的方法侧重于弱监督。但是弱监督的方式训练语义解析器有点困难，而且中间会产生一个中间步骤（logical forms）the generated logical forms are only used as an intermediate step prior to retrieving the denotation.

作者提出的TAPAS方法不需要产生logical forms，还可以从弱监督中进行训练。预训练的过程，是通过文本-表格对作为预训练的输入，希望能够学习到文本和表格之间，以及列的单元格和标题之间的关联。这种预训练的方法对于实验结果的成功是至关重要的（extend BERT`s masked language model objective to structured data）。但是模型输入的表格是需要特殊处理一下子的，形式比较简单。

7、Dual Reader-Parser on Hybrid Textual and Tabular Evidence for Open Domain Question Answering
这是第一篇将Text2SQL应用于ODQA任务的论文。论文提出了一个混合框架，该框架将文本和表格证据作为输入，并生成直接答案或SQL查询语句。（取决于哪种形式可以更好的回答问题）。

论文证明了，能够生成结构化的SQL查询总是能够带来收益，特别是对于那些需要复杂推理的问题

总的来说。论文的框架由三个阶段组成：检索，联合排序和双重阅读-解析（retrieval，joint ranking and dual reading-parsing）。首先，检索出文本和表格类型的支持性候选集，然后使用联合排名器（a joint reranker）预测每个支持性候选者与问题的相关程度，最后使用fusion-in-decoder （Izacard和Grave，2020年）用于reader-parser，它还采用了所有重新排序的候选人来生成直接答案或SQL查询。

表格处理：

对于表格，我们通过连接每一行的单元格值，将每个表格压扁（flattened）成段落。如果被压扁（flattened）的超过100字，我们就把它分成一个单独的的段落，并尊重行的边界。列标题被串联到每个表格段落中。为了将一个结构化的表格表示为一个段落，我们首先将每个表扁平为以下格式：每个扁平的表以完整的头名开始，然后是行。图1给出了此转换的一个示例。最后，一个表格的候选者是将表格的标题和内容扁平化为段落的串联，分别由特殊标记[table title]和[table content]来附加。我们使用表的ID作为标题，这样它就可以被复制到模型生成的SQL查询中。




检索用在线网站Elasticsearch的服务【https://www.elastic.co/】实现的，

重排序是用BERT reranker initialized with pretrained BERT-base-uncased model



8、TABERT: Pretraining for Joint Understanding of Textual and Tabular Data【ACL，2020】
TABERT是一个预训练的语言模型，可以共同学习自然语言( NL )句子和(半)结构化表格的表示，如果把TABBERT用在神经语义解析器中当作特征表示层，就可以得出很好的结果（因为它既学习了句子的特征又能同时学习表格的特征）。















13、CFGNN: Cross Flow Graph Neural Networks for Question Answering on Complex Tables
可以在多张表中，使用GNN进行推理，完成TB-QA任务。在GNN中不仅仅考虑了父子节点间的关系，更多的注意到了子节点的兄弟节点之间的潜在的相互关系，来为表格QA提供服务。在语义解析方面还是用了sequence-to-sequence的架构，在原来encoder-decoder的基础上，把encoder复杂化，分成层次编码层和推理层。decoder作为answer layer。在层次编码层使用多层RNN来完成encoding的从“粗到细”的变化。

但是这个东西吧，把一个问题和多张表格相连接送入bert的话，这个预训练的sentence的长度就很长了。




16、T-RAG: End-to-End Table Question Answering via Retrieval-Augmented
这篇文章居然只有四页（算上摘要，算上Related Work——这个居然占了两页？？？）。作者是http://rpi.edu的和http://ibm.com联合组成。

大多数现有的端到端表格问题回答（Table QA）模型包括一个两阶段的框架，其中一个检索器（retriever ）从语料库中选择相关的表格候选者，另一个读取器（reader ）从表格中找到正确的答案。 transformer-based的可以提升reader model的正确率，但是 retriever的错误率*reader的错误率还是很大。本文利用一个统一的管道自动搜索表格语料库，从表格单元中直接找到正确的答案（utilizes a unifified pipeline to automatically search through a table corpus to directly locate the correct answer from table cel）。把两个步骤省略为一个步骤，少了两者错误的乘机，整体的错误率自然就下来了。

然后还发现在表格检索方面的效果好像也还行。（开局一张表，后面全靠说）



1、OPEN QUESTION ANSWERING OVER TABLE AND TEXT
Motivation：

1.Most open QA systems have considered only retrieving information from unstructured text.

2.Prior open question answering systems focused only on retrieving and reading free-form passagesor documents.

3.However, a significant amount of real-world information is stored in other forms,such as semi-structured web tables due to its compact representation to aggregate related information. 之前的工作都是从文本段落之中进行QA的，但是现实生活中很多重要的数据都是存储在结构化的表格中，我们现在要把表格和段落结合起来搞。

Contributions:

1.consider for the first time open QA over both tabular and textual data and present a new large-scale dataset Open Table-and-Text Question Answering (OTT-QA) to evaluate performance on this task.提出了OTT-QA。

2.propose two novel techniques to address the challenge of retrieving and aggregating evidence for OTT-QA

(1)use “early fusion” to group multiple highly relevant tabular and textual units into a fused block, which provides more context for the retriever to search for.

(2)use a cross-block reader to model the cross-dependency between multiple retrieved evidence with global-local sparse attention.

we manage the increase the model’s effectiveness and effificiency by a large margin

Dataset：

we construct a new dataset, Open Table-and-Text Question Answering(OTT-QA). OTT-QA is built on the HybridQA dataset。

【和HybridQA一样，OTTQA的问题是多跳（multi-hop）问题，需要从表格和文本中聚合信息来回答。然而，与HybridQA不同的是，OTT-QA要求系统检索相关的表格和文本。与此相反，在HybridQA中，每个问题所需的真相表和文本段落都是给定的。为了产生OTT-QA的问题，我们首先重新注释了来自HybridQA的问题，以 "去语境化"也就是说，我们使问题适合于开放域的设置，这样就可以仅从问题中确定唯一的答案，而不需要从所提供的文本和表格中获得语境。然后我们增加新的问题，以消除潜在的偏见。】





Code：

https://github.com/wenhuchen/OTT-QA

2、HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data
Motivation：

1.Existing question answering datasets focus on dealing with homogeneous information, based either only on text or KB/Table information alone.

Contributions:

1、 we present HybridQA, a new large-scale question-answering dataset that requires reasoning on heterogeneous information.




advantages and disadvantages:

它是第一个混合数据集，同时包含段落内容和表格数据，并能够在他们之间相互推理。

缺点就是：表格和段落之间有超链接，使得表格和段落之间能够实现联系和推理。让异质信息之间可以互相推理。

超链接是需要人工去标注的。问题，答案对也需要人工去标注。

Dataset：

他自己

Code:

https://github.com/wenhuchen/HybridQA

3、TAT-QA： A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance
Motivation：

1.Hybrid data combining both tabular and textual content (e.g., financial reports) are quite pervasive in the real world. However, Question Answering (QA) over such hybrid data is largely neglected in existing research.

Contributions：

1.extract samples from real financial reports to build a new large-scale QA dataset containing both Tabular And Textual data, named TAT-QA，where numerical reasoning is usually required to infer the answer。

2.propose a novel QA model termed TAGOP, which is capable of reasoning over both tables and text.

Dataset：

we propose a new dataset, named TAT-QA (Tabular And Textual dataset for Question Answering)over such hybrid data.

The hybrid contexts in TAT-QA are extracted from real-world financial reports.



advantages and disadvantages:

这是一个新的来自真实世界的财经金融数据集。表格和段落之间联系不是通过超链接来完成的，它可以针对表格中的数据和相关的段落之间建模，来寻找两者之间的联系。然后在数字推理的过程中能够实现数字运算效果。

思考一下子：F1值只有58%，还有提升的空间。

我丢，我登上网站一看，F1 score都干到76.5了，以后提升的空间不是很大。但是这些高score的模型，我到目前为止还没有发现他们的论文。


Code：

https://github.com/NExTplusplus/tat-qa





4、Reasoning over Hybrid Chain for Table-and-Text Open Domain QA
Motivation：

1.Hybrid data combining both tabular and textual content (e.g., financial reports) are quite pervasive in the real world. However, Question Answering (QA) over such hybrid data is largely neglected in existing research.



Contributions：

1.propose a ChAin-centric Reasoning and Pre-training framework (CARP).

2.propose a novel chain-centric pretraining method, to enhance the pre-trained model in identifying the cross-modality reasoning process and alleviating the data sparsity problem.

Dataset：

evaluate our system on OTT-QA,a large-scale table-and-text open-domain question answering benchmark

提出了一个ChAincentric的推理和预训练框架(CARP)

引入了图结构：表格和文本的中间推理过程CARP首先制定了一个异质图，其结点是相关表格和段落中的信息块，以表示停留在混合知识中的互动。利用图结构来找一个最合理的推理链。





一种新的以链为中心的预训练方法，以加强预训练模型在分辨交叉模态推理过程、缓解数据稀疏问题的能力

增强了推理效果，通过增加了候选推理链的数量的方式来实现好的效果，但是需要构建一个很大的语料库。


Code:

None



5、Parameter-Effificient Abstractive Question Answering over Tables or Text
Motivation：

1.memory intensive pre-trained language models are adapted to downstream tasks such as QA by fine-tuning the model on QA data in a specific modality like unstructured text or structured tables.To avoid training such memory-hungry models.

说白了就是为了减少模型的参数。（通过降低一部分非必要的或者是影响不大的参数，来使得模型更容易被训练）

Contributions：

Our main contributions are summarized as:

(1) We perform parameter-effificient abstractive question answering over multi-modal context using only additional 1.5% of trainable parameters for each modality. Our adapter-tuned model outperforms existing work by a large margin on tabular QA datasets and achieves comparable performance on a textual QA dataset.

(2) We study tabular QA as a new modality that introduces massive input domain shift to pretrained language models. We propose a 2-step transformation of hierarchical tables to sequences, which produces a uniform representation to be used by a single, shared pre-trained language model and modality-specific adapter layers. To the best of our knowledge, this is the first work that explores tabular QA question answering in a parameter-efficient manner.

(3) We ablate adapter layers in both encoder and decoder modules to study their impact and show that beginning layers from both encoder and decoder can be eliminated without significant drop in performance. We also demonstrate that last encoder adapter layers are indispensable and have greater contribution than decoder layers at the same level.



Dataset：

Textual Question Answering
on the NarrativeQA dataset。

Tabular Question Answering
Tablesum (Zhang et al., 2020) and

FeTaQA (Nan et al., 2021)。



advantages and disadvantages:

first to study parameter-efficient transfer learning over tables and text for abstractive question answering using adapters.（对参数的effiicient进行探索。）

demonstrate that parameter efficient adapter-tuning outperforms fine-tuning on out-of-domain tabular data and achieves comparable results on in-domain textual data.（针对表格数据集和文本数据集分别进行了独立的研究，证明了在表格数据集上adapter-tuning的方式要比fine-tuning的方式表现好的多，但是在文本数据集上两者的效果差不多）

adapter layers from the end of the encoder is indispensable to encoding modality specifific information than decoder adapter layers at the same level.（adapter layers from the end of the encoder比同级别的decoder adapter layers at the same level要重要的多。）

缺点就是：models do not explicitly reason and aggregate over the table cells. This might lead to flfluent but factually incorrect answers on challenging Tablesum dataset.（没有在表格数据上进行显示的推理和聚集，这会导致在表格数据集上的生成的答案会有比较明显的错误）





Code:

https://github.com/kolk/Pea-QA





6、TAPAS: Weakly Supervised Table Parsing via Pre-training
Motivation：

To alleviate the collection cost of full logical forms, one popular approach focuses on weak supervision consisting of denotations instead of logical forms.However, training semantic parsers from weak supervision poses difficulties, and in addition, the generated logical forms are only used as an intermediate step prior to retrieving the denotation.

大家最近喜欢用弱监督的方式去减轻数据的收集成本，但是这种方法训练语义解析器的时候有点困难，而且中间有一个产生逻辑形式的步骤，是没有必要的，所以本文给它搞掉了，不用生成这个东西了。

Contributions：

1、 we present TAPAS, an approach to question answering over tables without generating logical forms.

本文对三个不同的语义解析数据集进行了实验，发现TAPAS的表现优于或可与在SQA上将最先进的准确性从55.1提高到67.2语义解析模型相媲美，在WIKISQL和WIKITQ上的表现则与最先进的技术表现相当，但模型架构更简单。我们还发现，从WIK-ISQL到WIKITQ的转移学习，在我们的设置中是无用的，它产生了48.7的准确率，比最先进的高4.2分

Dataset：

1、WIKITQ：This dataset consists of complex questions on Wikipedia tables.

2、SQA：把WIKITQ中的一个高度组合问题分解成了几个子集，其中每个分解后的问题都可以由一个或多个表格单元来回答。

3、WIKISQL：It was constructed by asking crowd workers to paraphrase a template-based question in natural language.

advantages and disadvantages:

实验还介绍了一些其他的预训练方法，简单论证了一下是否有用。

1、无效的数据增强的方式：We generated synthetic pairs of questions and denotations over real tables via a grammar, and augmented these to the end tasks training data.
2、预训练的目标问题：convert the extracted text-table pairs to pretraining examples。 use a masked language model pre-training objective。 We also experimented with adding a second objective of predicting whether the table belongs to the text or is a random table。

不足之处在于，实验中所用的表格形式比较简单（处理的是单个的表格），实验用的表格是经过专门的转换处理的（only contain horizontal tables with a header row with column names， transpose Infoboxes into a table with a single header and a single data row），具体形式见下图。但是这样处理有利于提升最后模型的表现。


Code:

https://github.com/google-research/tapas





8、TABERT: Pretraining for Joint Understanding of Textual and Tabular Data【ACL，2020】






Motivation：










Contributions：




Dataset：


Advantages and Disadvantages:


Code:

































13、CFGNN: Cross Flow Graph Neural Networks for Question Answering on Complex Tables
Motivation：
大多数传统的GNNs通常使用求和作为邻域聚合的功能。虽然它可以反映父节点和子节点之间的关系，但兄弟姐妹（孩子-孩子）节点的关系-船被忽略了。特别是父-子和子-子关系之间的权衡很少受到关注。此外，当节点有较少的度，但兄弟姐妹节点较多时，通过以前的GNN只能得到很少的信息。例如，数据库模式图中的列的节点可能无法获得足够的更新，因为它通常只有一个父节点，没有子节点。而在自然语言处理（NLP）的任务中，这些关系都是重要的上下文信息。为了理解像数据库模式一样的图的复杂结构，人类除了考虑邻接关系和隶属关系外，通常还会考虑间接关系，然后反复思考和推理。

Contributions：
we propose novel Cross Flow Graph Neural Networks (CFGNN) for question answering on complex tables inspired by the cognitive process of human beings。除了传统的父子关系，本文主要考虑了图中兄弟节点的关系。它可以帮助那些度较小但兄弟姐妹较多的节点获得更多的信息流。而在这个任务中，我们的方法还可以有效地对表内和表间的列节点进行建模，并将其视为语境（上下文）。 use two different recurrent neural networks (RNNs) with attention mechanism to integrate these cross flows.One of RNNs is used as the aggregation function among children nodes of the same parent node rather than parent-child nodes. The other is used for aggregating flows and reasoning between layers. And attention mechanism is also used for supplement of the relationship between parent and child nodes. 用两个RNN去整合和处理交叉流。
摘要里还提了一嘴hierarchical encoding layer to obtain contextualized representation in tables。使用sequence-to-sequence的结构来进行神经语义分析，但编码器部分比以前的模型更复杂。因此，将解码器视为答案层，并将编码器拆分为层次化的编码层和推理层。
层次编码层也是bert，将问题与表的所有的表和列的名称逐一串联。然后送进去，但是这样的话，你如果一个问题涉及到了多张表的话，或者答案所在表的规模很大，你这个sentence就很长了，很可能就超过了bert预训练的senten最大长度。然后作者说由粗到细的分层编码，即bert预训练刚出来的encoding被第一层RNN细化了还是很粗糙的，还需要用另一个RNN再次encoding。
GNN用在了推理层。GNN推理完了，又接了一层RNN。


Dataset：
Spider (Yu et al. 2018c)：a large-scale, complex and cross-domain text-to-SQL dataset annotated by human。作者说这是唯一的一个在多表格上进行TB-QA的数据集，里面的模式能建模成图(the schema in which can be modeled as graph)。具体的样子长这样：


The source of data comes from six datasets. There are 11,840 questions, 6,445 unique complex SQL queries, and 206 databases with multiple tables in this dataset.

数据集里还涉及到了两种评估度量方法：component matching and exact matching. component matching compares the different parts of SQL queries, such as SELECT, WHERE and so on.

advantages and disadvantages:


Code:

None







16、T-RAG: End-to-End Table Question Answering via Retrieval-Augmented
Motivation：
以前的表格QA都是检索器和阅读器的组合，两者都会发生错误，两者的错误率相乘导致整体的错误率更大了，本文想把两者结合在一起，从而降低错误率。

Contributions：
propose the first end-to-end Table QA pipeline, leveraging DPR along with the Seq2Seq component of RAG.



Dataset：
two open-domain benchmarks:

NQ-TABLES：is {q, T, a} format, where q, T, and a denote question, ground truth table, and answer


E2E_WTQ：

Advantages and Disadvantages:
我滴乖乖，摘要半页，Introduction一页，Related Work 1.7页， Conclusion and Future Work 1/4页，Experiments 部分满打满算，算一页，其中还插了三张表。里面共五段话，第一段介绍选用的benchmarks，第三段介绍了自己选用的评估方法。第四段介绍介绍了自己是怎么验证评估的，最后一段分析了一下子实验出来的数值。



本文下一步还计划在特定领域的数据集上验证T-RAG，如AIT-QA和TATQA（Katsis等人，2021年；Zhu等人，2021年），并将模型扩展到解决多模式的QA问题，语料库同时包含表格和段落，如OTT-QA和HybridQA的基准（Chen等人，2020a,b）。

Code:
无。
