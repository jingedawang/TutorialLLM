# 第1节 如何让AI学会说话

{% hint style="success" %}
**本节导读**

让我们先坐下来，不必动笔，也不必编程，只用大脑思考，想一想接下来要做的事。我们将从智能出发，思考语言的意义，寻找一条让AI学会说话的道路。读完本节，你将会了解：

* 语言与智能的关系
* 如何拆解说话这件事
* 什么是神经网络
* 让AI学会说话的方式
{% endhint %}

## 从智能到语言

无论是否身处AI（Artificial Intelligence，人工智能）行业，AI这个词一定早已频繁出现在你我的生活。敏感的话，你或许会发现在过去的10年间，人们的生活在不断被AI改善。智能汽车，智能手机，人脸识别，抖音和小红书的个性化推荐，以及各种企业级服务，不一而足。

当然，你或许不了解到底怎样算是AI。毕竟，商家为了蹭热点，硬说自家产品是AI的也不在少数。事实上，AI与否本来就没有判定的标准，我们只能说，越接近人类这样的高级智能，越算是AI。

而人相比于AI到底智慧在何处呢？简单来说，人可以思考各类问题，并融会贯通、举一反三。这种思考和学习的能力是AI所不具备的。AI往往只能做好某一件事，比如人脸识别的AI只能识别人脸，下围棋的AI只能下围棋。而人可以用很短的时间学会做某事，从而掌握多种技能。

至于人为何有这般能力，科学家也不知道。但有一个可能的原因，就是人掌握了语言。别以为语言只是人所有技能中的一小部分，事实上，它可能是最重要的部分。上学的时候，老师就说过人和其它动物的最大区别是人有语言。当时听完颇为不解，不过现在我倒是明白了其中的道理。

语言可以构建对世界上任何事物的描述，无论具体还是抽象。也就是说，宇宙中的所有事情，所有知识，都可以用语言表达出来。学习语言相当于打开了从内心到世界的通道，这是一种通用的接口，连接想象与现实。所以，学会说话，是迈向智能的一大步。

AI研究者或许早已明白了这一道理。从上世纪开始，自然语言处理（NLP，Natural Language Processing）就是人工智能的重要分支。时至今日，NLP发展日新月异，虽然大部分研究以实用性的目的出发，但那些令人惊艳的效果反过来说明了语言的重要性。ChatGPT的爆火也再次证明，学会语言，就能学会一切。

现在，让我们站在一个NLP研究者的角度，思考如何设计出一个懂得语言，会说话的AI。

## 借鉴人的说话方式

最直观的思路当然是借鉴一下人是怎么说话或写作的。

首先，我们往往会在头脑中形成一个模糊的概念，也许是几个关键词，锚定了接下来谈话的主题。接下来，我们会构思开头，一句一句地输出，最终形成一段完整的话。

对于写作，我们可以回过头修改之前写下的文字。但说话则是一口唾沫一个钉，说出的字无法收回。其中蕴含了一个重要的原则，语言是按时间顺序输出的，没有人会倒着说出一段话，他自己做不到，听众也绝无可能听懂。

于是，当我们打算让AI说话时，自然也应当让它按顺序输出文字，就像正常人说话一样。不过，聪明的人可能会想，AI是不是可以提高效率，一次输出一句话甚至一段话呢？理论上或许可以，但历史经验证明，人们还是选择让AI一次输出一个字。个中原因不难理解。句子长短不一，显然不如单个字更容易操作和管理。而且，一次输出一个字相当于给了AI更细致的思索空间，逐字推敲的质量肯定比囫囵吞枣更好。

与之俱来的另一个问题是，如何让AI能听懂别人的话。这一点很重要，我们要设计的是一个能与之交流的AI，而不是自言自语的机器。同样，人倾听的过程其实也是一个字一个字听进去的。只不过，我们处理信息的速度非常快，能够迅速从断续的话语中提取关键词，构建出大致的含义。

那么，会说话的AI应该是这么一种程序，它能够听取别人的输入，当输入结束后，它立即开始输出答复。就像两个人对话一样。如果你用过ChatGPT，很容易明白这是什么意思。

<figure><img src=".gitbook/assets/chat between human and ai.png" alt=""><figcaption><p>图1 Alice和名叫Kung的AI之间的对话</p></figcaption></figure>

带着这个目标，我们接下来需要知道如何才能实现这样的程序。无论普通人还是程序员，对这个问题可能都一时摸不着头脑。因为我们将要实现的，不是简单的问答机器，而是能适用任何场景，具备超强理解能力和表达能力的，像人一般的AI。这种AI不会基于有限的规则，因此传统的软件开发技术在这里都黯然失色。这种蕴含智能的程序只有一种现存的解法，那就是神经网络。

## 神经网络

“神经网络”这个词在生物学领域和人工智能领域分别代表两种不同但类似的东西。生物课上曾经学过，人脑就是一个由神经元构成的网络，神经元之间由树突和轴突连接，电信号和化学信号在网络中传递。

既然人脑的神经网络给人类带来了如此强大的智慧，科学家们自然希望AI可以借鉴这种生物结构。

于是人工神经网络出现了。与生物神经元不同的是，人工神经元只是简单运算规则的堆积，远没有生物神经网络那么复杂。但幸运的是，人工神经网络真的有用，它可以完成一些过去人们无法通过编程做到的任务。

<figure><img src=".gitbook/assets/biological neuron vs artifical neural network.png" alt=""><figcaption><p>图2 左：生物神经元，右：人工神经网络</p></figcaption></figure>

一个典型的案例是手写体数字识别。以前，人们无论如何都找不到一个好方法可以自动判断出手写体的数字属于0\~9的哪一个，因为每个人的字体各异，难以找到放之四海而皆准的规则。

<figure><img src=".gitbook/assets/mnist samples.png" alt="" width="375"><figcaption><p>图3 MNIST数据集中的手写体数字</p></figcaption></figure>

然而，人工神经网络的出现改变了这一局面。人们发现，原来根本不需要设计规则，只需要提供大量示例，每个示例包含一个手写体数字（一张图片）和其对应的真实数字（0\~9）。让神经网络不断地阅读这些数据，它就能自动发现规则，并内化于自己的神经元中。

这里有两个非常神奇但至关重要的地方。

第一，神经网络虽然由简单的神经元组成，但理论上它可以逼近任何复杂的函数映射。单个神经元做的事情非常简单，它先对输入做一次四则运算，相当于求输入的线性组合，然后再对结果做一次非线性运算，比如指数、次方等等。看起来虽然简单，但简单的东西聚集在一起往往会变得不简单。就像蚁群的群体智慧一样，单个蚂蚁的智力非常低，只能做最简单的任务，而一群蚂蚁合在一起却能表现出复杂的行为，仿佛蚁群拥有高级智慧。

第二，通过收集数据，我们可以定义期望的输入输出。神经网络通过学习逐步逼近数据所定义的函数映射，相当于实现了我们所需的算法。也就是说，在神经网络中，算法不是我们设计出来的，而是从数据中自动挖掘出来的。我们不必关心最后挖掘出的算法到底是什么，事实上也无从得知，因为算法隐含在神经网络参数中。我们只需要知道，有什么样的数据，神经网络模型就会学到对应的算法来拟合这批数据。

你或许想知道神经网络如何通过学习逼近目标映射。简单来说，神经网络的学习其实是一个优化过程。从随机的映射开始，根据实际输出与期望输出之间的差距，调整神经元的参数。通过反复的微调，神经网络的映射会越来越接近于我们期待的映射，实际输出也越来越接近期望输出。我们会在后续的章节亲自体会这个过程。

神经网络可能是科技史上最重要的范式转变。过去，人们总结自己的经验，打造各种各样的算法。现在，人们让出了主导权，以事实为本（即数据），让AI自己学习数据中蕴含的规律。从此，数据驱动AI的时代开启，这是智能真正的起点，奠定了往后几十年AI领域的繁荣。

{% hint style="info" %}
2024年的诺贝尔物理学奖授予AI教父Geoffrey Hinton，正是因为他对神经网络的贡献。虽然有人质疑其与物理学的关系，但时间将会证明，Hinton的成就或许远高于其他物理学家，而AI与物理的关系恐怕也没有想象中那么遥远。
{% endhint %}

言归正传，有了神经网络这个大杀器之后，如何让AI学会说话呢？

## 读遍天下文字

刚才提到，只要我们定义好期待的输入输出，神经网络就能学会其中蕴含的规律。然而回到语言这个任务上，我们期待的输入输出到底是什么呢？

回到我们最朴素的想法，让AI能够像人一样说话，那么输入输出自然就是人说的话。但人的语言太多了，有口语形式的语言、书面的语言，有小说、戏剧、闲聊、争吵，有私密的语言和公开的语言，有英语、汉语、法语、德语、西班牙语各种语言，有一个人的滔滔不绝也有两个人你问我答的对话。语言的形式数不胜数，任何一种语言形式都不能完全概括语言的魅力。

对于我们而言，希望打造一个在语言上与人类相当的AI，就必然要考虑所有的语言形式。虽然它们看起来可能差异极大，但回归文字本身，其最小元素都是一样的，无非是字或单词，以及标点符号。

{% hint style="info" %}
国际上大部分语言以单词为最小单位，汉语等语言以字为最小单位。但实际应用中，为了统一所有语言，一般采用独立于任何语言的词元（token）系统。一个词元可能对应于一个单词、半个单词、一个汉字、多个汉字等等。此处为了方便，用字或单词来解释，避免引入更多概念。
{% endhint %}

于是，我们便可以想出一个统一的输入输出映射模式。输入是任意长度的文本，输出是紧接着输入文本后面的下一个字或单词。

前文曾讨论过，让AI一个字一个字输出是最理想的方式。它不仅与人的行为一致，也更适配多样化的文本数据。试想，如果我们仅仅为了配合对话场景，输入一句话之后就让AI输出一句回答，这样的模式就无法利用其它类型的文本数据，因为你没法把一篇文章拆分成一句一句的对话。相反，一次预测一个词则完全没有问题，任何文本都必然由按顺序的一个个词构成，天然适合我们的规则。

也就是说，我们希望AI学习一种语言中蕴含的规律，这个规律可以在给定一段文本的情况下说出下一个词。实际上，我们自己也常常这样做，一边说一边想着下一个词，只不过我们通常想的很快而已。

那么，只要让AI博览天下文字，无论出处，无论语言，无论风格。当它把这些数据融会贯通之后，应该就能学会说话了。

至此，从“道”的层面，我们想明白了该如何做这件事。具体落实到行动上，自然还有许多细节需要推敲。我们从下一节开始正式完善并实现这个方案。