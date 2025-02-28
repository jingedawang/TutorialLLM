# 从零入门大模型

[![GitBook](https://img.shields.io/badge/GitBook-从零入门大模型-blue)](https://jingedawang.gitbook.io/tutorialllm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jingedawang/TutorialLLM/blob/main/TutorialLLM.ipynb)
[![Python](https://github.com/jingedawang/TutorialLLM/actions/workflows/python.yml/badge.svg)](https://github.com/jingedawang/TutorialLLM/actions/workflows/python.yml)

[![中文文档](https://img.shields.io/badge/中文-white)](README.md) [![中文文档](https://img.shields.io/badge/English-gray)](README-en.md)

本仓库包含教材《从零入门大模型》的代码实现和原始文本。

该教程引导你从零开始逐步训练一个大语言模型（LLM）。出于教学目的，我们使用了一个小数据集和一个小模型，但保留了大模型训练的完整流程和基本架构。

你可以先阅读教材的文本，再运行代码了解细节。代码中包含了详细的注释。

## 教材

点击 [![GitBook](https://img.shields.io/badge/GitBook-从零入门大模型-blue)](https://jingedawang.gitbook.io/tutorialllm) 阅读教材。教材的原始文档在本仓库的`book`目录中，由GitBook托管发布。欢迎通过Issue给我们提建议。

该教材同时与知乎官方合作作为付费专栏提供给知乎知识会员。你可以在知乎搜索“王金戈”，在我的作品中找到 **《会说话的AI：从零入门大模型》**。

## 代码

### 使用Google Colab运行

如果不想在自己电脑上搭建开发环境，建议点击 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jingedawang/TutorialLLM/blob/main/TutorialLLM.ipynb) ，在Google Colab中运行代码。

### 本地运行

请下载并安装[Python 3.12](https://www.python.org/downloads/)，然后执行如下命令：

```bash
git clone https://github.com/jingedawang/TutorialLLM.git
cd TutorialLLM
pip install -r requirements.txt
python run.py
```

程序的运行分为5个阶段，你将会依次看到每个阶段的输出。

#### 阶段1：准备数据。

对训练数据进行预处理，生成符合模型输入格式的数据集。

```
--------------------------------------------------
STAGE 1: PREPARE THE DATA
The whole pretrain data is a long text with all poems concatenated together. Here are the first 100 characters:
東歸留辭沈侍郎
一第久乖期，深心已自疑。
滄江歸恨遠，紫閣別愁遲。
稽古成何事，龍鍾負已知。
依門非近日，不慮舊恩移。

題玉芝趙尊師院
曉起磬房前，真經誦百篇。
漱流星入齒，照鏡石差肩。
靜閉街西觀
The instruction finetune data is a list of formatted texts. Here is the first item:
<INS>請用以下題目寫一首詩<INP>端午三殿侍宴應制探得魚字<RES>小暑夏弦應，徽音商管初。
願齎長命縷，來續大恩餘。
三殿褰珠箔，羣官上玉除。
助陽嘗麥彘，順節進龜魚。
甘露垂天酒，芝花捧御書。
合丹同蝘蜓，灰骨共蟾蜍。
今日傷蛇意，銜珠遂闕如。
The alignment data is a list of positive-negative pairs. Here is the first pair:
('<INS>請用以下題目寫一首詩<INP>宿壽安甘棠館 二<RES>山空蕙氣香，乳管折雲房。\n願值壺中客，親傳肘後方。\n三更禮星斗，寸匕服丹霜。\n默坐樹陰下，仙經橫石床。', '<INS>請用以下題目寫一首詩<INP>遣懷<RES>落魄江南載酒行，楚腰腸斷掌中輕。\n十年一覺揚州夢，贏得青樓薄倖名。')
Dataset length: 4052248, vocabulary size: 8548
Check a batch of pretrain data:
(tensor([[5605, 7964, 1180,  ..., 5851,  544, 1310],
        [4875, 1984, 8347,  ..., 2337, 8347, 5798],
        [6047, 7846, 7600,  ..., 3514,   20,    2],
        ...,
        [2784,  618, 7569,  ...,    2, 5151, 5102],
        [6425, 4875, 3297,  ...,  183, 8347, 2827],
        [ 178, 6864, 3875,  ..., 2745, 1855, 2995]], device='cuda:0'), tensor([[7964, 1180,   20,  ...,  544, 1310, 3523],
        [1984, 8347,  261,  ..., 8347, 5798, 1978],
        [7846, 7600, 7611,  ...,   20,    2, 3164],
        ...,
        [ 618, 7569,    2,  ..., 5151, 5102, 3587],
        [4875, 3297, 5384,  ..., 8347, 2827, 2149],
        [6864, 3875, 1534,  ..., 1855, 2995,   20]], device='cuda:0'))
Check a batch of finetune data:
(tensor([[5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0],
        ...,
        [5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0]], device='cuda:0'), tensor([[   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100],
        ...,
        [   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100]], device='cuda:0'))
Check a batch of alignment data:
(tensor([[5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0],
        ...,
        [5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0]], device='cuda:0'), tensor([[   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100],
        ...,
        [   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100]], device='cuda:0'), tensor([[5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0],
        ...,
        [5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0],
        [5, 8, 9,  ..., 0, 0, 0]], device='cuda:0'), tensor([[   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100],
        ...,
        [   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100],
        [   8,    9,   12,  ..., -100, -100, -100]], device='cuda:0'))
```
以上显示了预训练、微调和对齐三个阶段所需的数据格式。后续的训练阶段将会使用这些数据训练模型。

#### 阶段2：配置参数

```
--------------------------------------------------
STAGE 2: TRAINING CONFIGURATION
Our model has 1.318372 M parameters
```

#### 阶段3：预训练

通过让模型阅读大量文本，培养模型的基础语言能力。该过程称为预训练（pretrain）。

在本案例中，我们让模型阅读大量的唐诗，从而培养其写诗的语感。

```
--------------------------------------------------
STAGE 3: PRETRAIN
In this stage, the model will learn the basic knowledge of how to write poems.

Step 0, train loss 0.0000, evaluate loss 9.2029
Generate first 100 characters of poems starting with 春夜喜雨:
春夜喜雨墻鵷函媵分塏潏鍳母付菱莽换驃慚憮儲躕袗鯆溫沲芳罔窻倏菂弓匌莿尚茸茇培嵍鵝掣卵耽敧青魄叚𪆟瞑唱鄢懅齧泉綘躅鷂㦬烻超玃鯽敝俱惏廏鐐處翻矼奭媼悟出撾孃詠可碙媌鶂旐垤嵼鶤柘輩噇篲詮擲憇純絃蜘儔緬簇澎雨搰褚磐歙
Step 100, train loss 6.3026, evaluate loss 6.2829
Generate first 100 characters of poems starting with 春夜喜雨:
春夜喜雨剡冰 閒路。耕銅崔出玫胖，上惺尋。


樹天歸贈，。


強寥下戎景人。
火亂潔頴明戍流漸道一室勢。
摶磻四春相似
，年陽下了行爭癥追赤且木少塞是門相。
摘言迎來夢賦示營似吾首。
暮墜鬢纔征蕉春崖五牀

<Omit many iterations here...>

Step 5000, train loss 4.7134, evaluate loss 4.9221
Generate first 100 characters of poems starting with 春夜喜雨:
春夜喜雨發早，唯應尋與塵遠山。
尋敬況皆迷情重，見買腰量漢寒宵。

贈家石橋危師兼寄一題因作四使者池應制
梧桐葉朝廷塵，一曲鐘明冠黑。
西上國邀辭外，主詞高須禁圖違。
因循勢祿應承急，不知育符惟開。
舊玄元侍

<Omit many iterations here...>

Step 10000, train loss 4.3883, evaluate loss 4.6798
Generate first 100 characters of poems starting with 春夜喜雨:
春夜喜雨人掃轉，煙穗半開笑倚簷。
美酒飄生弱氈粉，記詩繫舊光桑榆。
鶯別夜悲行滿袖，簟前年態踏爲顏。
雲間葉絡長簾鉢，舊漏孤燈照水香。
尊容暮事黃雲伴，使我清光小舊狂。

重夜涼
自謂長安不速時，當來不得鷓鴣
```

以上输出展示了预训练模型的变化过程。
+ 最开始，生成的文字完全随机，没有任何含义。说明模型还什么都不懂。
+ 经过5000步训练，模型开始学会正确地使用词组和短语。
+ 经过10000步训练，模型开始学会规范句子结构，整体更为流畅。

由于模型大小和算力的原因，最终生成的诗文可能并不完美，但足以让我们体会到模型的学习过程。

#### 阶段4：微调

有了基础语言能力后，使用指令微调（instruction fine-tune）技术可以让模型根据指令生成更加符合要求的文本。

在之前的预训练阶段，并不存在一首诗的概念。因为训练数据是所有诗文的拼接，模型只学会不停地生成符合诗句格式的文本，而不会在生成完一首诗后停下来。

在微调阶段，通过设置训练数据的结构，可以让模型学会按照我们的指令写诗。在这个例子中，我们要求模型根据用户提供的题目生成一首完整的诗。

```
--------------------------------------------------
STAGE 4: FINETUNE
In this stage, the model will learn to generate poems based on the instruction.
In our case, we ask the model to generate a poem with a given title.

Epoch 0, step 0, train loss 0.0000, evaluate loss 8.5504
Generate a complete poem for title 春夜喜雨:
文請，曾說春來淚故花楊柳枝。

山中峰寺仙二首 二
高齋吟商賈居仙，欲嘯花開開却閉絲。
硯通四也應不見，却向宮江盡北山。
一字多堪杖不顧，紫芝多不見青山。
定開照鏡匳梅雨，上倚升階遶樹陰。
來此醉君

<Omit many iterations here...>

Epoch 4, step 800, train loss 3.2089, evaluate loss 3.3451
Generate a complete poem for title 春夜喜雨:
山村春色引，雨水曉光斜。
更問月中鏡，空啼千萬傳。
隱隱閑臥竹，神盤暗自風。
殘春望江島，不見磬聲王。
```

可以看到，4个周期的训练之后，模型的输出已经是一首完整的诗，而且与规定的题目契合。

#### 阶段5：对齐

对齐（alignment）是大模型之所以强大的重要原因之一，也是ChatGPT能够成功的关键因素。在实际应用场景中，有必要让大模型的输出符合人类价值观，避免危险、冒犯、歧视等话题产生。

在本案例中，由于作者本人更喜爱五言诗，我们尝试让模型与这一偏好对齐。

```
--------------------------------------------------
STAGE 5: ALIGN PREFERENCE
In this stage, the model will learn to generate poems that we prefer.
In our case, we prefer five-words poems than other poems.

<Omit many iterations here...>

Epoch 2, step 0, train loss 0.0000, evaluate loss 0.6818, train reward margin 0.0000, evaluate reward margin 0.2581
Generate a complete poem for title 春夜喜雨:
Aligned model:
翠紗復掩涴，色轉目猶輕。
素巷風無事，清池月不寒。
雲收爐尚望，日臥菊初明。
翠貼霏逾老，香藏遠獨紅。
Reference model:
黔城風光雪老遷，不將岸上林間樹。
莫道鸚鵡詩想熟，此催黃柳老風摧。
```

可以看到，在训练过程中，我们对比了原始模型与训练后模型的差异。训练前的参考模型（reference model）生成了一首七言诗，而训练后的模型（aligned model）生成了一首五言诗。模型成功对齐了这一偏好。

以上是训练的完整流程。虽然使用的案例非常简单，但从中足以了解到大模型训练各个阶段的基本原理。建议读者亲自尝试运行代码，感受模型学习的过程。

## 关于作者

我很喜欢用通俗的语言分享艰深的知识。在我看来，任何知识都应该能够被大众理解，不应成为少数人的专属。如果一篇教材让人读不懂，说明作者没能串联好知识传授的顺序。毕竟，教学是一门复杂的学问。希望这部作品能够让更多人了解到人工智能的魅力。

欢迎关注我的[知乎](https://www.zhihu.com/people/wang-jin-ge-67)、[B站](https://space.bilibili.com/69217382)、[GitHub](https://github.com/jingedawang)和[LinkedIn](https://www.linkedin.com/in/wangjinge/)。