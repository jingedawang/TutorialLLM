# Tutorial LLM

[![GitBook](https://img.shields.io/badge/GitBook-从零入门大模型-blue)](https://jingedawang.gitbook.io/tutorialllm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jingedawang/TutorialLLM/blob/main/TutorialLLM.ipynb)
[![Python](https://github.com/jingedawang/TutorialLLM/actions/workflows/python.yml/badge.svg)](https://github.com/jingedawang/TutorialLLM/actions/workflows/python.yml)

[![中文文档](https://img.shields.io/badge/中文-gray)](README.md) [![中文文档](https://img.shields.io/badge/English-white)](README-en.md)

This tutorial will guide you through the process of training a large language model (LLM) step
by step. For educational purposes, we choosed a small dataset and a small model, but the basic principles we want to convey is the same with larger models.

We provide a tutorial book and a code implementation. You can **read the book** to understand the theory and then **run the code** to see the practice. The code is well-documented and easy to understand.

## Book

Click [![GitBook](https://img.shields.io/badge/GitBook-从零入门大模型-blue)](https://jingedawang.gitbook.io/tutorialllm) to read the book.

The book is hosted on GitBook. But the content is located in the `book` folder in this repository in markdown format. Welcome any feedback or contribution through issues or pull requests.

This book is also published as a paid column for 知乎知识会员. You can search 王金戈 on 知乎 app and find the book in my homepage.

## Code

The overall architecture is pretty simple. The `run.py` is an entrance, which calls 4 modules accordingly. For more details, you can refer to chapter 10 in the book.
![code architecture](book/.gitbook/assets/code_architecture.png)
We provide 2 running modes as below.

### Run with Google Colab

If you don't have any hardware or don't want to set up the environment yourself, try using Google Colab.
Click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jingedawang/TutorialLLM/blob/main/TutorialLLM.ipynb) to open the notebook in Google Colab.

### Run locally

Please download and install [Python 3.12](https://www.python.org/downloads/), then run the following commands:

```bash
git clone https://github.com/jingedawang/TutorialLLM.git
cd TutorialLLM
pip install -r requirements.txt
python run.py
```

You will see the training process in 5 stages.

**In stage 1**, we prepare the data used for training. You will see output like this:

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

It shows the pretrain, finetune, alignment data in text and tensor format to give you a sense of the data. The training procedure will use these data to train the model.

**In stage 2**, we set up some training configurations. You will see output like this:

```
--------------------------------------------------
STAGE 2: TRAINING CONFIGURATION
Our model has 1.318372 M parameters
```

**In stage 3**, we pretrain the model. You will see output like this:

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

The changes of the generated poems is interesting through the training. At beginning, you can see the generated poem is completely random in the step 0, indicating the model has not learned anything. As the training goes on, the generated poem becomes more and more like a real poem.
It learns some words and phrases when it comes to step 5000. It learns the structure and more fluent expression when it comes to step 10000. Due to the limitation of the small dataset and small model, the generated poem may not looks very good, but you can feel the learning process in it.

**In stage 4**, we finetune the model to let it write a poem for a given title. You will see output like this:

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

In the previous pretrain stage, the model doesn't have a concept of one poem. It just learns to generate texts that resemble poems endlessly. In the finetune stage, we ask the model to generate a poem with a given title. You can see the model learns to generate a poem in a correct format after finetuning, and the topic of the generated poem is gradually closer to the given title.

**In stage 5**, we align the model to a specific preference of myself. I like 5-word poems than others. By providing the positive-negative pairs, the model will learn to generate poems more like the positive ones, which are 5-word poems in this case. You will see output like this:

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

You can see the aligned model generates a 5-word poem, while the reference model generates a 7-word poem. My preference is conveyed to the model through the alignment training.

Though this is a toy example, you should be able to see the basic principles of training a large language model. By looking into the code and running it on your local machine, you can get a deeper understanding of how large language models work.

## About

I'm a researcher in the field of AI. I like to share my knowledge with people. You can find me on [Zhihu](https://www.zhihu.com/people/wang-jin-ge-67), [LinkedIn](https://www.linkedin.com/in/wangjinge/), and [GitHub](https://github.com/jingedawang). Feel free to leave an issue or contact me if you have any questions.
