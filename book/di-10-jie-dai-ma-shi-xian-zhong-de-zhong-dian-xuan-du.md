# 第10节 代码实现中的重点（选读）

{% hint style="success" %}
**本节导读**

了解理论，只是懂了个大概。亲自实践，才算真正理解。那些理论上看起来简单的问题，实现的时候或许会遇到意想不到的麻烦。读完本节，你将会了解

* 代码的架构设计
* 数据处理模块如何实现
* 如何使用PyTorch搭建模型
* 如何训练模型

本节内容为选读，跳过本节并不会影响本书进度。
{% endhint %}

在前9节中，我们从头设计并实现了一个写诗AI。为了不让内容过于繁琐，我省略了代码实现的讲解。一方面，本书主要传达大模型的核心思想，用最短的篇幅把关键环节讲清楚。另一方面，读者未必真的需要从事大模型开发，细节对大部分人来说可能无关痛痒。

但为了满足另一部分真正想要了解实现细节的人，本节提供了一个方便的入口。其实，本书的配套代码并不复杂，自己阅读理解也并非难事。所以，我不打算逐行讲解代码，而是挑重要的部分拿出来与读者分享。

我们先来看看代码结构。打开GitHub的[TutorialLLM](https://github.com/jingedawang/TutorialLLM)仓库，可以看到如下文件和目录。

```
TutorialLLM/
├── book/
├── pre_work/
├── .gitignore
├── README.md
├── TutorialLLM.ipynb
├── data.json
├── dataset.py
├── evaluator.py
├── model.py
├── requirements.txt
├── run.py
└── trainer.py
```

非.py结尾的文件作用如下：

* `book`目录内部包含了本书MarkDown格式的草稿，由GitBook托管。
* `pre_work`目录内部包含了生成`data.json`所需的代码，这部分代码不属于本书讲解范畴。
* `.gitignore`是Git版本管理工具的一个配置文件，用于指定哪些文件类型不被Git管理。
* `README.md`是GitHub仓库的默认说明书，打开仓库主页，在文件列表下面看到的就是README的内容。
* `TutorialLLM.ipynb`是为读者提供的一个在线运行环境，读者可以根据README中的说明，在Google Colab中打开这个文件，运行整个程序。
* `data.json`中包含了2548位唐代诗人的47457首诗。数据源来自[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)仓库的全唐诗目录，并经过预处理，去除了一些格式不规整的条目。
* `requirements.txt`规定了运行代码所需的依赖库。

当然，以上都不是重点，剩下的以.py结尾的文件才是真正的代码。让我们对照下面的示意图解释一下代码的结构。

<figure><img src=".gitbook/assets/code_architecture.png" alt=""><figcaption></figcaption></figure>

