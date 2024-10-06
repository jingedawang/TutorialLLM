# Tutorial LLM

This tutorial will guide you through the process of training a large language model (LLM) step
by step. For educational purposes, we choosed a small dataset and a small model, but the basic principles we want to convey is the same with larger models.

There are 2 ways for you to learn:

- Run the notebook in Google Colab. With Google Colab, everyone can train an LLM from scratch with just a browser. To be done.
- Run the `run.py` script on your local machine. All code is well-documented and easy to understand.

Note that a free GPU like T4 in Google Colab is enough for the default setting. If you want to
run it with only CPU, you may need to reduce training iterations to avoid long waiting time.
But meanwhile, you cannot observe the expected effect of the training process.

## Run locally

Please install Python 3.12+ and then run the following commands:

```bash
pip install -r requirements.txt
python run.py
```

You will see the training process in the console. By examining the output poems, you can get a sense of how the model learns to generate text.

## About

I'm a researcher in the field of LLM. I like to share my knowledge with people. You can find me on [Zhihu](https://www.zhihu.com/people/wang-jin-ge-67), [LinkedIn](https://www.linkedin.com/in/wangjinge/), and [GitHub](https://github.com/jingedawang). Feel free to leave an issue or contact me if you have any questions.
