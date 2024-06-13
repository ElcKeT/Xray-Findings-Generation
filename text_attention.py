# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-03-29 16:10:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-12 09:56:12


## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.
import numpy as np
import torch
import seaborn as sns
from matplotlib import colorbar as colorbar
from matplotlib import colors as colors
from matplotlib import ticker as ticker
import sentencepiece as spm
latex_special_token = ["!@#$%^&*()"]
from matplotlib import pyplot as plt

def generate(text_list, attention_list, latex_file, color="red", rescale_value=False):
    assert len(text_list) == len(attention_list)
    if rescale_value:
        attention_list = rescale(attention_list)
    word_num = len(text_list)
    text_list = clean_word(text_list)
    with open(latex_file, "w") as f:
        f.write(
            r"""\documentclass[varwidth]{standalone}
					\special{papersize=210mm,297mm}
					\usepackage{color}
					\usepackage{tcolorbox}
					\usepackage{CJK}
					\usepackage{adjustbox}
					\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
					\begin{document}
					\begin{CJK*}{UTF8}{gbsn}"""
            + "\n"
        )

        string = (
            r"""{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{"""
            + "\n"
        )
        for idx in range(word_num):
            string += (
                "\\colorbox{%s!%s}{" % (color, attention_list[idx])
                + "\\strut "
                + text_list[idx]
                + "} "
            )
        string += "\n}}}"
        f.write(string + "\n")

        f.write(
            r"""\end{CJK*}
					\end{document}"""
        )


def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min) / (the_max - the_min) * 100
    return rescale.tolist()


def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_", "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, "\\" + latex_sensitive)
        new_word_list.append(word)
    return new_word_list


if __name__ == "__main__":
    ## This is a demo:

    """sent = negative for acute abnormality . the cardiomediastinal silhouette is normal in size and contour . no focal consolidation pneumothorax or large pleural effusion . negative for acute displaced rib fracture . 
    sent = 我 回忆 起 我 曾经 在 大学 年代 ， 我们 经常 喜欢 玩 “ Hawaii guitar ” 。 说起 Guitar ， 我 想起 了 西游记 里 的 琵琶精 。
	今年 下半年 ， 中 美 合拍 的 西游记 即将 正式 开机 ， 我 继续 扮演 美猴王 孙悟空 ， 我 会 用 美猴王 艺术 形象 努力 创造 一 个 正能量 的 形象 ， 文 体 两 开花 ， 弘扬 中华 文化 ， 希望 大家 能 多多 关注 。"""
    target_info = """xxxx-year-old female. chest pain."""
    vocab = spm.SentencePieceProcessor(model_file="./NLMCXR/nlmcxr_unigram_1000.model")
    encoded_target_info = (
        [vocab.bos_id()]
        + vocab.encode(target_info)
        + [vocab.eos_id()]
    )
    target_info = np.ones(1000, dtype=np.int64) * vocab.pad_id()
    target_info[: min(len(encoded_target_info), 1000)] = (
        encoded_target_info[: min(len(encoded_target_info), 1000)]
    )

    txt = []
    for i in range(999):
        t = vocab.decode_ids(target_info.tolist()[i])
        txt.append(t)

    word_num = len(txt)
    attention = torch.load("att_gen.pt")
    w = [2,25,36,37,38,39,42,44,45,71,77,936]
    x = []
    att = attention[0,0:13,w].tolist()
    w[0] = "EOS"
    # 使用sns画注意力热图
    xticklabels = [vocab.decode_ids(w[i]) for i in range(len(w))]

    yticklabels = [i for i in range(len(w))]
    sns.heatmap(
        att,
        xticklabels=xticklabels,
        square=True,
        yticklabels=yticklabels,
        vmin=0.0,
        vmax=1.0,
        cbar=True,
    )
    plt.show()

    att = torch.load("txt_att.pt")
    att = att[0, 0:14, 0:15].tolist()
    k = target_info.tolist()[0:15]
    k[0] = 'BOS'
    k[-1] = 'EOS'
    xticklabels = [vocab.decode_ids(k[i]) for i in range(15)]
    yticklabels = [i for i in range(14)]

    #cbar使用指数缩放
    from scipy.special import logit, expit

    log_att = np.log1p(att)
    # 创建一个函数来格式化颜色条的标签
    def format_tick(x, pos):
        return '{:.2f}'.format(np.exp(x) - 1)
    formatter = ticker.FuncFormatter(format_tick)

    ax = sns.heatmap(
        log_att,
        xticklabels=xticklabels,
        square=True,
        yticklabels=yticklabels,
        vmin=0.0,
        vmax=np.log1p(0.5),
        cbar=True,
        cbar_kws={'format': formatter}
    )

    plt.show()
    color = "red"
# generate(txt, att, "sample.tex", color)
