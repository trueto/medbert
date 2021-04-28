# medbert
本项目开源硕士毕业论文“BERT模型在中文临床自然语言处理中的应用探索与研究”相关模型

## 评估基准
构建了中文电子病历命名实体识别数据集（CEMRNER）、中文医学文本命名实体识别数据集（CMTNER）、

中文医学问句-问句识别数据集（CMedQQ）和中文临床文本分类数据集（CCTC）。

|  **数据集**   | **训练集**	| **验证集**	| **测试集**	| **任务类型**	| **语料来源**  |
|  ----    | ----   | ----  |----  |----  |:----:|
| CEMRNER	| 965	| 138	| 276	| 命名实体识别 | 医渡云 |
| CMTNER	| 14000	| 2000	| 4000	| 命名实体识别 |	CHIP2020 |
| CMedQQ	| 14000	| 2000	| 4000	| 句对识别 |	平安医疗 |
| CCTC	| 26837	| 3834 |	7669	| 句子分类 |	CHIP2019 |

## 开源模型
在6.5亿字符中文临床自然语言文本语料上基于BERT模型和Albert模型预训练获得了MedBERT和MedAlbert模型。

## 性能表现
在同等实验环境，相同训练参数和脚本下，各模型的性能表现

|  **模型**   | **CEMRNER**	| **CMTNER**	| **CMedQQ**	| **CCTC**	|
|  :----    | :----:   | :----:  |  :----:  |   :----:  |
|   [BERT](https://huggingface.co/bert-base-chinese)    |   81.17%  |   65.67%  |   87.77%  |   81.62%  |
| [MC-BERT](https://github.com/alibaba-research/ChineseBLUE)   |   80.93%  |   66.15%  |   89.04%  |   80.65%  |
| [PCL-BERT](https://code.ihub.org.cn/projects/1775)  |   81.58%  |   67.02%  |   88.81%  |   80.27%  |
| [MedBERT](https://huggingface.co/trueto/medbert-base-chinese)（ours）   |   82.29%  |   66.49%  |   88.32%  |   **81.77%**  |
|[MedBERT-wwm](https://huggingface.co/trueto/medbert-base-wwm-chinese) （ours）|   **82.60%**  |   67.11%  |   88.02%  |   81.72%  |
|[MedBERT-kd](https://huggingface.co/trueto/medbert-kd-chinese) （ours）|   82.58%  |   **67.27%**  |   **89.34%**  |   80.73%  |
|- |   -  |   -  |  -  |   -  |
|   [Albert](https://huggingface.co/voidful/albert_chinese_base) |   79.98%  |   62.42%  |   86.81%  |   79.83%  |
| [MedAlbert](https://huggingface.co/trueto/medalbert-base-chinese) （ours）|   81.03%  |   63.81%  |   87.56%  |   80.05%  |
|[MedAlbert-wwm](https://huggingface.co/trueto/medalbert-base-wwm-chinese)（ours）|   **81.28%**  |   **64.12%**  |   **87.71%**  |   **80.46%**  |

## 引用格式
```
XXX,XXX,XXX.BERT模型在中文临床自然语言处理中的应用探索与研究[EB/OL].https://github.com/trueto/medbert, 2021-03.
```
