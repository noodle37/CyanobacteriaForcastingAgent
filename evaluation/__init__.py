"""GIFT-Eval 评测适配层。

GIFT-Eval 的评测对象是一个「把上下文序列映射到预测分布」的函数,通过 GluonTS 的
Predictor 接口接入。本包把 tsfm-agent 的 HTTP 接口包装成该接口,使本项目的智能体
能够直接在基准的任务切分与指标体系下被评测。
"""
