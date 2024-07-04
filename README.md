## Interpret ESG Rating’s Impact on the Industrial Chain Using Graph Neural Networks

We conduct a quantitative analysis of the development of the industry chain from the environmental, social, and governance (ESG) perspective, which is an overall measure of sustainability. Factors that may impact the performance of the industrial chain have been studied in the literature, such as government regulation, monetary policy, etc. Our interest lies in how the sustainability changes (i.e., ESG shocks) affect the performance of the industrial chain. To achieve this goal, we model the industrial chain with a graph neural network (GNN) and conduct node regression on two financial performance metrics, namely, the aggregated profitability ratio and operating margin. To quantify the effects of ESG, we propose to compute the interaction between ESG shocks and industrial chain features with a cross-attention module, and then filter the original node features in the graph regression. Experiments on two real datasets demonstrate that (i) there are significant effects of ESG shocks on the industrial chain, and (ii) model parameters including regression coefficients and the attention map can explain how ESG shocks affect the performance of the industrial chain.

# citation
@inproceedings{ijcai2023p674,
  title     = {Interpret ESG Rating's Impact on the Industrial Chain Using Graph Neural Networks},
  author    = {Liu, Bin and He, Jiujun and Li, Ziyuan and Huang, Xiaoyang and Zhang, Xiang and Yin, Guosheng},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {6076--6084},
  year      = {2023},
  month     = {8},
  note      = {AI for Good},
  doi       = {10.24963/ijcai.2023/674},
  url       = {https://doi.org/10.24963/ijcai.2023/674},
}
