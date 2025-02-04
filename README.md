## Interpret ESG Rating’s Impact on the Industrial Chain Using Graph Neural Networks

We conduct a quantitative analysis of the development of the industry chain from the environmental, social, and governance (ESG) perspective, which is an overall measure of sustainability. Factors that may impact the performance of the industrial chain have been studied in the literature, such as government regulation, monetary policy, etc. Our interest lies in how the sustainability changes (i.e., ESG shocks) affect the performance of the industrial chain. To achieve this goal, we model the industrial chain with a graph neural network (GNN) and conduct node regression on two financial performance metrics, namely, the aggregated profitability ratio and operating margin. To quantify the effects of ESG, we propose to compute the interaction between ESG shocks and industrial chain features with a cross-attention module, and then filter the original node features in the graph regression. Experiments on two real datasets demonstrate that (i) there are significant effects of ESG shocks on the industrial chain, and (ii) model parameters including regression coefficients and the attention map can explain how ESG shocks affect the performance of the industrial chain.

# requirements
torch==1.10.1

torch-cluster==1.6.0

torch-geometric==2.2.0

torch-scatter==2.1.0

torch-sparse==0.6.16

numpy==1.21.5

# citation
@inproceedings{ijcai2023p674,

  title     = {Interpret ESG Rating's Impact on the Industrial Chain Using Graph Neural Networks},
  
  author    = {Liu, Bin and He, Jiujun and Li, Ziyuan and Huang, Xiaoyang and Zhang, Xiang and Yin, Guosheng},
  
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, {IJCAI-23}},
  
  publisher = {International Joint Conferences on Artificial Intelligence Organization},

  pages     = {6076--6084},
  
  year      = {2023}
}
