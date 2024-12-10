# Power System Fault Behavior Prediction  

## 目的  
利用隨機森林分類器預測電力系統的故障行為，並透過超參數優化提升模型的分類性能，降低錯誤預測率，支持更精確的電力系統管理與決策。

## 技術工具  
- Python  
- Scikit-learn  
- GridSearchCV  
- RandomizedSearchCV  

## 成果亮點  
1. 使用 **GridSearchCV** 進行超參數調整，最佳交叉驗證分數達到 **0.85**。  
2. 採用改進策略後，測試集加權 **F1 分數** 提升至 **0.86**，總體準確率從 **82%** 提升至 **88%**。  
3. 特別針對性能較弱的類別進行優化：  
   - **類別 2** 的 **F1 分數** 從 **0.55** 提升至 **0.58**。  
   - **類別 5** 的 **F1 分數** 從 **0.51** 提升至 **0.58**。  
   - 錯誤預測數顯著減少，提升模型在少數類別上的表現能力。

成果展示：https://powersystemfaultclassifier-flu2zajtp5h7cdwisgwfjq.streamlit.app/
