nohup: 忽略输入
=== Starting ProtSATT 3-Class Training on cuda:0 ===
📦 loading 3 dataset...
-> filtered remaining: 20663
⏳ aligning Sequence_ID...
✅ feature ID aligned - ESM2: (20663, 1280), UniRep: (20663, 1900), ProtT5: (20663, 1024)
📦 独立测试集: 1864 样本 | 训练数据: 18799 样本

==================================================
🚀 Outer Fold 1/10
/data/caoying/miniconda3/envs/ProtSATT/lib/python3.10/site-packages/torch/nn/modules/lazy.py:180: UserWarning: Lazy modules are a new feature under heavy development so changes to the API or functionality can happen at any moment.
  warnings.warn('Lazy modules are a new feature under heavy development '
   Train Epoch 001 [] | Train Loss: 1.0374814932274194
   Val Epoch 001 [*] | Val ACC: 0.3810 | Val AUC: 0.5996 | Val MCC: 0.0000 | LR: 1.00e-04 | LOSS: 1.0323
   Train Epoch 002 [] | Train Loss: 1.0254497542785252
   Val Epoch 002 [*] | Val ACC: 0.4490 | Val AUC: 0.6243 | Val MCC: 0.0000 | LR: 1.00e-04 | LOSS: 1.0291
   Train Epoch 003 [] | Train Loss: 1.024259890761901
   Val Epoch 003 [*] | Val ACC: 0.5013 | Val AUC: 0.6477 | Val MCC: 0.1511 | LR: 1.00e-04 | LOSS: 1.0281
   Train Epoch 004 [] | Train Loss: 1.0202726298857008
   Val Epoch 004 [ ] | Val ACC: 0.4496 | Val AUC: 0.5956 | Val MCC: 0.0234 | LR: 1.00e-04 | Patience: 1/100 | LOSS: 1.0227
   Train Epoch 005 [] | Train Loss: 1.0134086717766784
   Val Epoch 005 [ ] | Val ACC: 0.4774 | Val AUC: 0.6325 | Val MCC: 0.1665 | LR: 1.00e-04 | Patience: 2/100 | LOSS: 1.0159
   Train Epoch 006 [] | Train Loss: 1.001747095910277
   Val Epoch 006 [*] | Val ACC: 0.5142 | Val AUC: 0.6293 | Val MCC: 0.1781 | LR: 1.00e-04 | LOSS: 1.0033
   Train Epoch 007 [] | Train Loss: 0.9933581834798547
   Val Epoch 007 [ ] | Val ACC: 0.4562 | Val AUC: 0.6005 | Val MCC: 0.0665 | LR: 1.00e-04 | Patience: 1/100 | LOSS: 1.0225
   Train Epoch 008 [] | Train Loss: 1.011023294024245
   Val Epoch 008 [ ] | Val ACC: 0.4439 | Val AUC: 0.6545 | Val MCC: 0.1693 | LR: 1.00e-04 | Patience: 2/100 | LOSS: 1.0336
   Train Epoch 009 [] | Train Loss: 1.0016735641782286
   Val Epoch 009 [ ] | Val ACC: 0.4705 | Val AUC: 0.6377 | Val MCC: 0.1104 | LR: 1.00e-04 | Patience: 3/100 | LOSS: 1.0138
   Train Epoch 010 [] | Train Loss: 0.9896835807420453
   Val Epoch 010 [ ] | Val ACC: 0.4810 | Val AUC: 0.6290 | Val MCC: 0.1570 | LR: 1.00e-04 | Patience: 4/100 | LOSS: 1.0073
   Train Epoch 011 [] | Train Loss: 0.9801191282344651
   Val Epoch 011 [ ] | Val ACC: 0.4723 | Val AUC: 0.6304 | Val MCC: 0.1557 | LR: 1.00e-04 | Patience: 5/100 | LOSS: 1.0117
   Train Epoch 012 [] | Train Loss: 0.973304340025374
   Val Epoch 012 [ ] | Val ACC: 0.5034 | Val AUC: 0.6260 | Val MCC: 0.1685 | LR: 1.00e-04 | Patience: 6/100 | LOSS: 1.0110
   Train Epoch 013 [] | Train Loss: 0.9742551136947051
   Val Epoch 013 [*] | Val ACC: 0.5172 | Val AUC: 0.6386 | Val MCC: 0.2058 | LR: 1.00e-04 | LOSS: 0.9952
   Train Epoch 014 [] | Train Loss: 0.9721482566882603
   Val Epoch 014 [ ] | Val ACC: 0.5148 | Val AUC: 0.6327 | Val MCC: 0.2059 | LR: 1.00e-04 | Patience: 1/100 | LOSS: 0.9919
   Train Epoch 015 [] | Train Loss: 0.9777117066078662
   Val Epoch 015 [ ] | Val ACC: 0.5100 | Val AUC: 0.6381 | Val MCC: 0.2024 | LR: 1.00e-04 | Patience: 2/100 | LOSS: 0.9940
   Train Epoch 016 [] | Train Loss: 0.965057527356632
   Val Epoch 016 [*] | Val ACC: 0.5223 | Val AUC: 0.6423 | Val MCC: 0.1832 | LR: 1.00e-04 | LOSS: 0.9805
   Train Epoch 017 [] | Train Loss: 0.9569391892969858
   Val Epoch 017 [*] | Val ACC: 0.5286 | Val AUC: 0.6467 | Val MCC: 0.2100 | LR: 1.00e-04 | LOSS: 0.9799
   Train Epoch 018 [] | Train Loss: 0.9570483539938853
   Val Epoch 018 [*] | Val ACC: 0.5304 | Val AUC: 0.6487 | Val MCC: 0.2261 | LR: 1.00e-04 | LOSS: 0.9859
   Train Epoch 019 [] | Train Loss: 0.9554562669327251
   Val Epoch 019 [ ] | Val ACC: 0.5256 | Val AUC: 0.6512 | Val MCC: 0.1885 | LR: 1.00e-04 | Patience: 1/100 | LOSS: 0.9774
   Train Epoch 020 [] | Train Loss: 0.9501631068700168
   Val Epoch 020 [ ] | Val ACC: 0.5226 | Val AUC: 0.6528 | Val MCC: 0.2188 | LR: 1.00e-04 | Patience: 2/100 | LOSS: 0.9855
   Train Epoch 021 [] | Train Loss: 0.9514198945738294
   Val Epoch 021 [ ] | Val ACC: 0.4864 | Val AUC: 0.6478 | Val MCC: 0.1857 | LR: 1.00e-04 | Patience: 3/100 | LOSS: 1.0155
   Train Epoch 022 [] | Train Loss: 0.9738463605552174
   Val Epoch 022 [*] | Val ACC: 0.5465 | Val AUC: 0.6473 | Val MCC: 0.2288 | LR: 1.00e-04 | LOSS: 0.9760
   Train Epoch 023 [] | Train Loss: 0.9572814827891594
   Val Epoch 023 [ ] | Val ACC: 0.5166 | Val AUC: 0.6540 | Val MCC: 0.2103 | LR: 1.00e-04 | Patience: 1/100 | LOSS: 0.9871
   Train Epoch 024 [] | Train Loss: 0.956888260314308
   Val Epoch 024 [ ] | Val ACC: 0.5079 | Val AUC: 0.6537 | Val MCC: 0.1867 | LR: 1.00e-04 | Patience: 2/100 | LOSS: 0.9982
   Train Epoch 025 [] | Train Loss: 0.9545993985015158
   Val Epoch 025 [ ] | Val ACC: 0.5301 | Val AUC: 0.6695 | Val MCC: 0.2108 | LR: 1.00e-04 | Patience: 3/100 | LOSS: 0.9694
   Train Epoch 026 [] | Train Loss: 0.9420017055453755
   Val Epoch 026 [*] | Val ACC: 0.5492 | Val AUC: 0.6684 | Val MCC: 0.2336 | LR: 1.00e-04 | LOSS: 0.9676
   Train Epoch 027 [] | Train Loss: 0.9316549172463552
   Val Epoch 027 [ ] | Val ACC: 0.4900 | Val AUC: 0.6579 | Val MCC: 0.1970 | LR: 1.00e-04 | Patience: 1/100 | LOSS: 1.0113
   Train Epoch 028 [] | Train Loss: 0.9523499142937093
   Val Epoch 028 [ ] | Val ACC: 0.5274 | Val AUC: 0.6651 | Val MCC: 0.2077 | LR: 1.00e-04 | Patience: 2/100 | LOSS: 0.9698
   Train Epoch 029 [] | Train Loss: 0.9488718559841075
   Val Epoch 029 [ ] | Val ACC: 0.5022 | Val AUC: 0.6691 | Val MCC: 0.1714 | LR: 1.00e-04 | Patience: 3/100 | LOSS: 0.9829
   Train Epoch 030 [] | Train Loss: 0.9353394891578555
   Val Epoch 030 [ ] | Val ACC: 0.5172 | Val AUC: 0.6798 | Val MCC: 0.2177 | LR: 1.00e-04 | Patience: 4/100 | LOSS: 0.9796
   Train Epoch 031 [] | Train Loss: 0.9328294939855775
   Val Epoch 031 [*] | Val ACC: 0.5648 | Val AUC: 0.6806 | Val MCC: 0.2616 | LR: 1.00e-04 | LOSS: 0.9537
   Train Epoch 032 [] | Train Loss: 0.9227900962485447
   Val Epoch 032 [ ] | Val ACC: 0.5642 | Val AUC: 0.6847 | Val MCC: 0.2614 | LR: 1.00e-04 | Patience: 1/100 | LOSS: 0.9542
   Train Epoch 033 [] | Train Loss: 0.9219540226481691
   Val Epoch 033 [ ] | Val ACC: 0.5609 | Val AUC: 0.6844 | Val MCC: 0.2584 | LR: 1.00e-04 | Patience: 2/100 | LOSS: 0.9570
   Train Epoch 034 [] | Train Loss: 0.9218279874660888
   Val Epoch 034 [ ] | Val ACC: 0.5519 | Val AUC: 0.6770 | Val MCC: 0.2456 | LR: 1.00e-04 | Patience: 3/100 | LOSS: 0.9682
   Train Epoch 035 [] | Train Loss: 0.9283645725426096
   Val Epoch 035 [*] | Val ACC: 0.5681 | Val AUC: 0.6820 | Val MCC: 0.2673 | LR: 1.00e-04 | LOSS: 0.9555
   Train Epoch 036 [] | Train Loss: 0.9246871249864914
   Val Epoch 036 [ ] | Val ACC: 0.5406 | Val AUC: 0.6890 | Val MCC: 0.2350 | LR: 1.00e-04 | Patience: 1/100 | LOSS: 0.9616
   Train Epoch 037 [] | Train Loss: 0.918501812798533
   Val Epoch 037 [ ] | Val ACC: 0.5465 | Val AUC: 0.6865 | Val MCC: 0.2312 | LR: 1.00e-04 | Patience: 2/100 | LOSS: 0.9506
   Train Epoch 038 [] | Train Loss: 0.9104579484149656
   Val Epoch 038 [ ] | Val ACC: 0.4606 | Val AUC: 0.6748 | Val MCC: 0.2050 | LR: 1.00e-04 | Patience: 3/100 | LOSS: 1.0383
   Train Epoch 039 [] | Train Loss: 0.9500775350161585
   Val Epoch 039 [ ] | Val ACC: 0.5657 | Val AUC: 0.6928 | Val MCC: 0.2673 | LR: 1.00e-04 | Patience: 4/100 | LOSS: 0.9506
   Train Epoch 040 [] | Train Loss: 0.9187091260279466
   Val Epoch 040 [ ] | Val ACC: 0.5552 | Val AUC: 0.6927 | Val MCC: 0.2486 | LR: 1.00e-04 | Patience: 5/100 | LOSS: 0.9436
   Train Epoch 041 [] | Train Loss: 0.9077171616504298
   Val Epoch 041 [ ] | Val ACC: 0.5403 | Val AUC: 0.6969 | Val MCC: 0.2406 | LR: 1.00e-04 | Patience: 6/100 | LOSS: 0.9501
   Train Epoch 042 [] | Train Loss: 0.9147569534081269
   Val Epoch 042 [ ] | Val ACC: 0.5516 | Val AUC: 0.6951 | Val MCC: 0.2486 | LR: 1.00e-04 | Patience: 7/100 | LOSS: 0.9464
   Train Epoch 043 [] | Train Loss: 0.9075782027786585
   Val Epoch 043 [ ] | Val ACC: 0.5367 | Val AUC: 0.6890 | Val MCC: 0.2118 | LR: 1.00e-04 | Patience: 8/100 | LOSS: 0.9473
   Train Epoch 044 [] | Train Loss: 0.9018466711504484
   Val Epoch 044 [ ] | Val ACC: 0.5313 | Val AUC: 0.6950 | Val MCC: 0.2331 | LR: 1.00e-04 | Patience: 9/100 | LOSS: 0.9568
   Train Epoch 045 [] | Train Loss: 0.9006124356386785
   Val Epoch 045 [ ] | Val ACC: 0.5660 | Val AUC: 0.6992 | Val MCC: 0.2724 | LR: 1.00e-04 | Patience: 10/100 | LOSS: 0.9338
   Train Epoch 046 [] | Train Loss: 0.8969329732467068
   Val Epoch 046 [ ] | Val ACC: 0.5367 | Val AUC: 0.6978 | Val MCC: 0.2253 | LR: 1.00e-04 | Patience: 11/100 | LOSS: 0.9498
   Train Epoch 047 [] | Train Loss: 0.8996447305154396
   Val Epoch 047 [*] | Val ACC: 0.5684 | Val AUC: 0.7060 | Val MCC: 0.2858 | LR: 1.00e-04 | LOSS: 0.9333
   Train Epoch 048 [] | Train Loss: 0.908954313681112
   Val Epoch 048 [ ] | Val ACC: 0.4906 | Val AUC: 0.6920 | Val MCC: 0.2427 | LR: 1.00e-04 | Patience: 1/100 | LOSS: 1.0262
   Train Epoch 049 [] | Train Loss: 0.9381095603208042
   Val Epoch 049 [ ] | Val ACC: 0.5301 | Val AUC: 0.6967 | Val MCC: 0.2165 | LR: 1.00e-04 | Patience: 2/100 | LOSS: 0.9473
   Train Epoch 050 [] | Train Loss: 0.9055410881411275
   Val Epoch 050 [ ] | Val ACC: 0.5645 | Val AUC: 0.6941 | Val MCC: 0.2868 | LR: 1.00e-04 | Patience: 3/100 | LOSS: 0.9456
   Train Epoch 051 [] | Train Loss: 0.897518135932342
   Val Epoch 051 [*] | Val ACC: 0.5744 | Val AUC: 0.7008 | Val MCC: 0.2877 | LR: 1.00e-04 | LOSS: 0.9299
   Train Epoch 052 [] | Train Loss: 0.8980477185254916
   Val Epoch 052 [ ] | Val ACC: 0.5609 | Val AUC: 0.7037 | Val MCC: 0.2826 | LR: 1.00e-04 | Patience: 1/100 | LOSS: 0.9419
   Train Epoch 053 [] | Train Loss: 0.895030523629227
   Val Epoch 053 [ ] | Val ACC: 0.5678 | Val AUC: 0.7056 | Val MCC: 0.2827 | LR: 1.00e-04 | Patience: 2/100 | LOSS: 0.9312
   Train Epoch 054 [] | Train Loss: 0.8888042919096878
   Val Epoch 054 [ ] | Val ACC: 0.5531 | Val AUC: 0.7093 | Val MCC: 0.2576 | LR: 1.00e-04 | Patience: 3/100 | LOSS: 0.9354
   Train Epoch 055 [] | Train Loss: 0.8847687759109835
   Val Epoch 055 [ ] | Val ACC: 0.5331 | Val AUC: 0.7017 | Val MCC: 0.2591 | LR: 1.00e-04 | Patience: 4/100 | LOSS: 0.9620
   Train Epoch 056 [] | Train Loss: 0.8922206251537593
   Val Epoch 056 [ ] | Val ACC: 0.5295 | Val AUC: 0.6968 | Val MCC: 0.2303 | LR: 1.00e-04 | Patience: 5/100 | LOSS: 0.9541
   Train Epoch 057 [] | Train Loss: 0.8987387259494447
   Val Epoch 057 [ ] | Val ACC: 0.5163 | Val AUC: 0.6981 | Val MCC: 0.1947 | LR: 1.00e-04 | Patience: 6/100 | LOSS: 0.9545
   Train Epoch 058 [] | Train Loss: 0.897631812298611
   Val Epoch 058 [ ] | Val ACC: 0.5388 | Val AUC: 0.6966 | Val MCC: 0.2400 | LR: 1.00e-04 | Patience: 7/100 | LOSS: 0.9530
   Train Epoch 059 [] | Train Loss: 0.9002024309331285
   Val Epoch 059 [ ] | Val ACC: 0.5385 | Val AUC: 0.7021 | Val MCC: 0.2337 | LR: 1.00e-04 | Patience: 8/100 | LOSS: 0.9522
   Train Epoch 060 [] | Train Loss: 0.8849893401564297
   Val Epoch 060 [ ] | Val ACC: 0.5702 | Val AUC: 0.7012 | Val MCC: 0.2723 | LR: 1.00e-04 | Patience: 9/100 | LOSS: 0.9312
   Train Epoch 061 [] | Train Loss: 0.8805188222863964
   Val Epoch 061 [ ] | Val ACC: 0.5702 | Val AUC: 0.7016 | Val MCC: 0.2737 | LR: 1.00e-04 | Patience: 10/100 | LOSS: 0.9265
   Train Epoch 062 [] | Train Loss: 0.8833751788374031
   Val Epoch 062 [ ] | Val ACC: 0.5636 | Val AUC: 0.6991 | Val MCC: 0.2753 | LR: 1.00e-04 | Patience: 11/100 | LOSS: 0.9371
   Train Epoch 063 [] | Train Loss: 0.8883430850516512
   Val Epoch 063 [ ] | Val ACC: 0.5591 | Val AUC: 0.7071 | Val MCC: 0.2580 | LR: 1.00e-04 | Patience: 12/100 | LOSS: 0.9242
   Train Epoch 064 [] | Train Loss: 0.8798411084896134
   Val Epoch 064 [ ] | Val ACC: 0.5717 | Val AUC: 0.7159 | Val MCC: 0.2982 | LR: 1.00e-04 | Patience: 13/100 | LOSS: 0.9229
   Train Epoch 065 [] | Train Loss: 0.8940728749234252
   Val Epoch 065 [ ] | Val ACC: 0.5286 | Val AUC: 0.7048 | Val MCC: 0.2585 | LR: 1.00e-04 | Patience: 14/100 | LOSS: 0.9705
   Train Epoch 066 [] | Train Loss: 0.9017472244429869
   Val Epoch 066 [ ] | Val ACC: 0.5370 | Val AUC: 0.7140 | Val MCC: 0.2643 | LR: 1.00e-04 | Patience: 15/100 | LOSS: 0.9426
   Train Epoch 067 [] | Train Loss: 0.8894166880836574
   Val Epoch 067 [ ] | Val ACC: 0.5735 | Val AUC: 0.7034 | Val MCC: 0.3042 | LR: 1.00e-04 | Patience: 16/100 | LOSS: 0.9316
   Train Epoch 068 [] | Train Loss: 0.8833653535250917
   Val Epoch 068 [ ] | Val ACC: 0.5717 | Val AUC: 0.7115 | Val MCC: 0.3076 | LR: 1.00e-04 | Patience: 17/100 | LOSS: 0.9396
   Train Epoch 069 [] | Train Loss: 0.8877125143462475
   Val Epoch 069 [ ] | Val ACC: 0.5603 | Val AUC: 0.7199 | Val MCC: 0.2952 | LR: 1.00e-04 | Patience: 18/100 | LOSS: 0.9348
   Train Epoch 070 [] | Train Loss: 0.8818893678069829
   Val Epoch 070 [ ] | Val ACC: 0.5624 | Val AUC: 0.7178 | Val MCC: 0.2828 | LR: 1.00e-04 | Patience: 19/100 | LOSS: 0.9268
   Train Epoch 071 [] | Train Loss: 0.8730555859063687
   Val Epoch 071 [ ] | Val ACC: 0.5741 | Val AUC: 0.7095 | Val MCC: 0.3049 | LR: 1.00e-04 | Patience: 20/100 | LOSS: 0.9309
   Train Epoch 072 [] | Train Loss: 0.8801357946422498
   Val Epoch 072 [ ] | Val ACC: 0.5450 | Val AUC: 0.7121 | Val MCC: 0.2614 | LR: 5.00e-05 | Patience: 21/100 | LOSS: 0.9345
   Train Epoch 073 [] | Train Loss: 0.8752807631000887
   Val Epoch 073 [*] | Val ACC: 0.5783 | Val AUC: 0.7196 | Val MCC: 0.2961 | LR: 5.00e-05 | LOSS: 0.9110
   Train Epoch 074 [] | Train Loss: 0.8677273676011994
   Val Epoch 074 [ ] | Val ACC: 0.5630 | Val AUC: 0.7110 | Val MCC: 0.2744 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.9324
   Train Epoch 075 [] | Train Loss: 0.8714903161898302
   Val Epoch 075 [ ] | Val ACC: 0.5780 | Val AUC: 0.7186 | Val MCC: 0.3044 | LR: 5.00e-05 | Patience: 2/100 | LOSS: 0.9156
   Train Epoch 076 [] | Train Loss: 0.8666343581563906
   Val Epoch 076 [ ] | Val ACC: 0.5777 | Val AUC: 0.7178 | Val MCC: 0.2981 | LR: 5.00e-05 | Patience: 3/100 | LOSS: 0.9129
   Train Epoch 077 [] | Train Loss: 0.8685163515559267
   Val Epoch 077 [ ] | Val ACC: 0.5720 | Val AUC: 0.7188 | Val MCC: 0.3005 | LR: 5.00e-05 | Patience: 4/100 | LOSS: 0.9275
   Train Epoch 078 [] | Train Loss: 0.8713308541419924
   Val Epoch 078 [ ] | Val ACC: 0.5759 | Val AUC: 0.7270 | Val MCC: 0.2991 | LR: 5.00e-05 | Patience: 5/100 | LOSS: 0.9051
   Train Epoch 079 [] | Train Loss: 0.8652238213129307
   Val Epoch 079 [ ] | Val ACC: 0.5681 | Val AUC: 0.7120 | Val MCC: 0.2910 | LR: 5.00e-05 | Patience: 6/100 | LOSS: 0.9289
   Train Epoch 080 [] | Train Loss: 0.8773669521148113
   Val Epoch 080 [ ] | Val ACC: 0.5657 | Val AUC: 0.7209 | Val MCC: 0.2879 | LR: 5.00e-05 | Patience: 7/100 | LOSS: 0.9137
   Train Epoch 081 [] | Train Loss: 0.867680987730307
   Val Epoch 081 [*] | Val ACC: 0.5834 | Val AUC: 0.7241 | Val MCC: 0.3054 | LR: 5.00e-05 | LOSS: 0.9043
   Train Epoch 082 [] | Train Loss: 0.8635100032136582
   Val Epoch 082 [ ] | Val ACC: 0.5624 | Val AUC: 0.7192 | Val MCC: 0.2779 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.9209
   Train Epoch 083 [] | Train Loss: 0.8689832346069745
   Val Epoch 083 [ ] | Val ACC: 0.5549 | Val AUC: 0.7164 | Val MCC: 0.2601 | LR: 5.00e-05 | Patience: 2/100 | LOSS: 0.9302
   Train Epoch 084 [] | Train Loss: 0.868062596231701
   Val Epoch 084 [ ] | Val ACC: 0.5765 | Val AUC: 0.7221 | Val MCC: 0.2995 | LR: 5.00e-05 | Patience: 3/100 | LOSS: 0.9109
   Train Epoch 085 [] | Train Loss: 0.8668359008903317
   Val Epoch 085 [*] | Val ACC: 0.5837 | Val AUC: 0.7281 | Val MCC: 0.3152 | LR: 5.00e-05 | LOSS: 0.9029
   Train Epoch 086 [] | Train Loss: 0.8650911957479537
   Val Epoch 086 [ ] | Val ACC: 0.5825 | Val AUC: 0.7264 | Val MCC: 0.3021 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.9034
   Train Epoch 087 [] | Train Loss: 0.8575994394801261
   Val Epoch 087 [ ] | Val ACC: 0.5699 | Val AUC: 0.7270 | Val MCC: 0.2980 | LR: 5.00e-05 | Patience: 2/100 | LOSS: 0.9111
   Train Epoch 088 [] | Train Loss: 0.8624300298967228
   Val Epoch 088 [ ] | Val ACC: 0.5549 | Val AUC: 0.7180 | Val MCC: 0.2718 | LR: 5.00e-05 | Patience: 3/100 | LOSS: 0.9340
   Train Epoch 089 [] | Train Loss: 0.867430071375672
   Val Epoch 089 [*] | Val ACC: 0.5858 | Val AUC: 0.7264 | Val MCC: 0.3044 | LR: 5.00e-05 | LOSS: 0.9052
   Train Epoch 090 [] | Train Loss: 0.8634416727603258
   Val Epoch 090 [ ] | Val ACC: 0.5810 | Val AUC: 0.7273 | Val MCC: 0.3030 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.9030
   Train Epoch 091 [] | Train Loss: 0.8568139680743242
   Val Epoch 091 [ ] | Val ACC: 0.5690 | Val AUC: 0.7217 | Val MCC: 0.2838 | LR: 5.00e-05 | Patience: 2/100 | LOSS: 0.9186
   Train Epoch 092 [] | Train Loss: 0.8575877283075869
   Val Epoch 092 [ ] | Val ACC: 0.5855 | Val AUC: 0.7308 | Val MCC: 0.3057 | LR: 5.00e-05 | Patience: 3/100 | LOSS: 0.8981
   Train Epoch 093 [] | Train Loss: 0.8610583188828458
   Val Epoch 093 [ ] | Val ACC: 0.5687 | Val AUC: 0.7268 | Val MCC: 0.3016 | LR: 5.00e-05 | Patience: 4/100 | LOSS: 0.9164
   Train Epoch 094 [] | Train Loss: 0.8619169165390121
   Val Epoch 094 [ ] | Val ACC: 0.5780 | Val AUC: 0.7328 | Val MCC: 0.3039 | LR: 5.00e-05 | Patience: 5/100 | LOSS: 0.9001
   Train Epoch 095 [] | Train Loss: 0.8577786311995131
   Val Epoch 095 [ ] | Val ACC: 0.5825 | Val AUC: 0.7314 | Val MCC: 0.2992 | LR: 5.00e-05 | Patience: 6/100 | LOSS: 0.8974
   Train Epoch 096 [] | Train Loss: 0.8551197426343938
   Val Epoch 096 [ ] | Val ACC: 0.5483 | Val AUC: 0.7165 | Val MCC: 0.2618 | LR: 5.00e-05 | Patience: 7/100 | LOSS: 0.9434
   Train Epoch 097 [] | Train Loss: 0.8718667912098643
   Val Epoch 097 [*] | Val ACC: 0.5872 | Val AUC: 0.7315 | Val MCC: 0.3094 | LR: 5.00e-05 | LOSS: 0.8959
   Train Epoch 098 [] | Train Loss: 0.8510384932946163
   Val Epoch 098 [ ] | Val ACC: 0.5744 | Val AUC: 0.7289 | Val MCC: 0.2939 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.9052
   Train Epoch 099 [] | Train Loss: 0.8554543735955185
   Val Epoch 099 [ ] | Val ACC: 0.5693 | Val AUC: 0.7324 | Val MCC: 0.3161 | LR: 5.00e-05 | Patience: 2/100 | LOSS: 0.9115
   Train Epoch 100 [] | Train Loss: 0.8553419517083674
   Val Epoch 100 [ ] | Val ACC: 0.5304 | Val AUC: 0.7146 | Val MCC: 0.2351 | LR: 5.00e-05 | Patience: 3/100 | LOSS: 0.9599
   Train Epoch 101 [] | Train Loss: 0.8790989101873963
   Val Epoch 101 [ ] | Val ACC: 0.5591 | Val AUC: 0.7236 | Val MCC: 0.2847 | LR: 5.00e-05 | Patience: 4/100 | LOSS: 0.9188
   Train Epoch 102 [] | Train Loss: 0.8567114488166558
   Val Epoch 102 [*] | Val ACC: 0.5890 | Val AUC: 0.7356 | Val MCC: 0.3130 | LR: 5.00e-05 | LOSS: 0.8932
   Train Epoch 103 [] | Train Loss: 0.8483668504160984
   Val Epoch 103 [ ] | Val ACC: 0.5831 | Val AUC: 0.7307 | Val MCC: 0.3064 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.8978
   Train Epoch 104 [] | Train Loss: 0.8505949914607435
   Val Epoch 104 [*] | Val ACC: 0.5914 | Val AUC: 0.7365 | Val MCC: 0.3276 | LR: 5.00e-05 | LOSS: 0.8926
   Train Epoch 105 [] | Train Loss: 0.8464513317314443
   Val Epoch 105 [ ] | Val ACC: 0.5834 | Val AUC: 0.7304 | Val MCC: 0.3067 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.9010
   Train Epoch 106 [] | Train Loss: 0.8463442856474798
   Val Epoch 106 [ ] | Val ACC: 0.5759 | Val AUC: 0.7275 | Val MCC: 0.2904 | LR: 5.00e-05 | Patience: 2/100 | LOSS: 0.9033
   Train Epoch 107 [] | Train Loss: 0.8484657978817606
   Val Epoch 107 [ ] | Val ACC: 0.5908 | Val AUC: 0.7341 | Val MCC: 0.3201 | LR: 5.00e-05 | Patience: 3/100 | LOSS: 0.8937
   Train Epoch 108 [] | Train Loss: 0.8480094348265925
   Val Epoch 108 [ ] | Val ACC: 0.5810 | Val AUC: 0.7310 | Val MCC: 0.3072 | LR: 5.00e-05 | Patience: 4/100 | LOSS: 0.9058
   Train Epoch 109 [] | Train Loss: 0.8515579848587944
   Val Epoch 109 [ ] | Val ACC: 0.5899 | Val AUC: 0.7333 | Val MCC: 0.3137 | LR: 5.00e-05 | Patience: 5/100 | LOSS: 0.8963
   Train Epoch 110 [] | Train Loss: 0.846641238966545
   Val Epoch 110 [ ] | Val ACC: 0.5896 | Val AUC: 0.7330 | Val MCC: 0.3114 | LR: 5.00e-05 | Patience: 6/100 | LOSS: 0.9016
   Train Epoch 111 [] | Train Loss: 0.8489882841612804
   Val Epoch 111 [ ] | Val ACC: 0.5804 | Val AUC: 0.7324 | Val MCC: 0.3088 | LR: 5.00e-05 | Patience: 7/100 | LOSS: 0.9100
   Train Epoch 112 [] | Train Loss: 0.855849899126133
   Val Epoch 112 [ ] | Val ACC: 0.5762 | Val AUC: 0.7358 | Val MCC: 0.3004 | LR: 5.00e-05 | Patience: 8/100 | LOSS: 0.9049
   Train Epoch 113 [] | Train Loss: 0.849579647344684
   Val Epoch 113 [ ] | Val ACC: 0.5828 | Val AUC: 0.7346 | Val MCC: 0.3007 | LR: 5.00e-05 | Patience: 9/100 | LOSS: 0.8975
   Train Epoch 114 [] | Train Loss: 0.851810222976318
   Val Epoch 114 [ ] | Val ACC: 0.5911 | Val AUC: 0.7384 | Val MCC: 0.3204 | LR: 5.00e-05 | Patience: 10/100 | LOSS: 0.8927
   Train Epoch 115 [] | Train Loss: 0.846843161860365
   Val Epoch 115 [ ] | Val ACC: 0.5816 | Val AUC: 0.7332 | Val MCC: 0.3005 | LR: 5.00e-05 | Patience: 11/100 | LOSS: 0.8981
   Train Epoch 116 [] | Train Loss: 0.8452716970900611
   Val Epoch 116 [ ] | Val ACC: 0.5543 | Val AUC: 0.7281 | Val MCC: 0.2633 | LR: 5.00e-05 | Patience: 12/100 | LOSS: 0.9217
   Train Epoch 117 [] | Train Loss: 0.8505052872551148
   Val Epoch 117 [*] | Val ACC: 0.5926 | Val AUC: 0.7386 | Val MCC: 0.3278 | LR: 5.00e-05 | LOSS: 0.8881
   Train Epoch 118 [] | Train Loss: 0.8492669874970816
   Val Epoch 118 [ ] | Val ACC: 0.5884 | Val AUC: 0.7392 | Val MCC: 0.3138 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.8915
   Train Epoch 119 [] | Train Loss: 0.8543377146037531
   Val Epoch 119 [ ] | Val ACC: 0.5843 | Val AUC: 0.7344 | Val MCC: 0.3132 | LR: 5.00e-05 | Patience: 2/100 | LOSS: 0.8975
   Train Epoch 120 [] | Train Loss: 0.8450493174441688
   Val Epoch 120 [*] | Val ACC: 0.5953 | Val AUC: 0.7354 | Val MCC: 0.3208 | LR: 5.00e-05 | LOSS: 0.8922
   Train Epoch 121 [] | Train Loss: 0.8426229848378185
   Val Epoch 121 [*] | Val ACC: 0.5971 | Val AUC: 0.7320 | Val MCC: 0.3229 | LR: 5.00e-05 | LOSS: 0.8987
   Train Epoch 122 [] | Train Loss: 0.8457976039908629
   Val Epoch 122 [ ] | Val ACC: 0.5597 | Val AUC: 0.7280 | Val MCC: 0.2829 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.9240
   Train Epoch 123 [] | Train Loss: 0.8479042163047716
   Val Epoch 123 [ ] | Val ACC: 0.5861 | Val AUC: 0.7357 | Val MCC: 0.3170 | LR: 5.00e-05 | Patience: 2/100 | LOSS: 0.8989
   Train Epoch 124 [] | Train Loss: 0.8485942618206104
   Val Epoch 124 [ ] | Val ACC: 0.5750 | Val AUC: 0.7346 | Val MCC: 0.2999 | LR: 5.00e-05 | Patience: 3/100 | LOSS: 0.9036
   Train Epoch 125 [] | Train Loss: 0.8567274398878325
   Val Epoch 125 [ ] | Val ACC: 0.5687 | Val AUC: 0.7316 | Val MCC: 0.2852 | LR: 5.00e-05 | Patience: 4/100 | LOSS: 0.9077
   Train Epoch 126 [] | Train Loss: 0.8561845162533264
   Val Epoch 126 [ ] | Val ACC: 0.5648 | Val AUC: 0.7306 | Val MCC: 0.3140 | LR: 5.00e-05 | Patience: 5/100 | LOSS: 0.9239
   Train Epoch 127 [] | Train Loss: 0.854417385889495
   Val Epoch 127 [ ] | Val ACC: 0.5780 | Val AUC: 0.7391 | Val MCC: 0.2925 | LR: 5.00e-05 | Patience: 6/100 | LOSS: 0.8950
   Train Epoch 128 [] | Train Loss: 0.8450365724944555
   Val Epoch 128 [ ] | Val ACC: 0.5911 | Val AUC: 0.7336 | Val MCC: 0.3129 | LR: 5.00e-05 | Patience: 7/100 | LOSS: 0.9004
   Train Epoch 129 [] | Train Loss: 0.8398951365624452
   Val Epoch 129 [ ] | Val ACC: 0.5711 | Val AUC: 0.7283 | Val MCC: 0.2897 | LR: 5.00e-05 | Patience: 8/100 | LOSS: 0.9289
   Train Epoch 130 [] | Train Loss: 0.8544138191306239
   Val Epoch 130 [ ] | Val ACC: 0.5549 | Val AUC: 0.7295 | Val MCC: 0.3105 | LR: 5.00e-05 | Patience: 9/100 | LOSS: 0.9460
   Train Epoch 131 [] | Train Loss: 0.8615161579695726
   Val Epoch 131 [ ] | Val ACC: 0.5498 | Val AUC: 0.7300 | Val MCC: 0.2633 | LR: 5.00e-05 | Patience: 10/100 | LOSS: 0.9362
   Train Epoch 132 [] | Train Loss: 0.8615520823385613
   Val Epoch 132 [ ] | Val ACC: 0.5810 | Val AUC: 0.7400 | Val MCC: 0.3155 | LR: 5.00e-05 | Patience: 11/100 | LOSS: 0.8896
   Train Epoch 133 [] | Train Loss: 0.8492655656015125
   Val Epoch 133 [ ] | Val ACC: 0.5804 | Val AUC: 0.7357 | Val MCC: 0.3134 | LR: 5.00e-05 | Patience: 12/100 | LOSS: 0.8975
   Train Epoch 134 [] | Train Loss: 0.8545310325623215
   Val Epoch 134 [ ] | Val ACC: 0.5729 | Val AUC: 0.7372 | Val MCC: 0.2891 | LR: 5.00e-05 | Patience: 13/100 | LOSS: 0.8931
   Train Epoch 135 [] | Train Loss: 0.848214261691411
   Val Epoch 135 [ ] | Val ACC: 0.5786 | Val AUC: 0.7327 | Val MCC: 0.3164 | LR: 5.00e-05 | Patience: 14/100 | LOSS: 0.9146
   Train Epoch 136 [] | Train Loss: 0.8484869538862552
   Val Epoch 136 [ ] | Val ACC: 0.5941 | Val AUC: 0.7433 | Val MCC: 0.3229 | LR: 5.00e-05 | Patience: 15/100 | LOSS: 0.8805
   Train Epoch 137 [] | Train Loss: 0.8346741790034222
   Val Epoch 137 [ ] | Val ACC: 0.5837 | Val AUC: 0.7392 | Val MCC: 0.3174 | LR: 5.00e-05 | Patience: 16/100 | LOSS: 0.8978
   Train Epoch 138 [] | Train Loss: 0.8448487610526718
   Val Epoch 138 [*] | Val ACC: 0.5974 | Val AUC: 0.7444 | Val MCC: 0.3274 | LR: 5.00e-05 | LOSS: 0.8800
   Train Epoch 139 [] | Train Loss: 0.8343989130646502
   Val Epoch 139 [ ] | Val ACC: 0.5899 | Val AUC: 0.7426 | Val MCC: 0.3284 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.8863
   Train Epoch 140 [] | Train Loss: 0.8342465440517144
   Val Epoch 140 [ ] | Val ACC: 0.5911 | Val AUC: 0.7402 | Val MCC: 0.3194 | LR: 5.00e-05 | Patience: 2/100 | LOSS: 0.8925
   Train Epoch 141 [] | Train Loss: 0.8341167604454037
   Val Epoch 141 [*] | Val ACC: 0.6025 | Val AUC: 0.7409 | Val MCC: 0.3368 | LR: 5.00e-05 | LOSS: 0.8836
   Train Epoch 142 [] | Train Loss: 0.835919073368077
   Val Epoch 142 [ ] | Val ACC: 0.5849 | Val AUC: 0.7410 | Val MCC: 0.3220 | LR: 5.00e-05 | Patience: 1/100 | LOSS: 0.8999
   Train Epoch 143 [] | Train Loss: 0.8386952514412435
   Val Epoch 143 [ ] | Val ACC: 0.5995 | Val AUC: 0.7474 | Val MCC: 0.3313 | LR: 5.00e-05 | Patience: 2/100 | LOSS: 0.8780
   Train Epoch 144 [] | Train Loss: 0.8292144738215274
   Val Epoch 144 [ ] | Val ACC: 0.5896 | Val AUC: 0.7400 | Val MCC: 0.3214 | LR: 5.00e-05 | Patience: 3/100 | LOSS: 0.8886
   Train Epoch 145 [] | Train Loss: 0.8341372150527442
   Val Epoch 145 [ ] | Val ACC: 0.5881 | Val AUC: 0.7362 | Val MCC: 0.3073 | LR: 5.00e-05 | Patience: 4/100 | LOSS: 0.8922
   Train Epoch 146 [] | Train Loss: 0.8360732315352004
   Val Epoch 146 [ ] | Val ACC: 0.5810 | Val AUC: 0.7340 | Val MCC: 0.3161 | LR: 5.00e-05 | Patience: 5/100 | LOSS: 0.9090
   Train Epoch 147 [] | Train Loss: 0.8432849159475411
   Val Epoch 147 [ ] | Val ACC: 0.6001 | Val AUC: 0.7317 | Val MCC: 0.3289 | LR: 5.00e-05 | Patience: 6/100 | LOSS: 0.9008
   Train Epoch 148 [] | Train Loss: 0.8371407689862177
   Val Epoch 148 [ ] | Val ACC: 0.5959 | Val AUC: 0.7461 | Val MCC: 0.3243 | LR: 5.00e-05 | Patience: 7/100 | LOSS: 0.8780
   Train Epoch 149 [] | Train Loss: 0.8348809318990491
   Val Epoch 149 [ ] | Val ACC: 0.5929 | Val AUC: 0.7460 | Val MCC: 0.3220 | LR: 5.00e-05 | Patience: 8/100 | LOSS: 0.8839
   Train Epoch 150 [] | Train Loss: 0.8363511796172485
   Val Epoch 150 [ ] | Val ACC: 0.5992 | Val AUC: 0.7474 | Val MCC: 0.3323 | LR: 5.00e-05 | Patience: 9/100 | LOSS: 0.8769
   Train Epoch 151 [] | Train Loss: 0.8311117843041013
   Val Epoch 151 [ ] | Val ACC: 0.5926 | Val AUC: 0.7465 | Val MCC: 0.3302 | LR: 5.00e-05 | Patience: 10/100 | LOSS: 0.8832
   Train Epoch 152 [] | Train Loss: 0.8370926884580318
   Val Epoch 152 [ ] | Val ACC: 0.5804 | Val AUC: 0.7423 | Val MCC: 0.3138 | LR: 5.00e-05 | Patience: 11/100 | LOSS: 0.8942
   Train Epoch 153 [] | Train Loss: 0.8443707270560459
   Val Epoch 153 [ ] | Val ACC: 0.5905 | Val AUC: 0.7435 | Val MCC: 0.3166 | LR: 5.00e-05 | Patience: 12/100 | LOSS: 0.8820
   Train Epoch 154 [] | Train Loss: 0.8384753433276054
   Val Epoch 154 [ ] | Val ACC: 0.5843 | Val AUC: 0.7453 | Val MCC: 0.3158 | LR: 5.00e-05 | Patience: 13/100 | LOSS: 0.8823
   Train Epoch 155 [] | Train Loss: 0.8441957927854596
   Val Epoch 155 [ ] | Val ACC: 0.5795 | Val AUC: 0.7410 | Val MCC: 0.3291 | LR: 5.00e-05 | Patience: 14/100 | LOSS: 0.8999
   Train Epoch 156 [] | Train Loss: 0.8378711439447777
   Val Epoch 156 [ ] | Val ACC: 0.5917 | Val AUC: 0.7464 | Val MCC: 0.3180 | LR: 5.00e-05 | Patience: 15/100 | LOSS: 0.8763
   Train Epoch 157 [] | Train Loss: 0.8357226293724838
   Val Epoch 157 [ ] | Val ACC: 0.5394 | Val AUC: 0.7272 | Val MCC: 0.3023 | LR: 5.00e-05 | Patience: 16/100 | LOSS: 0.9737
   Train Epoch 158 [] | Train Loss: 0.8671521262477659
   Val Epoch 158 [ ] | Val ACC: 0.5849 | Val AUC: 0.7433 | Val MCC: 0.3059 | LR: 5.00e-05 | Patience: 17/100 | LOSS: 0.8860
   Train Epoch 159 [] | Train Loss: 0.8412789618639447
   Val Epoch 159 [ ] | Val ACC: 0.5914 | Val AUC: 0.7437 | Val MCC: 0.3164 | LR: 5.00e-05 | Patience: 18/100 | LOSS: 0.8878
   Train Epoch 160 [] | Train Loss: 0.8344715957746126
   Val Epoch 160 [ ] | Val ACC: 0.5911 | Val AUC: 0.7450 | Val MCC: 0.3270 | LR: 5.00e-05 | Patience: 19/100 | LOSS: 0.8806
   Train Epoch 161 [] | Train Loss: 0.8290966379574283
   Val Epoch 161 [ ] | Val ACC: 0.5744 | Val AUC: 0.7392 | Val MCC: 0.3096 | LR: 5.00e-05 | Patience: 20/100 | LOSS: 0.9127
   Train Epoch 162 [] | Train Loss: 0.8421282578921461
   Val Epoch 162 [ ] | Val ACC: 0.5828 | Val AUC: 0.7455 | Val MCC: 0.3170 | LR: 2.50e-05 | Patience: 21/100 | LOSS: 0.8891
   Train Epoch 163 [] | Train Loss: 0.8318781814511429
   Val Epoch 163 [ ] | Val ACC: 0.5861 | Val AUC: 0.7451 | Val MCC: 0.3193 | LR: 2.50e-05 | Patience: 22/100 | LOSS: 0.8842
   Train Epoch 164 [] | Train Loss: 0.825150244132618
   Val Epoch 164 [ ] | Val ACC: 0.5959 | Val AUC: 0.7471 | Val MCC: 0.3288 | LR: 2.50e-05 | Patience: 23/100 | LOSS: 0.8753
   Train Epoch 165 [] | Train Loss: 0.8244478514696241
   Val Epoch 165 [ ] | Val ACC: 0.5884 | Val AUC: 0.7450 | Val MCC: 0.3252 | LR: 2.50e-05 | Patience: 24/100 | LOSS: 0.8813
   Train Epoch 166 [] | Train Loss: 0.8293102483454551
   Val Epoch 166 [ ] | Val ACC: 0.5923 | Val AUC: 0.7456 | Val MCC: 0.3274 | LR: 2.50e-05 | Patience: 25/100 | LOSS: 0.8834
   Train Epoch 167 [] | Train Loss: 0.8245397734183499
   Val Epoch 167 [ ] | Val ACC: 0.5980 | Val AUC: 0.7466 | Val MCC: 0.3287 | LR: 2.50e-05 | Patience: 26/100 | LOSS: 0.8760
   Train Epoch 168 [] | Train Loss: 0.8312438875405967
   Val Epoch 168 [ ] | Val ACC: 0.5962 | Val AUC: 0.7488 | Val MCC: 0.3309 | LR: 2.50e-05 | Patience: 27/100 | LOSS: 0.8799
   Train Epoch 169 [] | Train Loss: 0.8288514750308021
   Val Epoch 169 [ ] | Val ACC: 0.5941 | Val AUC: 0.7493 | Val MCC: 0.3275 | LR: 2.50e-05 | Patience: 28/100 | LOSS: 0.8778
   Train Epoch 170 [] | Train Loss: 0.8260524982364763
   Val Epoch 170 [ ] | Val ACC: 0.5920 | Val AUC: 0.7455 | Val MCC: 0.3205 | LR: 2.50e-05 | Patience: 29/100 | LOSS: 0.8840
   Train Epoch 171 [] | Train Loss: 0.8276938191040022
   Val Epoch 171 [ ] | Val ACC: 0.5902 | Val AUC: 0.7447 | Val MCC: 0.3187 | LR: 2.50e-05 | Patience: 30/100 | LOSS: 0.8874
   Train Epoch 172 [] | Train Loss: 0.8316289500586374
   Val Epoch 172 [ ] | Val ACC: 0.5917 | Val AUC: 0.7448 | Val MCC: 0.3300 | LR: 2.50e-05 | Patience: 31/100 | LOSS: 0.8835
   Train Epoch 173 [] | Train Loss: 0.8245887556337034
   Val Epoch 173 [ ] | Val ACC: 0.5875 | Val AUC: 0.7474 | Val MCC: 0.3233 | LR: 2.50e-05 | Patience: 32/100 | LOSS: 0.8809
   Train Epoch 174 [] | Train Loss: 0.8217191960601744
   Val Epoch 174 [ ] | Val ACC: 0.5998 | Val AUC: 0.7493 | Val MCC: 0.3332 | LR: 2.50e-05 | Patience: 33/100 | LOSS: 0.8772
   Train Epoch 175 [] | Train Loss: 0.8253685494331076
   Val Epoch 175 [ ] | Val ACC: 0.5971 | Val AUC: 0.7491 | Val MCC: 0.3333 | LR: 2.50e-05 | Patience: 34/100 | LOSS: 0.8807
   Train Epoch 176 [] | Train Loss: 0.823721891996196
   Val Epoch 176 [ ] | Val ACC: 0.5938 | Val AUC: 0.7490 | Val MCC: 0.3259 | LR: 2.50e-05 | Patience: 35/100 | LOSS: 0.8740
   Train Epoch 177 [] | Train Loss: 0.8232154914272262
   Val Epoch 177 [ ] | Val ACC: 0.6001 | Val AUC: 0.7481 | Val MCC: 0.3373 | LR: 2.50e-05 | Patience: 36/100 | LOSS: 0.8765
   Train Epoch 178 [] | Train Loss: 0.8209167359377684
   Val Epoch 178 [ ] | Val ACC: 0.5855 | Val AUC: 0.7459 | Val MCC: 0.3312 | LR: 2.50e-05 | Patience: 37/100 | LOSS: 0.8934
   Train Epoch 179 [] | Train Loss: 0.8303510657125628
   Val Epoch 179 [ ] | Val ACC: 0.5926 | Val AUC: 0.7418 | Val MCC: 0.3201 | LR: 2.50e-05 | Patience: 38/100 | LOSS: 0.8850
   Train Epoch 180 [] | Train Loss: 0.8253365658444984
   Val Epoch 180 [ ] | Val ACC: 0.5992 | Val AUC: 0.7456 | Val MCC: 0.3335 | LR: 2.50e-05 | Patience: 39/100 | LOSS: 0.8834
   Train Epoch 181 [] | Train Loss: 0.829023376987501
   Val Epoch 181 [ ] | Val ACC: 0.5762 | Val AUC: 0.7416 | Val MCC: 0.3048 | LR: 2.50e-05 | Patience: 40/100 | LOSS: 0.9145
   Train Epoch 182 [] | Train Loss: 0.8354641502209385
   Val Epoch 182 [ ] | Val ACC: 0.5932 | Val AUC: 0.7464 | Val MCC: 0.3239 | LR: 2.50e-05 | Patience: 41/100 | LOSS: 0.8774
   Train Epoch 183 [] | Train Loss: 0.8265282042006634
   Val Epoch 183 [ ] | Val ACC: 0.5902 | Val AUC: 0.7490 | Val MCC: 0.3236 | LR: 1.25e-05 | Patience: 42/100 | LOSS: 0.8752
   Train Epoch 184 [] | Train Loss: 0.8197373234486937
   Val Epoch 184 [ ] | Val ACC: 0.5971 | Val AUC: 0.7500 | Val MCC: 0.3304 | LR: 1.25e-05 | Patience: 43/100 | LOSS: 0.8726
   Train Epoch 185 [] | Train Loss: 0.8207835135991071
   Val Epoch 185 [ ] | Val ACC: 0.5938 | Val AUC: 0.7494 | Val MCC: 0.3287 | LR: 1.25e-05 | Patience: 44/100 | LOSS: 0.8740
   Train Epoch 186 [] | Train Loss: 0.8202716332855219
   Val Epoch 186 [ ] | Val ACC: 0.5935 | Val AUC: 0.7475 | Val MCC: 0.3255 | LR: 1.25e-05 | Patience: 45/100 | LOSS: 0.8749
   Train Epoch 187 [] | Train Loss: 0.8173789804024676
   Val Epoch 187 [ ] | Val ACC: 0.5971 | Val AUC: 0.7471 | Val MCC: 0.3352 | LR: 1.25e-05 | Patience: 46/100 | LOSS: 0.8762
   Train Epoch 188 [] | Train Loss: 0.8153398726681177
   Val Epoch 188 [ ] | Val ACC: 0.6001 | Val AUC: 0.7472 | Val MCC: 0.3331 | LR: 1.25e-05 | Patience: 47/100 | LOSS: 0.8753
   Train Epoch 189 [] | Train Loss: 0.8189952598989088
   Val Epoch 189 [ ] | Val ACC: 0.5920 | Val AUC: 0.7502 | Val MCC: 0.3254 | LR: 1.25e-05 | Patience: 48/100 | LOSS: 0.8728
   Train Epoch 190 [] | Train Loss: 0.8217903424313487
   Val Epoch 190 [ ] | Val ACC: 0.5953 | Val AUC: 0.7483 | Val MCC: 0.3329 | LR: 1.25e-05 | Patience: 49/100 | LOSS: 0.8782
   Train Epoch 191 [] | Train Loss: 0.8187185543684756
   Val Epoch 191 [ ] | Val ACC: 0.5953 | Val AUC: 0.7486 | Val MCC: 0.3257 | LR: 1.25e-05 | Patience: 50/100 | LOSS: 0.8743
   Train Epoch 192 [] | Train Loss: 0.8190645252416651
   Val Epoch 192 [ ] | Val ACC: 0.5956 | Val AUC: 0.7506 | Val MCC: 0.3279 | LR: 1.25e-05 | Patience: 51/100 | LOSS: 0.8722
   Train Epoch 193 [] | Train Loss: 0.8185807454345261
   Val Epoch 193 [ ] | Val ACC: 0.5959 | Val AUC: 0.7478 | Val MCC: 0.3257 | LR: 1.25e-05 | Patience: 52/100 | LOSS: 0.8740
   Train Epoch 194 [] | Train Loss: 0.8158100141543478
   Val Epoch 194 [ ] | Val ACC: 0.5977 | Val AUC: 0.7521 | Val MCC: 0.3297 | LR: 1.25e-05 | Patience: 53/100 | LOSS: 0.8702
   Train Epoch 195 [] | Train Loss: 0.8171242293741567
   Val Epoch 195 [ ] | Val ACC: 0.5971 | Val AUC: 0.7489 | Val MCC: 0.3318 | LR: 1.25e-05 | Patience: 54/100 | LOSS: 0.8764
   Train Epoch 196 [] | Train Loss: 0.8190484248014915
   Val Epoch 196 [ ] | Val ACC: 0.5917 | Val AUC: 0.7500 | Val MCC: 0.3199 | LR: 1.25e-05 | Patience: 55/100 | LOSS: 0.8719
   Train Epoch 197 [] | Train Loss: 0.815087766352993
   Val Epoch 197 [ ] | Val ACC: 0.5911 | Val AUC: 0.7468 | Val MCC: 0.3190 | LR: 1.25e-05 | Patience: 56/100 | LOSS: 0.8766
   Train Epoch 198 [] | Train Loss: 0.8150577597264099
   Val Epoch 198 [*] | Val ACC: 0.6064 | Val AUC: 0.7486 | Val MCC: 0.3463 | LR: 1.25e-05 | LOSS: 0.8750
   Train Epoch 199 [] | Train Loss: 0.8183504060355539
   Val Epoch 199 [ ] | Val ACC: 0.6010 | Val AUC: 0.7508 | Val MCC: 0.3362 | LR: 1.25e-05 | Patience: 1/100 | LOSS: 0.8708
   Train Epoch 200 [] | Train Loss: 0.8155573599547417
   Val Epoch 200 [ ] | Val ACC: 0.5962 | Val AUC: 0.7501 | Val MCC: 0.3301 | LR: 1.25e-05 | Patience: 2/100 | LOSS: 0.8786
   Train Epoch 201 [] | Train Loss: 0.8190943812028918
   Val Epoch 201 [ ] | Val ACC: 0.5864 | Val AUC: 0.7480 | Val MCC: 0.3161 | LR: 1.25e-05 | Patience: 3/100 | LOSS: 0.8868
   Train Epoch 202 [] | Train Loss: 0.8194646678827464
   Val Epoch 202 [ ] | Val ACC: 0.5995 | Val AUC: 0.7507 | Val MCC: 0.3323 | LR: 1.25e-05 | Patience: 4/100 | LOSS: 0.8759
   Train Epoch 203 [] | Train Loss: 0.8205344763705705
   Val Epoch 203 [ ] | Val ACC: 0.5944 | Val AUC: 0.7500 | Val MCC: 0.3284 | LR: 1.25e-05 | Patience: 5/100 | LOSS: 0.8741
   Train Epoch 204 [] | Train Loss: 0.8173235672046308
   Val Epoch 204 [ ] | Val ACC: 0.5989 | Val AUC: 0.7520 | Val MCC: 0.3315 | LR: 1.25e-05 | Patience: 6/100 | LOSS: 0.8692
   Train Epoch 205 [] | Train Loss: 0.8153023473431046
   Val Epoch 205 [ ] | Val ACC: 0.5980 | Val AUC: 0.7506 | Val MCC: 0.3321 | LR: 1.25e-05 | Patience: 7/100 | LOSS: 0.8708
   Train Epoch 206 [] | Train Loss: 0.8149446530493959
   Val Epoch 206 [ ] | Val ACC: 0.5986 | Val AUC: 0.7498 | Val MCC: 0.3345 | LR: 1.25e-05 | Patience: 8/100 | LOSS: 0.8726
   Train Epoch 207 [] | Train Loss: 0.8179573433144627
   Val Epoch 207 [ ] | Val ACC: 0.5992 | Val AUC: 0.7491 | Val MCC: 0.3326 | LR: 1.25e-05 | Patience: 9/100 | LOSS: 0.8761
   Train Epoch 208 [] | Train Loss: 0.8165181353426312
   Val Epoch 208 [ ] | Val ACC: 0.5950 | Val AUC: 0.7514 | Val MCC: 0.3377 | LR: 1.25e-05 | Patience: 10/100 | LOSS: 0.8711
   Train Epoch 209 [] | Train Loss: 0.8160587828846468
   Val Epoch 209 [ ] | Val ACC: 0.5947 | Val AUC: 0.7498 | Val MCC: 0.3286 | LR: 1.25e-05 | Patience: 11/100 | LOSS: 0.8738
   Train Epoch 210 [] | Train Loss: 0.8157755426521707
   Val Epoch 210 [ ] | Val ACC: 0.5867 | Val AUC: 0.7466 | Val MCC: 0.3226 | LR: 1.25e-05 | Patience: 12/100 | LOSS: 0.8881
   Train Epoch 211 [] | Train Loss: 0.8196427523617315
   Val Epoch 211 [ ] | Val ACC: 0.5950 | Val AUC: 0.7509 | Val MCC: 0.3262 | LR: 1.25e-05 | Patience: 13/100 | LOSS: 0.8700
   Train Epoch 212 [] | Train Loss: 0.8114309059091988
   Val Epoch 212 [ ] | Val ACC: 0.5870 | Val AUC: 0.7476 | Val MCC: 0.3250 | LR: 1.25e-05 | Patience: 14/100 | LOSS: 0.8820
   Train Epoch 213 [] | Train Loss: 0.8168273968297888
   Val Epoch 213 [ ] | Val ACC: 0.5932 | Val AUC: 0.7493 | Val MCC: 0.3247 | LR: 1.25e-05 | Patience: 15/100 | LOSS: 0.8732
   Train Epoch 214 [] | Train Loss: 0.8142474183185264
   Val Epoch 214 [ ] | Val ACC: 0.5956 | Val AUC: 0.7532 | Val MCC: 0.3266 | LR: 1.25e-05 | Patience: 16/100 | LOSS: 0.8681
   Train Epoch 215 [] | Train Loss: 0.8152359910784381
   Val Epoch 215 [ ] | Val ACC: 0.5977 | Val AUC: 0.7488 | Val MCC: 0.3322 | LR: 1.25e-05 | Patience: 17/100 | LOSS: 0.8745
   Train Epoch 216 [] | Train Loss: 0.8147129040767909
   Val Epoch 216 [ ] | Val ACC: 0.5965 | Val AUC: 0.7498 | Val MCC: 0.3330 | LR: 1.25e-05 | Patience: 18/100 | LOSS: 0.8725
   Train Epoch 217 [] | Train Loss: 0.8144915477638037
   Val Epoch 217 [ ] | Val ACC: 0.5887 | Val AUC: 0.7496 | Val MCC: 0.3247 | LR: 1.25e-05 | Patience: 19/100 | LOSS: 0.8812
   Train Epoch 218 [] | Train Loss: 0.8143760361644824
   Val Epoch 218 [ ] | Val ACC: 0.5944 | Val AUC: 0.7485 | Val MCC: 0.3311 | LR: 1.25e-05 | Patience: 20/100 | LOSS: 0.8816
   Train Epoch 219 [] | Train Loss: 0.812990028183375
   Val Epoch 219 [ ] | Val ACC: 0.6019 | Val AUC: 0.7513 | Val MCC: 0.3417 | LR: 6.25e-06 | Patience: 21/100 | LOSS: 0.8714
   Train Epoch 220 [] | Train Loss: 0.8126038014498439
   Val Epoch 220 [ ] | Val ACC: 0.6010 | Val AUC: 0.7515 | Val MCC: 0.3354 | LR: 6.25e-06 | Patience: 22/100 | LOSS: 0.8699
   Train Epoch 221 [] | Train Loss: 0.8124658213276731
   Val Epoch 221 [ ] | Val ACC: 0.5977 | Val AUC: 0.7516 | Val MCC: 0.3318 | LR: 6.25e-06 | Patience: 23/100 | LOSS: 0.8694
   Train Epoch 222 [] | Train Loss: 0.8123948864180174
   Val Epoch 222 [ ] | Val ACC: 0.5986 | Val AUC: 0.7509 | Val MCC: 0.3305 | LR: 6.25e-06 | Patience: 24/100 | LOSS: 0.8705
   Train Epoch 223 [] | Train Loss: 0.8108348432814565
   Val Epoch 223 [ ] | Val ACC: 0.6022 | Val AUC: 0.7513 | Val MCC: 0.3377 | LR: 6.25e-06 | Patience: 25/100 | LOSS: 0.8705
   Train Epoch 224 [] | Train Loss: 0.8120197514527225
   Val Epoch 224 [ ] | Val ACC: 0.5989 | Val AUC: 0.7512 | Val MCC: 0.3335 | LR: 6.25e-06 | Patience: 26/100 | LOSS: 0.8711
   Train Epoch 225 [] | Train Loss: 0.8100609456579879
   Val Epoch 225 [ ] | Val ACC: 0.5941 | Val AUC: 0.7505 | Val MCC: 0.3241 | LR: 6.25e-06 | Patience: 27/100 | LOSS: 0.8708
   Train Epoch 226 [] | Train Loss: 0.810283682582511
   Val Epoch 226 [ ] | Val ACC: 0.5917 | Val AUC: 0.7512 | Val MCC: 0.3286 | LR: 6.25e-06 | Patience: 28/100 | LOSS: 0.8743
   Train Epoch 227 [] | Train Loss: 0.8127125797293422
   Val Epoch 227 [ ] | Val ACC: 0.6001 | Val AUC: 0.7519 | Val MCC: 0.3347 | LR: 6.25e-06 | Patience: 29/100 | LOSS: 0.8691
   Train Epoch 228 [] | Train Loss: 0.8103931501467699
   Val Epoch 228 [ ] | Val ACC: 0.5983 | Val AUC: 0.7523 | Val MCC: 0.3312 | LR: 6.25e-06 | Patience: 30/100 | LOSS: 0.8684
   Train Epoch 229 [] | Train Loss: 0.8091350408896116
   Val Epoch 229 [ ] | Val ACC: 0.5995 | Val AUC: 0.7528 | Val MCC: 0.3348 | LR: 6.25e-06 | Patience: 31/100 | LOSS: 0.8689
   Train Epoch 230 [] | Train Loss: 0.8091811520732672
   Val Epoch 230 [ ] | Val ACC: 0.5974 | Val AUC: 0.7526 | Val MCC: 0.3294 | LR: 6.25e-06 | Patience: 32/100 | LOSS: 0.8689
   Train Epoch 231 [] | Train Loss: 0.8117389360573337
   Val Epoch 231 [ ] | Val ACC: 0.5983 | Val AUC: 0.7515 | Val MCC: 0.3324 | LR: 6.25e-06 | Patience: 33/100 | LOSS: 0.8708
   Train Epoch 232 [] | Train Loss: 0.8120515931069239
   Val Epoch 232 [ ] | Val ACC: 0.5923 | Val AUC: 0.7497 | Val MCC: 0.3267 | LR: 6.25e-06 | Patience: 34/100 | LOSS: 0.8766
   Train Epoch 233 [] | Train Loss: 0.8096574031526843
   Val Epoch 233 [ ] | Val ACC: 0.5947 | Val AUC: 0.7499 | Val MCC: 0.3252 | LR: 6.25e-06 | Patience: 35/100 | LOSS: 0.8718
   Train Epoch 234 [] | Train Loss: 0.8083476234642653
   Val Epoch 234 [ ] | Val ACC: 0.5974 | Val AUC: 0.7498 | Val MCC: 0.3303 | LR: 6.25e-06 | Patience: 36/100 | LOSS: 0.8717
   Train Epoch 235 [] | Train Loss: 0.8111279624774091
   Val Epoch 235 [ ] | Val ACC: 0.5953 | Val AUC: 0.7501 | Val MCC: 0.3259 | LR: 6.25e-06 | Patience: 37/100 | LOSS: 0.8710
   Train Epoch 236 [] | Train Loss: 0.8092752792911329
   Val Epoch 236 [ ] | Val ACC: 0.5950 | Val AUC: 0.7512 | Val MCC: 0.3301 | LR: 6.25e-06 | Patience: 38/100 | LOSS: 0.8711
   Train Epoch 237 [] | Train Loss: 0.8086562934573944
   Val Epoch 237 [ ] | Val ACC: 0.5965 | Val AUC: 0.7509 | Val MCC: 0.3302 | LR: 6.25e-06 | Patience: 39/100 | LOSS: 0.8727
   Train Epoch 238 [] | Train Loss: 0.8058840186030398
   Val Epoch 238 [ ] | Val ACC: 0.5956 | Val AUC: 0.7512 | Val MCC: 0.3268 | LR: 6.25e-06 | Patience: 40/100 | LOSS: 0.8739
   Train Epoch 239 [] | Train Loss: 0.8107437712381831
   Val Epoch 239 [ ] | Val ACC: 0.5980 | Val AUC: 0.7507 | Val MCC: 0.3299 | LR: 6.25e-06 | Patience: 41/100 | LOSS: 0.8717
   Train Epoch 240 [] | Train Loss: 0.810031106180792
   Val Epoch 240 [ ] | Val ACC: 0.5908 | Val AUC: 0.7507 | Val MCC: 0.3247 | LR: 3.13e-06 | Patience: 42/100 | LOSS: 0.8755
   Train Epoch 241 [] | Train Loss: 0.8124779657072799
   Val Epoch 241 [ ] | Val ACC: 0.5971 | Val AUC: 0.7510 | Val MCC: 0.3285 | LR: 3.13e-06 | Patience: 43/100 | LOSS: 0.8709
   Train Epoch 242 [] | Train Loss: 0.8072279004642421
   Val Epoch 242 [ ] | Val ACC: 0.5971 | Val AUC: 0.7515 | Val MCC: 0.3315 | LR: 3.13e-06 | Patience: 44/100 | LOSS: 0.8707
   Train Epoch 243 [] | Train Loss: 0.8092399478131244
   Val Epoch 243 [ ] | Val ACC: 0.6007 | Val AUC: 0.7505 | Val MCC: 0.3351 | LR: 3.13e-06 | Patience: 45/100 | LOSS: 0.8727
   Train Epoch 244 [] | Train Loss: 0.8078659120825146
   Val Epoch 244 [ ] | Val ACC: 0.5950 | Val AUC: 0.7518 | Val MCC: 0.3287 | LR: 3.13e-06 | Patience: 46/100 | LOSS: 0.8709
   Train Epoch 245 [] | Train Loss: 0.8074127732536454
   Val Epoch 245 [ ] | Val ACC: 0.6016 | Val AUC: 0.7520 | Val MCC: 0.3375 | LR: 3.13e-06 | Patience: 47/100 | LOSS: 0.8690
   Train Epoch 246 [] | Train Loss: 0.8072118225067745
   Val Epoch 246 [ ] | Val ACC: 0.6004 | Val AUC: 0.7522 | Val MCC: 0.3360 | LR: 3.13e-06 | Patience: 48/100 | LOSS: 0.8687
   Train Epoch 247 [] | Train Loss: 0.8101386594363115
   Val Epoch 247 [ ] | Val ACC: 0.5992 | Val AUC: 0.7530 | Val MCC: 0.3346 | LR: 3.13e-06 | Patience: 49/100 | LOSS: 0.8680
   Train Epoch 248 [] | Train Loss: 0.8089211338301172
   Val Epoch 248 [ ] | Val ACC: 0.5992 | Val AUC: 0.7527 | Val MCC: 0.3340 | LR: 3.13e-06 | Patience: 50/100 | LOSS: 0.8685
   Train Epoch 249 [] | Train Loss: 0.8081900046531918
   Val Epoch 249 [ ] | Val ACC: 0.5938 | Val AUC: 0.7513 | Val MCC: 0.3246 | LR: 3.13e-06 | Patience: 51/100 | LOSS: 0.8704
   Train Epoch 250 [] | Train Loss: 0.807690872900851
   Val Epoch 250 [ ] | Val ACC: 0.5989 | Val AUC: 0.7514 | Val MCC: 0.3331 | LR: 3.13e-06 | Patience: 52/100 | LOSS: 0.8708
   Train Epoch 251 [] | Train Loss: 0.8072102802679196
   Val Epoch 251 [ ] | Val ACC: 0.5965 | Val AUC: 0.7511 | Val MCC: 0.3290 | LR: 3.13e-06 | Patience: 53/100 | LOSS: 0.8706
   Train Epoch 252 [] | Train Loss: 0.8057070476301962
   Val Epoch 252 [ ] | Val ACC: 0.5977 | Val AUC: 0.7517 | Val MCC: 0.3314 | LR: 3.13e-06 | Patience: 54/100 | LOSS: 0.8696
   Train Epoch 253 [] | Train Loss: 0.8069039204960944
   Val Epoch 253 [ ] | Val ACC: 0.6004 | Val AUC: 0.7513 | Val MCC: 0.3360 | LR: 3.13e-06 | Patience: 55/100 | LOSS: 0.8704
   Train Epoch 254 [] | Train Loss: 0.8044402999479223
   Val Epoch 254 [ ] | Val ACC: 0.5992 | Val AUC: 0.7505 | Val MCC: 0.3340 | LR: 3.13e-06 | Patience: 56/100 | LOSS: 0.8707
   Train Epoch 255 [] | Train Loss: 0.8073338453146972
   Val Epoch 255 [ ] | Val ACC: 0.5962 | Val AUC: 0.7511 | Val MCC: 0.3293 | LR: 3.13e-06 | Patience: 57/100 | LOSS: 0.8702
   Train Epoch 256 [] | Train Loss: 0.8104059604245214
   Val Epoch 256 [ ] | Val ACC: 0.6049 | Val AUC: 0.7513 | Val MCC: 0.3444 | LR: 3.13e-06 | Patience: 58/100 | LOSS: 0.8731
   Train Epoch 257 [] | Train Loss: 0.8091940769010633
   Val Epoch 257 [ ] | Val ACC: 0.5932 | Val AUC: 0.7521 | Val MCC: 0.3268 | LR: 3.13e-06 | Patience: 59/100 | LOSS: 0.8717
   Train Epoch 258 [] | Train Loss: 0.8071562502248928
   Val Epoch 258 [ ] | Val ACC: 0.5983 | Val AUC: 0.7511 | Val MCC: 0.3317 | LR: 3.13e-06 | Patience: 60/100 | LOSS: 0.8704
   Train Epoch 259 [] | Train Loss: 0.8096280425505528
   Val Epoch 259 [ ] | Val ACC: 0.5980 | Val AUC: 0.7513 | Val MCC: 0.3308 | LR: 3.13e-06 | Patience: 61/100 | LOSS: 0.8701
   Train Epoch 260 [] | Train Loss: 0.8073973536918697
   Val Epoch 260 [ ] | Val ACC: 0.5956 | Val AUC: 0.7505 | Val MCC: 0.3262 | LR: 3.13e-06 | Patience: 62/100 | LOSS: 0.8713
   Train Epoch 261 [] | Train Loss: 0.8080788810765742
   Val Epoch 261 [ ] | Val ACC: 0.5962 | Val AUC: 0.7520 | Val MCC: 0.3298 | LR: 1.56e-06 | Patience: 63/100 | LOSS: 0.8695
   Train Epoch 262 [] | Train Loss: 0.8066681889537086
   Val Epoch 262 [ ] | Val ACC: 0.5959 | Val AUC: 0.7516 | Val MCC: 0.3279 | LR: 1.56e-06 | Patience: 64/100 | LOSS: 0.8696
   Train Epoch 263 [] | Train Loss: 0.8099767574766333
   Val Epoch 263 [ ] | Val ACC: 0.5983 | Val AUC: 0.7515 | Val MCC: 0.3314 | LR: 1.56e-06 | Patience: 65/100 | LOSS: 0.8701
   Train Epoch 264 [] | Train Loss: 0.807320021610072
   Val Epoch 264 [ ] | Val ACC: 0.5971 | Val AUC: 0.7519 | Val MCC: 0.3322 | LR: 1.56e-06 | Patience: 66/100 | LOSS: 0.8708
   Train Epoch 265 [] | Train Loss: 0.8084488796762146
   Val Epoch 265 [ ] | Val ACC: 0.6001 | Val AUC: 0.7515 | Val MCC: 0.3339 | LR: 1.56e-06 | Patience: 67/100 | LOSS: 0.8703
   Train Epoch 266 [] | Train Loss: 0.8063637296032788
   Val Epoch 266 [ ] | Val ACC: 0.5977 | Val AUC: 0.7518 | Val MCC: 0.3304 | LR: 1.56e-06 | Patience: 68/100 | LOSS: 0.8691
   Train Epoch 267 [] | Train Loss: 0.8072418912237547
   Val Epoch 267 [ ] | Val ACC: 0.6004 | Val AUC: 0.7515 | Val MCC: 0.3352 | LR: 1.56e-06 | Patience: 69/100 | LOSS: 0.8699
   Train Epoch 268 [] | Train Loss: 0.809303901497432
   Val Epoch 268 [ ] | Val ACC: 0.5995 | Val AUC: 0.7511 | Val MCC: 0.3326 | LR: 1.56e-06 | Patience: 70/100 | LOSS: 0.8705
   Train Epoch 269 [] | Train Loss: 0.8077686877773953
   Val Epoch 269 [ ] | Val ACC: 0.5992 | Val AUC: 0.7516 | Val MCC: 0.3326 | LR: 1.56e-06 | Patience: 71/100 | LOSS: 0.8700
   Train Epoch 270 [] | Train Loss: 0.8061600538965605
   Val Epoch 270 [ ] | Val ACC: 0.5947 | Val AUC: 0.7522 | Val MCC: 0.3243 | LR: 1.56e-06 | Patience: 72/100 | LOSS: 0.8689
   Train Epoch 271 [] | Train Loss: 0.8084876591105641
   Val Epoch 271 [ ] | Val ACC: 0.5974 | Val AUC: 0.7526 | Val MCC: 0.3303 | LR: 1.56e-06 | Patience: 73/100 | LOSS: 0.8683
   Train Epoch 272 [] | Train Loss: 0.8068453106864082
   Val Epoch 272 [ ] | Val ACC: 0.5974 | Val AUC: 0.7518 | Val MCC: 0.3309 | LR: 1.56e-06 | Patience: 74/100 | LOSS: 0.8695
   Train Epoch 273 [] | Train Loss: 0.806564264863826
   Val Epoch 273 [ ] | Val ACC: 0.5953 | Val AUC: 0.7508 | Val MCC: 0.3255 | LR: 1.56e-06 | Patience: 75/100 | LOSS: 0.8706
   Train Epoch 274 [] | Train Loss: 0.8086640135716165
   Val Epoch 274 [ ] | Val ACC: 0.5965 | Val AUC: 0.7502 | Val MCC: 0.3281 | LR: 1.56e-06 | Patience: 76/100 | LOSS: 0.8713
   Train Epoch 275 [] | Train Loss: 0.8042526360817663
   Val Epoch 275 [ ] | Val ACC: 0.5986 | Val AUC: 0.7505 | Val MCC: 0.3335 | LR: 1.56e-06 | Patience: 77/100 | LOSS: 0.8713
   Train Epoch 276 [] | Train Loss: 0.8056612214000481
   Val Epoch 276 [ ] | Val ACC: 0.5977 | Val AUC: 0.7505 | Val MCC: 0.3316 | LR: 1.56e-06 | Patience: 78/100 | LOSS: 0.8713
   Train Epoch 277 [] | Train Loss: 0.8084138474211651
   Val Epoch 277 [ ] | Val ACC: 0.5983 | Val AUC: 0.7501 | Val MCC: 0.3320 | LR: 1.56e-06 | Patience: 79/100 | LOSS: 0.8718
   Train Epoch 278 [] | Train Loss: 0.8065738796883762
   Val Epoch 278 [ ] | Val ACC: 0.5944 | Val AUC: 0.7505 | Val MCC: 0.3260 | LR: 1.56e-06 | Patience: 80/100 | LOSS: 0.8715
   Train Epoch 279 [] | Train Loss: 0.8097311259425384
   Val Epoch 279 [ ] | Val ACC: 0.5965 | Val AUC: 0.7502 | Val MCC: 0.3276 | LR: 1.56e-06 | Patience: 81/100 | LOSS: 0.8717
   Train Epoch 280 [] | Train Loss: 0.8093267272413907
   Val Epoch 280 [ ] | Val ACC: 0.5968 | Val AUC: 0.7509 | Val MCC: 0.3286 | LR: 1.56e-06 | Patience: 82/100 | LOSS: 0.8709
   Train Epoch 281 [] | Train Loss: 0.8081547422092524
   Val Epoch 281 [ ] | Val ACC: 0.5992 | Val AUC: 0.7518 | Val MCC: 0.3367 | LR: 1.56e-06 | Patience: 83/100 | LOSS: 0.8701
   Train Epoch 282 [] | Train Loss: 0.8042490853056669
   Val Epoch 282 [ ] | Val ACC: 0.6013 | Val AUC: 0.7510 | Val MCC: 0.3377 | LR: 1.00e-06 | Patience: 84/100 | LOSS: 0.8707
   Train Epoch 283 [] | Train Loss: 0.8051847722246537
   Val Epoch 283 [ ] | Val ACC: 0.5989 | Val AUC: 0.7506 | Val MCC: 0.3325 | LR: 1.00e-06 | Patience: 85/100 | LOSS: 0.8713
   Train Epoch 284 [] | Train Loss: 0.8060487888251642
   Val Epoch 284 [ ] | Val ACC: 0.6001 | Val AUC: 0.7505 | Val MCC: 0.3346 | LR: 1.00e-06 | Patience: 86/100 | LOSS: 0.8716
   Train Epoch 285 [] | Train Loss: 0.8029959097411143
   Val Epoch 285 [ ] | Val ACC: 0.5977 | Val AUC: 0.7514 | Val MCC: 0.3318 | LR: 1.00e-06 | Patience: 87/100 | LOSS: 0.8706
   Train Epoch 286 [] | Train Loss: 0.8058054505270831
   Val Epoch 286 [ ] | Val ACC: 0.5968 | Val AUC: 0.7516 | Val MCC: 0.3304 | LR: 1.00e-06 | Patience: 88/100 | LOSS: 0.8702
   Train Epoch 287 [] | Train Loss: 0.8063257986363433
   Val Epoch 287 [ ] | Val ACC: 0.5974 | Val AUC: 0.7511 | Val MCC: 0.3298 | LR: 1.00e-06 | Patience: 89/100 | LOSS: 0.8702
   Train Epoch 288 [] | Train Loss: 0.8061721791831238
   Val Epoch 288 [ ] | Val ACC: 0.5974 | Val AUC: 0.7514 | Val MCC: 0.3299 | LR: 1.00e-06 | Patience: 90/100 | LOSS: 0.8698
   Train Epoch 289 [] | Train Loss: 0.8062026875495845
   Val Epoch 289 [ ] | Val ACC: 0.6010 | Val AUC: 0.7518 | Val MCC: 0.3373 | LR: 1.00e-06 | Patience: 91/100 | LOSS: 0.8696
   Train Epoch 290 [] | Train Loss: 0.8066231888109855
   Val Epoch 290 [ ] | Val ACC: 0.6037 | Val AUC: 0.7517 | Val MCC: 0.3421 | LR: 1.00e-06 | Patience: 92/100 | LOSS: 0.8698
   Train Epoch 291 [] | Train Loss: 0.8074448476605538
   Val Epoch 291 [ ] | Val ACC: 0.6025 | Val AUC: 0.7515 | Val MCC: 0.3391 | LR: 1.00e-06 | Patience: 93/100 | LOSS: 0.8700
   Train Epoch 292 [] | Train Loss: 0.8049699574176864
   Val Epoch 292 [ ] | Val ACC: 0.5980 | Val AUC: 0.7514 | Val MCC: 0.3318 | LR: 1.00e-06 | Patience: 94/100 | LOSS: 0.8704
   Train Epoch 293 [] | Train Loss: 0.8060646935441784
   Val Epoch 293 [ ] | Val ACC: 0.6010 | Val AUC: 0.7508 | Val MCC: 0.3355 | LR: 1.00e-06 | Patience: 95/100 | LOSS: 0.8706
   Train Epoch 294 [] | Train Loss: 0.8067797748761641
   Val Epoch 294 [ ] | Val ACC: 0.5980 | Val AUC: 0.7511 | Val MCC: 0.3311 | LR: 1.00e-06 | Patience: 96/100 | LOSS: 0.8700
   Train Epoch 295 [] | Train Loss: 0.8054208392532226
   Val Epoch 295 [ ] | Val ACC: 0.5989 | Val AUC: 0.7512 | Val MCC: 0.3331 | LR: 1.00e-06 | Patience: 97/100 | LOSS: 0.8700
   Train Epoch 296 [] | Train Loss: 0.8055639099456823
   Val Epoch 296 [ ] | Val ACC: 0.5986 | Val AUC: 0.7512 | Val MCC: 0.3328 | LR: 1.00e-06 | Patience: 98/100 | LOSS: 0.8702
   Train Epoch 297 [] | Train Loss: 0.8050342469250892
   Val Epoch 297 [ ] | Val ACC: 0.5992 | Val AUC: 0.7513 | Val MCC: 0.3351 | LR: 1.00e-06 | Patience: 99/100 | LOSS: 0.8700
   Train Epoch 298 [] | Train Loss: 0.806513087305908
   Val Epoch 298 [ ] | Val ACC: 0.5989 | Val AUC: 0.7511 | Val MCC: 0.3326 | LR: 1.00e-06 | Patience: 100/100 | LOSS: 0.8703
   🛑 early patience triggered！100 epoch didn't improve。
   ✨ Inner-Val 最佳点: Epoch 198 (AUC = 0.6064)
   🏆 Outer-Test 成绩 -> Macro_AUC: 0.7021 | MCC: 0.2650 | Macro_F1: 0.4710 | ACC: 0.5279
   💾 外层测试集保存至: /data/caoying/softwares/ProtSATT/new_train/result/3_lables_experiment_2500/checkpoints_2labels_lr1e-2_LambdaLR_monitorACC/ProtSATT_3Class_Sim_40/test_predictions_fold_1.csv
   🎯 独立测试集 -> Macro_AUC: 0.7353 | MCC: 0.2819 | Macro_F1: 0.4801 | ACC: 0.5553
