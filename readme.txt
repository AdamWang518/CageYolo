
全部的流程
    1. 用 /training_data/patches_640x640 訓練一個可以標註箱網跟浮標的權重(忘記留檔不知道哪一個了，其實用最後的權重也可以)
    2. 將之前只有船的標註資料，加上模型預測出的箱網跟浮標
    3. 再將全部資料(/RandomPick/RandomPick_v6)訓練一次

- /weight
    - best.pt     : 最終訓練出的權重
    - best.engine : 最終訓練出的權重

- predict.py : 
    使用時需要改 :
        model_path          : yolo權重
        input_folder        : 需要預測的圖片資料夾
        output_folder       : 輸出對應圖片的預測結果，照yolo的label格式儲存
        length_threshold    : 邊界閾值(寬跟高都會檢查)，小於閾值才可能合併，因為需要繼續檢查distance_threshold，具體公式看 '邊界長度相當判斷.jpg'
        distance_threshold  : 若邊界的高度相近，檢查水平距離小於distance_threshold才合併，合併;同理寬度相近檢查垂直距離
    程式功能 :
        先將圖片切成數個640x640，再根據權重預測，並將子圖預測出標籤轉回原圖位置，最後合併靠近的船和箱網物件，浮標不會合併
        輸出四個資料夾
        - all_object            : 全部物件
        - thresholded_objects   : 信心度超過閾值的物件
        - merged_objects        : 信心度超過閾值的物件，且靠近的物件會合併
        - labels                : merged_objects 的物件框資訊，照 yolo 的標籤格式

- divide_images.py : 切 patch 的方式，使用需要改輸入 image_dir、label_dir、output_dir，會在 output_dir 下加入分割後的圖片跟標籤

- /datasets : 只有 /training_data/patches_640x640 跟 /RandomPick/RandomPick_v6 有用，其他不重要，以下是各資料夾說明

    - /training_data : 包含 '箱網' 跟 '浮標' 的訓練資料，共 69 張
        - /original_2560x1920   : 原圖
        - /patches_640x640      : 裁切過的圖

    - /RandomPick : v1只有 '船' 的標註資料(共36105張)，經過一串處理，v6是最後訓練用的資料(包含浮標、箱網)
        - /RandomPick_v1 : 原本全部的資料
        - /RandomPick_v2 : 刪除缺少標註的圖
        - /RandomPick_v3 : 圖片檔名加上日期，並合併
        - /RandomPick_v4 : 刪除已在 training_data 的圖跟標籤
        - /RandomPick_v5 : 將圖片拆訓練、驗證、測試三個資料夾
        - /RandomPick_v6 : 用  training_data 訓練出的權重，對三個資料夾的圖片加上'箱網'和'浮標'的標籤

    - /predict : 放驗證模型效果的測試集

- /runs : 最後一次訓練(train6)是 /RandomPick/RandomPick_v6 包含船、箱網、浮標的訓練

- main.ipynb : 不想重複敲指令用而已， 切patch、 訓練、.pt轉.engine、預測