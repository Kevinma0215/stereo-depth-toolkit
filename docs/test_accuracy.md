使用方式：
拿一個平面物體（書、盒子都可以）放在相機前方，用捲尺量好距離，然後：
bashpython tools/verify_depth_accuracy.py \
  --calib outputs/calib/calib_strict.yaml \
  --left <拍的左圖> \
  --right <拍的右圖> \
  --distance <實際距離>
建議測三個距離：0.3m、0.5m、1.0m，把三組結果貼給我，我幫你判斷精度是否夠抓取用（一般抓取需要誤差 < 5%）。