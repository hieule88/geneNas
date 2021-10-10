# gene-nas
- Module evolution để cho những cái liên quan đến tiến hóa
- Network là để encode từ các gene ra thành 1 mạng RNN
- Module problem là chứa từ data đến define problem đến cách chạy pytorch các thứ
- Các file .py ở ngoài là cách chạy các thực nghiệm (các kịch bản thực nghiệm)

# Yêu cầu:
- Implement lại những module mà người ta sử dụng trong bài NASGEP  ( section III-C, III-D) (https://arxiv.org/abs/2005.07669). Trong code thì các module này sẽ ở cái file function_set.py, tuy nhiên hiện giờ chỉ có các module để tìm RNN thôi.
- 1 gene của mình sẽ có 2 main program, nhiệm vụ của 2 main program là xác định cái architecture của 2 cell normal cell và reduction cell. Sau khi tìm được xong 2 cell đó sẽ nhét vào 1 cái khung định sẵn (như figure 3 trong bài NASGEP) để thành một mạng conv net hoàn chỉnh. Các em sẽ code một model mới lo vụ này (có thể tham khảo code phần recurrent_net.py của a)
- Dựa vào framework pytorch-lightning để code cho phần image classification trên dataset CIFAR-10

# Project sử dụng Frame-work PyTorch Lightning
https://pytorch-lightning.readthedocs.io/en/latest/index.html?fbclid=IwAR3QeExQW_3NhdRZggdaFJSHn87eAmdAnzBAF2wdlmqn-1hCujKpvGjatXY

# Sử dụng toán tử, cấu trúc search của NasGep
- Nasgep: https://arxiv.org/pdf/2005.07669.pdf
- SepConv: https://viblo.asia/p/separable-convolutions-toward-realtime-object-detection-applications-aWj534bpK6m
