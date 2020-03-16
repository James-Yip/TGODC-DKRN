# # 构建镜像
# nvidia-docker build -t tgodc/dkrn_demo:1.0 .

# # 保存镜像到文件
# docker save tgodc/dkrn_demo:1.0 -o dkrn_demo.tar

# # 运行交互式容器，调试时用
# nvidia-docker run -it tgodc/dkrn_demo:1.0 /bin/bash

# 加载镜像
docker load -i dkrn_demo.tar

# 运行容器，上线时用
# NV_GPU指定使用哪张显卡
NV_GPU=1 nvidia-docker run -p 8080:8080 --name dkrn tgodc/dkrn_demo:1.0