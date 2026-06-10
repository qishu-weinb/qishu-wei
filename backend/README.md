# 后端服务

本目录是一阶段业务后端：完成注册、登录、图片上传、历史记录、个人统计和健康知识接口。

当前仓库没有 AI 模型文件或外部推理服务配置，因此上传图片后会保存记录并返回 `code=3001`，表示“AI模型未配置，暂不能生成诊断结果”。后端不会返回随机或固定诊断结果。

## 启动

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

服务地址：`http://localhost:5000/api`

