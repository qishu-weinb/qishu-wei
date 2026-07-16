# 后端服务

本目录是乳腺超声辅助分析后端：提供注册、登录、图片上传、模型推理、病灶掩膜保存、历史记录、个人统计和健康知识接口。

训练权重位于 `backend/models/model_busbusi_multitask_fold1.pt`。上传图片后，后端使用 EfficientNet-B0 多任务模型返回 `normal`、`benign` 或 `malignant`，同时保存模型生成的病灶掩膜。

## 启动

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

服务地址：`http://localhost:5000/api`

模型状态检查：`GET /api/model/health`。

小程序本地调试默认请求 `http://localhost:5000/api`；真机或上线时，将 `miniprogram/util/request.js` 中的地址替换为已配置 HTTPS 合法域名。

模型输出仅用于科研和辅助参考，不能替代医生诊断或病理检查。
