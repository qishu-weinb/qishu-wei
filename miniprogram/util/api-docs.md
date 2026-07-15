# 医疗AI小程序 - 接口文档

## 1. 基础信息

### 1.1. 服务地址
| 环境 | 地址 |
|------|------|
| 开发环境 | `http://localhost:5000/api` |
| 测试环境 | 待确认 |
| 生产环境 | 待确认 |

### 1.2. 协议约定
- **协议**: HTTP/HTTPS
- **字符编码**: UTF-8
- **请求格式**: JSON (Content-Type: application/json)
- **响应格式**: JSON

### 1.3. 状态码说明
| HTTP状态码 | 含义 |
|-----------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 401 | 未登录或登录过期 |
| 403 | 无权限访问 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

### 1.4. 响应格式统一约定

```json
{
  "code": 0,
  "message": "success",
  "data": {}
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| code | number | 业务状态码，0表示成功，非0表示失败 |
| message | string | 提示信息 |
| data | any | 业务数据 |

### 1.5. 错误码定义
| 错误码 | 含义 |
|--------|------|
| 0 | 成功 |
| 401 | 未登录或登录已过期 |
| 403 | 无权限操作 |
| 1001 | 参数校验失败 |
| 1002 | 用户不存在 |
| 1003 | 密码错误 |
| 1004 | 用户已存在 |
| 2001 | 文件上传失败 |
| 2002 | 图片格式不支持 |
| 5000 | 服务器内部错误 |

---

## 2. 认证接口

### 2.1. 用户登录

**接口地址**: `POST /api/login`

**功能描述**: 用户通过手机号和密码登录

**请求体**:
```json
{
  "phone": "13800138000",
  "password": "string"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| phone | string | 是 | 手机号 |
| password | string | 是 | 密码（建议长度6-20位） |

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "登录成功",
  "data": {
    "token": "string",
    "userInfo": {
      "id": "number",
      "phone": "string",
      "name": "string",
      "avatar": "string",
      "createdAt": "2024-01-01 00:00:00"
    }
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| token | string | 登录令牌，后续请求需携带在请求头 |
| userInfo.id | number | 用户ID |
| userInfo.phone | string | 用户手机号 |
| userInfo.name | string | 用户姓名 |
| userInfo.avatar | string | 头像URL |
| userInfo.createdAt | string | 创建时间 |

**失败响应** (code=1002):
```json
{
  "code": 1002,
  "message": "用户不存在",
  "data": null
}
```

---

### 2.2. 用户注册

**接口地址**: `POST /api/register`

**功能描述**: 用户注册新账号

**请求体**:
```json
{
  "phone": "13800138000",
  "password": "string",
  "confirmPassword": "string",
  "name": "string"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| phone | string | 是 | 手机号 |
| password | string | 是 | 密码 |
| confirmPassword | string | 是 | 确认密码 |
| name | string | 否 | 用户姓名 |

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "注册成功",
  "data": {
    "id": "number",
    "phone": "string",
    "name": "string",
    "createdAt": "2024-01-01 00:00:00"
  }
}
```

---

### 2.3. 退出登录

**接口地址**: `POST /api/logout`

**功能描述**: 用户退出登录，清除服务端session

**请求头**:
```
Authorization: Bearer <token>
```

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "退出成功",
  "data": null
}
```

---

## 3. 用户接口

### 3.1. 获取用户信息

**接口地址**: `GET /api/user/info`

**功能描述**: 获取当前登录用户的详细信息

**请求头**:
```
Authorization: Bearer <token>
```

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "number",
    "phone": "string",
    "name": "string",
    "avatar": "string",
    "createdAt": "2024-01-01 00:00:00",
    "updatedAt": "2024-01-01 00:00:00"
  }
}
```

---

### 3.2. 更新用户信息

**接口地址**: `PUT /api/user/info`

**功能描述**: 更新用户基本信息

**请求头**:
```
Authorization: Bearer <token>
```

**请求体**:
```json
{
  "name": "string",
  "avatar": "string"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | string | 否 | 用户姓名 |
| avatar | string | 否 | 头像URL |

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "更新成功",
  "data": {
    "id": "number",
    "phone": "string",
    "name": "string",
    "avatar": "string",
    "updatedAt": "2024-01-01 00:00:00"
  }
}
```

---

## 4. 诊断接口

### 4.1. 上传病理图片

**接口地址**: `POST /api/upload/image`

**功能描述**: 上传病理图片进行诊断

**请求头**:
```
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**请求体**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件（支持jpg、png格式，大小不超过5MB） |

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "上传成功",
  "data": {
    "imageId": "string",
    "url": "string",
    "fileName": "string",
    "uploadedAt": "2024-01-01 00:00:00"
  }
}
```

---

### 4.2. 获取诊断结果

**接口地址**: `GET /api/diagnosis/:imageId`

**功能描述**: 获取图片的AI诊断结果

**请求头**:
```
Authorization: Bearer <token>
```

**路径参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| imageId | string | 图片ID |

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "string",
    "imageId": "string",
    "imageUrl": "string",
    "result": "benign",
    "confidence": 0.95,
    "analysis": "string",
    "suggestion": "string",
    "diagnosedAt": "2024-01-01 00:00:00"
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| result | string | 诊断结果：benign（良性）/ malignant（恶性） |
| confidence | number | 置信度（0-1） |
| analysis | string | AI分析报告 |
| suggestion | string | 建议 |

---

## 5. 历史记录接口

### 5.1. 获取历史记录列表

**接口地址**: `GET /api/history`

**功能描述**: 获取用户的诊断历史记录列表

**请求头**:
```
Authorization: Bearer <token>
```

**查询参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| page | number | 否 | 页码，默认1 |
| size | number | 否 | 每页数量，默认10 |

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "list": [
      {
        "id": "string",
        "imageUrl": "string",
        "result": "benign",
        "confidence": 0.95,
        "createdAt": "2024-01-01 00:00:00"
      }
    ],
    "total": 100,
    "page": 1,
    "size": 10
  }
}
```

---

### 5.2. 删除单条记录

**接口地址**: `DELETE /api/history/:id`

**功能描述**: 删除指定的诊断记录

**请求头**:
```
Authorization: Bearer <token>
```

**路径参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| id | string | 记录ID |

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "删除成功",
  "data": null
}
```

---

## 6. 健康知识接口

### 6.1. 获取知识列表

**接口地址**: `GET /api/knowledge`

**功能描述**: 获取健康知识列表

**查询参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| page | number | 否 | 页码，默认1 |
| size | number | 否 | 每页数量，默认10 |
| keyword | string | 否 | 搜索关键词 |

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "list": [
      {
        "id": "string",
        "title": "string",
        "summary": "string",
        "cover": "string",
        "createdAt": "2024-01-01 00:00:00"
      }
    ],
    "total": 50,
    "page": 1,
    "size": 10
  }
}
```

---

### 6.2. 获取知识详情

**接口地址**: `GET /api/knowledge/:id`

**功能描述**: 获取健康知识详情

**路径参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| id | string | 知识ID |

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": "string",
    "title": "string",
    "content": "string",
    "cover": "string",
    "createdAt": "2024-01-01 00:00:00"
  }
}
```

---

## 7. 微信登录接口（可选）

### 7.1. 微信快捷登录

**接口地址**: `POST /api/login/wechat`

**功能描述**: 通过微信授权登录

**请求体**:
```json
{
  "code": "string"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| code | string | 是 | 微信小程序登录code |

**成功响应** (code=0):
```json
{
  "code": 0,
  "message": "登录成功",
  "data": {
    "token": "string",
    "userInfo": {
      "id": "number",
      "phone": "string",
      "name": "string",
      "avatar": "string"
    }
  }
}
```

---

## 附录：请求头说明

所有需要登录的接口，请求头需携带：
```
Authorization: Bearer <token>
```

其中 `<token>` 为登录接口返回的 token 值。